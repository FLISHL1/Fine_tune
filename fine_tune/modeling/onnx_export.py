import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from loguru import logger
import typer

from fine_tune.config import Config, MODELS_DIR
from fine_tune.modeling.models import create_model


class ONNXExporter:
    """Класс для экспорта PyTorch моделей в ONNX формат"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def export_model(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        input_size: Tuple[int, int, int] = (3, 224, 224),
        opset_version: int = 11,
        optimize: bool = True
    ) -> Path:
        """
        Экспортирует PyTorch модель в ONNX формат
        
        Args:
            model_path: Путь к сохраненной PyTorch модели
            output_path: Путь для сохранения ONNX модели
            input_size: Размер входного тензора (C, H, W)
            opset_version: Версия ONNX opset
            optimize: Оптимизировать ли модель после экспорта
        
        Returns:
            Путь к экспортированной ONNX модели
        """
        logger.info(f"Экспортируем модель {model_path} в ONNX формат")
        
        # Загружаем модель
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Создаем модель
        model = create_model(
            model_config=self.config.model,
            num_classes=self.config.data.num_classes,
            model_name=checkpoint.get('model_name', 'resnet50.a1_in1k')
        )
        
        # Загружаем веса
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(self.device)
        
        # Создаем пример входного тензора
        dummy_input = torch.randn(1, *input_size).to(self.device)
        
        # Экспортируем в ONNX
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.success(f"Модель успешно экспортирована в {output_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте модели: {e}")
            raise
        
        # Оптимизируем модель если нужно
        if optimize:
            self.optimize_onnx_model(output_path)
        
        # Проверяем модель
        self.verify_onnx_model(output_path, dummy_input.cpu().numpy())
        
        return output_path
    
    def optimize_onnx_model(self, onnx_path: Path):
        """Оптимизирует ONNX модель"""
        logger.info(f"Оптимизируем ONNX модель: {onnx_path}")
        
        try:
            # Загружаем модель
            model = onnx.load(str(onnx_path))
            
            # Оптимизируем
            from onnx import optimizer
            optimized_model = optimizer.optimize(model)
            
            # Сохраняем оптимизированную модель
            optimized_path = onnx_path.parent / f"{onnx_path.stem}_optimized.onnx"
            onnx.save(optimized_model, str(optimized_path))
            
            logger.success(f"Оптимизированная модель сохранена: {optimized_path}")
            
        except Exception as e:
            logger.warning(f"Не удалось оптимизировать модель: {e}")
    
    def verify_onnx_model(self, onnx_path: Path, dummy_input: np.ndarray):
        """Проверяет корректность ONNX модели"""
        logger.info(f"Проверяем ONNX модель: {onnx_path}")
        
        try:
            # Проверяем модель
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            logger.success("ONNX модель прошла проверку")
            
            # Тестируем инференс
            session = ort.InferenceSession(str(onnx_path))
            
            # Получаем имена входов и выходов
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            # Выполняем инференс
            result = session.run([output_name], {input_name: dummy_input})
            
            logger.info(f"Тестовый инференс выполнен успешно")
            logger.info(f"Размер выхода: {result[0].shape}")
            logger.info(f"Вероятности: {result[0][0]}")
            
        except Exception as e:
            logger.error(f"Ошибка при проверке ONNX модели: {e}")
            raise
    
    def compare_pytorch_onnx(
        self,
        pytorch_model_path: Path,
        onnx_model_path: Path,
        test_input: torch.Tensor,
        tolerance: float = 1e-3
    ) -> bool:
        """Сравнивает выходы PyTorch и ONNX моделей"""
        logger.info("Сравниваем выходы PyTorch и ONNX моделей")
        
        # Загружаем PyTorch модель
        checkpoint = torch.load(pytorch_model_path, map_location=self.device, weights_only=False)
        pytorch_model = create_model(
            model_config=self.config.model,
            num_classes=self.config.data.num_classes,
            model_name=checkpoint.get('model_name', 'resnet50.a1_in1k')
        )
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        pytorch_model.eval()
        pytorch_model = pytorch_model.to(self.device)
        
        # Получаем выход PyTorch модели
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input.to(self.device))
            pytorch_output = pytorch_output.cpu().numpy()
        
        # Получаем выход ONNX модели
        session = ort.InferenceSession(str(onnx_model_path))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        onnx_output = session.run([output_name], {input_name: test_input.numpy()})[0]
        
        # Сравниваем выходы
        diff = np.abs(pytorch_output - onnx_output).max()
        
        logger.info(f"Максимальная разность: {diff}")
        logger.info(f"Допустимая разность: {tolerance}")
        
        if diff < tolerance:
            logger.success("Выходы моделей совпадают в пределах допуска")
            return True
        else:
            logger.warning("Выходы моделей не совпадают")
            return False


class ONNXInference:
    """Класс для инференса с ONNX моделями"""
    
    def __init__(self, onnx_model_path: Union[str, Path]):
        self.model_path = Path(onnx_model_path)
        self.session = ort.InferenceSession(str(self.model_path))
        
        # Получаем информацию о модели
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        logger.info(f"ONNX модель загружена: {self.model_path}")
        logger.info(f"Вход: {self.input_name}, форма: {self.input_shape}")
        logger.info(f"Выход: {self.output_name}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Выполняет предсказание"""
        # Проверяем форму входных данных
        if len(input_data.shape) == 3:
            input_data = input_data[np.newaxis, ...]  # Добавляем batch dimension
        
        # Выполняем инференс
        result = self.session.run([self.output_name], {self.input_name: input_data})
        return result[0]
    
    def predict_proba(self, input_data: np.ndarray) -> np.ndarray:
        """Возвращает вероятности классов"""
        logits = self.predict(input_data)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1)
        return probabilities.numpy()
    
    def predict_class(self, input_data: np.ndarray) -> int:
        """Возвращает предсказанный класс"""
        logits = self.predict(input_data)
        return np.argmax(logits, axis=1)[0]


def export_trained_model_to_onnx(
    config: Config,
    model_name: str,
    input_size: Tuple[int, int, int] = (3, 224, 224)
) -> Path:
    """Экспортирует обученную модель в ONNX формат"""
    
    # Путь к обученной модели
    model_path = MODELS_DIR / f"{model_name}_best.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    # Путь для ONNX модели
    onnx_path = MODELS_DIR / f"{model_name}_best.onnx"
    
    # Создаем экспортер
    exporter = ONNXExporter(config)
    
    # Экспортируем модель
    exported_path = exporter.export_model(
        model_path=model_path,
        output_path=onnx_path,
        input_size=input_size
    )
    
    return exported_path


app = typer.Typer()


@app.command()
def export_model(
    model_name: str,
    config_path: Optional[Path] = None,
    input_size: str = "3,224,224",
):
    """Экспортирует обученную модель в ONNX формат"""
    
    # Парсим размер входа
    try:
        input_size_tuple = tuple(map(int, input_size.split(',')))
        if len(input_size_tuple) != 3:
            raise ValueError("Размер входа должен содержать 3 значения: C,H,W")
    except ValueError as e:
        logger.error(f"Ошибка в формате размера входа: {e}")
        return
    
    # Загружаем конфигурацию
    if config_path and config_path.exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        config = Config(**config_dict)
    else:
        config = Config()
    
    try:
        # Экспортируем модель
        onnx_path = export_trained_model_to_onnx(
            config=config,
            model_name=model_name,
            input_size=input_size_tuple
        )
        
        logger.success(f"Модель {model_name} успешно экспортирована в {onnx_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при экспорте модели: {e}")


@app.command()
def test_onnx_model(
    onnx_model_path: Path,
    test_image_path: Optional[Path] = None
):
    """Тестирует ONNX модель"""
    
    if not onnx_model_path.exists():
        logger.error(f"ONNX модель не найдена: {onnx_model_path}")
        return
    
    # Создаем инференс-объект
    inference = ONNXInference(onnx_model_path)
    
    if test_image_path and test_image_path.exists():
        # Тестируем на реальном изображении
        from PIL import Image
        import albumentations as A
        
        # Загружаем и обрабатываем изображение
        image = Image.open(test_image_path).convert('RGB')
        image = np.array(image)
        
        # Применяем трансформации
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transformed = transform(image=image)
        input_data = transformed['image'].transpose(2, 0, 1)  # HWC -> CHW
        
        # Выполняем предсказание
        probabilities = inference.predict_proba(input_data)
        predicted_class = inference.predict_class(input_data)
        
        logger.info(f"Предсказанный класс: {predicted_class}")
        logger.info(f"Вероятности: {probabilities[0]}")
        
    else:
        # Тестируем на случайных данных
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Измеряем время инференса
        import time
        start_time = time.time()
        
        for _ in range(100):
            _ = inference.predict(dummy_input)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100

        logger.info(f"Выход модели: {inference.predict(dummy_input)}")


if __name__ == "__main__":
    app()

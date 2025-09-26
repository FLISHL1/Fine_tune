import warnings
import urllib3
import gradio as gr
import numpy as np
from PIL import Image
import albumentations as A
from pathlib import Path
from typing import Optional, Tuple
import json
import typer
from loguru import logger

# Подавляем лишние предупреждения
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from fine_tune.config import Config, MODELS_DIR
from fine_tune.modeling.onnx_export import ONNXInference


class ImageClassifierApp:
    """Gradio приложение для классификации изображений"""
    
    def __init__(self, config: Config, model_path: Optional[Path] = None):
        self.config = config
        self.model = None
        self.onnx_inference = None
        
        # Настраиваем трансформации
        self.transform = A.Compose([
            A.Resize(config.data.image_size[0], config.data.image_size[1]),
            A.Normalize(mean=config.augmentation.mean, std=config.augmentation.std)
        ])
        
        # Загружаем модель
        if model_path:
            self.load_model(model_path)
        else:
            # Ищем лучшую модель автоматически
            self.find_and_load_best_model()
    
    def find_and_load_best_model(self):
        """Автоматически находит и загружает лучшую модель"""
        # Ищем файл с результатами обучения
        summary_path = MODELS_DIR / "training_summary.json"
        
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            best_model_name = summary.get('best_model')
            if best_model_name:
                # Пробуем загрузить ONNX модель
                onnx_path = MODELS_DIR / f"{best_model_name}_best.onnx"
                if onnx_path.exists():
                    self.load_onnx_model(onnx_path)
                    return
        logger.warning("Не удалось найти обученную модель")
    
    def load_model(self, model_path: Path):
        """Загружает модель из указанного пути"""
        if model_path.suffix == '.onnx':
            self.load_onnx_model(model_path)

    def load_onnx_model(self, onnx_path: Path):
        """Загружает ONNX модель"""
        try:
            self.onnx_inference = ONNXInference(onnx_path)
            logger.info(f"ONNX модель загружена: {onnx_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки ONNX модели: {e}")
            raise

    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Предобрабатывает изображение для модели"""
        # Конвертируем в numpy array
        image_array = np.array(image.convert('RGB'))
        
        # Применяем трансформации
        transformed = self.transform(image=image_array)
        processed_image = transformed['image']
        
        # Конвертируем в формат CHW
        processed_image = processed_image.transpose(2, 0, 1)
        
        return processed_image
    
    def predict(self, image: Image.Image) -> Tuple[str, dict]:
        """Выполняет предсказание на изображении"""
        if image is None:
            return "Ошибка: изображение не загружено", {}
        
        try:
            # Предобрабатываем изображение
            processed_image = self.preprocess_image(image)
            
            # Используем ONNX модель
            probabilities = self.onnx_inference.predict_proba(processed_image)
            predicted_class_idx = self.onnx_inference.predict_class(processed_image)
            
            # Получаем предсказанный класс
            predicted_class = self.config.data.class_names[predicted_class_idx]
            
            # Создаем словарь с вероятностями для всех классов (для gr.Label)
            class_probabilities = {}
            for i, class_name in enumerate(self.config.data.class_names):
                class_probabilities[class_name] = float(probabilities[0][i])
            
            return predicted_class, class_probabilities
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            return f"Ошибка: {str(e)}", {}
    
    def create_interface(self) -> gr.Blocks:
        """Создает интерфейс Gradio"""
        
        # Создаем примеры изображений
        examples = []
        for class_name in self.config.data.class_names:
            class_dir = self.config.data.dataset_path / class_name
            if class_dir.exists():
                for img_path in list(class_dir.glob("*.jpg"))[:3]:  # Берем первые 3 изображения
                    examples.append([str(img_path)])
        
        with gr.Blocks(
            title="Классификатор изображений",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 800px !important;
                margin: auto !important;
            }
            """
        ) as interface:
            
            gr.Markdown(
                """
                # 🖼️ Классификатор изображений
                
                Загрузите изображение для классификации. Модель определит, к какому классу оно относится.
                
                **Доступные классы:**
                """ + ", ".join([f"`{name}`" for name in self.config.data.class_names])
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Входное изображение
                    input_image = gr.Image(
                        type="pil",
                        label="Загрузите изображение",
                        height=300
                    )
                    
                    # Кнопка предсказания
                    predict_btn = gr.Button(
                        "🔍 Классифицировать",
                        variant="primary",
                        size="lg"
                    )
                    
                    # Примеры
                    if examples:
                        gr.Examples(
                            examples=examples,
                            inputs=input_image,
                            label="Примеры изображений"
                        )
                
                with gr.Column(scale=1):
                    # Результат предсказания
                    prediction_output = gr.Textbox(
                        label="Предсказанный класс",
                        interactive=False,
                        value="Загрузите изображение для классификации"
                    )
                    
                    # Вероятности классов
                    probabilities_output = gr.Label(
                        label="Вероятности классов",
                        num_top_classes=len(self.config.data.class_names)
                    )
            
            # Обработчик события
            predict_btn.click(
                fn=self.predict,
                inputs=input_image,
                outputs=[prediction_output, probabilities_output]
            )
            
            # Автоматическое предсказание при загрузке изображения
            input_image.change(
                fn=self.predict,
                inputs=input_image,
                outputs=[prediction_output, probabilities_output]
            )
            
            # Информация о модели
            with gr.Accordion("ℹ️ Информация о модели", open=False):
                gr.Markdown(f"""
                **Тип модели:** ONNX  
                **Путь к модели:** {self.onnx_inference.model_path}  
                **Размер входа:** {self.onnx_inference.input_shape}  
                **Количество классов:** {len(self.config.data.class_names)}
                """)
        return interface
    
    def launch(
        self, 
        share: bool = False, 
        server_name: str = "127.0.0.1", 
        server_port: int = 7860,
        **kwargs
    ):
        """Запускает Gradio приложение"""
        interface = self.create_interface()
        
        logger.info(f"Запускаем Gradio приложение на {server_name}:{server_port}")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            **kwargs
        )


def create_app_from_config(
    config_path: Optional[Path] = None,
    model_path: Optional[Path] = None
) -> ImageClassifierApp:
    """Создает приложение из конфигурации"""
    
    # Загружаем конфигурацию
    if config_path and config_path.exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        config = Config(**config_dict)
    else:
        config = Config()
    
    # Создаем приложение
    app = ImageClassifierApp(config, model_path)
    
    return app


# CLI интерфейс
app_cli = typer.Typer()


@app_cli.command()
def launch_app(
    config_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    share: bool = False,
    port: int = 7860,
    host: str = "127.0.0.1"
):
    """Запускает Gradio приложение для классификации изображений"""
    
    try:
        # Создаем приложение
        app = create_app_from_config(config_path, model_path)
        
        # Запускаем
        app.launch(
            share=share,
            server_name=host,
            server_port=port
        )
        
    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {e}")
        raise


if __name__ == "__main__":
    app_cli()

import json
import time
import warnings
import urllib3
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
import typer
import yaml


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Настраиваем PyTorch для подавления предупреждений
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

from fine_tune.config import (
    Config, 
    MODELS_DIR,
    set_seed
)
from fine_tune.dataset import create_data_loaders
from fine_tune.modeling.models import create_model, print_model_summary
from fine_tune.modeling.metrics import (
    MetricsCalculator, 
    EarlyStopping, 
    LearningCurveTracker,
    ModelEvaluator,
)

app = typer.Typer()


class Trainer:
    """Основной класс для обучения моделей"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {self.device}")
        
        # Устанавливаем seed для воспроизводимости
        set_seed(config.experiment.seed)
        
        # Инициализируем компоненты
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        
        # Метрики и отслеживание
        self.metrics_calculator = MetricsCalculator(
            config.data.num_classes, 
            config.data.class_names
        )
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.min_delta
        )
        self.learning_curve_tracker = LearningCurveTracker()
        
        # История обучения
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.training_history = []
    
    def setup_model(self, model_name: str) -> nn.Module:
        """Настраивает модель"""
        logger.info(f"Создаем модель: {model_name}")
        
        self.model = create_model(
            model_config=self.config.model,
            num_classes=self.config.data.num_classes,
            model_name=model_name
        )
        
        self.model = self.model.to(self.device)
        print_model_summary(self.model)
        
        return self.model
    
    def setup_optimizer_and_scheduler(self):
        """Настраивает оптимизатор и планировщик"""
        # Оптимизатор
        if self.config.training.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Неизвестный оптимизатор: {self.config.training.optimizer}")
        
        # Планировщик
        if self.config.training.scheduler.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.training.epochs
            )
        elif self.config.training.scheduler.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma
            )
        elif self.config.training.scheduler.lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.training.gamma,
                patience=self.config.training.patience
            )
        
        # Критерий потерь
        self.criterion = nn.CrossEntropyLoss()
        
        
        logger.info(f"Оптимизатор: {self.config.training.optimizer}")
        logger.info(f"Планировщик: {self.config.training.scheduler}")
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Обучает модель одну эпоху"""
        self.model.train()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Обновляем метрики
            with torch.no_grad():
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                self.metrics_calculator.update(predictions, target, probabilities)
            
            total_loss += loss.item()
            
            # Обновляем прогресс-бар
            if batch_idx % self.config.experiment.log_interval == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        # Вычисляем средние метрики
        avg_loss = total_loss / num_batches
        metrics = self.metrics_calculator.compute_metrics()
        accuracy = metrics.get('accuracy', 0.0)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader) -> Tuple[float, float]:
        """Валидирует модель одну эпоху"""
        self.model.eval()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Обновляем метрики
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                self.metrics_calculator.update(predictions, target, probabilities)
                
                total_loss += loss.item()
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Вычисляем средние метрики
        avg_loss = total_loss / num_batches
        metrics = self.metrics_calculator.compute_metrics()
        accuracy = metrics.get('accuracy', 0.0)
        
        return avg_loss, accuracy
    
    def train(
        self, 
        train_loader, 
        val_loader, 
        model_name: str
    ) -> Dict:
        """Основной цикл обучения"""
        logger.info(f"Начинаем обучение модели {model_name}")
        logger.info(f"Эпох: {self.config.training.epochs}")
        logger.info(f"Размер батча: {self.config.data.batch_size}")
        logger.info(f"Learning rate: {self.config.training.learning_rate}")
        

        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            epoch_start = time.time()
            
            # Обучение
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Валидация
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Обновляем планировщик
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Получаем текущий learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Обновляем кривые обучения
            self.learning_curve_tracker.update(
                epoch, train_loss, val_loss, train_acc, val_acc, current_lr
            )
            
            # Сохраняем лучшую модель
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
            
            # Логирование
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                f"LR: {current_lr:.6f} - Time: {epoch_time:.2f}s"
            )
            
            # Сохраняем историю
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'lr': current_lr
            })
            
            # Ранняя остановка
            if self.early_stopping(val_acc):
                logger.info(f"Ранняя остановка на эпохе {epoch+1}")
                break
            
            # Сохранение модели
            if (epoch + 1) % self.config.experiment.save_interval == 0:
                self.save_checkpoint(epoch, model_name)
        
        total_time = time.time() - start_time
        logger.info(f"Обучение завершено за {total_time:.2f} секунд")
        logger.info(f"Лучшая точность валидации: {self.best_val_acc:.4f}")
        
        # Загружаем лучшую модель
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'best_val_acc': self.best_val_acc,
            'training_time': total_time,
            'total_epochs': len(self.training_history),
            'history': self.training_history
        }
    
    def save_checkpoint(self, epoch: int, model_name: str):
        """Сохраняет чекпоинт модели"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'model_name': model_name
        }
        
        checkpoint_path = MODELS_DIR / f"{model_name}_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Чекпоинт сохранен: {checkpoint_path}")
    
    def save_best_model(self, model_name: str):
        """Сохраняет лучшую модель"""
        model_path = MODELS_DIR / f"{model_name}_best.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'model_name': model_name,
            'best_val_acc': self.best_val_acc
        }, model_path)
        logger.info(f"Лучшая модель сохранена: {model_path}")
        
        # Сохраняем кривые обучения
        curves_path = MODELS_DIR / f"{model_name}_learning_curves.png"
        self.learning_curve_tracker.plot_curves(str(curves_path))
        
        return model_path


def train_single_model(
    config: Config,
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
) -> Dict:
    """Обучает одну модель"""
    logger.info(f"Обучаем модель: {model_name}")
    
    # Создаем тренер
    trainer = Trainer(config)
    
    # Настраиваем модель
    trainer.setup_model(model_name)
    trainer.setup_optimizer_and_scheduler()
    
    # Обучаем
    training_results = trainer.train(train_loader, val_loader, model_name)
    
    # Сохраняем лучшую модель
    model_path = trainer.save_best_model(model_name)
    
    # Оцениваем на тестовом наборе
    evaluator = ModelEvaluator(trainer.model, trainer.device, config.data.class_names)
    test_metrics = evaluator.evaluate(test_loader, trainer.criterion)
    
    # Сохраняем результаты
    results = {
        'model_name': model_name,
        'model_path': str(model_path),
        'training_results': training_results,
        'test_metrics': test_metrics,
        'config': config
    }
    
    # Сохраняем в JSON
    results_path = MODELS_DIR / f"{model_name}_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Результаты сохранены: {results_path}")
    
    return results


@app.command()
def main(
    config_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    freezing_strategy: Optional[str] = None,
):
    """Основная функция обучения"""
    
    # Настраиваем логирование - показываем только важные сообщения
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        filter=lambda record: record["level"].name in ["INFO", "WARNING", "ERROR"]
    )
    # Загружаем конфигурацию
    if config_path and config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        config = Config(**config_dict)
    else:
        config = Config()
    
    # Переопределяем модель если указана
    if model_name:
        config.model.model_names = [model_name]
    
    logger.info("Конфигурация загружена")
    logger.info(f"Модели для обучения: {config.model.model_names}")
    
    # Создаем DataLoader'ы
    train_loader, val_loader, test_loader = create_data_loaders(
        config.data, 
        config.augmentation, 
        config.experiment.seed
    )
    
    # Обучаем каждую модель
    all_results = []
    
    for model_name in config.model.model_names:
        try:
            results = train_single_model(
                config=config,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"Ошибка при обучении модели {model_name}: {e}")
            continue
    
    # Анализируем результаты
    if all_results:
        logger.info("=" * 50)
        logger.info("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
        logger.info("=" * 50)
        
        best_model = max(all_results, key=lambda x: x['test_metrics']['accuracy'])
        
        for results in all_results:
            model_name = results['model_name']
            test_acc = results['test_metrics']['accuracy']
            logger.info(f"{model_name}: Test Accuracy = {test_acc:.4f}")
        
        logger.info(f"Лучшая модель: {best_model['model_name']} (Accuracy: {best_model['test_metrics']['accuracy']:.4f})")
        
        # Сохраняем сводку
        summary = {
            'best_model': best_model['model_name'],
            'best_accuracy': best_model['test_metrics']['accuracy'],
            'all_results': all_results
        }
        
        summary_path = MODELS_DIR / "training_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Сводка сохранена: {summary_path}")
    
    logger.success("Обучение завершено!")


if __name__ == "__main__":
    app()

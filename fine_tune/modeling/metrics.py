import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

class MetricsCalculator:
    """Калькулятор метрик для классификации"""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Сбрасывает накопленные метрики"""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, probabilities: Optional[torch.Tensor] = None):
        """Обновляет накопленные метрики"""
        # Конвертируем в numpy
        preds = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        self.all_predictions.extend(preds)
        self.all_targets.extend(targets)
        
        if probabilities is not None:
            probs = probabilities.cpu().numpy()
            self.all_probabilities.extend(probs)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Вычисляет все метрики"""
        if not self.all_predictions:
            return {}
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        # Основные метрики
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        # Метрики по классам
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Матрица ошибок
        cm = confusion_matrix(targets, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support': support
        }
        
        return metrics
    
    def get_classification_report(self) -> str:
        """Возвращает детальный отчет по классификации"""
        if not self.all_predictions:
            return "Нет данных для отчета"
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        return classification_report(
            targets, predictions, 
            target_names=self.class_names,
            zero_division=0
        )


class EarlyStopping:
    """Ранняя остановка обучения"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float) -> bool:
        """Проверяет, нужно ли остановить обучение"""
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class LearningCurveTracker:
    """Отслеживание кривых обучения"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Сбрасывает историю"""
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epochs = []
    
    def update(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float, 
        train_acc: float, 
        val_acc: float,
        lr: float
    ):
        """Обновляет историю"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
    
    def plot_curves(self, save_path: Optional[str] = None):
        """Строит графики кривых обучения"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.epochs, self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.epochs, self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.epochs, self.train_accuracies, label='Train Accuracy', color='blue')
        axes[0, 1].plot(self.epochs, self.val_accuracies, label='Val Accuracy', color='red')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.epochs, self.learning_rates, label='Learning Rate', color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # Combined metrics
        axes[1, 1].plot(self.epochs, self.train_losses, label='Train Loss', color='blue', alpha=0.7)
        axes[1, 1].plot(self.epochs, self.val_losses, label='Val Loss', color='red', alpha=0.7)
        ax2 = axes[1, 1].twinx()
        ax2.plot(self.epochs, self.train_accuracies, label='Train Acc', color='blue', linestyle='--', alpha=0.7)
        ax2.plot(self.epochs, self.val_accuracies, label='Val Acc', color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Combined Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        ax2.set_ylabel('Accuracy')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Кривые обучения сохранены в {save_path}")
        
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray, 
    class_names: List[str], 
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
):
    """Строит матрицу ошибок"""
    plt.figure(figsize=(10, 8))
    
    # Нормализуем матрицу
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Создаем heatmap
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Матрица ошибок сохранена в {save_path}")
    
    plt.show()


def calculate_model_size(model: nn.Module) -> Dict[str, float]:
    """Вычисляет размер модели в различных единицах"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Примерный размер в байтах (float32)
    size_bytes = total_params * 4
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'size_mb': size_bytes / (1024 * 1024),
        'size_gb': size_bytes / (1024 * 1024 * 1024)
    }


class ModelEvaluator:
    """Комплексный оценщик модели"""
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str]):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.metrics_calculator = MetricsCalculator(len(class_names), class_names)
    
    def evaluate(
        self, 
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Оценивает модель на датасете"""
        self.model.eval()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Прямой проход
                output = self.model(data)
                loss = criterion(output, target)
                
                # Получаем предсказания
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                # Обновляем метрики
                self.metrics_calculator.update(predictions, target, probabilities)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Вычисляем средний loss
        avg_loss = total_loss / num_batches
        
        # Получаем метрики
        metrics = self.metrics_calculator.compute_metrics()
        metrics['loss'] = avg_loss
        
        return metrics
    
    def generate_report(self, save_dir: str):
        """Генерирует полный отчет по модели"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Получаем метрики
        metrics = self.metrics_calculator.compute_metrics()
        
        # Строим матрицу ошибок
        if 'confusion_matrix' in metrics:
            cm_path = save_dir / "confusion_matrix.png"
            plot_confusion_matrix(
                metrics['confusion_matrix'], 
                self.class_names,
                str(cm_path)
            )
        
        # Сохраняем отчет
        report_path = save_dir / "classification_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО КЛАССИФИКАЦИИ\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Точность: {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall: {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-score: {metrics.get('f1', 0):.4f}\n\n")
            f.write(self.metrics_calculator.get_classification_report())
        
        logger.info(f"Отчет сохранен в {save_dir}")
        
        return metrics

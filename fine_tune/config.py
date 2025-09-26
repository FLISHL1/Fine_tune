from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


@dataclass
class DataConfig:
    """Конфигурация для загрузки и обработки данных"""
    dataset_path: Path = RAW_DATA_DIR / "dataset"
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ["cow", "ironclad", "turtle"])


@dataclass
class AugmentationConfig:
    """Конфигурация для аугментации данных"""
    # Основные аугментации
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_limit: int = 15
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    saturation_limit: float = 0.2
    hue_limit: float = 0.1
    
    # Дополнительные аугментации
    blur_limit: int = 3
    noise_limit: float = 0.1
    cutout_prob: float = 0.5
    cutout_size: float = 0.1
    
    # Нормализация
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    """Конфигурация для модели"""
    # Архитектуры моделей (из разных семейств timm)
    model_names: List[str] = field(default_factory=lambda: [
        "resnet50.a1_in1k",  # ResNet семейство
        "efficientnet_b0.ra_in1k"  # EfficientNet семейство
    ])
    
    # Transfer learning настройки
    pretrained: bool = True
    freeze_backbone: bool = True  # Замораживаем backbone, обучаем только classifier


@dataclass
class TrainingConfig:
    """Конфигурация для обучения"""
    # Основные параметры
    epochs: int = 5
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    
    # Оптимизатор
    optimizer: str = "adamw"  # adam, adamw, sgd
    momentum: float = 0.9  # для SGD
    
    # Планировщик
    scheduler: str = "cosine"  # cosine, step, plateau
    step_size: int = 10  # для step scheduler
    gamma: float = 0.1  # для step scheduler
    patience: int = 5  # для plateau scheduler
    
    # Ранняя остановка
    early_stopping_patience: int = 10
    min_delta: float = 0.001
    
    # Другие параметры
    gradient_clip_norm: float = 1.0


@dataclass
class ExperimentConfig:
    """Конфигурация для экспериментов"""
    # Воспроизводимость
    seed: int = 42
    deterministic: bool = True
    
    # Логирование
    log_interval: int = 10
    save_interval: int = 5

    # Сохранение моделей
    save_best_only: bool = True
    save_last: bool = True
    
    # Метрики
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    
    # Экспорт
    export_onnx: bool = True
    onnx_export_path: Path = MODELS_DIR / "model.onnx"


@dataclass
class Config:
    """Главная конфигурация, объединяющая все остальные"""
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """Создаем необходимые директории после инициализации"""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    """Устанавливает seed для воспроизводимости"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

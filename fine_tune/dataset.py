from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import typer

from fine_tune.config import (
    PROCESSED_DATA_DIR, 
    RAW_DATA_DIR, 
    DataConfig, 
    AugmentationConfig,
    set_seed
)

app = typer.Typer()


class ImageClassificationDataset(Dataset):
    """Датасет для классификации изображений с поддержкой аугментаций"""
    
    def __init__(
        self, 
        data_dir: Path, 
        class_names: List[str],
        image_size: Tuple[int, int] = (224, 224),
        augmentations: Optional[AugmentationConfig] = None,
        is_training: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.class_names = class_names
        self.image_size = image_size
        self.is_training = is_training
        
        # Создаем маппинг классов
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        
        # Загружаем данные
        self.samples = self._load_samples()
        
        # Настраиваем аугментации
        self.transform = self._get_transforms(augmentations)
        
        logger.info(f"Загружено {len(self.samples)} образцов из {len(class_names)} классов")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Загружает список образцов (путь к изображению, класс)"""
        samples = []
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Директория {class_dir} не найдена")
                continue
                
            # Объединяем все поддерживаемые форматы изображений
            image_extensions = ["*.jpg", "*.jpeg", "*.png"]
            for extension in image_extensions:
                for img_path in class_dir.glob(extension):
                    class_idx = self.class_to_idx[class_name]
                    samples.append((img_path, class_idx))
        
        return samples
    
    def _get_transforms(self, aug_config: Optional[AugmentationConfig]) -> A.Compose:
        """Создает трансформации для аугментации данных"""
        if aug_config is None:
            # Базовые трансформации без аугментаций
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        if self.is_training:
            # Аугментации для обучения
            transforms_list = [
                A.Resize(self.image_size[0], self.image_size[1]),
            ]
            
            # Основные аугментации
            if aug_config.horizontal_flip:
                transforms_list.append(A.HorizontalFlip(p=0.5))
            if aug_config.vertical_flip:
                transforms_list.append(A.VerticalFlip(p=0.3))
            if aug_config.rotation_limit > 0:
                transforms_list.append(A.Rotate(limit=aug_config.rotation_limit, p=0.5))
            
            # Цветовые аугментации
            if any([aug_config.brightness_limit, aug_config.contrast_limit, 
                   aug_config.saturation_limit, aug_config.hue_limit]):
                transforms_list.append(A.ColorJitter(
                    brightness=aug_config.brightness_limit,
                    contrast=aug_config.contrast_limit,
                    saturation=aug_config.saturation_limit,
                    hue=aug_config.hue_limit,
                    p=0.5
                ))
            
            # Дополнительные аугментации
            if aug_config.blur_limit > 0:
                transforms_list.append(A.OneOf([
                    A.MotionBlur(blur_limit=aug_config.blur_limit, p=0.3),
                    A.GaussianBlur(blur_limit=aug_config.blur_limit, p=0.3),
                ], p=0.3))
            
            if aug_config.noise_limit > 0:
                transforms_list.append(A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.ISONoise(p=0.3),
                ], p=0.3))
            
            if aug_config.cutout_prob > 0:
                transforms_list.append(A.CoarseDropout(
                    holes=8,
                    height=int(self.image_size[0] * aug_config.cutout_size),
                    width=int(self.image_size[1] * aug_config.cutout_size),
                    p=aug_config.cutout_prob
                ))
            
            # Нормализация
            transforms_list.append(A.Normalize(mean=aug_config.mean, std=aug_config.std))
            
            return A.Compose(transforms_list)
        else:
            # Только базовые трансформации для валидации/тестирования
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=aug_config.mean, std=aug_config.std),
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_idx = self.samples[idx]
        
        # Загружаем изображение
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения {img_path}: {e}")
            # Возвращаем случайное изображение в случае ошибки
            image = np.random.randint(0, 255, (self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        
        # Применяем трансформации
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # Конвертируем в тензор
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image, class_idx


def create_data_loaders(
    data_config: DataConfig,
    aug_config: AugmentationConfig,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Создает DataLoader'ы для обучения, валидации и тестирования"""
    
    set_seed(seed)
    
    # Создаем полный датасет
    full_dataset = ImageClassificationDataset(
        data_dir=data_config.dataset_path,
        class_names=data_config.class_names,
        image_size=data_config.image_size,
        augmentations=aug_config,
        is_training=True
    )
    
    # Разделяем на train/val/test
    total_size = len(full_dataset)
    train_size = int(data_config.train_split * total_size)
    val_size = int(data_config.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Создаем отдельные датасеты для валидации и тестирования без аугментаций
    val_dataset.dataset.is_training = False
    test_dataset.dataset.is_training = False
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Созданы DataLoader'ы:")
    logger.info(f"  Train: {len(train_loader)} батчей ({len(train_dataset)} образцов)")
    logger.info(f"  Val: {len(val_loader)} батчей ({len(val_dataset)} образцов)")
    logger.info(f"  Test: {len(test_loader)} батчей ({len(test_dataset)} образцов)")
    
    return train_loader, val_loader, test_loader


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset",
    output_path: Path = PROCESSED_DATA_DIR / "dataset_info.json",
):
    """Анализирует датасет и сохраняет информацию о нем"""
    from collections import Counter
    import json
    
    logger.info("Анализируем датасет...")
    
    dataset = ImageClassificationDataset(
        data_dir=input_path,
        class_names=["cow", "ironclad", "turtle"],
        is_training=False
    )
    
    # Подсчитываем количество образцов по классам
    class_counts = Counter([dataset.samples[i][1] for i in range(len(dataset))])
    
    dataset_info = {
        "total_samples": len(dataset),
        "num_classes": len(dataset.class_names),
        "class_names": dataset.class_names,
        "class_counts": {dataset.idx_to_class[idx]: count for idx, count in class_counts.items()},
        "image_size": dataset.image_size
    }
    
    # Сохраняем информацию
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Информация о датасете сохранена в {output_path}")
    logger.info(f"Всего образцов: {dataset_info['total_samples']}")
    for class_name, count in dataset_info['class_counts'].items():
        logger.info(f"  {class_name}: {count}")


def create_dataset_structure(data_dir: Path):
    """Создает структуру папок для датасета"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ["cow", "ironclad", "turtle"]
    
    for class_name in class_names:
        class_dir = data_dir / class_name
        class_dir.mkdir(exist_ok=True)
        logger.info(f"Создана папка для класса: {class_name}")
    
    logger.success(f"Структура датасета создана в {data_dir}")


@app.command()
def create_sample_data(
    output_path: Path = RAW_DATA_DIR / "dataset"
):
    """Создает структуру папок для датасета"""
    create_dataset_structure(output_path)


if __name__ == "__main__":
    app()

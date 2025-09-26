import torch
import torch.nn as nn
from typing import Optional, Tuple
import timm
from loguru import logger

from fine_tune.config import ModelConfig


class TransferLearningModel(nn.Module):
    """Базовая модель для transfer learning с замораживанием backbone"""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Загружаем предобученную модель из timm с нужным количеством классов
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=3,  # Устанавливаем 3 класса
            global_pool='avg'  # Глобальное усреднение
        )
        
        # Замораживаем backbone если нужно
        if freeze_backbone:
            self._freeze_backbone()
            logger.info(f"Backbone модели {model_name} заморожен")
        else:
            logger.info(f"Backbone модели {model_name} разморожен для дообучения")
    
    def _freeze_backbone(self):
        """Замораживает параметры backbone (все слои кроме классификатора)"""
        # Замораживаем все параметры кроме последнего слоя (классификатора)
        for name, param in self.model.named_parameters():
            if 'classifier' not in name and 'head' not in name and 'fc' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Размораживает параметры backbone для fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        logger.info(f"Backbone модели {self.model_name} разморожен")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход"""
        # Прямой проход через всю модель
        output = self.model(x)
        return output
    
    def get_feature_extractor(self):
        """Возвращает модель без последнего слоя для извлечения признаков"""
        # Создаем копию модели без классификатора
        feature_extractor = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )
        # Копируем веса backbone
        state_dict = self.model.state_dict()
        # Удаляем веса классификатора
        feature_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k and 'head' not in k and 'fc' not in k}
        feature_extractor.load_state_dict(feature_dict, strict=False)
        return feature_extractor


class ResNetTransferModel(TransferLearningModel):
    """ResNet модель для transfer learning"""
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet50.a1_in1k",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        logger.info(f"Создана ResNet модель: {model_name}")


class EfficientNetTransferModel(TransferLearningModel):
    """EfficientNet модель для transfer learning"""
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "efficientnet_b0.ra_in1k",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        logger.info(f"Создана EfficientNet модель: {model_name}")


def create_model(
    model_config: ModelConfig,
    num_classes: int,
    model_name: Optional[str] = None
) -> TransferLearningModel:
    """Создает модель на основе конфигурации"""
    
    if model_name is None:
        model_name = model_config.model_names[0]
    
    # Определяем семейство модели и создаем соответствующую модель
    if "resnet" in model_name.lower():
        model = ResNetTransferModel(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=model_config.pretrained,
            freeze_backbone=model_config.freeze_backbone,
        )
    elif "efficientnet" in model_name.lower():
        model = EfficientNetTransferModel(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=model_config.pretrained,
            freeze_backbone=model_config.freeze_backbone,
        )
    else:
        # Универсальная модель для других архитектур
        model = TransferLearningModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=model_config.pretrained,
            freeze_backbone=model_config.freeze_backbone,
        )
    
    return model


def get_model_info(model: TransferLearningModel) -> dict:
    """Возвращает информацию о модели"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "model_name": model.model_name,
        "num_classes": model.num_classes,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "freeze_backbone": model.freeze_backbone
    }


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Подсчитывает общее количество параметров и обучаемых параметров"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
    """Выводит краткую сводку о модели"""
    total_params, trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params
    
    logger.info("=" * 50)
    logger.info("СВОДКА МОДЕЛИ")
    logger.info("=" * 50)
    logger.info(f"Архитектура: {model.model_name if hasattr(model, 'model_name') else 'Unknown'}")
    logger.info(f"Входной размер: {input_size}")
    logger.info(f"Количество классов: {model.num_classes if hasattr(model, 'num_classes') else 'Unknown'}")
    logger.info(f"Всего параметров: {total_params:,}")
    logger.info(f"Обучаемых параметров: {trainable_params:,}")
    logger.info(f"Замороженных параметров: {frozen_params:,}")
    logger.info(f"Процент обучаемых: {100 * trainable_params / total_params:.2f}%")
    logger.info("=" * 50)


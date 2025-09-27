# Fine_Tune - Transfer Learning для классификации изображений

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Проект для дообучения предварительно обученных моделей из библиотеки `timm` с использованием transfer learning для классификации изображений. Включает в себя экспериментирование с различными архитектурами, стратегиями замораживания, подбором гиперпараметров и создание веб-интерфейса для инференса.

## 🚀 Основные возможности

- **Transfer Learning**: Дообучение предварительно обученных моделей из разных семейств (ResNet, EfficientNet)
- **Стратегии замораживания**: Замораживание/размораживание слоев с различными стратегиями
- **Аугментация данных**: Комплексная система аугментаций с использованием Albumentations
- **Экспериментирование**: Jupyter ноутбук для сравнения моделей и подбора гиперпараметров
- **Экспорт в ONNX**: Конвертация моделей в ONNX формат для быстрого инференса на CPU
- **Веб-интерфейс**: Gradio приложение для интерактивной классификации изображений
- **Воспроизводимость**: Фиксированные генераторы случайных чисел и структурированная конфигурация

## 📁 Структура проекта

```
├── Makefile           <- Команды для удобной работы с проектом
├── README.md          <- Документация проекта
├── requirements.txt   <- Зависимости Python
├── pyproject.toml     <- Конфигурация проекта
│
├── data/              <- Данные проекта
│   └── raw/           <- Исходные данные
│
├── models/            <- Обученные модели и результаты
│   └── *.pth          <- Сохраненные модели PyTorch
│   └── *.onnx         <- Модели в формате ONNX
│
├── notebooks/         <- Jupyter ноутбуки
│   └── 1.0-experiments-and-model-selection.ipynb
│
│
└── fine_tune/         <- Исходный код
    ├── __init__.py
    ├── config.py      <- Конфигурация с датаклассами
    ├── dataset.py     <- Загрузка и аугментация данных
    └── modeling/      <- Модели и обучение
        ├── __init__.py
        ├── models.py      <- Архитектуры моделей
        ├── train.py       <- Скрипт обучения
        ├── metrics.py     <- Метрики и оценка
        ├── onnx_export.py <- Экспорт в ONNX
        └── gradio_app.py  <- Веб-приложение
```

## 🛠️ Установка и настройка

### 1. Клонирование репозитория
```bash
git clone <repository-url>
cd fine_tune
```

### 2. Создание виртуального окружения
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Обучение моделей
```bash

# Обучение конкретной модели
python -m fine_tune.modeling.train --model-name "resnet50.a1_in1k"

```

### 3. Экспорт модели в ONNX
```bash
python -m fine_tune.modeling.onnx_export export-model resnet50.a1_in1k
```

### 4. Запуск веб-приложения
```bash
python -m fine_tune.modeling.gradio_app launch-app
```

## 🔬 Экспериментирование

Для проведения экспериментов и выбора лучшей модели используйте Jupyter ноутбук:

```bash
jupyter lab notebooks/1.0-experiments-and-model-selection.ipynb
```

## ⚙️ Конфигурация

Проект использует структурированную конфигурацию через датаклассы Python:

```python
from fine_tune.config import Config

config = Config()
config.data.batch_size = 32
config.training.learning_rate = 0.001
config.model.model_names = ["resnet50.a1_in1k", "efficientnet_b0.ra_in1k"]
```

### Основные параметры конфигурации:

- **DataConfig**: Настройки данных (размер изображений, размер батча, разделение на train/val/test)
- **AugmentationConfig**: Параметры аугментации данных
- **ModelConfig**: Настройки моделей (архитектуры, transfer learning)
- **TrainingConfig**: Параметры обучения (оптимизатор, планировщик, количество эпох)
- **ExperimentConfig**: Настройки экспериментов (seed, логирование, сохранение)

## 🏗️ Архитектура моделей

Проект поддерживает модели из разных семейств:

### ResNet семейство
- `resnet50.a1_in1k`

### EfficientNet семейство
- `efficientnet_b0.ra_in1k`

## 📈 Метрики и оценка

Проект включает комплексную систему оценки:

- **Accuracy**: Общая точность классификации
- **Precision, Recall, F1-score**: Метрики по классам
- **Confusion Matrix**: Матрица ошибок
- **Learning Curves**: Кривые обучения
- **Model Size**: Размер модели и количество параметров

## 🌐 Веб-интерфейс

Gradio приложение предоставляет интерактивный интерфейс для:
- Загрузки изображений
- Классификации в реальном времени
- Отображения вероятностей классов
- Примеров изображений

## 🔧 Команды Makefile

```bash
make install          # Установка зависимостей
make train            # Обучение моделей
make export-onnx-resnet      # Экспорт в ONNX resnet
make export-onnx-efficientnet      # Экспорт в ONNX efficientnet
make launch-app       # Запуск веб-приложения
```
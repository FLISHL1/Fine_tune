.PHONY: help install train export-onnx launch-app

# Переменные
PYTHON := python3
PIP := pip3
PROJECT_NAME := fine_tune

# Цвета для вывода
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Показать справку по командам
	@echo "$(GREEN)Доступные команды:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Установить зависимости
	@echo "$(GREEN)Устанавливаем зависимости...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)Зависимости установлены!$(NC)"


create-data: ## Создать пример датасета
	@echo "$(GREEN)Создаем пример датасета...$(NC)"
	$(PYTHON) -m $(PROJECT_NAME).dataset create-sample-data --num-samples 100
	@echo "$(GREEN)Пример датасета создан!$(NC)"

train: ## Обучить модели
	@echo "$(GREEN)Начинаем обучение моделей...$(NC)"
	$(PYTHON) -m $(PROJECT_NAME).modeling.train 
	@echo "$(GREEN)Обучение завершено!$(NC)"

train-resnet: ## Обучить ResNet модель
	@echo "$(GREEN)Обучаем ResNet модель...$(NC)"
	$(PYTHON) -m $(PROJECT_NAME).modeling.train --model-name "resnet50.a1_in1k" 
	@echo "$(GREEN)ResNet модель обучена!$(NC)"

train-efficientnet: ## Обучить EfficientNet модель
	@echo "$(GREEN)Обучаем EfficientNet модель...$(NC)"
	$(PYTHON) -m $(PROJECT_NAME).modeling.train --model-name "efficientnet_b0.ra_in1k"
	@echo "$(GREEN)EfficientNet модель обучена!$(NC)"



export-onnx-resnet: ## Экспортировать лучшую модель в ONNX
	@echo "$(GREEN)Экспортируем модель в ONNX...$(NC)"
	$(PYTHON) -m $(PROJECT_NAME).modeling.onnx_export export-model resnet50.a1_in1k
	@echo "$(GREEN)Модель экспортирована в ONNX!$(NC)"

export-onnx-efficientnet: ## Экспортировать лучшую модель в ONNX
	@echo "$(GREEN)Экспортируем модель в ONNX...$(NC)"
	$(PYTHON) -m $(PROJECT_NAME).modeling.onnx_export export-model efficientnet_b0.ra_in1k
	@echo "$(GREEN)Модель экспортирована в ONNX!$(NC)"

test-onnx: ## Тестировать ONNX модель
	@echo "$(GREEN)Тестируем ONNX модель...$(NC)"
	$(PYTHON) -m $(PROJECT_NAME).modeling.onnx_export test-onnx models/resnet50.a1_in1k_best.onnx
	@echo "$(GREEN)Тестирование ONNX модели завершено!$(NC)"

launch-app: ## Запустить веб-приложение
	@echo "$(GREEN)Запускаем веб-приложение...$(NC)"
	$(PYTHON) -m $(PROJECT_NAME).modeling.gradio_app
	@echo "$(GREEN)Веб-приложение запущено!$(NC)"

status: ## Показать статус проекта
	@echo "$(GREEN)Статус проекта:$(NC)"
	@echo ""
	@echo "$(YELLOW)Структура данных:$(NC)"
	@if [ -d "data/raw/dataset" ]; then \
		echo "  ✓ Датасет существует"; \
		for class_dir in data/raw/dataset/*/; do \
			if [ -d "$$class_dir" ]; then \
				class_name=$$(basename "$$class_dir"); \
				count=$$(find "$$class_dir" -name "*.jpg" | wc -l); \
				echo "    - $$class_name: $$count изображений"; \
			fi; \
		done; \
	else \
		echo "  ✗ Датасет не найден"; \
	fi
	@echo ""
	@echo "$(YELLOW)Обученные модели:$(NC)"
	@if [ -d "models" ]; then \
		pth_count=$$(find models -name "*.pth" | wc -l); \
		onnx_count=$$(find models -name "*.onnx" | wc -l); \
		echo "  - PyTorch модели: $$pth_count"; \
		echo "  - ONNX модели: $$onnx_count"; \
	else \
		echo "  ✗ Папка models не найдена"; \
	fi
	@echo ""
	@echo "$(YELLOW)Зависимости:$(NC)"
	@$(PYTHON) -c "import torch; print(f'  - PyTorch: {torch.__version__}')" 2>/dev/null || echo "  ✗ PyTorch не установлен"
	@$(PYTHON) -c "import timm; print(f'  - timm: {timm.__version__}')" 2>/dev/null || echo "  ✗ timm не установлен"
	@$(PYTHON) -c "import gradio; print(f'  - Gradio: {gradio.__version__}')" 2>/dev/null || echo "  ✗ Gradio не установлен"

# Команда по умолчанию
.DEFAULT_GOAL := help
# ============================================================================
# DeepLearning-Phoneme Makefile
# ============================================================================
# Uso: make <comando>
# Per Windows, usare con Git Bash o WSL, oppure usare i comandi Python direttamente
# ============================================================================

.PHONY: help install setup scrape build-dataset preprocess train evaluate inference clean clean-all test lint

# Variabili
PYTHON := python
CONFIG := configs/training_config.yaml
MODEL_DIR := outputs/final_model
OUTPUT_DIR := outputs
DATA_DIR := data
AUDIO ?= ""

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "============================================"
	@echo "  DeepLearning-Phoneme - Comandi Make"
	@echo "============================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Installa dipendenze"
	@echo "  make setup            Setup completo (install + dirs)"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make build-dataset    Crea CSV metadata"
	@echo "  make preprocess       Preprocessa dataset e crea vocab"
	@echo "  make data-pipeline    Esegue tutto il data pipeline"
	@echo ""
	@echo "Data Augmentation:"
	@echo "  make augment          Augmentation completa (acoustic + TTS)"
	@echo "  make augment-acoustic Solo augmentation acustica (no TTS)"
	@echo ""
	@echo "Training:"
	@echo "  make train            Avvia training"
	@echo "  make train-augmented  Training su dataset augmentato"
	@echo "  make train-debug      Training in modalita debug"
	@echo ""
	@echo "Evaluation & Inference:"
	@echo "  make evaluate         Valuta modello su test set"
	@echo "  make evaluate-so      Valuta su SpeechOcean762"
	@echo "  make inference AUDIO=path  Inferenza su file"
	@echo ""
	@echo "Utility:"
	@echo "  make clean            Pulisce file temporanei"
	@echo "  make clean-all        Pulisce tutto (inclusi modelli)"
	@echo "  make info             Mostra info dataset"
	@echo ""

# ============================================================================
# SETUP
# ============================================================================

install:
	@echo ">>> Installazione dipendenze..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo ">>> Dipendenze installate!"

setup: install
	@echo ">>> Creazione struttura directory..."
	@mkdir -p $(DATA_DIR)/raw/phonemeref_data
	@mkdir -p $(DATA_DIR)/processed
	@mkdir -p $(DATA_DIR)/wordlists
	@mkdir -p $(OUTPUT_DIR)
	@mkdir -p configs logs tests
	@echo ">>> Setup completato!"

# ============================================================================
# DATA PIPELINE
# ============================================================================

build-dataset:
	@echo ">>> Costruzione dataset CSV..."
	$(PYTHON) -m scripts.01_build_dataset \
		--data-dir $(DATA_DIR)/raw/phonemeref_data \
		--output $(DATA_DIR)/processed/phonemeref_metadata.csv

preprocess:
	@echo ">>> Preprocessing e creazione vocabolario..."
	$(PYTHON) -m scripts.02_preprocess \
		--input $(DATA_DIR)/processed/phonemeref_metadata.csv \
		--output-csv $(DATA_DIR)/processed/phonemeref_processed.csv \
		--output-vocab $(DATA_DIR)/processed/vocab.json

data-pipeline: build-dataset preprocess
	@echo ">>> Data pipeline completato!"

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

augment:
	@echo ">>> Avvio augmentation dataset..."
	$(PYTHON) -m scripts.build_augmented_dataset \
		--input $(DATA_DIR)/processed/phonemeref_processed.csv \
		--output $(DATA_DIR)/processed/phonemeref_augmented.csv \
		--output-dir $(DATA_DIR)/augmented

augment-acoustic:
	@echo ">>> Augmentation solo acustica (no TTS)..."
	$(PYTHON) -m scripts.build_augmented_dataset \
		--input $(DATA_DIR)/processed/phonemeref_processed.csv \
		--output $(DATA_DIR)/processed/phonemeref_augmented.csv \
		--output-dir $(DATA_DIR)/augmented \
		--no-tts

train-augmented:
	@echo ">>> Training su dataset augmentato..."
	$(PYTHON) scripts/03_train.py \
		--config $(CONFIG) \
		--data-csv $(DATA_DIR)/processed/phonemeref_augmented.csv

# ============================================================================
# TRAINING
# ============================================================================

train:
	@echo ">>> Avvio training..."
	$(PYTHON) scripts/03_train.py --config $(CONFIG)

train-baseline:
	@echo ">>> Training baseline (solo dati originali)..."
	$(PYTHON) scripts/03_train.py --config configs/training_config_baseline.yaml

train-debug:
	@echo ">>> Training in modalita debug..."
	$(PYTHON) scripts/03_train.py --config $(CONFIG) --debug

train-resume:
	@echo ">>> Ripresa training da checkpoint..."
	$(PYTHON) scripts/03_train.py --config $(CONFIG) --resume

# ============================================================================
# EVALUATION & INFERENCE
# ============================================================================

evaluate:
	@echo ">>> Valutazione modello..."
	$(PYTHON) scripts/04_evaluate.py \
		--model-path $(MODEL_DIR) \
		--test-csv $(DATA_DIR)/processed/phonemeref_augmented.csv \
		--audio-base .

evaluate-so:
	@echo ">>> Valutazione su SpeechOcean762..."
	$(PYTHON) scripts/05_evaluate_speechocean.py \
		--model-path $(MODEL_DIR)

inference:
	@echo ">>> Inferenza su $(AUDIO)..."
	$(PYTHON) scripts/04_evaluate.py \
		--model-path $(MODEL_DIR) \
		--audio $(AUDIO)

inference-interactive:
	@echo ">>> Modalita inferenza interattiva..."
	$(PYTHON) scripts/04_evaluate.py \
		--model-path $(MODEL_DIR) \
		--interactive

# ============================================================================
# UTILITY
# ============================================================================

info:
	@echo ">>> Informazioni dataset..."
	@if [ -f $(DATA_DIR)/processed/phonemeref_processed.csv ]; then \
		echo "Righe totali: $$(wc -l < $(DATA_DIR)/processed/phonemeref_processed.csv)"; \
	else \
		echo "Dataset non ancora creato. Esegui 'make data-pipeline'"; \
	fi

clean:
	@echo ">>> Pulizia file temporanei..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	rm -rf logs/*.log 2>/dev/null || true
	@echo ">>> Pulizia completata!"

clean-all: clean
	@echo ">>> Pulizia completa (inclusi modelli)..."
	rm -rf $(OUTPUT_DIR)/* 2>/dev/null || true
	@echo ">>> Pulizia completa!"

# 🎙️ DeepLearning-Phoneme

**Riconoscimento fonetico IPA da audio con architettura Ensemble (WavLM + XLS-R)**

Sistema di Automatic Phoneme Recognition (APR) e Pronunciation Scoring per speaker non-nativi, basato su ensemble di modelli WavLM e XLS-R con Late Fusion.

## 📋 Indice

- [Features](#features)
- [Requisiti](#requisiti)
- [Installazione](#installazione)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Struttura Progetto](#struttura-progetto)
- [Benchmark](#benchmark)
- [Ensemble Architecture](#ensemble-architecture)
- [Colab Training](#colab-training)

## ✨ Features

- **Riconoscimento fonemi IPA** da audio inglese
- **Ensemble SOTA** con WavLM + XLS-R e Late Fusion
- **Weighted Layer Sum** per combinazione ottimale dei layer
- **Benchmark scientifico** su SpeechOcean762 (speaker non-nativi)
- **3 Task di valutazione**: ASR Robustness, Scoring Correlation, Mispronunciation Detection

## Requisiti

- Python 3.9+
- CUDA 11.8+ (opzionale, per GPU)
- ~10GB spazio disco per dataset
- ~8GB VRAM (WavLM) / ~12GB VRAM (XLS-R)

## Installazione

```bash
# 1. Clona repository
git clone https://github.com/maurocarlu/DeepLearning-Phoneme.git
cd DeepLearning-Phoneme

# 2. Crea ambiente virtuale
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure: venv\Scripts\activate  # Windows

# 3. Installa dipendenze
pip install -r requirements.txt
```

## Quick Start

```bash
# Valutazione modello su SpeechOcean762
python scripts/05_evaluate_speechocean.py --model-path outputs/final_model

# Inferenza su file audio
python scripts/04_evaluate.py --model-path outputs/final_model --audio audio.mp3
```

## Workflow

### 1. Preparazione Dati

```bash
# Costruisci dataset da WordReference
python scripts/01_build_dataset.py --data-dir data/raw/phonemeref_data

# Preprocessa e crea vocabolario IPA
python scripts/02_preprocess.py

# Aggiungi SpeechOcean e augmentation
python scripts/build_combined_dataset.py --min-score 8
```

### 2. Training

```bash
# Training WavLM standard
python scripts/03_train.py --config configs/training_config.yaml

# Training WavLM Weighted (Ensemble Model A)
python scripts/train_weighted.py --config configs/training_config.yaml

# Training XLS-R (Ensemble Model B)
python scripts/train_xlsr.py --config configs/training_config.yaml
```

### 3. Valutazione

```bash
# Benchmark su SpeechOcean762 (3 task)
python scripts/05_evaluate_speechocean.py --model-path outputs/final_model

# Late Fusion Ensemble
python scripts/evaluate_fusion.py \
    --model-a outputs/wavlm_weighted \
    --model-b outputs/xlsr \
    --weight 0.6
```

## Struttura Progetto

```
DeepLearning-Phoneme/
├── configs/                    # Configurazioni YAML
│   └── training_config.yaml
├── data/
│   ├── raw/                    # Audio + JSON scaricati
│   ├── processed/              # CSV processati
│   │   ├── combined_augmented.csv  # Dataset principale
│   │   └── vocab.json              # Vocabolario IPA
│   └── speechocean/            # Audio SpeechOcean
├── outputs/                    # Modelli salvati
├── scripts/                    # Script eseguibili
│   ├── 01_build_dataset.py     # Costruzione dataset
│   ├── 02_preprocess.py        # Preprocessing IPA
│   ├── 03_train.py             # Training standard
│   ├── 04_evaluate.py          # Valutazione interna
│   ├── 05_evaluate_speechocean.py  # Benchmark SpeechOcean
│   ├── train_weighted.py       # Training WavLM Weighted
│   ├── train_xlsr.py           # Training XLS-R
│   └── evaluate_fusion.py      # Late Fusion eval
├── src/                        # Moduli Python
│   ├── data/                   # Gestione dati e IPA
│   ├── training/               # Training loop
│   └── inference/              # Inferenza
├── tests/                      # Unit tests
├── colab_train_augmented.ipynb     # Notebook Colab
├── colab_train_wavlm_weighted.ipynb
├── colab_train_xlsr.ipynb
├── ENSEMBLE_GUIDE.md           # Guida Ensemble
└── README.md
```

## Benchmark

### SpeechOcean762 (Speaker Non-Nativi)

Il benchmark valuta 3 task su 2500 samples con punteggi umani (1-10):

| Task | Metrica | Valore |
|------|---------|--------
| **A - ASR Robustness** | PER (score ≥8) | ~15% |
| **B - Scoring Correlation** | Spearman ρ | ~0.51 |
| **C - Mispronunciation Detection** | AUC-ROC | ~0.84 |

### Interpretazione

- **TASK A**: Il modello trascrive correttamente pronunce di alta qualità
- **TASK B**: Correlazione significativa tra PER e giudizio umano
- **TASK C**: Il modello può identificare pronunce errate (AUC > 0.8)

## Ensemble Architecture

L'architettura Ensemble combina due modelli per migliorare robustezza:

```
Audio → WavLM (Weighted Layers) → logits_A ─┐
                                            ├─→ Late Fusion → Prediction
Audio → XLS-R (Multilingual)    → logits_B ─┘

Fusion: final = w * logits_A + (1-w) * logits_B
```

### Componenti

| Modello | Caratteristica | Forza |
|---------|---------------|-------|
| **WavLM Weighted** | Somma pesata 12 layer | Info acustiche + fonetiche |
| **XLS-R 300M** | Pre-training 128 lingue | Varietà accenti |

📚 Vedi [ENSEMBLE_GUIDE.md](ENSEMBLE_GUIDE.md) per dettagli tecnici.

## Colab Training

Per training su Google Colab con GPU gratuita:

1. Carica `phonemeRef.zip` su Google Drive
2. Apri uno dei notebook:
   - `colab_train_augmented.ipynb` - Training standard
   - `colab_train_wavlm_weighted.ipynb` - WavLM Weighted
   - `colab_train_xlsr.ipynb` - XLS-R + Late Fusion

## Dataset

### Fonti

- **WordReference**: ~15k parole inglesi con IPA
- **SpeechOcean762**: 2500 samples speaker non-nativi (train: score≥8)

### Pipeline

```
WordReference (15k) ─┬─→ Augmentation ─→ combined_augmented.csv
SpeechOcean (≥8)   ──┘                   (~40k samples)
```

## Riferimenti

- [WavLM Paper](https://arxiv.org/abs/2110.13900)
- [XLS-R Paper](https://arxiv.org/abs/2111.09296)
- [SpeechOcean762](https://arxiv.org/abs/2104.01378)
- [Weighted Layer Sum](https://arxiv.org/abs/2111.00346)

## Licenza

MIT License

## Autori

Progetto universitario - Deep Learning, Magistrale Anno 2

# 🎙️ DeepLearning-Phoneme

**Riconoscimento fonetico IPA da audio usando WavLM + CTC**

Sistema di Automatic Phoneme Recognition (APR) che converte audio di parole inglesi nella loro trascrizione fonetica IPA utilizzando il modello WavLM fine-tunato con CTC loss.

## 📋 Indice

- [Requisiti](#requisiti)
- [Installazione](#installazione)
- [Quick Start](#quick-start)
- [Workflow Completo](#workflow-completo)
- [Struttura Progetto](#struttura-progetto)
- [Comandi](#comandi)
- [Configurazione](#configurazione)
- [Risultati](#risultati)

## Requisiti

- Python 3.9+
- CUDA 11.8+ (opzionale, per GPU)
- ~10GB spazio disco per dataset completo
- ~4GB VRAM GPU (consigliato)

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

Se hai già il modello trainato:

```bash
# Inferenza su un file audio
python -m scripts.04_evaluate --model-path outputs/wavlm-phoneme-recognizer/final_model --audio path/to/audio.mp3

# Modalità interattiva
python -m scripts.04_evaluate --model-path outputs/wavlm-phoneme-recognizer/final_model --interactive
```

## Workflow Completo

### Fase 1: Preparazione Dati

```bash
# 1.1 Costruisci il dataset CSV dai file scaricati
python -m scripts.01_build_dataset \
    --data-dir data/raw/phonemeref_data \
    --output data/processed/phonemeref_metadata.csv

# 1.2 Preprocessa e crea vocabolario IPA
python -m scripts.02_preprocess \
    --input data/processed/phonemeref_metadata.csv \
    --output-csv data/processed/phonemeref_processed.csv \
    --output-vocab data/processed/vocab.json
```

### Fase 2: Training

```bash
# Training completo (~10 ore su CPU, ~2 ore su GPU)
python -m scripts.03_train --config configs/training_config.yaml

# Training debug (1 epoca, batch piccolo)
python -m scripts.03_train --config configs/training_config.yaml --debug

# Riprendi da checkpoint
python -m scripts.03_train --config configs/training_config.yaml --resume
```

### Fase 3: Valutazione

```bash
# Valuta sul test set interno
python -m scripts.04_evaluate \
    --model-path outputs/wavlm-phoneme-recognizer/final_model \
    --test-csv data/processed/phonemeref_processed.csv

# Valuta su SpeechOcean762 (speaker non-nativi)
python -m scripts.05_evaluate_speechocean \
    --model-path outputs/wavlm-phoneme-recognizer/final_model
```

## Struttura Progetto

```
DeepLearning-Phoneme/
├── configs/                    # File di configurazione
│   └── training_config.yaml    # Config training principale
├── data/
│   ├── raw/                    # Dati grezzi
│   │   └── phonemeref_data/    # Audio + JSON scaricati
│   ├── processed/              # Dati processati
│   │   ├── phonemeref_metadata.csv
│   │   ├── phonemeref_processed.csv
│   │   └── vocab.json
│   └── wordlists/              # Liste parole sorgente
├── outputs/                    # Modelli salvati
│   └── wavlm-phoneme-recognizer/
│       ├── checkpoint-*/
│       └── final_model/
├── scripts/                    # Script eseguibili
│   ├── 01_build_dataset.py
│   ├── 02_preprocess.py
│   ├── 03_train.py
│   ├── 04_evaluate.py
│   └── 05_evaluate_speechocean.py
├── src/                        # Moduli Python
│   ├── data/                   # Gestione dati
│   │   └── preprocessor.py
│   ├── training/               # Training
│   │   ├── dataset.py
│   │   └── trainer.py
│   └── inference/              # Inferenza
│       └── predictor.py
├── tests/                      # Unit tests
├── Makefile                    # Comandi make
├── requirements.txt            # Dipendenze Python
└── README.md
```

## Comandi

### Con Make (Linux/Mac/WSL)

| Comando | Descrizione |
|---------|-------------|
| `make install` | Installa dipendenze |
| `make build-dataset` | Crea CSV metadata |
| `make preprocess` | Preprocessa e crea vocab |
| `make train` | Avvia training |
| `make train-debug` | Training debug (1 epoca) |
| `make evaluate` | Valuta su test set |
| `make evaluate-so` | Valuta su SpeechOcean762 |
| `make inference AUDIO=file.mp3` | Inferenza su file |
| `make clean` | Pulisce file temporanei |

### Con PowerShell (Windows)

```powershell
# Build dataset
python -m scripts.01_build_dataset

# Preprocess
python -m scripts.02_preprocess

# Train
python -m scripts.03_train --config configs/training_config.yaml

# Evaluate
python -m scripts.04_evaluate

# Evaluate SpeechOcean
python -m scripts.05_evaluate_speechocean
```

## Configurazione

Modifica `configs/training_config.yaml`:

```yaml
model:
  name: "microsoft/wavlm-base-plus"
  freeze_feature_encoder: true

training:
  num_train_epochs: 10
  per_device_train_batch_size: 8
  learning_rate: 1e-4
  warmup_steps: 200
  fp16: true  # Disabilita se non hai GPU

data:
  test_size: 0.1
  sampling_rate: 16000
```

## Risultati

### Test Set Interno (PhonemeRef)

| Metrica | Valore |
|---------|--------|
| **PER** | 5.52% |
| **Accuratezza** | 94.48% |

### SpeechOcean762 (Non-Native Speakers)

| Qualità Pronuncia | PER |
|-------------------|-----|
| Alta (8-10) | ~15-20% |
| Media (5-7) | ~25-35% |
| Bassa (1-4) | ~40-50% |

## Architettura

```
Audio (16kHz) → WavLM Feature Encoder → Transformer → CTC Head → Fonemi IPA
                    (frozen)              (trainable)   (trainable)
```

- **Modello base**: microsoft/wavlm-base-plus
- **Loss**: CTC (Connectionist Temporal Classification)
- **Vocabolario**: ~50 simboli IPA

## Metrica

Il modello viene valutato usando **PER** (Phoneme Error Rate):

$$PER = \frac{S + D + I}{N}$$

Dove:
- S = Sostituzioni
- D = Cancellazioni
- I = Inserimenti
- N = Numero totale fonemi reference

## Riferimenti

- [WavLM Paper](https://arxiv.org/abs/2110.13900)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [IPA Chart](https://www.internationalphoneticassociation.org/content/ipa-chart)
- [SpeechOcean762](https://arxiv.org/abs/2104.01378)

## Licenza

MIT License

## Autori

- Progetto universitario - Deep Learning, Magistrale Anno 2


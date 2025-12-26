# 📜 Scripts

Pipeline di training e valutazione organizzata in moduli.

## Struttura

```
scripts/
├── data/               # Preparazione dati
├── training/           # Training modelli
└── evaluation/         # Valutazione e benchmark
```

---

## 📂 `data/` - Data Processing

| Script | Descrizione | Comando |
|--------|-------------|---------|
| `build_dataset.py` | Costruisce dataset base da sorgenti | `python scripts/data/build_dataset.py` |
| `preprocess.py` | Preprocessa audio e genera vocab | `python scripts/data/preprocess.py` |
| `build_augmented.py` | Crea dataset con data augmentation (pitch, noise) | `python scripts/data/build_augmented.py` |
| `build_combined.py` | Combina tutti i dataset in uno singolo | `python scripts/data/build_combined.py` |
| `build_focused.py` | Augmentation mirata sui fonemi difficili | `python scripts/data/build_focused.py` |

---

## 🏋️ `training/` - Model Training

| Script | Modello | Comando Esempio |
|--------|---------|-----------------|
| `train_wavlm.py` | WavLM-Large standard | `python scripts/training/train_wavlm.py` |
| `train_weighted.py` | WavLM con Weighted Layer Sum | `python scripts/training/train_weighted.py --epochs 5` |
| `train_baseline_mlp.py` | WavLM frozen + MLP (Linear Probe) | `python scripts/training/train_baseline_mlp.py --epochs 10` |
| `train_hubert.py` | HuBERT-Large | `python scripts/training/train_hubert.py --epochs 5` |
| `train_xlsr.py` | XLS-R 300M multilingual | `python scripts/training/train_xlsr.py --epochs 5` |

### Argomenti Comuni
- `--data-csv`: Path al CSV del dataset
- `--vocab-path`: Path a vocab.json
- `--output-dir`: Directory per checkpoint e modello finale
- `--epochs`: Numero di epoche
- `--batch-size`: Batch size

### Resume Automatico
Tutti gli script di training rilevano automaticamente i checkpoint esistenti e riprendono il training.

---

## 📊 `evaluation/` - Evaluation & Benchmark

| Script | Descrizione | Comando |
|--------|-------------|---------|
| `evaluate_model.py` | Valutazione semplice su test set | `python scripts/evaluation/evaluate_model.py --model-path <path>` |
| `evaluate_speechocean.py` | **Benchmark principale** su SpeechOcean762 | `python scripts/evaluation/evaluate_speechocean.py --model-path <path>` |
| `evaluate_fusion.py` | Late Fusion di due modelli | `python scripts/evaluation/evaluate_fusion.py --model-a <path> --model-b <path>` |
| `track_benchmark.py` | Salva risultati in Excel | `python scripts/evaluation/track_benchmark.py --model_name "..." --per X.X` |

---

## Quick Start

```bash
# 1. Prepara dati
python scripts/data/preprocess.py

# 2. Training (esempio WavLM weighted)
python scripts/training/train_weighted.py --epochs 5 --output-dir outputs/wavlm_weighted

# 3. Valutazione benchmark
python scripts/evaluation/evaluate_speechocean.py --model-path outputs/wavlm_weighted/final_model

# 4. Traccia risultati
python scripts/evaluation/track_benchmark.py --model_name "WavLM Weighted" --per 32.5 --pearson 0.58
```

# 🎙️ DeepLearning-Phoneme

**Riconoscimento fonetico IPA da audio con architettura Ensemble (WavLM + XLS-R)**

Sistema di Automatic Phoneme Recognition (APR) e Pronunciation Scoring per speaker non-nativi, basato su ensemble di modelli WavLM e XLS-R con Late Fusion.

---

## 📚 Documentazione

La documentazione completa del progetto è disponibile nella cartella [`docs/`](docs/):

- **[🦁 Model Zoo](docs/MODEL_ZOO.md)**: Dettagli su tutti i modelli implementati (WavLM, HuBERT, Whisper, SpeechTokenizer, Qwen2-Audio, MMS).
- **[🏗️ Architecture Details](docs/ARCHITECTURE_DETAILS.md)**: Approfondimenti tecnici su Weighted Layer Sum, Ensemble e Custom CTC Heads.
- **[🧪 Benchmark Guide](docs/BENCHMARK_GUIDE.md)**: Metriche, Dataset (SpeechOcean762) e protocolli di valutazione.
- **[📓 Experiments Log](EXPERIMENTS.md)**: Diario cronologico degli esperimenti e risultati.
- **[📚 References](docs/REFERENCES.md)**: Bibliografia e paper citati.

---

## ✨ Features

- **Riconoscimento fonemi IPA** da audio inglese
- **Ensemble SOTA** con WavLM + XLS-R e Late Fusion
- **Weighted Layer Sum** per combinazione ottimale dei layer
- **Benchmark scientifico** su SpeechOcean762 (speaker non-nativi)
- **Supporto multi-ambiente**: Locale, Google Colab, Kaggle

---

## 🤖 Modelli Implementati

| Modello | Params | Mode | VRAM | Status | Script |
|---------|--------|------|------|--------|--------|
| **HuBERT Large** | 317M | Fine-tuning | ~12GB | 🏆 **Best PER: 8.84%** | `train_hubert.py` |
| **WavLM Weighted** | 317M | Fine-tuning | ~12GB | 🏆 **Best AUC: 0.8523** | `train_weighted.py` |
| **Late Fusion** | 634M | Inference | ~16GB | 🆕 NEW | `evaluate_hubert_fusion.py` |
| **Early Fusion** | 634M | Frozen+CTC | ~20GB | 🆕 NEW | `train_early_fusion.py` |
| WavLM Large | 317M | Fine-tuning | ~12GB | ✅ | `train_wavlm.py` |
| XLS-R 300M | 300M | Fine-tuning | ~10GB | ✅ | `train_xlsr.py` |
| Whisper (Encoder) | 244M | Last 4 layers | ~8GB | ❌ | `train_whisper_encoder.py` |
| Baseline MLP | 2M | Linear Probe | ~4GB | ✅ | `train_baseline_mlp.py` |

> Dettagli completi: [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md)

## Quick Start

### Installazione

```bash
git clone https://github.com/maurocarlu/DeepLearning-Phoneme.git
cd DeepLearning-Phoneme
pip install -r requirements.txt
```

### Valutazione Modelli

```bash
# Valutazione modelli standard (WavLM, HuBERT, XLS-R)
python scripts/evaluation/evaluate_speechocean.py --model-path outputs/final_model

# Valutazione SpeechTokenizer (Discrete)
python scripts/evaluation/evaluate_speechtokenizer.py --model-path outputs/speechtokenizer
```

### Training

Vedi il notebook unificato per addestrare qualsiasi modello:
[**`notebooks/unified_trainer.ipynb`**](notebooks/unified_trainer.ipynb)

## Struttura Progetto

```
DeepLearning-Phoneme/
├── docs/                       # 📘 DOCUMENTAZIONE COMPLETA
│   ├── MODEL_ZOO.md            # Dettagli modelli
│   ├── ARCHITECTURE_DETAILS.md # Dettagli tecnici
│   └── BENCHMARK_GUIDE.md      # Guida valutazione
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # Script Python
│   ├── training/               # Script di training
│   ├── evaluation/             # Script di valutazione
│   └── data/                   # Processing dati
├── src/                        # Moduli sorgente
└── EXPERIMENTS.md              # Log risultati
```

## Autori

Progetto universitario - Deep Learning, Magistrale Anno 2

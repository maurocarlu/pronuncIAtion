# 🎯 Ensemble Architecture for Pronunciation Scoring

## Guida Tecnica Completa

Questa guida documenta l'architettura SOTA (State of the Art) implementata per il sistema di Pronunciation Scoring, basata su un ensemble di modelli WavLM e XLS-R con Late Fusion.

---

## 📚 Motivazione Scientifica

### Perché Weighted Layer Sum?

I modelli Transformer come WavLM producono **12 hidden states** (uno per layer). Tradizionalmente, si usa solo l'ultimo layer per le predizioni finali. Tuttavia, ricerche recenti mostrano che:

| Layer | Informazione Catturata |
|-------|------------------------|
| 1-4 (bassi) | **Acustica**: formanti, pitch, energia |
| 5-8 (medi) | **Fonetica**: confini fonetici, transizioni |
| 9-12 (alti) | **Semantica**: contesto, co-articolazione |

**Weighted Layer Sum** combina TUTTI i layer con pesi apprendibili, permettendo al modello di:
1. Catturare informazioni acustiche E fonetiche simultaneamente
2. Apprendere automaticamente quale combinazione è ottimale
3. Adattarsi al task specifico (riconoscimento fonemi)

### Perché XLS-R insieme a WavLM?

L'ensemble combina due modelli con caratteristiche complementari:

| Aspetto | WavLM | XLS-R |
|---------|-------|-------|
| **Pre-training** | Inglese (focus) | 128 lingue |
| **Dimensione** | 316M parametri | 300M parametri |
| **Forza** | Robustezza rumore | Varietà fonetica |
| **Debolezza** | Accenti non-nativi | Più lento |

La **diversità** tra i modelli è cruciale: se entrambi facessero gli stessi errori, combinarli non aiuterebbe. XLS-R, avendo visto 128 lingue, ha una percezione fonetica più ampia che copre accenti e pronunce non standard.

---

## 🏗️ Architettura

### Diagramma del Flusso

```
                              ┌─────────────────────────────────────┐
                              │           INPUT AUDIO               │
                              │       (16kHz waveform)              │
                              └───────────────┬─────────────────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   │                   ▼
           ┌──────────────────────────┐       │    ┌──────────────────────────┐
           │    MODEL A: WavLM        │       │    │    MODEL B: XLS-R        │
           │  (Weighted Layer Sum)    │       │    │   (wav2vec2-xls-r-300m)  │
           └──────────────────────────┘       │    └──────────────────────────┘
                          │                   │                   │
                          ▼                   │                   ▼
           ┌──────────────────────────┐       │    ┌──────────────────────────┐
           │  12 Hidden States        │       │    │  24 Hidden States        │
           │  h_1, h_2, ..., h_12     │       │    │  Standard CTC Head       │
           └──────────────────────────┘       │    └──────────────────────────┘
                          │                   │                   │
                          ▼                   │                   ▼
           ┌──────────────────────────┐       │    ┌──────────────────────────┐
           │  Weighted Sum:           │       │    │                          │
           │  Σ(softmax(w_i) * h_i)   │       │    │      logits_B            │
           └──────────────────────────┘       │    └──────────────────────────┘
                          │                   │                   │
                          ▼                   │                   ▼
           ┌──────────────────────────┐       │    ┌──────────────────────────┐
           │  Dropout + CTC Head      │       │    │                          │
           │      logits_A            │       │    │                          │
           └──────────────────────────┘       │    └──────────────────────────┘
                          │                   │                   │
                          └───────────────────┼───────────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────────┐
                              │         LATE FUSION               │
                              │  final = w*A + (1-w)*B            │
                              └───────────────────────────────────┘
                                              │
                                              ▼
                              ┌───────────────────────────────────┐
                              │       ARGMAX + DECODE             │
                              │     → Trascrizione IPA            │
                              └───────────────────────────────────┘
```

### Dettaglio: Weighted Layer Sum

```python
# Formula matematica implementata in WavLMWithWeightedLayers

# 1. Ottieni hidden states da tutti i layer
hidden_states = wavlm(audio, output_hidden_states=True).hidden_states
# Shape: tuple di 13 tensori [batch, time, 768]

# 2. Normalizza pesi con softmax
weights = softmax(learnable_weights)  # [13] -> somma = 1.0

# 3. Stack e weighted sum
stacked = torch.stack(hidden_states, dim=0)  # [13, batch, time, 768]
weighted = (stacked * weights.view(-1,1,1,1)).sum(dim=0)  # [batch, time, 768]

# 4. CTC head
logits = linear(dropout(weighted))  # [batch, time, vocab_size]
```

---

## 📁 Struttura dei File

```
scripts/
├── train_weighted.py      # Training WavLM con Weighted Layer Sum
├── train_xlsr.py          # Training XLS-R (modello ausiliario)
├── evaluate_fusion.py     # Valutazione Late Fusion
└── 05_evaluate_speechocean.py  # Benchmark singolo modello

configs/
└── training_config.yaml   # Configurazione training

outputs/
├── final_model_weighted/  # WavLM Weighted addestrato
└── xlsr/
    └── final_model_xlsr/  # XLS-R addestrato
```

---

## 🚀 Guida all'Uso

### 1. Training WavLM Weighted

```bash
# Training con configurazione default
python scripts/train_weighted.py \
    --config configs/training_config.yaml \
    --data-csv data/processed/combined_augmented.csv

# Training con override output
python scripts/train_weighted.py \
    --config configs/training_config.yaml \
    --output-dir outputs/wavlm_weighted_v2
```

**Output**: `outputs/final_model_weighted/`

### 2. Training XLS-R (Modello Ausiliario)

```bash
# IMPORTANTE: Usa lo STESSO dataset e vocab.json
python scripts/train_xlsr.py \
    --config configs/training_config.yaml \
    --data-csv data/processed/combined_augmented.csv

# Nota: XLS-R richiede ~12GB VRAM
# Se OOM, riduci batch_size nel config
```

**Output**: `outputs/xlsr/final_model_xlsr/`

### 3. Valutazione Late Fusion

```bash
# Fusion con peso 0.5 (media semplice)
python scripts/evaluate_fusion.py \
    --model-a outputs/final_model_weighted \
    --model-b outputs/xlsr/final_model_xlsr \
    --weight 0.5

# Fusion con peso a favore di WavLM
python scripts/evaluate_fusion.py \
    --model-a outputs/final_model_weighted \
    --model-b outputs/xlsr/final_model_xlsr \
    --weight 0.7  # 70% WavLM, 30% XLS-R
```

### 4. Ottimizzazione del Peso

Per trovare il peso ottimale:

```bash
# Test multipli pesi
for w in 0.3 0.4 0.5 0.6 0.7; do
    echo "=== Weight: $w ==="
    python scripts/evaluate_fusion.py \
        --model-a outputs/final_model_weighted \
        --model-b outputs/xlsr/final_model_xlsr \
        --weight $w --quiet
done
```

---

## 📊 Spiegazione delle Classi

### `WavLMWithWeightedLayers`

Classe custom che implementa WavLM con weighted layer sum.

```python
class WavLMWithWeightedLayers(nn.Module):
    """
    Attributi chiave:
    - wavlm: Modello base WavLM
    - layer_weights: nn.Parameter con pesi apprendibili [13]
    - lm_head: Linear layer per CTC (768 → vocab_size)
    
    Metodi:
    - forward(): Inferenza con weighted sum
    - get_layer_weights_info(): Stampa pesi finali per analisi
    """
```

**Pesi tipici dopo training:**
```
layer_0:  0.0423  (feature encoder output)
layer_1:  0.0512
layer_2:  0.0634  
...
layer_10: 0.1245  ← Layer più importanti
layer_11: 0.1523  ← per riconoscimento fonemi
layer_12: 0.0987
```

### `LateFusionEnsemble`

Classe per combinare predizioni di due modelli.

```python
class LateFusionEnsemble:
    """
    Attributi:
    - model_a, model_b: I due modelli caricati
    - processor_a, processor_b: Processor per preprocessing
    - weight: Peso del modello A (0-1)
    
    Metodi:
    - predict_single(audio): Predice con fusion
    - predict_batch(audios): Batch prediction
    """
```

---

## ⚠️ Note Importanti

### Allineamento Vocabolario

**CRITICO**: Entrambi i modelli DEVONO usare lo stesso `vocab.json`.

```
Se vocab_A ≠ vocab_B → I logits non sono comparabili → Fusion fallisce
```

Lo script `train_xlsr.py` usa automaticamente `data/processed/vocab.json` come WavLM.

### Requisiti VRAM

| Modello | VRAM Training | VRAM Inference |
|---------|---------------|----------------|
| WavLM Weighted | ~8 GB | ~4 GB |
| XLS-R | ~12 GB | ~6 GB |
| Fusion | N/A | ~10 GB |

### Allineamento Temporale

I due modelli possono avere output di lunghezza diversa. `evaluate_fusion.py` gestisce questo automaticamente:

```python
min_len = min(logits_a.size(1), logits_b.size(1))
logits_a = logits_a[:, :min_len, :]
logits_b = logits_b[:, :min_len, :]
```

---

## 📈 Risultati Attesi

Con l'architettura ensemble, ci aspettiamo miglioramenti su:

| Metrica | Singolo Modello | Ensemble |
|---------|-----------------|----------|
| PER (High Quality) | ~15% | ~12-13% |
| Pearson Correlation | ~0.55 | ~0.60+ |
| AUC-ROC Detection | ~0.84 | ~0.87+ |

I miglioramenti sono dovuti a:
1. **Diversità**: Errori diversi si compensano
2. **Robustezza**: Due modelli = più stabile
3. **Weighted Layers**: Informazioni acustiche preservate

---

## 📚 Riferimenti

- [WavLM Paper](https://arxiv.org/abs/2110.13900)
- [XLS-R Paper](https://arxiv.org/abs/2111.09296)
- [Weighted Layer Sum in Speech](https://arxiv.org/abs/2111.00346)
- [SpeechOcean762 Dataset](https://arxiv.org/abs/2104.01378)

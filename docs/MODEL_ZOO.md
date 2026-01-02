# 🦁 Model Zoo - Phoneme Recognition

Questo documento descrive in dettaglio tutte le architetture implementate e testate nel benchmark.

---

## 1. WavLM (Large)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `microsoft/wavlm-large` |
| **Piattaforma** | HuggingFace Transformers |
| **Parametri** | ~317M |
| **Pre-training** | Masked Speech Prediction + Denoising |

### Caratteristiche
WavLM estende HuBERT aggiungendo un task di **denoising** durante il pre-training. Maschera parti dell'input e aggiunge rumore/sovrapposizioni, costringendo il modello a imparare non solo la struttura fonetica ma anche caratteristiche acustiche robuste.

### Implementazione
- **Standard**: Fine-tuning classico CTC sull'ultimo layer.
- **Weighted**: Somma pesata dei 25 layer (vedi [Architecture Details](ARCHITECTURE_DETAILS.md)).

---

## 2. HuBERT (Large)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `facebook/hubert-large-ls960-ft` |
| **Parametri** | ~317M |
| **Pre-training** | Masked Prediction con target discreti (K-means clustering) |

### Caratteristiche
HuBERT impara mappando l'audio continuo a cluster discreti. Questo forza il modello a scartare dettagli acustici inutili (come il canale) e concentrarsi sulle unità linguistiche. È stato il primo modello a dimostrare che ASL self-supervised scala bene.

---

## 3. XLS-R (300M)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `facebook/wav2vec2-xls-r-300m` |
| **Lingue** | 128 (Pre-training multilingua) |
| **Parametri** | ~300M |

### Caratteristiche
Versione multilingua di Wav2Vec2. Addestrato su mezza milionata di ore di audio in 128 lingue diverse.
**Motivazione**: Teoria che i modelli multilingua gestiscano meglio gli accenti non-nativi (come quelli in SpeechOcean762).

---

## 5. Whisper (Encoder Only)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `openai/whisper-small` (Encoder) |
| **Input** | Log-Mel Spectrogram (80 bins) |
| **Parametri** | ~87M (Encoder) + ~150M (Embeddings iniziali) |

### Architettura Custom
Whisper è nativamente un Seq2Seq (Encoder-Decoder). Per il phoneme recognition frame-level:
1.  **Isoliamo l'Encoder**: Scartiamo il decoder autoregressivo.
2.  **CTC Head**: Aggiungiamo un layer lineare sopra gli hidden states dell'encoder.
3.  **Input**: Richiede conversione Audio → Mel Spectrogram (non raw waveform).

### Implementazione
Vedi `scripts/training/train_whisper_encoder.py`.

---

## 6. SpeechTokenizer (Discrete)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `fnlp/SpeechTokenizer` (HuBERT-based) |
| **Tipo** | Discrete Token Representation |
| **Codebook** | 1024 token |

### Architettura a 2 Stadi
Approccio radicalmente diverso basato su token discreti.

1.  **Fase 1 (Encoder Frozen)**: L'audio passa per SpeechTokenizer che usa RVQ (Residual Vector Quantization) per produrre una sequenza di indici interi.
2.  **Fase 2 (Classifier Trainable)**: Un piccolo Transformer trainato da zero prende gli indici e predice i fonemi.

**Motivazione**: Testare se la discretizzazione "pulisce" il segnale mantenendo solo l'info fonetica.

---

## 7. Qwen2-Audio (Linear Probe)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `Qwen/Qwen2-Audio-7B-Instruct` (Audio Tower) |
| **Quantizzazione** | 4-bit (bitsandbytes NF4) |
| **Params Totali** | ~1B (FROZEN) |
| **Params Trainabili** | ~260k (CTC Head only) |
| **VRAM** | ~5-6GB |

### Caratteristiche
Modello multimodale (LLM + Audio). Estraiamo solo la "Audio Tower" pre-trainata.
**Modalità**: Linear Probe (encoder completamente frozen, solo CTC head trainabile).
**Obiettivo**: Valutare feature "zero-shot" del modello multimodale.

---

## 8. Wav2Vec2-BERT 2.0 (⭐ Recommended)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `facebook/w2v-bert-2.0` |
| **Parametri** | ~600M |
| **Pre-training** | Contrastive + Masked Language Modeling |
| **Input** | Log-Mel Spectrogram (80 bins) |
| **VRAM** | ~10-12GB |

### Caratteristiche
Combina il contrastive learning di Wav2Vec2 con MLM di BERT.

> **⚠️ IMPORTANTE**: A differenza di Wav2Vec2/WavLM, W2V-BERT 2.0 richiede **spettrogrammi log-mel** (`input_features`), NON audio raw (`input_values`)!

### Configurazione Corretta
| Componente | Classe |
|------------|--------|
| Feature Extractor | `SeamlessM4TFeatureExtractor` (non Wav2Vec2FeatureExtractor!) |
| Processor | `Wav2Vec2BertProcessor` (non Wav2Vec2Processor!) |
| Model | `Wav2Vec2BertForCTC` |

### Training
- **LR**: 5e-6 (conservativo per evitare CTC collapse)
- **Warmup**: 2000 steps
- **Freeze**: `feature_projection` (non esiste `freeze_feature_encoder()`)
- **Subsampling**: Fattore 2 sui frame spettrogramma

---

## 9. MMS (Massively Multilingual Speech)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `facebook/mms-1b-all` |
| **Parametri** | ~1B |
| **Pre-training** | 1000+ lingue, contrastive learning |
| **VRAM** | ~16GB (FP16) o ~8GB (4-bit) |

### Caratteristiche
MMS è un modello massiccio (1 miliardo di parametri) addestrato su oltre 1000 lingue.
Usa adapter per lingua, ma qui lo usiamo con CTC head custom sul nostro vocab IPA.

### Configurazione
- **FP16**: Obbligatorio per VRAM
- **4-bit quantization**: Opzionale per GPU con <16GB VRAM
- **Batch Size**: 8 (con FP16)

**Motivazione**: Testare se il pre-training multilingue massivo migliora il riconoscimento fonetico.

---

## 10. Baseline MLP (Linear Probe)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | WavLM Base (Frozen) |
| **Aggregazione** | Global Average Pooling |
| **Classifier** | MLP (2 layer) |

### Caratteristiche
Modello "stupido" di controllo. Se un classificatore semplice su feature medie funziona bene, il task è troppo facile. Serve come *lower bound*.

---

## Tabella Riassuntiva

| Modello | Params | Training Mode | VRAM | Script |
|---------|--------|---------------|------|--------|
| WavLM Large | 317M | Fine-tuning | ~12GB | - |
| HuBERT Large | 317M | Fine-tuning | ~12GB | - |
| XLS-R 300M | 300M | Fine-tuning | ~10GB | - |
| Whisper Small (Encoder) | 244M | Partial Fine-tuning | ~8GB | `train_whisper_encoder.py` |
| SpeechTokenizer | 256K train | 2-Stage (Classifier) | ~4GB | `train_speechtokenizer.py` |
| **Qwen2-Audio** | **260K train** | **Linear Probe** | **~5GB** | `train_qwen_audio.py` |
| **Wav2Vec2-BERT** | **600M** | **Fine-tuning** | **~12GB** | `train_w2v2_bert.py` |
| **MMS 1B** | **1B** | **Fine-tuning** | **~16GB** | `train_mms.py` |
| Baseline MLP | 2M train | Linear Probe | ~4GB | `train_baseline_mlp.py` |

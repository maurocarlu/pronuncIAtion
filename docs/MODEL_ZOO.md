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

## 3. Wav2Vec 2.0 (Large)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `facebook/wav2vec2-large-960h-lv60-self` |
| **Parametri** | ~317M |
| **Pre-training** | Contrastive Learning (Quantized vector selection) |

### Caratteristiche
Predecessore di HuBERT e WavLM. Usa una loss contrastiva per distinguere la vera rappresentazione quantizzata da distrattori.
**Motivazione nel Benchmark**: Serve per isolare l'effetto dei pre-training più avanzati (HuBERT/WavLM).

---

## 4. XLS-R (300M)

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

## 7. Qwen2-Audio (4-bit)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `Qwen/Qwen2-Audio-7B-Instruct` (Audio Tower) |
| **Quantizzazione** | 4-bit (bitsandbytes NF4) |
| **Parametri Audio** | ~1B (originale) |

### Caratteristiche
Modello multimodale (LLM + Audio). Estraiamo solo la "Audio Tower" pre-trainata.
**Sfida**: Enorme consumo di memoria.
**Soluzione**: Quantizzazione 4-bit (`load_in_4bit=True`). L'encoder è completamente congelato; alleniamo solo la CTC head finale.

---

## 8. Baseline MLP (Linear Probe)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | WavLM Base (Frozen) |
| **Aggregazione** | Global Average Pooling |
| **Classifier** | MLP (2 layer) |

### Caratteristiche
Modello "stupido" di controllo. Se un classificatore semplice su feature medie funziona bene, il task è troppo facile. Serve come *lower bound*.

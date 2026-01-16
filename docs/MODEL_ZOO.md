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

## 2b. DistilHuBERT ⭐ NEW

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `ntu-spml/distilhubert` |
| **Parametri** | ~75% in meno vs HuBERT Large (distillazione) |
| **Input** | Raw Waveform (16kHz) |
| **Head** | CTC (`HubertForCTC`) |

### Caratteristiche
DistilHuBERT applica **Knowledge Distillation** per ottenere un encoder più leggero e veloce, cercando di mantenere le feature fonetiche utili per il task.

**Motivazione**: verificare se una compressione importante del backbone (≈75% parametri in meno) mantiene performance competitive su SpeechOcean762.

### Note di Implementazione
- **Vocab custom**: `data/processed/vocab.json`
- **Tokenizer**: `Wav2Vec2CTCTokenizer` con `bos_token=None`, `eos_token=None`
- **Stabilità CTC**: `ctc_zero_infinity=True` + re-init `lm_head` (std=0.02)
- **Memoria**: `fp16=True`, `gradient_checkpointing=True`
- **Monitoring**: `PredictionMonitorCallback` ogni 100 step

### Script
Vedi `scripts/training/train_distilhubert.py`.

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

## 3b. XLS-R (1B) ⭐ NEW

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `facebook/wav2vec2-xlsr-1b` |
| **Lingue** | 128 (Pre-training multilingua) |
| **Parametri** | ~1B |
| **Input** | Raw Waveform (16kHz) |
| **Head** | CTC (`Wav2Vec2ForCTC`) |

### Caratteristiche
Versione “scaled” di XLS-R per testare l’ipotesi: più capacità → miglior AUC (detection) e/o PER su accenti non-nativi.

### Note di Implementazione
- **Vocab custom**: `data/processed/vocab.json`
- **Tokenizer**: `bos_token=None`, `eos_token=None`
- **Stabilità CTC**: `ctc_zero_infinity=True` + re-init `lm_head`
- **Memoria**: `fp16=True`, `gradient_checkpointing=True`
- **Anti-OOM (1B)**: usare `--max-audio-seconds` (truncate/drop), `--eval-batch-size` separato e `group_by_length=True` (bucketing) per ridurre il peak VRAM
- **Fallback VRAM (QLoRA)**: `--load-in-4bit` o `--load-in-8bit` + LoRA (richiede `bitsandbytes` + `peft`)
- **Monitoring**: `PredictionMonitorCallback` ogni 100 step

### Script
Vedi `scripts/training/train_xlsr_1b.py`.

---

## 3c. Data2Vec 2.0 (Large) ⭐ NEW

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `facebook/data2vec2-large-960h` |
| **Input** | Raw Waveform (16kHz) |
| **Head** | CTC (`Wav2Vec2ForCTC`) |
| **Pre-training** | Self-distillation (teacher/student) |

### Caratteristiche
Evoluzione di Wav2Vec2: tende a convergere più rapidamente e con training più stabile grazie all’obiettivo di self-distillation.

### Note di Implementazione
- **Vocab custom**: `data/processed/vocab.json`
- **Tokenizer**: `bos_token=None`, `eos_token=None`
- **Hyperparams benchmark**: LR `3e-5`, `warmup_ratio=0.1`, `gradient_accumulation_steps=4`
- **Memoria**: `fp16=True`, `gradient_checkpointing=True`
- **Monitoring**: `PredictionMonitorCallback` ogni 100 step

### Script
Vedi `scripts/training/train_data2vec2.py`.

---

## 4. Wav2Vec2 Phoneme (Domain Init)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | `facebook/wav2vec2-xlsr-53-espeak-cv-ft` |
| **Input** | Raw Waveform (16kHz) |
| **Head** | CTC (`Wav2Vec2ForCTC`) |
| **Tokenizer** | `Wav2Vec2PhonemeCTCTokenizer` (vocab custom) |

### Caratteristiche
Questo modello usa l'architettura standard di Wav2Vec2 ma parte da un checkpoint basato su **XLSR-53** (multilingua) già fine-tunato su fonemi (dataset CommonVoice + espeak).

**Motivazione scientifica**: "Test di un modello con inizializzazione specifica e multilingua per il dominio dei fonemi invece che SSL generico".

### Note di Implementazione
- **Vocab custom**: usa `data/processed/vocab.json`.
- **do_phonemize=False**: i target `ipa_clean` sono già in formato fonetico.
- **CTC safety**: `bos_token` e `eos_token` sono impostati a `None` se non presenti nel vocab (evita mismatch negli indici della loss CTC).
- **Stabilità**: re-inizializzazione `lm_head` per ridurre rischio di blank collapse.
- **Monitoring**: `PredictionMonitorCallback` ogni 100 step.

### Script
Vedi `scripts/training/train_wav2vec2_phoneme.py`.

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

> **⚠️ Nota**: A causa dei requisiti computazionali (~10+ ore di training su GPU consumer), Qwen2-Audio viene trainato su **10.000 samples** invece del dataset completo (~40k).

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
- **FP16**: consigliato/necessario per VRAM
- **Anti-OOM (1B)**: `--max-audio-seconds`, `--eval-batch-size`, bucketing `group_by_length=True`
- **Fallback VRAM (QLoRA)**: `--load-in-4bit` o `--load-in-8bit` + LoRA (richiede `bitsandbytes` + `peft`)
- **Batch Size**: tipicamente 1 su 16GB VRAM (aumentare con `--gradient-accumulation-steps`)

**Motivazione**: Testare se il pre-training multilingue massivo migliora il riconoscimento fonetico.

### Script
- Benchmark script: `scripts/training/train_mms_1b.py`
- Script legacy/esteso: `scripts/training/train_mms.py`

---

## 9b. M-CTC-T (Meta) ⭐ NEW

| Parametro | Valore |
|-----------|--------|
| **Checkpoint** | `speechbrain/m-ctc-t-large` |
| **Input** | Mel Spectrogram (80 bins) |
| **Processor** | `MCTCTProcessor` |
| **Model** | `MCTCTForCTC` |

### Caratteristiche
M-CTC-T è un modello CTC di Meta progettato per lavorare su **spettrogrammi Mel** (invece che su waveform raw).
Questo esperimento serve a confrontare in modo “fair” l’efficacia di:
- **Mel-CTC (Meta)**: feature acustiche esplicite (Mel)
- **Raw-CTC (Meta)**: SSL su waveform (es. XLS-R, MMS)

### Note di Implementazione
- **Vocab custom**: `data/processed/vocab.json`
- **Tokenizer**: `bos_token=None`, `eos_token=None`
- **Init head**: re-init `lm_head` (std=0.02), `ignore_mismatched_sizes=True`
- **Hyperparams benchmark**: LR `3e-5`, `warmup_ratio=0.1`, `gradient_accumulation_steps=4`
- **Memoria**: `fp16=True`, `gradient_checkpointing=True`
- **Anti-OOM**: bucketing `group_by_length=True` e (se necessario) `--max-audio-seconds`
- **Monitoring**: `PredictionMonitorCallback` ogni 100 step

### Script
Vedi `scripts/training/train_mctct.py`.

---

## 9c. Parakeet-CTC 1.1B (NVIDIA) ⭐ NEW

| Parametro | Valore |
|-----------|--------|
| **Checkpoint** | `nvidia/parakeet-ctc-1.1b` |
| **Architettura** | FastConformer-CTC |
| **Input** | Audio 16kHz via `ParakeetProcessor` |
| **Processor** | `ParakeetProcessor` |
| **Model** | `ParakeetForCTC` |

### Note di Implementazione
- **Vocab custom**: `data/processed/vocab.json`
- **Tokenizer**: `bos_token=None`, `eos_token=None`
- **Init head**: re-init `lm_head` (std=0.02), `ignore_mismatched_sizes=True`
- **Memoria (consigliato)**: carico in **4-bit** (NF4) con `BitsAndBytesConfig` quando la VRAM è limitata
- **Training mode**: tipicamente **linear probing** (solo `lm_head` trainabile) per stare entro 16GB
- **Hyperparams benchmark**: LR `3e-5`, `warmup_ratio=0.1`, `gradient_accumulation_steps=4`, `fp16=True`
- **Monitoring**: `PredictionMonitorCallback` ogni 100 step

### Script
Vedi `scripts/training/train_parakeet.py`.

---

## 10. Late Fusion (HuBERT + WavLM) ⭐ NEW

| Parametro | Valore |
|-----------|--------|
| **Modello A** | HuBERT Large (Best PER: 8.84%) |
| **Modello B** | WavLM Weighted (Best AUC: 0.8523) |
| **Strategia** | Logit-level ensemble |
| **VRAM** | ~16GB (2x Large models) |

### Formula
```
final_logits = α × logits_HuBERT + (1-α) × logits_WavLM
```

### Caratteristiche
Combina i due top-performer senza re-training:
- **HuBERT**: Pre-training con target discreti → trascrizione precisa
- **WavLM Weighted**: Pre-training con denoising → detection robusta

Pesi testati: α ∈ {0.3, 0.5, 0.7}

### Script
- Evaluation: `scripts/evaluation/evaluate_hubert_fusion.py`

---

## 11. Early Fusion (Multi-Backbone) ⭐ UPDATED

| Parametro | Valore |
|-----------|--------|
| **Backbone 1** | HuBERT Large (frozen, fine-tuned encoder) |
| **Backbone 2** | WavLM Base (frozen, fine-tuned encoder) |
| **Concatenazione** | 1024 + 768 = 1792D |
| **CTC Head** | Linear(1792, vocab_size) |
| **VRAM** | ~8-10GB (con 4-bit quantization) |

### Architettura
```
Audio → HuBERT Large → 1024D ─┐
                               ├→ concat(1792D) → CTC Head → Phonemes
Audio → WavLM Base  → 768D ──┘
```

### Caratteristiche
- Usa i **tuoi encoder già fine-tuned** per phoneme recognition
- Solo la CTC head viene trainata (frozen backbones)
- Supporta 4-bit quantization per ridurre VRAM
- Il classificatore ha accesso simultaneo a rappresentazioni fonetiche (HuBERT) e acustiche (WavLM)

### Script
- Training: `scripts/training/train_early_fusion.py`
- Argomenti custom: `--wavlm-path` e `--hubert-path` per usare checkpoint fine-tuned


---

## 12. Baseline MLP (Linear Probe)

| Parametro | Valore |
|-----------|--------|
| **Backbone** | WavLM Base (Frozen) |
| **Aggregazione** | Global Average Pooling |
| **Classifier** | MLP (2 layer) |

### Caratteristiche
Modello "stupido" di controllo. Se un classificatore semplice su feature medie funziona bene, il task è troppo facile. Serve come *lower bound*.

---

## Tabella Riassuntiva

| Modello | Input Type | Params | Training Mode | VRAM | Status | Script |
|---------|------------|--------|---------------|------|--------|--------|
| **HuBERT Large** | Raw Waveform | 317M | Fine-tuning | ~12GB | ✅ **Best PER** | `train_hubert.py` |
| **WavLM Weighted** | Raw Waveform | 317M | Fine-tuning | ~12GB | ✅ **Best AUC** | `train_weighted.py` |
| **Late Fusion** | Raw Waveform | 634M | Inference Only | ~16GB | 🆕 NEW | `evaluate_hubert_fusion.py` |
| **Early Fusion** | Raw Waveform | 413M+2K | Frozen+CTC | ~8GB | 🆕 UPDATED | `train_early_fusion.py` |
| WavLM Base/Large | Raw Waveform | 317M | Fine-tuning | ~12GB | ✅ Works | `train_wavlm.py` |
| XLS-R 300M | Raw Waveform | 300M | Fine-tuning | ~10GB | ✅ Works | `train_xlsr.py` |
| XLS-R 1B | Raw Waveform | 1B | Fine-tuning / QLoRA | ~16GB / ~8GB | ⏳ TBD | `train_xlsr_1b.py` |
| Baseline MLP | Raw Waveform | 2M train | Linear Probe | ~4GB | ✅ Works | `train_baseline_mlp.py` |
| Whisper Small (Encoder) | Mel Spectrogram | 244M | Partial Fine-tuning | ~8GB | ❌ Failed | `train_whisper_encoder.py` |
| Wav2Vec2-BERT | Mel Spectrogram | 600M | Fine-tuning | ~12GB | ❌ Failed | `train_w2v2_bert.py` |
| Qwen2-Audio | Mel Spectrogram | 260K train | Linear Probe | ~5GB | ⏳ TBD | `train_qwen_audio.py` |
| SpeechTokenizer | Discrete Tokens | 256K train | 2-Stage | ~4GB | ⚠️ Partial | `train_speechtokenizer.py` |
| Data2Vec2 Large | Raw Waveform | 317M | Fine-tuning | ~12GB | ⏳ TBD | `train_data2vec2.py` |
| MMS 1B | Raw Waveform | 1B | Fine-tuning / QLoRA | ~16GB / ~8GB | ⏳ TBD | `train_mms_1b.py` |
| M-CTC-T (Meta) | Mel Spectrogram | ~? | Fine-tuning | ~10-12GB | ⏳ TBD | `train_mctct.py` |
| Parakeet-CTC 1.1B | Audio 16kHz | 1.1B | Linear Probe (4-bit) | ~? | ⏳ TBD | `train_parakeet.py` |

> **💡 Key Insight**: L'input type conta, ma la differenza vera la fanno **pre-training + preprocessing corretto**. Alcuni modelli mel-based (es. Whisper Encoder) hanno fallito per mismatch architetturale/CTC; altri modelli CTC nativi (es. M-CTC-T) sono progettati per lavorare su feature 2D.

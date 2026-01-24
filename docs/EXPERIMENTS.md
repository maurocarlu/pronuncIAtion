# 📓 Experiments Log - Phoneme Recognition Benchmark

Questo documento traccia cronologicamente gli esperimenti condotti e i risultati ottenuti sul benchmark **SpeechOcean762**.

Per dettagli tecnici sui modelli, vedi **[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md)**.
Per dettagli sulle metriche, vedi **[docs/BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md)**.

---

## 📊 Tabella Riassuntiva Risultati (Gennaio 2025)

Questa è una **vista sintetica** (3 metriche chiave). La tabella completa (con F1/Precision/Recall/Threshold e note) è in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

| Modello | Input Type | Task A PER (HQ) ↓ | Task B Pearson ↑ | Task C AUC ↑ | Note |
|---------|------------|------------------:|-----------------:|-------------:|------|
| **HuBERT Large** | Raw Waveform | **8.84** | **0.5932** | 0.8426 | 🏆 Best PER + Best Pearson |
| EarlyFusion (HuBERT+WavLM) | Raw Waveform | 9.46 | 0.5829 | 0.8404 | Frozen encoders + CTC head only |
| Early Fusion (HuBERT+WavLM) | Raw Waveform | 9.52 | 0.5886 | **0.8479** | Frozen encoders + CTC head only |
| WavLM Base (Aug_Comb) | Raw Waveform | 14.91 | 0.5550 | 0.8369 | Baseline fine-tuned |
| WavLM Large (Aug_Comb) | Raw Waveform | 17.91 | 0.5736 | 0.8382 | Ablation study |
| Baseline MLP | Raw Waveform | 25.92 | 0.5754 | 0.8427 | WavLM-base frozen + MLP |
| MMS-1B (PEFT) | Raw Waveform | 27.46 | 0.5773 | **0.8479** | LoRA/QLoRA pipeline (adapter) |
| WavLM Large Weighted | Raw Waveform | 30.41 | 0.5812 | **0.8523** | 🥇 Best AUC (detection) |
| Data2Vec2 Large | Raw Waveform | 31.22 | 0.5694 | 0.8420 | Full FT, feature encoder frozen |
| XLS-R 300M | Raw Waveform | 39.40 | 0.5589 | 0.8435 | Multilingual |
| Qwen2-Audio 7B (Linear Probe) | Log-Mel | 46.66 | 0.3994 | 0.7531 | 4-bit, encoder frozen, CTC head only |
| DistilHuBERT V2 | Raw Waveform | 58.08 | 0.4694 | 0.7883 | Distillation (v2) |
| DistilHuBERT (CTC) | Raw Waveform | 71.19 | 0.3290 | 0.7146 | Distillation (v1) |
| XLS-R 1B (PEFT) | Raw Waveform | 65.70 | 0.5636 | 0.8366 | Adapter su backbone 1B (risultati deboli) |
| SpeechTokenizer | Discrete | 60.85 | 0.3842 | 0.7311 | RVQ codes + Transformer classifier |
| Wav2Vec2-BERT 2.0 | Log-Mel | 88.58 | 0.3247 | 0.6936 | ❌ Probabile mismatch preprocessing spettrogrammi |
| Whisper Encoder + CTC | Log-Mel | 236.77 | 0.4510 | 0.7963 | ❌ Non convergente (CTC decode fix applicata) |
| M-CTC-T Large | Log-Mel | 267.18 | 0.2002 | 0.6367 | ❌ Fallito (instabilità training) |
| Parakeet-CTC 1.1B | Audio 16kHz | 94.73 | -0.0274 | 0.4790 | ❌ Linear probe head-only non informativo |

> **⚠️ Key Finding (updated)**: I modelli raw waveform hanno funzionato in modo più "plug-and-play". I modelli con feature 2D hanno richiesto setup molto più delicato (feature extractor + shape + CTC), e alcuni esperimenti sono falliti per mismatch. Vedi [ARCHITECTURE_DETAILS.md](ARCHITECTURE_DETAILS.md#8--input-types-vs-performance---critical-analysis).

---

## 📅 Log Esperimenti

### [Planned] DistilHuBERT vs HuBERT Large (Knowledge Distillation)
- **Motivazione scientifica**: Valutazione dell'efficacia della distillazione (≈75% di parametri in meno) rispetto al backbone HuBERT Large nel task di Phoneme Recognition.
- **Checkpoint**: `ntu-spml/distilhubert`
- **Script**: `scripts/training/train_distilhubert.py`

### [26/12/2024] Benchmark SpeechTokenizer
- **Config**: HuBERT encoder (frozen) → RVQ (8 layers) → Classificatore Transformer (2 layers).
- **Risultati**: PER alto (60%).
- **Analisi**: La discretizzazione sembra perdere troppe informazioni fonetiche fini necessarie per la trascrizione precisa, ma mantiene una discreta capacità di detection (AUC 0.73). Probabilmente il codebook (1024 token) è troppo compresso per IPA quality transcription.

### [25/12/2024] Benchmark HuBERT Large
- **Config**: `facebook/hubert-large-ls960-ft`, 10 epoche.
- **Risultati**: **SOTA!** PER 8.84% su high-quality samples.
- **Analisi**: HuBERT (masked prediction con target discreti) sembra imparare feature fonetiche più robuste di WavLM per questo task specifico.

### [24/12/2024] Ensemble Experiments (Late Fusion)
- **Config**: Fusione di logits tra WavLM Weighted e XLS-R.
- **Risultati**:
    - Weight 0.5: PER 39.95%, AUC 0.8270
    - Weight 0.7 (Favore WavLM): PER 31.41%, AUC 0.8478
- **Conclusioni**: L'ensemble migliora leggermente la robustezza generale, ma pesare di più il modello "specializzato" (WavLM) aiuta.

### [20/12/2024] Introduzione Weighted Layer Sum
- **Ipotesi**: I layer intermedi contengono info fonetiche migliori dell'ultimo layer.
- **Risultato**: PER peggiore (30% vs 15%) ma AUC migliore (0.85 vs 0.83).
- **Conclusioni**: L'ultimo layer è specializzato per ASR (trascrizione), i layer medi per caratteristiche acustiche (detection errori).

---

## 📐 Configurazioni Training

| Modello | LR | Batch | Epochs | Mode | Script |
|---------|-----|-------|--------|------|--------|
| Wav2Vec2 | 3e-4 | 4 | 10 | Fine-tuning | `train_wav2vec2.py` |
| Whisper Enc | 1e-4 | 4 | 10 | Partial (unfreeze last 4 encoder layers) | `train_whisper_encoder.py` |
| Qwen2-Audio | 1e-4 | 2 | 10 | **Linear Probe** (encoder frozen, 4-bit) | `train_qwen_audio.py` |
| Wav2Vec2-BERT | 5e-5 | 4 | 10 | Fine-tuning (freeze feature_projection) | `train_w2v2_bert.py` |
| SpeechTokenizer | 1e-4 | 8 | 10 | Discrete tokens → Transformer classifier | `train_speechtokenizer.py` |
| **Early Fusion** | 1e-4 | 2 | 5 | **Frozen+CTC** | `train_early_fusion.py` |
| **Wav2Vec2 Phoneme (lv60-pmp)** | **3e-5** | 4 | 10 | Fine-tuning (domain init) | `train_wav2vec2_phoneme.py` |
| **Data2Vec2 Large** | **3e-5** | 4 | 10 | Fine-tuning (CTC) | `train_data2vec2.py` |
| **DistilHuBERT** | **3e-5** | 4 | 10 | Fine-tuning (CTC) | `train_distilhubert.py` |
| **XLS-R 1B** | **3e-5** | 1 | 10 | Fine-tuning / QLoRA (CTC) | `train_xlsr_1b.py` |
| **MMS-1B** | **3e-5** | 1 | 10 | Fine-tuning / QLoRA (CTC) | `train_mms_1b.py` |
| **M-CTC-T (Meta)** | **1e-5** | 2 | 10 | Fine-tuning (CTC) | `train_mctct.py` |
| **Parakeet-CTC 1.1B** | **3e-5** | 1 | 10 | **Linear Probe (4-bit)** | `train_parakeet.py` |

> **Nota VRAM (1B)**: su 16GB spesso serve `--max-audio-seconds`, bucketing (`group_by_length`) e/o QLoRA (`--load-in-4bit`).

---

## 🧪 Fusion Experiments (Gennaio 2025)

### Late Fusion "Dream Team" (HuBERT + WavLM)

Combinazione dei due top-performer a livello logit:

| α (HuBERT) | PER (HQ) | AUC-ROC | Note |
|------------|----------|---------|------|
| 0.3 | TBD | TBD | Peso maggiore WavLM |
| 0.5 | TBD | TBD | Media semplice |
| 0.7 | TBD | TBD | Peso maggiore HuBERT |

**Script**: `scripts/evaluation/evaluate_hubert_fusion.py`

### Early Fusion Multi-Backbone (UPDATED)

| Parametro | Valore |
|-----------|--------|
| Backbone 1 | HuBERT Large (frozen encoder) |
| Backbone 2 | WavLM Base o WavLM Large (frozen encoder) |
| Features | 1024 + 1024 = 2048D (con WavLM Large) |
| CTC Head | Linear(2048, vocab) |
| fp16 | ✓ |
| 4-bit quantization | ✗ (disabilitata nello script per stabilità) |
| VRAM stimata | ~16-20GB (Large+Large, fp16+checkpointing) |

**Ottimizzazioni memoria**: 
- WavLM Base invece di Large (~95M vs ~317M)
- 4-bit quantization per backbone frozen
- Supporto per encoder già fine-tuned (estrazione automatica da ForCTC)

**Script**: `scripts/training/train_early_fusion.py`
**Args**: `--wavlm-path <your_checkpoint>` `--hubert-path <your_checkpoint>`

### [FAILED] Gated Fusion Experiments
- **Motivazione**: Testare se un gate apprendibile può dinamicamente scegliere il backbone migliore per ogni frame.
- **Config**: Gate Network (Linear + Sigmoid) su concatenazione feature HuBERT+WavLM.
- **Risultati**: ❌ **Total Failure**
    - **HuBERT + WavLM Base**: Gate output collassa a 0.001 (solo WavLM), poi con regolarizzazione a 0.5. PER 96.02%, AUC 0.58. Output corrotto.
    - **HuBERT + WavLM Large**: Stesso comportamento. PER 89.68%, output casuale.
- **Analisi Post-Mortem**: 
    1. **Optimization Difficulty**: Il gate network faticava a convergere senza pesanti regolarizzazioni.
    2. **Feature Mismatch**: La somma pesata di embedding provenienti da spazi latenti diversi (HuBERT vs WavLM) distrugge la semantica necessaria per la CTC.
    3. **Conclusioni**: La concatenazione (Early Fusion) funziona meglio perché preserva entrambe le feature per il layer successivo, invece di mischiarle linearmente.


---

## ✅ Lista Modelli Implementati

1. **Baseline MLP** (`train_baseline_mlp.py`)
2. **WavLM Standard** (`train_wavlm.py`)
3. **WavLM Weighted** (`train_weighted.py`)
4. **HuBERT Large** (`train_hubert.py`)
5. **XLS-R** (`train_xlsr.py`)
6. **Wav2Vec2** (`train_wav2vec2.py`)
7. **Whisper Encoder** (`train_whisper_encoder.py`)
8. **SpeechTokenizer** (`train_speechtokenizer.py`)
9. **Qwen2-Audio** (`train_qwen_audio.py`) - Linear Probe
10. **Wav2Vec2-BERT** (`train_w2v2_bert.py`)
11. **Wav2Vec2 Phoneme (lv60-pmp)** (`train_wav2vec2_phoneme.py`) - Domain Init
12. **Data2Vec2 Large** (`train_data2vec2.py`) - 🆕 NEW
13. **XLS-R 1B** (`train_xlsr_1b.py`) - 🆕 NEW
14. **MMS-1B** (`train_mms_1b.py`) - 🆕 NEW
15. **M-CTC-T (Meta)** (`train_mctct.py`) - 🆕 NEW
16. **Late Fusion HuBERT+WavLM** (`evaluate_hubert_fusion.py`) - 🆕 NEW
17. **Early Fusion Multi-Backbone** (`train_early_fusion.py`) - 🆕 NEW
18. **Parakeet-CTC 1.1B** (`train_parakeet.py`) - 🆕 NEW
19. **DistilHuBERT** (`train_distilhubert.py`) - 🆕 NEW

---

## 🔜 Next Steps

1. ✅ ~~Implementare Late Fusion HuBERT + WavLM~~
2. ✅ ~~Implementare Early Fusion Multi-Backbone~~
3. Eseguire sweep pesi Late Fusion (α ∈ {0.3, 0.5, 0.7})
4. Training Early Fusion su GPU con ≥20GB VRAM
5. Generare report qualitativo con `analyze_model_gap.py`
6. Aggiornare tabella risultati con nuovo record AUC
7. Lanciare benchmark: Data2Vec2 Large / XLS-R 1B / MMS-1B / M-CTC-T

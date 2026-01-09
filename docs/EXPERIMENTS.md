# 📓 Experiments Log - Phoneme Recognition Benchmark

Questo documento traccia cronologicamente gli esperimenti condotti e i risultati ottenuti sul benchmark **SpeechOcean762**.

Per dettagli tecnici sui modelli, vedi **[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md)**.
Per dettagli sulle metriche, vedi **[docs/BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md)**.

---

## 📊 Tabella Riassuntiva Risultati (Gennaio 2025)

| Modello | Input Type | Task A (PER) ↓ | Task B (Pearson) ↑ | Task C (AUC) ↑ | Note |
|---------|------------|----------------|-------------------|----------------|------|
| **HuBERT Large** | Raw Waveform | **8.84%** | **0.5932** | 0.8426 | **🏆 Best Overall** |
| WavLM Base | Raw Waveform | 14.91% | 0.5550 | 0.8369 | Baseline fine-tuned |
| WavLM Large | Raw Waveform | 17.91% | 0.5736 | 0.8382 | Standard CTC |
| Baseline MLP | Raw Waveform | 25.92% | 0.5754 | 0.8427 | Linear Probe |
| WavLM Weighted | Raw Waveform | 30.41% | 0.5812 | **0.8523** | **Best Detection** |
| XLS-R 300M | Raw Waveform | 39.40% | 0.5589 | 0.8435 | Multilingual |
| SpeechTokenizer | Discrete | 60.85% | 0.3842 | 0.7311 | Discrete tokens (lossy) |
| Wav2Vec2-BERT | Mel Spectrogram | 88.58% | 0.3247 | 0.6936 | ❌ Failed (input mismatch) |
| Whisper Encoder | Mel Spectrogram | ~237% | 0.4510 | 0.7963 | ❌ Failed (CTC not aligned) |
| Qwen2-Audio | Mel Spectrogram | TBD | TBD | TBD | Linear Probe (encoder frozen) |

> **⚠️ Key Finding**: Models using **raw waveform input** (`input_values`) all work (PER < 50%). Models using **mel spectrogram** (`input_features`) all fail (PER > 80%). See [ARCHITECTURE_DETAILS.md](ARCHITECTURE_DETAILS.md#8--input-types-vs-performance---critical-analysis) for analysis.

---

## 📅 Log Esperimenti

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
| Whisper Enc | 3e-4 | 4 | 10 | Partial (last 4) | `train_whisper_encoder.py` |
| Qwen2-Audio | 1e-3 | 2 | 10 | **Linear Probe** | `train_qwen_audio.py` |
| Wav2Vec2-BERT | 3e-4 | 4 | 10 | Fine-tuning | `train_w2v2_bert.py` |
| SpeechTokenizer | 1e-3 | 4 | 10 | 2-Stage | `train_speechtokenizer.py` |
| **Early Fusion** | 1e-4 | 2 | 5 | **Frozen+CTC** | `train_early_fusion.py` |
| **Wav2Vec2 Phoneme (lv60-pmp)** | **3e-5** | 4 | 10 | Fine-tuning (domain init) | `train_wav2vec2_phoneme.py` |

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
| Backbone 1 | HuBERT Large (frozen, fine-tuned encoder) |
| Backbone 2 | WavLM Base (frozen, fine-tuned encoder) |
| Features | 1024 + 768 = 1792D |
| CTC Head | Linear(1792, 45) |
| fp16 | ✓ |
| 4-bit quantization | ✓ (bitsandbytes) |
| VRAM stimata | ~8-10GB |

**Ottimizzazioni memoria**: 
- WavLM Base invece di Large (~95M vs ~317M)
- 4-bit quantization per backbone frozen
- Supporto per encoder già fine-tuned (estrazione automatica da ForCTC)

**Script**: `scripts/training/train_early_fusion.py`
**Args**: `--wavlm-path <your_checkpoint>` `--hubert-path <your_checkpoint>`

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
12. **Late Fusion HuBERT+WavLM** (`evaluate_hubert_fusion.py`) - 🆕 NEW
13. **Early Fusion Multi-Backbone** (`train_early_fusion.py`) - 🆕 NEW

---

## 🔜 Next Steps

1. ✅ ~~Implementare Late Fusion HuBERT + WavLM~~
2. ✅ ~~Implementare Early Fusion Multi-Backbone~~
3. Eseguire sweep pesi Late Fusion (α ∈ {0.3, 0.5, 0.7})
4. Training Early Fusion su GPU con ≥20GB VRAM
5. Generare report qualitativo con `analyze_model_gap.py`
6. Aggiornare tabella risultati con nuovo record AUC

# 📓 Experiments Log - Phoneme Recognition Benchmark

Questo documento traccia cronologicamente gli esperimenti condotti e i risultati ottenuti sul benchmark **SpeechOcean762**.

Per dettagli tecnici sui modelli, vedi **[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md)**.
Per dettagli sulle metriche, vedi **[docs/BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md)**.

---

## 📊 Tabella Riassuntiva Risultati (Dicembre 2024)

| Modello | Task A (PER) | Task B (Pearson) | Task C (AUC) | Note |
|---------|--------------|------------------|--------------|------|
| **WavLM Standard** | 14.91% | 0.5550 | 0.8369 | Baseline fine-tuned |
| **WavLM Weighted** | 30.41% | 0.5812 | 0.8523 | Layer sum (ottimo detection) |
| **XLS-R 300M** | 39.40% | 0.5589 | 0.8435 | Multilingual |
| **HuBERT Large** | 8.84% | 0.5932 | 0.8426 | **Best PER & Correlation** |
| **SpeechTokenizer** | 60.85% | 0.3842 | 0.7311 | Discrete tokens (lossy) |
| **Wav2Vec2** | TBD | TBD | TBD | LR=3e-4, 10 epochs |
| **Whisper Enc** | TBD | TBD | TBD | Small encoder, last 4 layers |
| **Qwen2-Audio** | TBD | TBD | TBD | Linear Probe (encoder frozen) |
| **Wav2Vec2-BERT** | TBD | TBD | TBD | Fine-tuning, 10 epochs |

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
10. **Wav2Vec2-BERT** (`train_w2v2_bert.py`) - NEW

---

## 🔜 Next Steps

1. Completare training **Wav2Vec2-BERT** e confrontare con Wav2Vec2 standard.
2. Testare **Qwen2-Audio** Linear Probe per valutare feature multimodali.
3. Investigare perché **HuBERT** performa così meglio di WavLM.

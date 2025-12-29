# 🧪 Benchmark Guide & Metrics

Guida al benchmark scientifico condotto sul dataset SpeechOcean762.

---

## 1. Il Dataset: SpeechOcean762

- **Fonte**: [mispeech/speechocean762](https://huggingface.co/datasets/mispeech/speechocean762)
- **Contenuto**: 2500 frasi pronunciate da bambini cinesi che imparano l'inglese.
- **Annotazioni**:
    - Trascrizione fonetica manuale (ARPABET).
    - Punteggi di qualità pronuncia per parola e frase (0-10) assegnati da 5 esperti umani.
    - Punteggi di intonazione, stress e fluidità.

### Preprocessing per il Benchmark
1.  **Conversione ARPABET → IPA**: Usiamo un mapping custom (vedi `src/data/normalize_ipa.py`) per convertire le annotazioni in IPA standard.
2.  **Split Test**: Usiamo lo split `test` ufficiale (2500 sample) per tutte le valutazioni.

---

## 2. Methodology

Il benchmark valuta i modelli su **3 Task Complementari** per misurare diverse sfaccettature della comprensione fonetica.

### TASK A: ASR Robustness
*Quanto è accurato il modello nel trascrivere pronunce corrette?*

- **Filtro**: Selezioniamo solo i campioni con score umano di pronuncia ≥ 9 (o 8).
- **Metrica**: **PER (Phoneme Error Rate)**.
- **Obiettivo**: Minimizzare il PER (ideale: < 10%).

$$ PER = \frac{Inserzioni + Cancellazioni + Sostituzioni}{Numero Totale Fonemi} $$

### TASK B: Human Score Correlation
*Il modello "soffre" sugli stessi errori che gli umani giudicano negativamente?*

- **Metrica**: Correlazione tra il PER del modello su una frase e lo score di qualità umano.
- **Coefficiente Pearson ($r$)**: Correlazione lineare.
- **Coefficiente Spearman ($\rho$)**: Correlazione di rango (più robusta).
- **Interpretazione**: Ci aspettiamo una **correlazione negativa forte** (Alto PER = Basso Score). Nel codice usiamo `(1-PER)` per ottenere correlazione positiva.

### TASK C: Mispronunciation Detection
*Possiamo usare il modello per flaggare errori di pronuncia?*

- **Setup**: Classificazione binaria.
    - Classe Positiva (Errore): Score umano ≤ 6.
    - Classe Negativa (Corretto): Score umano > 6.
- **Predittore**: Usiamo il PER calcolato dal modello come score di "difettosità".
- **Metriche**:
    - **AUC-ROC**: Capacità di separazione indipendentemente dalla soglia.
    - **F1-Score**: Ottimizzato cercando la soglia di PER migliore.

---

## 3. Comandi di Valutazione

### Valutazione Singolo Modello
Lo script `evaluate_speechocean.py` rileva automaticamente il tipo di modello (WavLM, HuBERT, Whisper, ecc.) dal config.

```bash
python scripts/evaluation/evaluate_speechocean.py --model-path outputs/wavlm_weighted
```

### Valutazione Ensemble
Per valutare la fusione di due modelli:

```bash
python scripts/evaluation/evaluate_fusion.py \
    --model-a outputs/wavlm_weighted \
    --model-b outputs/xlsr \
    --weight 0.6
```

### Valutazione SpeechTokenizer
SpeechTokenizer richiede uno script dedicato a causa del preprocessing diverso (estrazione codici discreti):

```bash
python scripts/evaluation/evaluate_speechtokenizer.py --model-path outputs/speechtokenizer
```

---

## 4. Troubleshooting

### CTC Loss = 0.0, grad_norm = NaN

**Sintomo**: Il training mostra `loss: 0.0` e `grad_norm: nan` fin dal primo step.

**Causa**: Il `Wav2Vec2CTCTokenizer` aggiunge automaticamente token BOS/EOS che non sono nel `vocab.json`, causando un mismatch di vocab size (es. 43 vs 45).

**Diagnosi**: Controlla il log per:
```
Updated tokens: {'eos_token_id': 44, 'bos_token_id': 43}
```

**Soluzione**: Aggiungi `bos_token=None, eos_token=None` alla creazione del tokenizer:
```python
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_path,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
    bos_token=None,  # CRITICAL
    eos_token=None,  # CRITICAL
)
```

### Label Length >= Output Frames

**Sintomo**: CTC fallisce silenziosamente (loss = 0 con `ctc_zero_infinity=True`).

**Causa**: Le trascrizioni IPA sono troppo lunghe rispetto ai frame audio dopo il downsampling (320x per Wav2Vec2).

**Soluzione**: Filtra i sample nel preprocessing:
```python
ds = ds.filter(lambda x: x["label_length"] < x["input_length"] // 320)
```

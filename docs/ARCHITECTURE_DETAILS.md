# 🏗️ Dettagli Implementativi e Architetture Custom

Questo documento fornisce un'analisi approfondita delle soluzioni architetturali personalizzate implementate nel framework. Non si limita a descrivere i modelli standard, ma dettaglia le **modifiche al codice**, le **strategie di adattamento** e le **logiche matematiche** sviluppate specificamente per questo benchmark.

---

## 1. Weighted Layer Sum Strategy (SUPERB)

### 🔬 Il Problema Scientifico
I modelli Self-Supervised Learning (SSL) come WavLM e HuBERT imparano rappresentazioni gerarchiche:
- **Layer Bassi (1-6)**: Codificano feature acustiche grezze (pitch, formanti, rumore di fondo).
- **Layer Medi (7-18)**: Codificano unità fonetiche e sub-lessicali (fonemi, sillabe).
- **Layer Alti (19-24)**: Codificano semantica e sintassi a lungo raggio.

Per un task di **Phoneme Recognition e Valutazione**, l'ultimo layer (output standard di HuggingFace) è spesso sub-ottimale perché troppo specializzato sulla semantica o sul pre-training objective (Masked Prediction). L'informazione fonetica pura risiede spesso nei layer intermedi.

### 🛠️ La Soluzione Implementativa
Abbiamo implementato una classe custom `WavLMWithWeightedLayers` che non usa l'output finale, ma impara a combinare **tutti gli hidden states**.

#### Formula Matematica
L'output $H_{out}$ è una somma pesata dei $L$ hidden states, dove i pesi $w_i$ sono parametri addestrabili.

$$ w'_i = \text{Softmax}(\alpha_i) = \frac{e^{\alpha_i}}{\sum_{j=1}^L e^{\alpha_j}} $$
$$ H_{out} = \sum_{i=1}^L w'_i \cdot H_i $$

#### Codice Custom (`src/training/weighted_model.py`)
Ecco come abbiamo modificato il forward pass standard:

```python
class WavLMWithWeightedLayers(nn.Module):
    def __init__(self, model_name, num_layers=25):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        # Pesi apprendibili inizializzati a 0 (Softmax uniforme all'inizio)
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        
    def forward(self, input_values):
        # 1. Output con hidden_states di tutti i layer
        outputs = self.wavlm(input_values, output_hidden_states=True)
        all_layers = outputs.hidden_states  # Lista di 25 tensori (B, T, 1024)
        
        # 2. Calcolo pesi normalizzati
        weights = F.softmax(self.layer_weights, dim=0)
        
        # 3. Somma pesata
        weighted_sum = 0
        for i, layer_output in enumerate(all_layers):
            weighted_sum += weights[i] * layer_output
            
        return weighted_sum
```

**Vantaggio**: Il modello decide autonomamente quali layer sono utili. Per la pronuncia, tipicamente converge dando peso alto ai layer 12-18.

---

## 2. Whisper Encoder Adapter

### 🔬 La Sfida
Whisper è un modello Encoder-Decoder nato per seq2seq (generazione testo). Per il nostro benchmark serve una predizione **frame-level** (un fonema per ogni timestep audio), simile a CTC. Il decoder autoregressivo di Whisper è inutile e lento per questo scopo.

### 🛠️ Modifiche Custom (`scripts/training/train_whisper_encoder.py`)
Abbiamo "chirurgicamente" isolato l'encoder e aggiunto una testa CTC.

1.  **Rimozione Decoder**: Carichiamo `WhisperModel`, ma scartiamo `model.decoder`.
2.  **CTC Head**: Aggiungiamo un layer lineare sopra l'encoder.
3.  **Input Mismatch**: Whisper vuole Log-Mel Spectrograms, non Raw Audio.

#### Implementazione
```python
class WhisperEncoderCTC(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Carica solo il modulo encoder
        whisper = WhisperModel.from_pretrained("openai/whisper-small")
        self.encoder = whisper.encoder
        
        # CTC Projection: 768 (Hidden) -> Vocab Size
        self.ctc_head = nn.Linear(768, vocab_size)

    def forward(self, input_features):
        # Input: [Batch, 80, Time] (Mel Spectrogram)
        # L'encoder di Whisper fa downsampling 2x
        hidden_states = self.encoder(input_features).last_hidden_state
        
        # Output: [Batch, Time/2, 768]
        logits = self.ctc_head(hidden_states)
        return logits
```

**Nota Tecnica**: L'encoder di Whisper riduce la risoluzione temporale di un fattore 2 rispetto all'input mel. Questo è gestito automaticamente dalla CTC loss che allinea sequenze di lunghezza diversa.

---

## 3. SpeechTokenizer: Architettura Ibrida Discreta

### 🔬 Concetto
SpeechTokenizer converte l'audio continuo in **token discreti** (codici interi da 0 a 1023) usando RVQ (Residual Vector Quantization). L'idea è che la quantizzazione rimuova rumore e variabilità del parlatore, lasciando solo l'informazione linguistica.

### 🛠️ Pipeline a Due Stadi
A differenza degli altri modelli end-to-end, qui abbiamo una catena disgiunta.

#### Stage 1: Estrazione Codici (Frozen)
Usiamo il modello `fnlp/SpeechTokenizer` pre-addestrato.
- **Input**: Audio 16kHz.
- **Processo**: Encoder HuBERT -> Quantizzatore RVQ (8 codebooks).
- **Selezione**: Prendiamo solo il **primo codebook** (RVQ-0) perché contiene l'informazione semantica/fonetica più ricca.
- **Output**: Sequenza di interi, es. `[45, 12, 998, ...]`

#### Stage 2: Classificatore Transformer (Trainable)
Non possiamo usare CTC direttamente su interi. Abbiamo costruito un piccolo Transformer da zero.

```python
class DiscreteTokenClassifier(nn.Module):
    def __init__(self, vocab_size, codebook_size=1024):
        # Embedding: Converte intero 12 -> Vettore denso
        self.emb = nn.Embedding(codebook_size, 256)
        
        # Transformer leggero (2 layer) per contestualizzare i token
        self.encoder = nn.TransformerEncoder(..., num_layers=2)
        
        # Positional Encoding Leaerning
        self.pos_enc = nn.Parameter(...)
        
        self.head = nn.Linear(256, vocab_size) # CTC Output
```

**Perché un Transformer qui?** I codici discreti perdono contesto temporale. Il Transformer permette al modello di guardare "avanti e indietro" nella sequenza di codici per decidere il fonema corretto.

---

## 4. Qwen2-Audio: Linear Probe Mode

### 🔬 La Sfida
Qwen2-Audio ha un encoder audio massiccio (~1 miliardo di parametri). È impossibile fare fine-tuning completo su GPU consumer.

### 🛠️ Soluzione: Linear Probe (Feature Extraction)
**L'encoder è COMPLETAMENTE FROZEN**. Solo una piccola CTC head viene addestrata.
Questo valuta le feature "zero-shot" del modello multimodale.

#### Caricamento 4-bit
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# VRAM: ~5GB invece di ~16GB
```

#### Architettura
```
Audio → Qwen2-Audio Encoder (FROZEN, 4-bit) → CTC Head (TRAINABLE, ~260k params)
```

```python
# Forward pass: encoder completamente disabilitato per gradients
with torch.no_grad():
    hidden = self.audio_encoder(mel_features)
logits = self.ctc_head(hidden)  # Solo questo layer apprende
```

**Parametri Trainabili**: ~260k (solo CTC head)
**VRAM Stimata**: ~5-6GB

---

## 5. Wav2Vec2-BERT 2.0 (Recommended)

### 🔬 Innovazione
W2V-BERT 2.0 combina il **contrastive learning** di Wav2Vec2 con la **masked language modeling** di BERT, ottenendo rappresentazioni audio più ricche e contestualizzate.

### ⚠️ Differenza Chiave da Wav2Vec2
**W2V-BERT 2.0 NON usa audio raw!** Richiede **spettrogrammi log-mel a 80 bin** come input, a differenza di Wav2Vec2/WavLM che lavorano su waveform grezze.

| Aspetto | Wav2Vec2 | W2V-BERT 2.0 |
|---------|----------|---------------|
| **Input** | Raw audio (`input_values`) | Log-mel spectrogram (`input_features`) |
| **Feature Extractor** | `Wav2Vec2FeatureExtractor` | `SeamlessM4TFeatureExtractor` |
| **Processor** | `Wav2Vec2Processor` | `Wav2Vec2BertProcessor` |
| **Subsampling Factor** | 320 (campioni audio) | 2 (frame spettrogramma) |

### 🛠️ Architettura
```
Audio 16kHz → SeamlessM4TFeatureExtractor → Log-Mel (80 bins) → Transformer (24L) → CTC Head
```

**Modello**: `facebook/w2v-bert-2.0`
- 24 Transformer layers, 1024 hidden
- ~600M parametri totali
- SOTA su LibriSpeech

### Implementazione Corretta
```python
from transformers import (
    Wav2Vec2BertProcessor,
    Wav2Vec2BertForCTC,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2CTCTokenizer,
)

# Feature extractor per spettrogrammi log-mel (NON Wav2Vec2FeatureExtractor!)
feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

# Tokenizer IPA custom
tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]")

# Processor che combina entrambi
processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Modello CTC
model = Wav2Vec2BertForCTC.from_pretrained(
    "facebook/w2v-bert-2.0",
    vocab_size=45,
    ctc_loss_reduction="mean",
    ignore_mismatched_sizes=True,
)

# ⚠️ W2V-BERT non ha freeze_feature_encoder()! Usare:
for param in model.wav2vec2_bert.feature_projection.parameters():
    param.requires_grad = False

model.gradient_checkpointing_enable()  # Riduce VRAM

# ⚠️ CRITICO: Reinizializzare lm_head per evitare CTC collapse
nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
nn.init.zeros_(model.lm_head.bias)
```

### Preprocessing Audio
```python
def preprocess(batch):
    audio, sr = librosa.load(batch["audio_path"], sr=16000)
    
    # Il processor genera automaticamente spettrogrammi log-mel
    inputs = processor(audio, sampling_rate=16000, return_tensors=None)
    
    # ⚠️ Usare input_features (spettrogramma), NON input_values (audio raw)
    input_features = inputs.input_features[0]
    
    # Subsampling: W2V-BERT usa fattore 2 sui frame spettrogramma
    input_frames = len(input_features) // 2
    
    return {"input_features": input_features, "labels": labels}
```

**Differenza da Wav2Vec2**: L'obiettivo MLM aggiuntivo migliora la comprensione contestuale, utile per fonemi co-articolati.

---

## 6. Ensemble via Late Fusion

### 🛠️ Implementazione
La fusione avviene a livello di **logits** (prima della softmax), per preservare l'incertezza del modello.

$$ P(y|x) = \text{CTC\_Decode}(\alpha \cdot L_A + (1-\alpha) \cdot L_B) $$

#### Allineamento delle Sequenze
WavLM e XLS-R hanno lo stesso stride (20ms), quindi producono sequenze di lunghezza identica per lo stesso input audio.
Se per qualche motivo (padding diverso) le lunghezze differiscono di 1-2 frame, troncataiamo alla lunghezza minima:

```python
min_len = min(logits_A.shape[1], logits_B.shape[1])
lA = logits_A[:, :min_len, :]
lB = logits_B[:, :min_len, :]
fused = weight * lA + (1 - weight) * lB
```

Questa operazione è sicura perché le differenze sono solo nei bordi di silenzio.

---

## 7. CTC Head Architectures - Confronto Dettagliato

### 🎯 Panoramica

Ogni modello nel benchmark usa una CTC (Connectionist Temporal Classification) head per mappare le rappresentazioni audio ai fonemi IPA. Le architetture variano in base alla **natura dell'input** (continuo vs discreto) e alla **ricchezza contestuale** dell'encoder.

### 📊 Tabella Comparativa

| Modello | Input alla Head | Architettura CTC Head | Params Head | Motivazione |
|---------|-----------------|----------------------|-------------|-------------|
| **W2V-BERT 2.0** | Vettori 1024D (da log-mel) | `Linear(1024 → 43)` | ~44K | Encoder già contestualizzato |
| **MMS 1B** | Vettori 1024D | `Linear(1024 → 43)` | ~44K | Identico a W2V-BERT |
| **Whisper** | Vettori 768D | `Dropout(0.1) → Linear(768 → 43)` | ~33K | Regularizzazione aggiuntiva |
| **Qwen2-Audio** | Vettori 1280D | `Linear(1280→512) → GELU → Dropout(0.1) → Linear(512→43)` | ~680K | MLP più espressivo per linear probe |
| **SpeechTokenizer** | Indici discreti 0-1023 | `Embedding → Transformer(2L) → Linear` | ~1.4M | Token discreti richiedono contestualizzazione |

### 🔬 Dettagli Implementativi

#### W2V-BERT 2.0 / MMS (Pretrained Head)

```python
# Head built-in da HuggingFace
self.lm_head = nn.Linear(hidden_size, vocab_size)  # 1024 → 43
```

**⚠️ Fix Critico - Reinizializzazione lm_head:**

Il checkpoint pretrained ha `lm_head` con shape diversa (vocab originale ≠ nostro IPA vocab). HuggingFace la reinizializza random, ma la distribuzione può causare **CTC collapse** (il modello predice solo blank token).

```python
# Reinizializzazione esplicita per prevenire collapse
nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
nn.init.zeros_(model.lm_head.bias)
```

Questa inizializzazione con std=0.02 (stile BERT) assicura che i logits iniziali siano bilanciati tra tutti i token, evitando il minimo locale dove il modello predice solo blank.

---

#### Whisper Encoder (Custom Head)

```python
class WhisperEncoderForCTC(nn.Module):
    def __init__(self, vocab_size):
        # ...
        hidden_size = 768  # whisper-small
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
```

**Motivazione Dropout**: Whisper encoder è molto potente e tende a overfittare rapidamente. Il dropout aggiuntivo prima della proiezione aiuta la generalizzazione.

---

#### Qwen2-Audio (MLP Head)

```python
self.ctc_head = nn.Sequential(
    nn.Linear(1280, 512),   # Riduzione dimensionalità
    nn.GELU(),              # Non-linearità
    nn.Dropout(0.1),        # Regularizzazione
    nn.Linear(512, vocab_size),
)
```

**Motivazione MLP a 2 Layer**:
1. **Linear Probe Mode**: L'encoder è completamente frozen, quindi la head deve compensare con maggiore capacità
2. **Bottleneck 512**: Riduce da 1280 a 512, forzando una compressione delle feature
3. **GELU**: Attivazione smoother di ReLU, migliora gradient flow
4. **680K params**: Abbastanza espressiva per adattarsi al task senza fine-tuning dell'encoder

---

#### SpeechTokenizer (Transformer Head)

```python
class DiscreteTokenClassifier(nn.Module):
    def __init__(self, vocab_size, codebook_size=1024, embed_dim=256):
        # 1. Embedding per token discreti
        self.embedding = nn.Embedding(codebook_size, embed_dim)  # 1024 → 256
        
        # 2. Positional Encoding apprendibile
        self.pos_encoding = nn.Parameter(torch.zeros(1, 2048, embed_dim))
        
        # 3. Transformer leggero
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024),
            num_layers=2
        )
        
        # 4. CTC projection
        self.lm_head = nn.Linear(embed_dim, vocab_size)
```

**Motivazione Architettura Complessa**:

| Componente | Perché è Necessario |
|------------|---------------------|
| **Embedding** | I token RVQ sono indici interi (0-1023), non vettori - serve trasformarli in spazio continuo |
| **Positional Encoding** | I codici discreti perdono ogni informazione temporale durante la quantizzazione RVQ |
| **Transformer** | Senza un encoder contestuale, i singoli token non sanno nulla dei token adiacenti - il Transformer ricostruisce il contesto |
| **2 Layer** | Compromesso tra capacità e velocità - sufficiente per modellare dipendenze locali fonetiche |

---



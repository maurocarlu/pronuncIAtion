# 🔀 Tecniche di Fusione per Phoneme Recognition

Questo documento descrive in dettaglio le **tre tecniche di fusione** implementate nel progetto per migliorare le performance di riconoscimento fonetico combinando i migliori modelli del benchmark.

---

## Sommario

1. [Introduzione](#introduzione)
2. [Early Fusion](#1-early-fusion---concatenazione-feature)
3. [Late Fusion](#2-late-fusion---combinazione-logits)
4. [Gated Fusion](#3-gated-fusion---gate-apprendibile)
5. [Tabella Comparativa](#tabella-comparativa)
6. [Raccomandazioni](#raccomandazioni)

---

## Introduzione

### Modelli Base Utilizzati

Le tecniche di fusione combinano i seguenti modelli pre-addestrati (fine-tuned sul dataset Aug_Comb):

| Modello | Punti di Forza | Metrica Migliore |
|---------|----------------|------------------|
| **HuBERT Large** | Trascrizione fonetica precisa | PER: 8.84% |
| **WavLM Large Weighted** | Detection errori di pronuncia | AUC: 0.8523 |
| **WavLM Base** | Buon bilanciamento PER/AUC | PER: 14.91% |

### Perché la Fusione?

Ogni modello ha caratteristiche complementari:
- **HuBERT**: Pre-training con target discreti (k-means clustering) → ottimo per trascrizione
- **WavLM**: Pre-training con denoising + contrastive learning → robusto al rumore

La fusione mira a combinare questi punti di forza per ottenere:
- PER migliore del singolo modello
- AUC più alta per detection errori
- Maggiore robustezza su input difficili

---

## 1. Early Fusion - Concatenazione Feature

### 📐 Formula Matematica

```
h_hubert = HuBERT_encoder(audio)      # [batch, time, 1024]
h_wavlm = WavLM_encoder(audio)        # [batch, time, 1024]

h_combined = concat([h_hubert, h_wavlm], dim=-1)  # [batch, time, 2048]

logits = CTC_head(dropout(h_combined))  # [batch, time, vocab]
```

### 🏗️ Architettura

```mermaid
flowchart LR
    A[Audio 16kHz] --> B[HuBERT Large<br/>FROZEN]
    A --> C[WavLM Weighted<br/>FROZEN]
    B --> D[h_hubert<br/>1024D]
    C --> E[h_wavlm<br/>1024D]
    D --> F[Concatenazione]
    E --> F
    F --> G[2048D Features]
    G --> H[Dropout 0.1]
    H --> I[Linear CTC Head<br/>TRAINABLE]
    I --> J[Phonemes IPA]
    
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style I fill:#c8e6c9
```

### ⚙️ Dettagli Tecnici Implementativi

Il cuore dell'implementazione risiede nella classe `EarlyFusionModel` (o `MultiModelEarlyFusion` per n>2 modelli).

**1. Gestione dei Backbone**
I modelli (HuBERT, WavLM) vengono caricati usando `AutoModel`. È fondamentale notare che:
- Vengono congelati (`requires_grad=False`) per evitare "catastrophic forgetting" e ridurre l'impronta VRAM.
- Se attivato `use_weighted_wavlm`, per WavLM viene inizializzato un vettore di pesi paramterico `layer_weights` che apprende a combinare tutti i layer intermedi (weighted sum), mentre per HuBERT si usa tipicamente l'output dell'ultimo layer (`last_hidden_state`).

**2. Allineamento e Concatenazione**
```python
# Forward pass semplificata
with torch.no_grad():
    h_hubert = self.hubert(audio).last_hidden_state  # [B, T, 1024]
    
    # WavLM (opzionale weighted sum)
    outputs_w = self.wavlm(audio, output_hidden_states=True)
    if self.use_weighted:
        weights = F.softmax(self.layer_weights, dim=0)
        h_wavlm = (torch.stack(outputs_w.hidden_states) * weights.view(-1,1,1,1)).sum(dim=0)
    else:
        h_wavlm = outputs_w.last_hidden_state

# Concatenazione
# Importante: Gestione discrepanze di lunghezza (1-2 frame) dovute a padding
min_len = min(h_hubert.size(1), h_wavlm.size(1))
combined = torch.cat([h_hubert[:, :min_len], h_wavlm[:, :min_len]], dim=-1) # [B, T, 2048]
```

**3. La CTC Head**
Per evitare il problema del "CTC Collapse" (dove il modello predice solo blank token all'inizio del training), l'inizializzazione della head è critica:
```python
self.ctc_head = nn.Linear(2048, vocab_size)
# Inizializzazione a media 0 e std 0.02 (stile BERT)
nn.init.normal_(self.ctc_head.weight, mean=0.0, std=0.02)
nn.init.zeros_(self.ctc_head.bias)
```
Il dropout (`0.1`) è applicato *prima* della proiezione lineare.

**4. Quantizzazione**
Abbiamo osservato instabilità con la quantizzazione 4-bit (`bitsandbytes`) quando si concatenano tensori di modelli diversi. Pertanto, l'Early Fusion è eseguita in **FP16** nativo (o FP32), disabilitando esplicitamente il caricamento 4-bit per stabilità.

### 📝 Uso
```bash
python scripts/training/train_early_fusion.py \
    --hubert-path outputs/backup/hubert/final_model \
    --wavlm-path outputs/backup/wavlm_weighted/final_model \
    --epochs 5 \
    --batch-size 2
```

---

## 2. Late Fusion - Combinazione Logits

### 📐 Formula Matematica

```
logits_h = HuBERT_model(audio)   # [batch, time, vocab]
logits_w = WavLM_model(audio)    # [batch, time, vocab]

# Late Fusion con peso α
logits_fused = α * logits_h + (1 - α) * logits_w

# Decodifica CTC
prediction = CTC_decode(logits_fused)
```

### 🏗️ Architettura

```mermaid
flowchart LR
    A[Audio 16kHz] --> B[HuBERT Large<br/>+ CTC Head]
    A --> C[WavLM Weighted<br/>+ CTC Head]
    B --> D[logits_h<br/>vocab D]
    C --> E[logits_w<br/>vocab D]
    D --> F["α · logits_h +<br/>(1-α) · logits_w"]
    E --> F
    F --> G[CTC Decode]
    G --> H[Phonemes IPA]
    
    style F fill:#fff3e0
```

### 🔍 Grid Search dei Pesi

Per trovare il peso ottimale α, testiamo diversi valori:

| α (peso HuBERT) | Trade-off |
|-----------------|-----------|
| 0.3 | Favorisce WavLM (detection) |
| 0.5 | Bilanciato |
| 0.7 | Favorisce HuBERT (trascrizione) |
| 0.9 | Quasi solo HuBERT |

**Risultati attesi** (basati su benchmark esistenti):
- α = 0.9 → Migliore PER (più vicino a HuBERT puro)
- α = 0.5 → Migliore AUC (bilanciamento detection)

### 💡 Vantaggi e Svantaggi

| ✅ Vantaggi | ❌ Svantaggi |
|------------|-------------|
| Nessun training aggiuntivo | Peso fisso per tutti i timestep |
| Facile da implementare | Richiede 2 forward pass |
| Interpretabile (α = importance) | |

### 📝 Uso

```bash
# Singolo peso
python scripts/evaluation/evaluate_hubert_fusion.py \
    --hubert-path outputs/backup/hubert/final_model \
    --wavlm-path outputs/backup/wavlm_weighted/final_model \
    --weight 0.7

# Grid search automatico
python scripts/evaluation/evaluate_hubert_fusion.py \
    --hubert-path outputs/backup/hubert/final_model \
    --wavlm-path outputs/backup/wavlm_weighted/final_model \
    --weight-grid
```

---

## 3. Gated Fusion - Gate Apprendibile

### 📐 Formula Matematica

A differenza di Late Fusion che usa un peso fisso α, Gated Fusion apprende un **gate dinamico** per ogni timestep. La formulazione cambia leggermente se usiamo 2 o 3 modelli.

**Caso 2 Modelli (Sigmoid Gate):**
```python
gate_input = concat([h_a, h_b])          # [batch, time, dim_a+dim_b]
gate = sigmoid(Linear(gate_input))       # [batch, time, 1] ∈ (0, 1)

h_fused = gate * h_a + (1 - gate) * h_b
```

**Caso 3 Modelli (Softmax Gate):**
```python
gate_input = concat([h_a, h_b, h_c])     # [batch, time, dim_tot]
gates = softmax(Linear(gate_input))      # [batch, time, 3]

h_fused = gates[0]*h_a + gates[1]*h_b + gates[2]*h_c
```

### 🏗️ Architettura (Caso 2 Modelli)

```mermaid
flowchart TB
    A[Audio 16kHz] --> B[HuBERT Large<br/>FROZEN]
    A --> C[WavLM Weighted<br/>FROZEN]
    B --> D[h_hubert<br/>1024D]
    C --> E[h_wavlm<br/>1024D]
    
    D --> F[Concat 2048D]
    E --> F
    F --> G["Gate Network<br/>Linear(2048→1) + σ<br/>TRAINABLE"]
    G --> H[gate ∈ 0,1]
    
    D --> I["gate · h_hubert"]
    E --> J["(1-gate) · h_wavlm"]
    H --> I
    H --> J
    I --> K[h_fused 1024D]
    J --> K
    
    K --> L[Dropout 0.1]
    L --> M["CTC Head<br/>Linear(1024→vocab)<br/>TRAINABLE"]
    M --> N[Phonemes IPA]
    
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style G fill:#c8e6c9
    style M fill:#c8e6c9
```

### ⚙️ Parametri Trainabili

| Componente | Parametri | Trainabile |
|------------|-----------|------------|
| HuBERT Large | 317M | ❌ Frozen |
| WavLM Large | 317M | ❌ Frozen |
| Gate Network (2048→1) | 2049 | ✅ |
| CTC Head (1024→vocab) | ~44K | ✅ |
| **Totale Trainabile** | ~46K | - |

### 🧠 Interpretabilità del Gate

Il gate fornisce insight su quale backbone è preferito per ogni contesto:

```python
# Dopo il training, analizzare i gate values
gate_values = model.get_gate_statistics(audio)

# gate ≈ 1.0 → il modello preferisce HuBERT (fonemi chiari)
# gate ≈ 0.0 → il modello preferisce WavLM (audio rumoroso)
# gate ≈ 0.5 → entrambi ugualmente utili
```

**Pattern attesi**:
- Fonemi vocalici chiari → gate alto (HuBERT)
- Consonanti in contesto rumoroso → gate basso (WavLM)
- Transizioni fonetiche → gate intermedio

### 💡 Vantaggi e Svantaggi

| ✅ Vantaggi | ❌ Svantaggi |
|------------|-------------|
| Peso adattivo per contesto | Richiede training |
| Interpretabilità (analisi gate) | Parametri extra (pochi) |
| Output 1024D (meno CTC params) | Leggermente più complesso |
| Può specializzarsi per tipo fonema | **Instabilità Training** (vedi sotto) |

### 📝 Uso

```bash
python scripts/training/train_multi_fusion.py \
    --model-a outputs/backup/hubert/final_model \
    --model-b outputs/backup/wavlm_weighted/final_model \
    --fusion-type gated \
    --epochs 5
```

### ⚠️ Analisi Fallimenti (Gennaio 2026)

Durante i test approfonditi sul dataset SpeechOcean762, questa tecnica ha mostrato **gravi problemi di ottimizzazione**, rendendola MENO efficace di Early e Late Fusion.

**Problemi riscontrati:**
1. **Gate Collapse**: Senza regolarizzazione, il gate tende a collassare su 0 o 1 nei primi step, ignorando completamente uno dei due backbone.
   - *Fix tentato*: Introdotta entropy regularization (`--gate-reg`). Il gate si bilancia (~0.5), ma il modello non impara comunque.
2. **CTC Loss Convergence Failure**: Anche con gate bilanciato, la loss scende ma l'output rimane "garbage" (CER/PER ≈ 100%).
3. **Incompatibilità Rappresentazionale**: È probabile che gli spazi latenti di HuBERT e WavLM siano troppo disallineati perché una somma pesata lineare (anche se gated) produca feature coerenti per la CTC head.

**Raccomandazione attuale**: Usare **Early Fusion** (concatenazione) o **Late Fusion** (ensemble soft), che si sono dimostrate molto più robuste.


---

## Tabella Comparativa

| Aspetto | Early Fusion | Late Fusion | Gated Fusion |
|---------|--------------|-------------|--------------|
| **Trainable Params** | ~88K | 0 | ~46K |
| **Richiede Training** | ✅ Sì | ❌ No | ✅ Sì |
| **Peso Dinamico** | Implicito | ❌ Fisso | ✅ Per timestep |
| **VRAM Inference** | ~8GB | ~16GB (2x) | ~8GB |
| **Interpretabilità** | Bassa | Alta (α) | Alta (gate) |
| **Output Dim** | 2048D | vocab | 1024D |
| **Complessità** | Media | Bassa | Media |

### Quando Usare Cosa

```mermaid
flowchart TD
    A[Scegli Tecnica] --> B{Training possibile?}
    B -->|No| C[Late Fusion]
    B -->|Sì| D{Interpretabilità importante?}
    D -->|Sì| E[Gated Fusion]
    D -->|No| F{VRAM limitata?}
    F -->|Sì| G[Gated Fusion]
    F -->|No| H[Early Fusion]
    
    style C fill:#fff3e0
    style E fill:#c8e6c9
    style G fill:#c8e6c9
    style H fill:#e1f5fe
```

---

## Raccomandazioni

### Per Migliore PER (Trascrizione)
1. **Late Fusion α=0.9** - Quasi solo HuBERT, minimo overhead
2. **Early Fusion** - Se hai tempo per training

### Per Migliore AUC (Detection Errori)
1. **Gated Fusion** - Può specializzare gate per errori
2. **Late Fusion α=0.5** - Bilanciamento detection

### Per Massima Interpretabilità
1. **Gated Fusion** - Analisi gate values per timestep
2. **Late Fusion** - α fisso e interpretabile

### Per Risorse Limitate
1. **Late Fusion** - Zero training
2. **Gated Fusion** - Meno params di Early

---

## 4. Multi-Model Fusion (2 o 3 modelli)

Per combinazioni flessibili di più modelli, usare `train_multi_fusion.py`:

### Combinazioni Supportate

| Comb. | Modello A | Modello B | Modello C |
|-------|-----------|-----------|-----------|
| 2-way | HuBERT | WavLM Weighted | - |
| 2-way | HuBERT | WavLM Base | - |
| 2-way | WavLM Weighted | WavLM Base | - |
| 3-way | HuBERT | WavLM Weighted | WavLM Base |

### Esempi Uso

```bash
# Early Fusion con 2 modelli
python scripts/training/train_multi_fusion.py \
    --model-a outputs/backup/hubert/final_model \
    --model-b outputs/backup/wavlm_weighted/final_model \
    --fusion-type early \
    --epochs 5

# Gated Fusion con 3 modelli (tripla fusione)
python scripts/training/train_multi_fusion.py \
    --model-a outputs/backup/hubert/final_model \
    --model-b outputs/backup/wavlm_weighted/final_model \
    --model-c outputs/backup/wavlm_base/final_model \
    --fusion-type gated \
    --epochs 5

# Late Fusion Grid Search con 2 modelli
python scripts/training/train_multi_fusion.py \
    --model-a outputs/backup/hubert/final_model \
    --model-b outputs/backup/wavlm_base/final_model \
    --fusion-type late \
    --weight-grid
```

### Architettura Tripla Fusione

```mermaid
flowchart LR
    A[Audio] --> B[HuBERT]
    A --> C[WavLM Weighted]
    A --> D[WavLM Base]
    B --> E[h_a 1024D]
    C --> F[h_b 1024D]
    D --> G[h_c 1024D]
    E --> H[Concat 3072D]
    F --> H
    G --> H
    H --> I[CTC Head]
    I --> J[Phonemes]
```

---

## Riferimenti

- [SUPERB Benchmark](https://superbbenchmark.org/) - Weighted layer approach
- [HuBERT Paper](https://arxiv.org/abs/2106.07447) - Self-supervised speech
- [WavLM Paper](https://arxiv.org/abs/2110.13900) - Denoising pre-training
- [Gated Multimodal Units](https://arxiv.org/abs/1702.01992) - Gating for fusion

---

*Ultimo aggiornamento: Gennaio 2026*

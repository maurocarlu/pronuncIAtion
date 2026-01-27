# L-MAC Technical Reference

## 1. Overview
L‑MAC (Listenable Maps for Audio Classifiers) è un metodo di interpretabilità **post‑hoc** progettato per modelli di riconoscimento fonemico. Utilizza un decoder leggero (1D U-Net) per generare una maschera temporale $m \in [0,1]$ che, applicata alla waveform originale, isola le componenti acustiche "causali" per la predizione di un determinato fonema.

### Concetto Chiave
Dato un audio $x$, L-MAC produce:
$$x_{map} = x \odot m$$
Dove $x_{map}$ è l'audio "spiegazione" che, se ascoltato, dovrebbe evidenziare solo i suoni che il modello ha utilizzato per riconoscere il fonema target.

---

## 2. Architettura del Sistema

Il sistema L-MAC si compone di tre parti:
1.  **Backbone (Congelato)**: Il modello di classificazione da spiegare (HuBERT, Early Fusion, o Late Fusion).
2.  **Feature Extractor**: Selezione e concatenazione di hidden states specifici dal backbone.
3.  **L-MAC Decoder (Trainable)**: Una U-Net 1D leggera che mappa le feature in una maschera scalare.

### 2.1 Feature Extraction & Hidden States
L-MAC non utilizza l'ultimo layer, ma una combinazione di layer intermedi per catturare informazioni sia acustiche che linguistiche. 

**Configurazione Standard:**
- **Layer IDs**: 6, 12, 18, 24 (su 24 layer totali di un Transformer Large).

La strategia di estrazione varia in base al backbone:

| Backbone | Strategia di Estrazione Stati | Dimensione Feature ($C_{in}$) |
| :--- | :--- | :--- |
| **HuBERT Large** | Estrae i layer [6, 12, 18, 24] e li concatena lungo il canale. | $1024 \times 4 = 4096$ |
| **Early Fusion** | Estrae i layer [6, 12, 18, 24] da **HuBERT** E i layer [6, 12, 18, 24] da **WavLM**. Concatena tutto. <br> *Nota: Usa gli stati raw di WavLM, non la somma pesata usata dalla head.* | $(1024 + 1024) \times 4 = 8192$ |
| **Late Fusion** | Identifica il **modello dominante** (quello con peso maggiore nell'ensemble). Estrae i layer [6, 12, 18, 24] solo da esso. | $1024 \times 4 = 4096$ |

### 2.2 Il Decoder (1D U-Net)
Il decoder è progettato per essere leggero (~1.5M parametri) per evitare che impari a classificare autonomamente (camuffando il backbone).
- **Input**: Feature map $[B, C_{in}, T_{feat}]$ (dove $T_{feat}$ è downsampled di 320x rispetto all'audio).
- **Struttura**: 3 blocchi di Encoder, bottleneck, 3 blocchi di Decoder con skip connections.
- **Output**: Maschera $[B, 1, T_{feat}]$, interpolata linearmente a lunghezza waveform $T_{wave}$.
- **Conditioning (Multi-Phoneme)**: Se il sistema è in modalità multi-fonema, l'ID del fonema target viene embeddato ($Dim=64$) e concatenato alle feature di input, permettendo al decoder di generare maschere diverse per fonemi diversi sullo stesso audio.

---

## 3. Training & Loss Function

Il decoder viene addestrato mantenendo il backbone congelato. La loss function bilancia tre obiettivi:

$$L_{total} = L_{fid\_in} + \lambda_{out} L_{fid\_out} + \lambda_{reg} L_{reg}$$

1.  **In-Mask Fidelity ($L_{fid\_in}$)**:
    Minimizza la differenza tra la probabilità del target sull'audio originale e sull'audio mascherato.
    $$L_{fid\_in} = | P(y|x) - P(y|x \odot m) |$$
    *Obiettivo: La spiegazione deve essere sufficiente per la predizione.*

2.  **Out-Mask Fidelity ($L_{fid\_out}$)**:
    Massimizza la distanza (o minimizza la probabilità target) sulla parte di audio rimossa (maschera inversa $1-m$).
    *Obiettivo: Le parti scartate non devono contenere il fonema.*

3.  **Regularization ($L_{reg}$)**:
    Norma $L_1$ sulla maschera per incoraggiare la sparsità.
    $$L_{reg} = ||m||_1$$
    *Obiettivo: Selezionare solo i millisecondi strettamente necessari.*

---

## 4. Metriche di Valutazione (XAI)

Le spiegazioni vengono valutate quantitativamente usando metriche standard di fedeltà (usando SpeechOcean762 come test set).

### Average Deletion (AD) $\downarrow$ (Lower is better)
Misura quanto cala la predizione rimuovendo la spiegazione.
- Un valore basso (es. < 0.2) è ottimo: significa che rimuovendo la maschera (quindi tenendo solo il rumore di fondo), il modello non riconosce più il fonema.
- **Formula**: $AD = \text{Mean} \left( \frac{P_{orig} - P_{masked}}{P_{orig}} \right)$
- *Interpretazione*: Vogliamo che $P_{masked}$ sia alta (la spiegazione contiene le info), quindi $P_{orig} - P_{masked}$ deve essere piccolo.

### Average Increase (AI) $\uparrow$ (Higher is better)
Misura la percentuale di casi in cui la confidenza del modello sulla spiegazione isolata è addirittura **superiore** a quella sull'audio completo (rimuovendo rumore distrattore).
- **Formula**: $\text{Count}(P_{masked} > P_{orig}) / N \times 100$

---

## 5. Note Operative e Parametri

- **Layer Selezionati**: [6, 12, 18, 24] (Configurabile in `lmac_core.py`).
- **Frequenza Campionamento**: 16kHz.
- **Vocabulary Size**: 43 fonemi (standard CMU/IPA mappati).
- **Dataset**: Addestrato e valutato su SpeechOcean762 (Full).

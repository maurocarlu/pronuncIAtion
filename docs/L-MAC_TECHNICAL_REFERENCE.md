# L-MAC Technical Reference

## Overview
L‑MAC (Listenable Maps for Audio Classifiers) è un metodo di interpretabilità **post‑hoc** basato su un decoder leggero che genera una maschera temporale $m \in [0,1]$ applicata alla waveform.

Dato un audio $x$, la maschera produce l’audio spiegazione:

$$x_{map} = x \odot m$$

L’obiettivo è isolare **i millisecondi critici** che causano una specifica decisione del modello (es. un fonema errato), mantenendo la “listenability” dell’audio spiegazione.

## Post‑hoc decoder‑based interpretation
Il backbone rimane **congelato** (HuBERT Large o Early Fusion). Un decoder 1D (U‑Net leggero) apprende a filtrare il segnale acustico in modo che la predizione del modello resti coerente sulla porzione evidenziata.

Questa strategia è *post‑hoc* perché:
- non modifica i pesi del classificatore;
- non richiede gradienti rispetto ai layer del backbone per spiegare;
- produce un segnale direttamente ascoltabile (waveform mascherata).

## Loss L‑MAC (Fedeltà + Sparsità)
L’obiettivo ottimizza tre componenti:

1. **Fedeltà In‑mask** $L_{fid\_in}$:
   mantiene la predizione del modello quando l’input è mascherato.

2. **Fedeltà Out‑mask** $L_{fid\_out}$:
   riduce la probabilità della classe target quando l’input è la parte rimossa.

3. **Regolarizzazione** $L_{reg}$:
   penalizza maschere dense (sparsità), usando $L_1$.

## Vantaggi rispetto ai metodi a gradiente (IG, Saliency, Grad‑CAM)
Rispetto a metodi basati su gradiente, L‑MAC:
- **Produce spiegazioni più stabili** (meno rumore ad alta frequenza).
- **Migliora la fedeltà** (AI/AD superiori nel paper).
- **Permette l’ascolto diretto** delle regioni causali (listenability).

## Interpretazione della maschera in relazione ai fonemi IPA
La maschera può essere letta come una **selezione temporale** dei segmenti acustici responsabili della decisione del fonema target. In pratica:
- un picco della maschera indica la porzione audio **causale**;
- la sovrapposizione con la trascrizione IPA consente di localizzare **errori di pronuncia**;
- quando il target è un singolo fonema IPA, L‑MAC enfatizza i segmenti associati a quel fonema.

## Metriche AI/AD (Sezione 4.1 del paper)
**Average Drop (AD) ↓**:

$$AD = \frac{1}{N} \sum_{i=1}^N \frac{\max(0, f(x)_c - f(x \odot m)_c)}{f(x)_c}$$

**Average Increase (AI) ↑**:

Percentuale di casi in cui $f(x \odot m)_c > f(x)_c$.

## Note Implementative
- Layer consigliati per HuBERT Large: **[6, 12, 18, 24]**.
- Maschera applicata alla **waveform comune**.
- Decoder leggero (U‑Net 1D) per evitare che “impari a classificare”.

## Riferimenti
- L‑MAC: Listenable Maps for Audio Classifiers (paper di riferimento).
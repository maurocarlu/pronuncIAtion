# 📚 Riferimenti Bibliografici

Questo documento contiene le citazioni scientifiche per le tecnologie utilizzate nel progetto.

---

## WavLM (Large)

Paper fondamentale che introduce il modello e la pre-training strategy "Masked Speech Prediction + Denoising".

> **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing**
> 
> *Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, Furu Wei*
> 
> **IEEE Journal of Selected Topics in Signal Processing**, 2022
> 
> [arXiv:2110.13900](https://arxiv.org/abs/2110.13900)

**Concetto Chiave:** Dimostra che il pre-training con rumore (denoising) rende il modello robusto per task non solo semantici (ASR) ma anche acustici (Speaker Verification).

---

## XLS-R (Modello Multilingua)

Giustifica l'uso di un modello diverso per catturare accenti non nativi (L2 speakers).

> **XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale**
> 
> *Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, Michael Auli*
> 
> **Interspeech 2022**
> 
> [arXiv:2111.09296](https://arxiv.org/abs/2111.09296)

**Concetto Chiave:** Addestrato su 128 lingue, impara rappresentazioni fonetiche universali che aiutano quando l'input è pronunciato da speaker non nativi.

---

## Data2Vec 2.0

Paper che introduce l'obiettivo di self-distillation per SSL su audio (e altre modalità).

> **data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language**
>
> *Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli*
>
> **ICML 2022**
>
> [arXiv:2202.03555](https://arxiv.org/abs/2202.03555)

---

## MMS (Massively Multilingual Speech)

Risorsa di riferimento per il pre-training multilingua su 1000+ lingue (1B params) e il rilascio dei checkpoint MMS.

> **Scaling Speech Technology to 1,000+ Languages**
>
> *Meta AI*
>
> [Blog/Release](https://ai.facebook.com/blog/massively-multilingual-speech/)

---

## Whisper

Paper che descrive Whisper e il training su larga scala.

> **Robust Speech Recognition via Large-Scale Weak Supervision**
>
> *Alec Radford et al.*
>
> [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)

---

## M-CTC-T / Parakeet-CTC (Model Cards)

Per i modelli CTC specifici usati nel benchmark, i dettagli più aggiornati (architettura, input, processor) sono nelle rispettive model card:

- `speechbrain/m-ctc-t-large`: https://huggingface.co/speechbrain/m-ctc-t-large
- `nvidia/parakeet-ctc-1.1b`: https://huggingface.co/nvidia/parakeet-ctc-1.1b

---

## SUPERB Benchmark (Weighted Layers)

L'idea di sommare pesatamente i layer per task diversi. Dimostra che ogni layer di un modello speech contiene informazioni diverse:
- **Layer bassi** → Speaker/Acustica
- **Layer alti** → Contenuto/Semantica

> **SUPERB: Speech processing Universal PERformance Benchmark**
> 
> *Shu-wen Yang, Po-Han Chi, Yung-Sung Chuang, Cheng-I Jeff Lai, Kushal Lakhotia, Yist Y. Lin, Andy T. Liu, Jiatong Shi, Xuankai Chang, Guan-Ting Lin, Tzu-Hsien Huang, Wei-Cheng Tseng, Ko-tik Lee, Da-Rong Liu, Zili Huang, Shuyan Dong, Shang-Wen Li, Shinji Watanabe, Abdelrahman Mohamed, Hung-yi Lee*
> 
> **Interspeech 2021**
> 
> [arXiv:2105.01051](https://arxiv.org/abs/2105.01051)

**Concetto Chiave (Sezione 3):** Descrivono esplicitamente la tecnica "Weighted Sum" usata nei task downstream:

> *"The downstream model takes the weighted sum of hidden states from different layers of the upstream model... allow[ing] the downstream model to gather information from different layers."*

---

## Ablation Study Design

Per un confronto rigoroso, il progetto include tre configurazioni:

| Modello | Architettura | Scopo |
|---------|--------------|-------|
| WavLM Base (Standard) | Baseline | Riferimento iniziale |
| WavLM Large (Standard) | Ablation | Isola il contributo della dimensione del modello |
| WavLM Large (Weighted) | Proposto | Valuta il contributo dei layer pesati |

**Ipotesi Scientifica:**
- Il Large Standard avrà un **PER più basso** del Weighted (l'ultimo layer è ottimo per trascrivere correttamente)
- Ma avrà una **AUC più bassa** nel detection rispetto al Weighted (ignora i dettagli acustici dei primi layer)

Se questa ipotesi si verifica, si ha la prova che l'architettura a layer pesati è specifica per la valutazione della pronuncia.

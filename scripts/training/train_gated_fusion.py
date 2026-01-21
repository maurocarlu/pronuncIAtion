#!/usr/bin/env python3
"""
==============================================================================
GATED FUSION TRAINING - Phoneme Recognition con Gate Apprendibile
==============================================================================

Questo script implementa la tecnica di Gated Fusion per combinare due backbone
audio (HuBERT e WavLM) usando un meccanismo di gating apprendibile.

MOTIVAZIONE SCIENTIFICA:
------------------------
A differenza di Early Fusion (concatenazione statica, 2048D) e Late Fusion 
(peso α fisso), Gated Fusion apprende un gate DINAMICO per ogni timestep:

    gate = σ(W · [h_hubert; h_wavlm] + b)    # gate ∈ [0, 1]
    h_fused = gate * h_hubert + (1 - gate) * h_wavlm

VANTAGGI:
---------
1. Il gate può adattarsi al contesto acustico:
   - Fonemi vocalici chiari → gate ≈ 1 (preferisce HuBERT, migliore per PER)
   - Audio rumoroso/consonanti → gate ≈ 0 (preferisce WavLM, più robusto)
   
2. Interpretabilità: analizzando i gate values si può capire quale backbone
   contribuisce di più per ogni tipo di fonema.
   
3. Output a 1024D invece di 2048D → CTC head più leggera

ARCHITETTURA:
-------------
    Audio → HuBERT (frozen) → h₁ (1024D) ─┐
                                           ├→ Gate σ(W·[h₁;h₂]) ─→ gate*h₁ + (1-gate)*h₂
    Audio → WavLM (frozen)  → h₂ (1024D) ─┘                                    │
                                                                               ↓
                                                                    Dropout → CTC Head → Phonemes

PARAMETRI TRAINABILI:
---------------------
- Gate Network: Linear(2048 → 1) = 2049 parametri
- CTC Head: Linear(1024 → vocab) ≈ 44K parametri
- Totale: ~46K parametri (vs 88K di Early Fusion)

REQUISITI MEMORIA:
------------------
- Entrambi i modelli Large (~317M each) = ~2.2GB pesi (frozen, no gradients)
- Con fp16 e gradient_checkpointing: ~12-16GB VRAM (meno di Early Fusion)
- Inference: ~8GB VRAM

USO:
----
    # Training standard
    python scripts/training/train_gated_fusion.py \\
        --hubert-path outputs/backup/hubert/final_model \\
        --wavlm-path outputs/backup/wavlm_weighted/final_model \\
        --epochs 5 \\
        --batch-size 2 \\
        --output-dir outputs/gated_fusion

    # Resume da checkpoint
    python scripts/training/train_gated_fusion.py \\
        --resume \\
        --output-dir outputs/gated_fusion

    # Analisi gate values post-training
    python scripts/training/train_gated_fusion.py \\
        --analyze-gates \\
        --model-path outputs/gated_fusion/final_model

Autore: DeepLearning-Phoneme Project
Creato: Gennaio 2026
Riferimenti:
  - Gated Multimodal Units: https://arxiv.org/abs/1702.01992
  - HuBERT: https://arxiv.org/abs/2106.07447
  - WavLM: https://arxiv.org/abs/2110.13900
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import sys
import os
import warnings
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
import evaluate
from datasets import load_dataset, Audio, Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    HubertModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.models.wavlm import WavLMModel
import shutil
from tqdm import tqdm

# Sopprime warning non critici
warnings.filterwarnings("ignore")

# Aggiungi project root al path per import locali
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# DRIVE BACKUP CALLBACK
# =============================================================================

class DriveBackupCallback(TrainerCallback):
    """
    Callback per backup automatico checkpoint su Google Drive o Kaggle.
    
    Rileva automaticamente l'ambiente (Colab/Kaggle/Local) e copia i checkpoint
    nella directory appropriata per prevenire perdita di progresso durante
    sessioni lunghe su piattaforme cloud.
    
    Attributes:
        backup_dir: Directory di destinazione per backup.
        env: Ambiente rilevato ('colab', 'kaggle', 'local').
    """
    
    def __init__(self, backup_dir: str = None):
        """
        Inizializza callback con rilevamento automatico ambiente.
        
        Args:
            backup_dir: Path opzionale per backup. Se None, usa default per ambiente.
        """
        self.backup_dir = backup_dir
        
        # Rileva ambiente automaticamente
        if '/content' in os.getcwd() or 'COLAB_GPU' in os.environ:
            self.env = 'colab'
            if not backup_dir:
                self.backup_dir = '/content/drive/MyDrive/phoneme_checkpoints'
        elif '/kaggle' in os.getcwd():
            self.env = 'kaggle'
            if not backup_dir:
                self.backup_dir = '/kaggle/working/drive_backup'
        else:
            self.env = 'local'
            if not backup_dir:
                self.backup_dir = None  # Nessun backup su locale di default
    
    def on_save(self, args, state, control, **kwargs):
        """
        Callback chiamato dopo ogni salvataggio checkpoint.
        
        Copia il checkpoint nella directory di backup se configurata.
        """
        if not self.backup_dir:
            return
        
        # Evita copia ricorsiva se già su Drive
        if self.env == 'colab' and '/drive/' in str(args.output_dir):
            return
        
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        
        if checkpoint_dir.exists():
            os.makedirs(self.backup_dir, exist_ok=True)
            model_name = Path(args.output_dir).name
            backup_path = Path(self.backup_dir) / model_name / checkpoint_dir.name
            
            # Evita copia su sé stesso
            if checkpoint_dir.resolve() == backup_path.resolve():
                return
            
            try:
                if self.env == 'colab':
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.copytree(checkpoint_dir, backup_path)
                    print(f"\n💾 Checkpoint copiato su Drive: {backup_path}")
                elif self.env == 'kaggle':
                    # Su Kaggle, comprimi per ridurre I/O
                    zip_path = Path(self.backup_dir) / f"{model_name}_checkpoint-{state.global_step}"
                    shutil.make_archive(str(zip_path), 'zip', checkpoint_dir)
                    print(f"\n💾 Checkpoint compresso: {zip_path}.zip")
            except Exception as e:
                print(f"\n⚠️ Backup fallito: {e}")


# =============================================================================
# GATED FUSION MODEL
# =============================================================================

class GatedFusionModel(nn.Module):
    """
    Gated Fusion per Phoneme Recognition con gate apprendibile.
    
    A differenza di Early Fusion (concatenazione 2048D) e Late Fusion (peso α fisso),
    questo modello apprende un gate DINAMICO per ogni timestep che decide quanto
    pesare ciascun backbone.
    
    Formula:
    --------
        h_hubert = HuBERT_encoder(audio)              # [batch, time, 1024]
        h_wavlm = WavLM_encoder(audio)                # [batch, time, 1024]
        
        gate_input = concat([h_hubert, h_wavlm])      # [batch, time, 2048]
        gate = sigmoid(Linear(gate_input))            # [batch, time, 1], range [0,1]
        
        h_fused = gate * h_hubert + (1 - gate) * h_wavlm  # [batch, time, 1024]
        logits = CTC_head(dropout(h_fused))           # [batch, time, vocab]
    
    Interpretazione Gate:
    ---------------------
        gate ≈ 1.0 → Il modello preferisce HuBERT (fonemi chiari, trascrizione)
        gate ≈ 0.0 → Il modello preferisce WavLM (rumore, robustezza)
        gate ≈ 0.5 → Entrambi i backbone contribuiscono ugualmente
    
    Args:
        vocab_size: Dimensione vocabolario IPA (tipicamente 43 tokens).
        hubert_name: Nome/path modello HuBERT HuggingFace o checkpoint locale.
        wavlm_name: Nome/path modello WavLM HuggingFace o checkpoint locale.
        freeze_backbones: Se True (default), congela completamente i backbone.
        dropout_rate: Dropout prima della CTC head (default 0.1).
        use_weighted_wavlm: Se True, usa Weighted Layer Sum per WavLM.
    
    Attributes:
        hubert: HuBERT encoder (frozen).
        wavlm: WavLM encoder (frozen).
        gate_network: Linear layer per calcolo gate.
        dropout: Dropout layer.
        ctc_head: Linear projection to vocab.
        layer_weights: Pesi apprendibili per WavLM layers (se use_weighted).
    """
    
    def __init__(
        self,
        vocab_size: int = 43,
        hubert_name: str = "facebook/hubert-large-ls960-ft",
        wavlm_name: str = "microsoft/wavlm-large",
        freeze_backbones: bool = True,
        dropout_rate: float = 0.1,
        use_weighted_wavlm: bool = True,
    ):
        super().__init__()
        
        # =====================================================================
        # STEP 1: Logging configurazione
        # =====================================================================
        print(f"\n{'='*60}")
        print("GATED FUSION MODEL - Inizializzazione")
        print(f"{'='*60}")
        print(f"   HuBERT: {hubert_name}")
        print(f"   WavLM:  {wavlm_name}")
        print(f"   Weighted WavLM: {use_weighted_wavlm}")
        print(f"   Freeze backbones: {freeze_backbones}")
        print(f"   Dropout rate: {dropout_rate}")
        
        # =====================================================================
        # STEP 2: Carica HuBERT backbone
        # Supporta sia HubertModel base che HubertForCTC (checkpoint fine-tuned)
        # =====================================================================
        print(f"\n📦 Loading HuBERT from: {hubert_name}")
        try:
            # Prima prova a caricare come HubertForCTC (modello fine-tuned)
            from transformers import HubertForCTC
            hubert_full = HubertForCTC.from_pretrained(
                hubert_name,
                torch_dtype=torch.float16,  # FP16 per ridurre VRAM
            )
            self.hubert = hubert_full.hubert
            hubert_size = "Large" if self.hubert.config.hidden_size >= 1024 else "Base"
            print(f"   ✓ HuBERT: Loaded from ForCTC ({hubert_size}, {self.hubert.config.hidden_size}D)")
        except Exception as e:
            # Fallback: carica come HubertModel base
            print(f"   ⚠️ ForCTC load failed: {e}")
            self.hubert = HubertModel.from_pretrained(
                hubert_name,
                output_hidden_states=False,
                torch_dtype=torch.float16,
            )
            hubert_size = "Large" if self.hubert.config.hidden_size >= 1024 else "Base"
            print(f"   ✓ HuBERT: Loaded as base Model ({hubert_size}, {self.hubert.config.hidden_size}D)")
        
        # =====================================================================
        # STEP 3: Carica WavLM backbone
        # Supporta sia WavLMModel base che WavLMForCTC (checkpoint fine-tuned)
        # =====================================================================
        print(f"\n📦 Loading WavLM from: {wavlm_name}")
        try:
            # Prima prova a caricare come WavLMForCTC (modello fine-tuned)
            from transformers import WavLMForCTC
            wavlm_full = WavLMForCTC.from_pretrained(
                wavlm_name,
                torch_dtype=torch.float16,
            )
            self.wavlm = wavlm_full.wavlm
            wavlm_size = "Large" if self.wavlm.config.hidden_size >= 1024 else "Base"
            print(f"   ✓ WavLM: Loaded from ForCTC ({wavlm_size}, {self.wavlm.config.hidden_size}D)")
        except Exception as e:
            # Fallback: carica come WavLMModel base
            print(f"   ⚠️ ForCTC load failed: {e}")
            self.wavlm = WavLMModel.from_pretrained(
                wavlm_name,
                output_hidden_states=use_weighted_wavlm,
                torch_dtype=torch.float16,
            )
            wavlm_size = "Large" if self.wavlm.config.hidden_size >= 1024 else "Base"
            print(f"   ✓ WavLM: Loaded as base Model ({wavlm_size}, {self.wavlm.config.hidden_size}D)")
        
        # Assicura che output_hidden_states sia attivo per weighted sum
        if use_weighted_wavlm:
            self.wavlm.config.output_hidden_states = True
        
        # =====================================================================
        # STEP 4: Weighted Layer Sum per WavLM (opzionale)
        # Permette di combinare tutti i layer transformer invece del solo ultimo
        # =====================================================================
        self.use_weighted = use_weighted_wavlm
        if use_weighted_wavlm:
            num_layers = self.wavlm.config.num_hidden_layers + 1  # +1 per embedding layer
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
            print(f"   WavLM layers: {num_layers} (weighted sum enabled)")
        
        # =====================================================================
        # STEP 5: Freeze backbone (CRITICO per efficienza)
        # I backbone sono congelati: solo gate e CTC head sono trainabili
        # =====================================================================
        if freeze_backbones:
            for param in self.hubert.parameters():
                param.requires_grad = False
            for param in self.wavlm.parameters():
                param.requires_grad = False
            print(f"   ✓ Backbones FROZEN (no gradients)")
        
        # =====================================================================
        # STEP 6: Dimension Alignment (NUOVO per supportare hidden_size diversi)
        # Se HuBERT e WavLM hanno dimensioni diverse, proiettiamo alla dimensione maggiore
        # =====================================================================
        hidden_size_h = self.hubert.config.hidden_size  # 1024 per HuBERT Large
        hidden_size_w = self.wavlm.config.hidden_size   # 768 per WavLM Base, 1024 per Large
        
        # Dimensione target per fusione (la maggiore delle due)
        self.fused_hidden_size = max(hidden_size_h, hidden_size_w)
        
        # Proiezione per allineare le dimensioni se diverse
        self.proj_hubert = None
        self.proj_wavlm = None
        
        if hidden_size_h != self.fused_hidden_size:
            self.proj_hubert = nn.Linear(hidden_size_h, self.fused_hidden_size)
            nn.init.xavier_uniform_(self.proj_hubert.weight)
            nn.init.zeros_(self.proj_hubert.bias)
            print(f"   ⚠️ HuBERT projection: {hidden_size_h}D → {self.fused_hidden_size}D")
        
        if hidden_size_w != self.fused_hidden_size:
            self.proj_wavlm = nn.Linear(hidden_size_w, self.fused_hidden_size)
            nn.init.xavier_uniform_(self.proj_wavlm.weight)
            nn.init.zeros_(self.proj_wavlm.bias)
            print(f"   ⚠️ WavLM projection: {hidden_size_w}D → {self.fused_hidden_size}D")
        
        if hidden_size_h == hidden_size_w:
            print(f"   ✓ Dimensioni già allineate: {hidden_size_h}D")
        
        combined_size = self.fused_hidden_size * 2  # Gate prende concat delle feature proiettate
        
        # =====================================================================
        # STEP 7: Gate Network
        # Questo è il cuore di Gated Fusion: un layer lineare che prende
        # la concatenazione di h_hubert e h_wavlm e produce un gate in [0,1]
        # =====================================================================
        # Gate Network: 2*fused_hidden_size → 1 + sigmoid
        # Input: concatenazione [h_hubert_proj; h_wavlm_proj]
        # Output: gate value per ogni timestep
        self.gate_network = nn.Linear(combined_size, 1)
        
        # Inizializzazione gate a ~0.5 (bias = 0 → sigmoid(0) = 0.5)
        # Questo permette all'ottimizzatore di partire da un mix bilanciato
        nn.init.xavier_uniform_(self.gate_network.weight)
        nn.init.zeros_(self.gate_network.bias)
        
        print(f"\n🚪 Gate Network:")
        print(f"   Input size: {combined_size}D (concat h_hubert + h_wavlm projected)")
        print(f"   Output: 1D gate per timestep [0, 1]")
        
        # =====================================================================
        # STEP 8: CTC Head
        # L'output fused ha dimensione fused_hidden_size
        # =====================================================================
        self.dropout = nn.Dropout(dropout_rate)
        self.ctc_head = nn.Linear(self.fused_hidden_size, vocab_size)
        
        # Reinizializza CTC head per prevenire CTC collapse
        # std=0.02 è lo standard BERT, funziona bene per CTC
        nn.init.normal_(self.ctc_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.ctc_head.bias)
        
        print(f"\n📊 CTC Head:")
        print(f"   Input: {self.fused_hidden_size}D (h_fused)")
        print(f"   Output: {vocab_size} (vocab size)")
        
        # =====================================================================
        # STEP 9: Statistiche parametri
        # =====================================================================
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📈 Parametri:")
        print(f"   Totali: {total_params / 1e6:.1f}M")
        print(f"   Trainabili: {trainable_params / 1e3:.1f}K")
        print(f"   Frozen: {(total_params - trainable_params) / 1e6:.1f}M")
        print(f"{'='*60}\n")
    
    def _get_wavlm_weighted_output(
        self, 
        hidden_states: Tuple[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calcola somma pesata dei layer WavLM.
        
        Invece di usare solo l'ultimo layer, combina tutti i layer transformer
        con pesi apprendibili (approccio SUPERB).
        
        Args:
            hidden_states: Tuple di tensori [batch, time, hidden] per ogni layer.
            
        Returns:
            Tensor [batch, time, hidden] con output pesato.
        """
        # Normalizza pesi con softmax per garantire che sommino a 1
        weights = F.softmax(self.layer_weights, dim=0)
        
        # Stack tutti i layer: [num_layers, batch, time, hidden]
        stacked = torch.stack(hidden_states, dim=0)
        
        # Reshape weights per broadcasting: [num_layers, 1, 1, 1]
        weights_view = weights.view(-1, 1, 1, 1)
        
        # Somma pesata lungo dimensione layer
        weighted = (stacked * weights_view).sum(dim=0)
        
        return weighted
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass con Gated Fusion.
        
        Pipeline:
        1. Passa audio attraverso entrambi i backbone (frozen, no gradients)
        2. Calcola gate per ogni timestep
        3. Fonde le rappresentazioni con il gate
        4. Passa attraverso CTC head
        5. Calcola loss se labels forniti
        
        Args:
            input_values: Audio waveform [batch, samples].
            attention_mask: Maschera per padding [batch, samples] (opzionale).
            labels: Target phoneme IDs [batch, label_len] con -100 per padding.
            
        Returns:
            Dict con:
                - logits: Predizioni [batch, time, vocab].
                - loss: CTC loss (se labels forniti).
                - gate_values: Valori gate per analisi [batch, time, 1].
        """
        # =====================================================================
        # STEP 1: Converti input al dtype del modello
        # =====================================================================
        target_dtype = next(self.hubert.parameters()).dtype
        if input_values.dtype != target_dtype:
            input_values = input_values.to(target_dtype)
        
        # =====================================================================
        # STEP 2: Forward pass backbone (FROZEN, no gradients)
        # Usiamo torch.no_grad() per efficienza e per evitare problemi
        # con operazioni in-place in modelli quantizzati
        # =====================================================================
        with torch.no_grad():
            # HuBERT forward
            outputs_h = self.hubert(
                input_values,
                attention_mask=attention_mask,
            )
            # Clone per staccare dal graph e evitare memory leak
            hidden_h = outputs_h.last_hidden_state.clone()  # [batch, time, 1024]
            
            # WavLM forward
            outputs_w = self.wavlm(
                input_values,
                attention_mask=attention_mask,
            )
            
            # Usa weighted sum se abilitato, altrimenti ultimo layer
            if self.use_weighted:
                hidden_w = self._get_wavlm_weighted_output(outputs_w.hidden_states).clone()
            else:
                hidden_w = outputs_w.last_hidden_state.clone()  # [batch, time, 1024]
        
        # =====================================================================
        # STEP 3: Allineamento temporale
        # In rari casi le due sequenze possono differire di 1-2 frame
        # a causa di arrotondamenti nel subsampling
        # =====================================================================
        min_len = min(hidden_h.size(1), hidden_w.size(1))
        hidden_h = hidden_h[:, :min_len, :]
        hidden_w = hidden_w[:, :min_len, :]
        
        # =====================================================================
        # STEP 4: Proiezione dimensioni (se necessario)
        # Allinea le dimensioni alla fused_hidden_size
        # =====================================================================
        if self.proj_hubert is not None:
            hidden_h = self.proj_hubert(hidden_h)
        if self.proj_wavlm is not None:
            hidden_w = self.proj_wavlm(hidden_w)
        
        # =====================================================================
        # STEP 5: Calcolo Gate (CUORE del Gated Fusion)
        # Concateniamo le due rappresentazioni e passiamo attraverso gate network
        # =====================================================================
        # Concatenazione: [batch, time, 2*fused_hidden_size]
        gate_input = torch.cat([hidden_h, hidden_w], dim=-1)
        
        # Gate network + sigmoid: [batch, time, 1] in range [0, 1]
        gate = torch.sigmoid(self.gate_network(gate_input))
        
        # =====================================================================
        # STEP 6: Fusione pesata con gate
        # h_fused = gate * h_hubert + (1 - gate) * h_wavlm
        # Ora funziona perché entrambi hanno la stessa dimensione
        # =====================================================================
        h_fused = gate * hidden_h + (1 - gate) * hidden_w  # [batch, time, fused_hidden_size]
        
        # =====================================================================
        # STEP 6: CTC Head
        # =====================================================================
        h_fused = self.dropout(h_fused)
        logits = self.ctc_head(h_fused)  # [batch, time, vocab]
        
        # =====================================================================
        # STEP 7: Calcola loss se labels forniti
        # =====================================================================
        loss = None
        if labels is not None:
            loss = self._compute_ctc_loss(logits, labels, attention_mask)
        
        return {
            "logits": logits,
            "loss": loss,
            "gate_values": gate,  # Per analisi post-training
        }
    
    def _compute_ctc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Calcola CTC loss.
        
        Args:
            logits: [batch, time, vocab].
            labels: [batch, label_len] con -100 per padding.
            attention_mask: [batch, samples] (opzionale).
            
        Returns:
            Scalar CTC loss.
        """
        # Log-softmax per CTC
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # [time, batch, vocab] per CTC
        
        batch_size = logits.size(0)
        
        # Input lengths: assumiamo nessun padding nei logits
        input_lengths = torch.full(
            (batch_size,), logits.size(1), dtype=torch.long, device=logits.device
        )
        
        # Prepara labels: sostituisci -100 (padding) con 0
        labels_no_pad = labels.clone()
        labels_no_pad[labels_no_pad == -100] = 0
        
        # Target lengths: conta token non-padding
        target_lengths = (labels != -100).sum(dim=-1)
        
        # CTC Loss
        loss = F.ctc_loss(
            log_probs,
            labels_no_pad,
            input_lengths,
            target_lengths,
            blank=0,  # Token blank è indice 0
            reduction="mean",
            zero_infinity=True,  # Previene NaN per sequenze troppo corte
        )
        
        return loss
    
    def get_layer_weights_info(self) -> Dict[str, float]:
        """
        Restituisce i pesi normalizzati dei layer WavLM per analisi.
        
        Utile per capire quali layer transformer contribuiscono di più
        alla rappresentazione finale.
        
        Returns:
            Dict layer_name → peso normalizzato.
        """
        if not self.use_weighted:
            return {}
        
        weights = F.softmax(self.layer_weights, dim=0).detach().cpu().numpy()
        return {f"layer_{i}": float(w) for i, w in enumerate(weights)}
    
    def get_gate_statistics(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Calcola statistiche sui gate values per un batch di audio.
        
        Utile per analisi post-training: capire quanto il modello preferisce
        HuBERT vs WavLM in media.
        
        Args:
            input_values: Audio waveform [batch, samples].
            attention_mask: Maschera per padding (opzionale).
            
        Returns:
            Dict con statistiche: mean, std, min, max, hubert_preference.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_values, attention_mask)
            gate = outputs["gate_values"]  # [batch, time, 1]
            
            gate_flat = gate.view(-1).cpu().numpy()
            
            return {
                "gate_mean": float(np.mean(gate_flat)),
                "gate_std": float(np.std(gate_flat)),
                "gate_min": float(np.min(gate_flat)),
                "gate_max": float(np.max(gate_flat)),
                "hubert_preference": float(np.mean(gate_flat > 0.5)),  # % timestep che preferiscono HuBERT
            }


# =============================================================================
# DATA COLLATOR
# =============================================================================

class DataCollatorCTCWithPadding:
    """
    Data Collator per CTC training con padding dinamico.
    
    Gestisce il padding di input audio e labels, mascherando
    le posizioni di padding con -100 per ignorarle nella loss CTC.
    
    Args:
        processor: Wav2Vec2Processor per padding.
        padding: Se True, applica padding dinamico al batch.
    """
    
    def __init__(self, processor: Wav2Vec2Processor, padding: bool = True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch con padding.
        
        Args:
            features: Lista di dict con "input_values" e "labels".
            
        Returns:
            Batch dict con tensori paddati.
        """
        # Separa input_values e labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        # Pad input_values
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Pad labels
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Maschera posizioni di padding con -100 (ignorato dalla loss)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        
        return batch


# =============================================================================
# CUSTOM TRAINER
# =============================================================================

class GatedFusionTrainer(Trainer):
    """
    Custom Trainer per GatedFusionModel.
    
    Override compute_loss per gestire il dict output del modello e
    estrarre correttamente la loss.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute CTC loss from model outputs.
        
        Args:
            model: GatedFusionModel.
            inputs: Dict con input_values, attention_mask, labels.
            return_outputs: Se True, ritorna anche gli output del modello.
            
        Returns:
            Loss (e outputs se return_outputs=True).
        """
        labels = inputs.pop("labels", None)
        
        outputs = model(
            input_values=inputs["input_values"],
            attention_mask=inputs.get("attention_mask"),
            labels=labels,
        )
        
        loss = outputs["loss"]
        
        if return_outputs:
            return loss, outputs
        return loss


# =============================================================================
# TRAINING WRAPPER
# =============================================================================

class GatedFusionTrainerWrapper:
    """
    Wrapper per training Gated Fusion.
    
    Gestisce:
    - Caricamento dataset
    - Preprocessing audio
    - Training con HuggingFace Trainer
    - Salvataggio modello
    - Analisi gate values post-training
    
    Args:
        config: Dizionario configurazione.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GatedFusion] Device: {self.device}")
        
        self._setup_processor()
        self._setup_model()
    
    def _setup_processor(self):
        """Inizializza tokenizer e processor per audio."""
        vocab_path = self.config["data"]["vocab_path"]
        
        print(f"[GatedFusion] Caricamento vocab: {vocab_path}")
        
        # Tokenizer per fonemi IPA
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
        
        # Feature extractor per audio raw
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        
        # Processor che combina tokenizer e feature extractor
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
        )
        
        print(f"[GatedFusion] Vocab size: {len(self.tokenizer)}")
    
    def _setup_model(self):
        """Inizializza GatedFusionModel."""
        self.model = GatedFusionModel(
            vocab_size=len(self.tokenizer),
            hubert_name=self.config["model"].get("hubert_name", "facebook/hubert-large-ls960-ft"),
            wavlm_name=self.config["model"].get("wavlm_name", "microsoft/wavlm-large"),
            freeze_backbones=self.config["model"].get("freeze_backbones", True),
            dropout_rate=self.config["model"].get("dropout_rate", 0.1),
            use_weighted_wavlm=self.config["model"].get("use_weighted_wavlm", True),
        )
        
        # Abilita gradient checkpointing per ridurre VRAM
        if self.config["training"].get("gradient_checkpointing", True):
            self.model.hubert.gradient_checkpointing_enable()
            self.model.wavlm.gradient_checkpointing_enable()
            print("[GatedFusion] ✓ Gradient checkpointing enabled")
    
    def load_and_prepare_dataset(self):
        """
        Carica e preprocessa dataset.
        
        Usa for-loop invece di dataset.map() per evitare problemi di
        serializzazione con librosa e altri moduli.
        """
        csv_path = self.config["data"]["csv_path"]
        audio_base = self.config["data"]["audio_base_path"]
        
        print(f"[GatedFusion] Caricamento dataset: {csv_path}")
        
        # Carica CSV con pandas
        df = pd.read_csv(csv_path)
        print(f"[GatedFusion] Samples totali: {len(df)}")
        
        # Split train/val
        val_size = self.config["data"].get("val_size", 0.1)
        val_count = int(len(df) * val_size)
        
        # Shuffle e split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = df.iloc[val_count:]
        val_df = df.iloc[:val_count]
        
        print(f"[GatedFusion] Train: {len(train_df)}, Val: {len(val_df)}")
        
        base_path = Path(audio_base)
        
        def preprocess_dataframe(dataframe, desc="Processing"):
            """Preprocessa dataframe in lista di dizionari."""
            processed = []
            import librosa
            
            for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=desc):
                audio_path = str(row.get("audio_path", "")).replace("\\", "/")
                full_path = base_path / audio_path
                
                try:
                    audio, sr = librosa.load(full_path, sr=16000)
                except Exception:
                    audio = np.zeros(16000, dtype=np.float32)
                
                inputs = self.processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors=None,
                )
                
                ipa = row.get("ipa_clean", "")
                if pd.isna(ipa) or not ipa:
                    ipa = ""
                labels = self.tokenizer(str(ipa)).input_ids
                
                processed.append({
                    "input_values": inputs.input_values[0],
                    "labels": labels,
                })
            
            return processed
        
        print("\n🔄 Preprocessing TRAIN set...")
        train_data = preprocess_dataframe(train_df, "Train preprocessing")
        
        print("\n🔄 Preprocessing VAL set...")
        val_data = preprocess_dataframe(val_df, "Val preprocessing")
        
        # Converti in Dataset HuggingFace
        self.train_dataset = Dataset.from_list(train_data)
        self.val_dataset = Dataset.from_list(val_data)
        
        print(f"✓ Preprocessing completato!")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}\n")
    
    def train(self):
        """Esegue training."""
        output_dir = self.config["training"]["output_dir"]
        
        # Data collator per batch con padding
        data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True,
        )
        
        # Metrica CER (Character Error Rate, equivalente a PER per fonemi)
        cer_metric = evaluate.load("cer")
        
        def compute_metrics(pred):
            """Calcola metriche di valutazione."""
            pred_logits = pred.predictions
            if isinstance(pred_logits, dict):
                pred_logits = pred_logits["logits"]
            pred_ids = np.argmax(pred_logits, axis=-1)
            
            pred_str = self.processor.batch_decode(pred_ids)
            
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
            label_str = self.processor.batch_decode(label_ids, group_tokens=False)
            
            cer = cer_metric.compute(predictions=pred_str, references=label_str)
            
            return {"cer": cer, "per": cer}
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            group_by_length=False,
            per_device_train_batch_size=self.config["training"].get("batch_size", 2),
            per_device_eval_batch_size=self.config["training"].get("batch_size", 2),
            gradient_accumulation_steps=self.config["training"].get("gradient_accumulation", 8),
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.config["training"].get("epochs", 5),
            fp16=self.config["training"].get("fp16", True),
            logging_steps=25,
            learning_rate=self.config["training"].get("learning_rate", 1e-4),
            warmup_steps=self.config["training"].get("warmup_steps", 200),
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            report_to="none",
            dataloader_num_workers=0,  # Evita memory leak
        )
        
        # Trainer
        trainer = GatedFusionTrainer(
            model=self.model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.processor,
            callbacks=[DriveBackupCallback()],
        )
        
        print("\n" + "=" * 60)
        print("TRAINING GATED FUSION (HuBERT + WavLM)")
        print("=" * 60)
        
        # Resume da checkpoint se esiste
        checkpoint = None
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            checkpoint = str(checkpoints[-1])
            print(f"\n🔄 Resume da checkpoint: {checkpoint}")
        else:
            print("\n📝 Inizio training da zero")
        
        try:
            trainer.train(resume_from_checkpoint=checkpoint)
        except AttributeError as e:
            if "scaler" in str(e):
                print(f"\n⚠️ Errore scaler: {e}")
                if checkpoint:
                    scaler_file = Path(checkpoint) / "scaler.pt"
                    if scaler_file.exists():
                        scaler_file.unlink()
                        trainer.train(resume_from_checkpoint=checkpoint)
                    else:
                        trainer.train(resume_from_checkpoint=None)
                else:
                    trainer.train()
            else:
                raise
        
        # Salva modello finale
        final_path = Path(output_dir) / "final_model_gated_fusion"
        trainer.save_model(str(final_path))
        self.processor.save_pretrained(str(final_path))
        
        # Salva config
        config_save = {
            "model_type": "gated_fusion",
            "vocab_size": len(self.tokenizer),
            "hubert_name": self.config["model"].get("hubert_name", "facebook/hubert-large-ls960-ft"),
            "wavlm_name": self.config["model"].get("wavlm_name", "microsoft/wavlm-large"),
            "use_weighted_wavlm": self.config["model"].get("use_weighted_wavlm", True),
        }
        with open(final_path / "config.json", "w") as f:
            json.dump(config_save, f, indent=2)
        
        # Log layer weights (WavLM)
        if self.model.use_weighted:
            weights_info = self.model.get_layer_weights_info()
            print("\n📊 WavLM Layer Weights (final):")
            for name, weight in sorted(weights_info.items(), key=lambda x: -x[1])[:5]:
                print(f"   {name}: {weight:.4f}")
        
        # Log gate statistics
        print("\n🚪 Gate Statistics (su validation set):")
        self._analyze_gate_statistics(trainer)
        
        print("\n" + "=" * 60)
        print(f"✓ Training completato!")
        print(f"  Modello salvato in: {final_path}")
        print("=" * 60)
    
    def _analyze_gate_statistics(self, trainer):
        """Analizza statistiche gate sul validation set."""
        self.model.eval()
        
        gate_means = []
        
        # Campiona alcuni batch dal validation set
        for i, batch in enumerate(trainer.get_eval_dataloader()):
            if i >= 5:  # Solo primi 5 batch per velocità
                break
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model(
                    input_values=batch["input_values"],
                    attention_mask=batch.get("attention_mask"),
                )
                gate = outputs["gate_values"]
                gate_means.append(gate.mean().cpu().item())
        
        if gate_means:
            avg_gate = np.mean(gate_means)
            print(f"   Gate medio: {avg_gate:.4f}")
            print(f"   Interpretazione: {'Preferisce HuBERT' if avg_gate > 0.5 else 'Preferisce WavLM'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Entry point principale."""
    parser = argparse.ArgumentParser(
        description="Training Gated Fusion (HuBERT + WavLM) per Phoneme Recognition"
    )
    
    # Argomenti config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path al config YAML (opzionale)"
    )
    
    # Argomenti data
    parser.add_argument(
        "--data-csv",
        type=str,
        default="data/processed/combined_augmented.csv",
        help="Path al CSV training"
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="data/processed/vocab.json",
        help="Path al vocab.json"
    )
    parser.add_argument(
        "--audio-base",
        type=str,
        default="",
        help="Path base per audio files"
    )
    
    # Argomenti modello
    parser.add_argument(
        "--hubert-path",
        type=str,
        default="facebook/hubert-large-ls960-ft",
        help="Path modello HuBERT (HuggingFace o locale)"
    )
    parser.add_argument(
        "--wavlm-path",
        type=str,
        default="microsoft/wavlm-large",
        help="Path modello WavLM (HuggingFace o locale)"
    )
    parser.add_argument(
        "--no-weighted",
        action="store_true",
        help="Disabilita Weighted Layer Sum per WavLM"
    )
    
    # Argomenti training
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gated_fusion",
        help="Directory output"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Numero epoche"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume da ultimo checkpoint"
    )
    
    args = parser.parse_args()
    
    # Costruisci config
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data": {
                "csv_path": args.data_csv,
                "vocab_path": args.vocab_path,
                "audio_base_path": args.audio_base,
                "val_size": 0.1,
            },
            "model": {
                "hubert_name": args.hubert_path,
                "wavlm_name": args.wavlm_path,
                "freeze_backbones": True,
                "dropout_rate": 0.1,
                "use_weighted_wavlm": not args.no_weighted,
            },
            "training": {
                "output_dir": args.output_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "gradient_accumulation": args.gradient_accumulation,
                "fp16": True,
                "gradient_checkpointing": True,
                "warmup_steps": 200,
            },
        }
    
    # Crea output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Training
    trainer_wrapper = GatedFusionTrainerWrapper(config)
    trainer_wrapper.load_and_prepare_dataset()
    trainer_wrapper.train()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Early Fusion Training - Multi-Backbone Architecture.

Questo script implementa Early Fusion concatenando le feature di due backbone:
- HuBERT Large: Pre-training con target discreti (k-means clustering)
- WavLM Large: Pre-training con denoising + contrastive learning

ARCHITETTURA:
    Audio → ┌─────────────────┐
            │ HuBERT Large    │ → hidden_h (1024D)  ┐
            │ (frozen/low-lr) │                      │
            └─────────────────┘                      ├→ concat (2048D) → CTC Head → Phonemes
            ┌─────────────────┐                      │
    Audio → │ WavLM Weighted  │ → hidden_w (1024D)  ┘
            │ (frozen/low-lr) │
            └─────────────────┘

MOTIVAZIONE SCIENTIFICA:
La concatenazione permette al classificatore di accedere simultaneamente a:
- Rappresentazioni fonetiche discrete di HuBERT (ottimali per PER)
- Rappresentazioni acustiche robuste di WavLM (ottimali per detection)

Il vantaggio rispetto alla Late Fusion è che il classificatore può imparare
a pesare dinamicamente le feature in base al contesto, invece di usare
un peso fisso α.

REQUISITI MEMORIA:
- Entrambi i modelli Large (~317M each) = ~2.2GB pesi
- Con fp16 e gradient_checkpointing: ~16-20GB VRAM
- Senza gradient_checkpointing: ~24GB VRAM

Uso:
    # Training standard
    python scripts/training/train_early_fusion.py \\
        --epochs 5 \\
        --batch-size 2 \\
        --output-dir outputs/early_fusion

    # Resume da checkpoint
    python scripts/training/train_early_fusion.py \\
        --resume \\
        --output-dir outputs/early_fusion

Autore: DeepLearning-Phoneme Project
"""

import argparse
import sys
import os
import warnings
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
import evaluate
from datasets import load_dataset, Audio
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

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# DRIVE BACKUP CALLBACK (from existing scripts)
# =============================================================================

class DriveBackupCallback(TrainerCallback):
    """
    Callback per backup automatico checkpoint su Google Drive o Kaggle.
    
    Rileva automaticamente l'ambiente (Colab/Kaggle/Local) e copia i checkpoint
    nella directory appropriata per prevenire perdita di progresso.
    """
    
    def __init__(self, backup_dir: str = None):
        self.backup_dir = backup_dir
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
                self.backup_dir = None
    
    def on_save(self, args, state, control, **kwargs):
        """Chiamato dopo ogni salvataggio checkpoint."""
        if not self.backup_dir:
            return
        
        if self.env == 'colab' and '/drive/' in str(args.output_dir):
            return
        
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        
        if checkpoint_dir.exists():
            os.makedirs(self.backup_dir, exist_ok=True)
            model_name = Path(args.output_dir).name
            backup_path = Path(self.backup_dir) / model_name / checkpoint_dir.name
            
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
                    zip_path = Path(self.backup_dir) / f"{model_name}_checkpoint-{state.global_step}"
                    shutil.make_archive(str(zip_path), 'zip', checkpoint_dir)
                    print(f"\n💾 Checkpoint compresso: {zip_path}.zip")
            except Exception as e:
                print(f"\n⚠️ Backup fallito: {e}")


# =============================================================================
# EARLY FUSION MODEL
# =============================================================================

class EarlyFusionModel(nn.Module):
    """
    Multi-Backbone Early Fusion per Phoneme Recognition.
    
    Concatena le rappresentazioni di HuBERT e WavLM prima della CTC head.
    I backbone sono frozen (o con LR molto basso) per stabilità e memoria.
    
    Architecture:
        Audio → HuBERT → hidden_h (1024D) ─┐
                                            ├→ concat → Dropout → Linear → Phonemes
        Audio → WavLM  → hidden_w (1024D) ─┘
                                              (2048D)           (vocab_size)
    
    Args:
        vocab_size: Dimensione vocabolario IPA (43 tokens)
        hubert_name: Nome modello HuBERT HuggingFace
        wavlm_name: Nome modello WavLM HuggingFace
        freeze_backbones: Se True, congela completamente i backbone
        dropout_rate: Dropout prima della CTC head
        use_weighted_wavlm: Se True, usa Weighted Layer Sum per WavLM
    
    Attributes:
        hubert: HuBERT encoder (frozen)
        wavlm: WavLM encoder (frozen)
        dropout: Dropout layer
        ctc_head: Linear projection to vocab
        use_weighted: Flag per weighted layer sum
        layer_weights: Pesi apprendibili per WavLM layers (se use_weighted)
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
        
        print(f"[EarlyFusion] Inizializzazione Multi-Backbone...")
        print(f"   HuBERT: {hubert_name}")
        print(f"   WavLM:  {wavlm_name}")
        print(f"   Weighted WavLM: {use_weighted_wavlm}")
        print(f"   Freeze backbones: {freeze_backbones}")
        
        # 4-bit quantization per ridurre VRAM
        use_4bit = freeze_backbones  # Solo se frozen
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                print(f"   4-bit quantization: ENABLED")
            except ImportError:
                print(f"   ⚠️ bitsandbytes not installed, using fp16")
                bnb_config = None
                use_4bit = False
        else:
            bnb_config = None
        
        # Carica HuBERT (supporta sia HubertModel che HubertForCTC checkpoints)
        print(f"   Loading HuBERT from: {hubert_name}")
        try:
            # Prima prova a caricare come HubertForCTC (modello fine-tuned)
            from transformers import HubertForCTC
            hubert_full = HubertForCTC.from_pretrained(
                hubert_name,
                quantization_config=bnb_config if use_4bit else None,
                torch_dtype=torch.float16 if not use_4bit else None,
            )
            # Estrai solo l'encoder (hubert interno)
            self.hubert = hubert_full.hubert
            print(f"   ✓ HuBERT: Loaded from ForCTC checkpoint (encoder extracted)")
        except Exception:
            # Fallback: carica come HubertModel base
            self.hubert = HubertModel.from_pretrained(
                hubert_name,
                output_hidden_states=False,
                quantization_config=bnb_config if use_4bit else None,
                torch_dtype=torch.float16 if not use_4bit else None,
            )
            print(f"   ✓ HuBERT: Loaded as base Model")
        
        # Carica WavLM (supporta sia WavLMModel che WavLMForCTC checkpoints)
        print(f"   Loading WavLM from: {wavlm_name}")
        try:
            # Prima prova a caricare come WavLMForCTC (modello fine-tuned)
            from transformers import WavLMForCTC
            wavlm_full = WavLMForCTC.from_pretrained(
                wavlm_name,
                quantization_config=bnb_config if use_4bit else None,
                torch_dtype=torch.float16 if not use_4bit else None,
            )
            # Estrai solo l'encoder (wavlm interno)
            self.wavlm = wavlm_full.wavlm
            print(f"   ✓ WavLM: Loaded from ForCTC checkpoint (encoder extracted)")
        except Exception:
            # Fallback: carica come WavLMModel base
            self.wavlm = WavLMModel.from_pretrained(
                wavlm_name,
                output_hidden_states=use_weighted_wavlm,
                quantization_config=bnb_config if use_4bit else None,
                torch_dtype=torch.float16 if not use_4bit else None,
            )
            print(f"   ✓ WavLM: Loaded as base Model")
        
        # Ensure output_hidden_states is set for weighted sum
        if use_weighted_wavlm:
            self.wavlm.config.output_hidden_states = True
        
        # Weighted Layer Sum per WavLM
        self.use_weighted = use_weighted_wavlm
        if use_weighted_wavlm:
            num_layers = self.wavlm.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
            print(f"   WavLM layers: {num_layers} (weighted)")
        
        # Freeze backbones
        if freeze_backbones:
            for param in self.hubert.parameters():
                param.requires_grad = False
            for param in self.wavlm.parameters():
                param.requires_grad = False
            print(f"   ✓ Backbone frozen")
        
        # CTC Head
        hidden_size_h = self.hubert.config.hidden_size  # 1024
        hidden_size_w = self.wavlm.config.hidden_size   # 1024
        combined_size = hidden_size_h + hidden_size_w   # 2048
        
        self.dropout = nn.Dropout(dropout_rate)
        self.ctc_head = nn.Linear(combined_size, vocab_size)
        
        # Reinitialize CTC head (critical for preventing CTC collapse)
        nn.init.normal_(self.ctc_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.ctc_head.bias)
        
        print(f"   Combined features: {combined_size}D")
        print(f"   CTC head: {combined_size} → {vocab_size}")
        
        # Conta parametri
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[EarlyFusion] Parametri totali: {total / 1e6:.1f}M")
        print(f"[EarlyFusion] Parametri trainabili: {trainable / 1e6:.3f}M")
    
    def _get_wavlm_weighted_output(self, hidden_states: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Calcola somma pesata dei layer WavLM.
        
        Args:
            hidden_states: Tuple di tensori [batch, time, hidden] per ogni layer
            
        Returns:
            Tensor [batch, time, hidden] con output pesato
        """
        weights = F.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(hidden_states, dim=0)  # [layers, batch, time, hidden]
        weights_view = weights.view(-1, 1, 1, 1)
        weighted = (stacked * weights_view).sum(dim=0)
        return weighted
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass con Early Fusion.
        
        Args:
            input_values: Audio waveform [batch, samples]
            attention_mask: Maschera per padding [batch, samples]
            labels: Target phoneme IDs [batch, label_len] con -100 per padding
            
        Returns:
            Dict con:
                - logits: Predizioni [batch, time, vocab]
                - loss: CTC loss (se labels forniti)
        """
        # HuBERT forward
        outputs_h = self.hubert(
            input_values,
            attention_mask=attention_mask,
        )
        hidden_h = outputs_h.last_hidden_state  # [batch, time, 1024]
        
        # WavLM forward
        outputs_w = self.wavlm(
            input_values,
            attention_mask=attention_mask,
        )
        
        if self.use_weighted:
            hidden_w = self._get_wavlm_weighted_output(outputs_w.hidden_states)
        else:
            hidden_w = outputs_w.last_hidden_state
        
        # Allineamento temporale (dovrebbero essere identici, ma per sicurezza)
        min_len = min(hidden_h.size(1), hidden_w.size(1))
        hidden_h = hidden_h[:, :min_len, :]
        hidden_w = hidden_w[:, :min_len, :]
        
        # Concatenazione Early Fusion
        combined = torch.cat([hidden_h, hidden_w], dim=-1)  # [batch, time, 2048]
        
        # CTC Head
        combined = self.dropout(combined)
        logits = self.ctc_head(combined)  # [batch, time, vocab]
        
        # Calcola loss se labels forniti
        loss = None
        if labels is not None:
            loss = self._compute_ctc_loss(logits, labels, attention_mask)
        
        return {"logits": logits, "loss": loss}
    
    def _compute_ctc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Calcola CTC loss.
        
        Args:
            logits: [batch, time, vocab]
            labels: [batch, label_len] con -100 per padding
            attention_mask: [batch, samples] (opzionale)
            
        Returns:
            Scalar CTC loss
        """
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # [time, batch, vocab]
        
        batch_size = logits.size(0)
        input_lengths = torch.full(
            (batch_size,), logits.size(1), dtype=torch.long, device=logits.device
        )
        
        # Rimuovi padding da labels
        labels_no_pad = labels.clone()
        labels_no_pad[labels_no_pad == -100] = 0
        
        target_lengths = (labels != -100).sum(dim=-1)
        
        loss = F.ctc_loss(
            log_probs,
            labels_no_pad,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )
        
        return loss
    
    def get_layer_weights_info(self) -> Dict[str, float]:
        """
        Restituisce i pesi normalizzati dei layer WavLM per analisi.
        
        Returns:
            Dict layer_name → peso normalizzato
        """
        if not self.use_weighted:
            return {}
        
        weights = F.softmax(self.layer_weights, dim=0).detach().cpu().numpy()
        return {f"layer_{i}": float(w) for i, w in enumerate(weights)}


# =============================================================================
# DATA COLLATOR
# =============================================================================

class DataCollatorCTCWithPadding:
    """
    Data Collator per CTC training con padding dinamico.
    
    Gestisce il padding di input audio e labels, mascherando
    le posizioni di padding con -100 per ignorarle nella loss CTC.
    
    Args:
        processor: Wav2Vec2Processor per padding
        padding: Se True, applica padding dinamico
    """
    
    def __init__(self, processor: Wav2Vec2Processor, padding: bool = True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch con padding.
        
        Args:
            features: Lista di dict con "input_values" e "labels"
            
        Returns:
            Batch dict con tensori paddati
        """
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        
        return batch


# =============================================================================
# CUSTOM TRAINER
# =============================================================================

class EarlyFusionTrainer(Trainer):
    """
    Custom Trainer per EarlyFusionModel.
    
    Override compute_loss per gestire il dict output del modello.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute CTC loss from model outputs."""
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
# TRAINING CLASS
# =============================================================================

class EarlyFusionTrainerWrapper:
    """
    Wrapper per training Early Fusion Multi-Backbone.
    
    Gestisce:
    - Caricamento dataset
    - Preprocessing audio
    - Training con HuggingFace Trainer
    - Salvataggio modello
    
    Args:
        config: Dizionario configurazione
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EarlyFusion] Device: {self.device}")
        
        self._setup_processor()
        self._setup_model()
    
    def _setup_processor(self):
        """Inizializza tokenizer e processor."""
        vocab_path = self.config["data"]["vocab_path"]
        
        print(f"[EarlyFusion] Caricamento vocab: {vocab_path}")
        
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
        
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
        )
        
        print(f"[EarlyFusion] Vocab size: {len(self.tokenizer)}")
    
    def _setup_model(self):
        """Inizializza EarlyFusionModel."""
        self.model = EarlyFusionModel(
            vocab_size=len(self.tokenizer),
            hubert_name=self.config["model"].get("hubert_name", "facebook/hubert-large-ls960-ft"),
            wavlm_name=self.config["model"].get("wavlm_name", "microsoft/wavlm-large"),
            freeze_backbones=self.config["model"].get("freeze_backbones", True),
            dropout_rate=self.config["model"].get("dropout_rate", 0.1),
            use_weighted_wavlm=self.config["model"].get("use_weighted_wavlm", True),
        )
        
        # Enable gradient checkpointing per ridurre memoria
        if self.config["training"].get("gradient_checkpointing", True):
            self.model.hubert.gradient_checkpointing_enable()
            self.model.wavlm.gradient_checkpointing_enable()
            print("[EarlyFusion] ✓ Gradient checkpointing enabled")
    
    def load_and_prepare_dataset(self):
        """Carica e preprocessa dataset usando for-loop (evita pickle issues)."""
        csv_path = self.config["data"]["csv_path"]
        audio_base = self.config["data"]["audio_base_path"]
        
        print(f"[EarlyFusion] Caricamento dataset: {csv_path}")
        
        # Carica CSV direttamente con pandas per evitare problemi di serializzazione
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"[EarlyFusion] Samples totali: {len(df)}")
        
        # Split
        val_size = self.config["data"].get("val_size", 0.1)
        val_count = int(len(df) * val_size)
        
        # Shuffle and split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = df.iloc[val_count:]
        val_df = df.iloc[:val_count]
        
        print(f"[EarlyFusion] Train: {len(train_df)}, Val: {len(val_df)}")
        
        base_path = Path(audio_base)
        
        def preprocess_dataframe(dataframe, desc="Processing"):
            """Preprocessa dataframe in lista di dizionari."""
            processed = []
            import librosa
            from tqdm import tqdm
            
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
        from datasets import Dataset
        self.train_dataset = Dataset.from_list(train_data)
        self.val_dataset = Dataset.from_list(val_data)
        
        print(f"✓ Preprocessing completato!")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}\n")
    
    def train(self):
        """Esegue training."""
        output_dir = self.config["training"]["output_dir"]
        
        data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True,
        )
        
        cer_metric = evaluate.load("cer")
        
        def compute_metrics(pred):
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
        
        trainer = EarlyFusionTrainer(
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
        print("TRAINING EARLY FUSION (HuBERT + WavLM)")
        print("=" * 60)
        
        # Resume da checkpoint
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
        final_path = Path(output_dir) / "final_model_early_fusion"
        trainer.save_model(str(final_path))
        self.processor.save_pretrained(str(final_path))
        
        # Salva config
        config_save = {
            "model_type": "early_fusion",
            "vocab_size": len(self.tokenizer),
            "hubert_name": self.config["model"].get("hubert_name", "facebook/hubert-large-ls960-ft"),
            "wavlm_name": self.config["model"].get("wavlm_name", "microsoft/wavlm-large"),
            "use_weighted_wavlm": self.config["model"].get("use_weighted_wavlm", True),
        }
        with open(final_path / "config.json", "w") as f:
            json.dump(config_save, f, indent=2)
        
        # Log layer weights
        if self.model.use_weighted:
            weights_info = self.model.get_layer_weights_info()
            print("\n📊 WavLM Layer Weights (final):")
            for name, weight in sorted(weights_info.items(), key=lambda x: -x[1])[:5]:
                print(f"   {name}: {weight:.4f}")
        
        print("\n" + "=" * 60)
        print(f"✓ Training completato!")
        print(f"  Modello salvato in: {final_path}")
        print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Training Early Fusion (HuBERT + WavLM) per Phoneme Recognition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path al config YAML"
    )
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
        default=".",
        help="Base path per audio files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/early_fusion",
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
        help="Batch size (ridotto per memoria, 2 modelli Large)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (alto perché backbone frozen)"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disabilita fp16 (richiede più VRAM)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Riprendi dall'ultimo checkpoint"
    )
    parser.add_argument(
        "--wavlm-path",
        type=str,
        default=None,
        help="Path to custom WavLM checkpoint (default: microsoft/wavlm-base)"
    )
    parser.add_argument(
        "--hubert-path",
        type=str,
        default=None,
        help="Path to custom HuBERT checkpoint (default: facebook/hubert-large-ls960-ft)"
    )
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model": {
                "hubert_name": args.hubert_path or "facebook/hubert-large-ls960-ft",
                "wavlm_name": args.wavlm_path or "microsoft/wavlm-base",  # Changed to base!
                "freeze_backbones": True,
                "dropout_rate": 0.1,
                "use_weighted_wavlm": True,
            },
            "data": {
                "csv_path": args.data_csv,
                "vocab_path": args.vocab_path,
                "audio_base_path": args.audio_base,
                "val_size": 0.1,
            },
            "training": {
                "output_dir": args.output_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "gradient_accumulation": args.gradient_accumulation,
                "warmup_steps": 200,
                "fp16": not args.no_fp16,
                "gradient_checkpointing": True,
            },
        }
    
    trainer = EarlyFusionTrainerWrapper(config)
    trainer.load_and_prepare_dataset()
    trainer.train()


if __name__ == "__main__":
    main()

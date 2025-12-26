#!/usr/bin/env python3
"""
Training script per Baseline MLP (Linear Probe) con CTC Loss.

=============================================================================
MOTIVAZIONE SCIENTIFICA - LINEAR PROBING
=============================================================================

Questo script implementa un "Linear Probe" per misurare QUANTA INFORMAZIONE 
FONETICA è già presente linearmente negli hidden states di WavLM SENZA che 
il modello venga minimamente adattato al task.

COSA SIGNIFICA LINEAR PROBING?
Il linear probing è una tecnica di valutazione delle rappresentazioni:
- Si congela COMPLETAMENTE il backbone pre-trained
- Si aggiunge SOLO un classificatore lineare (o MLP shallow)
- Se funziona bene → le feature pre-trained contengono già l'informazione
- Se funziona male → serve fine-tuning del backbone

DIFFERENZA CON FINE-TUNING:
┌─────────────────────┬──────────────────┬──────────────────────┐
│ Aspetto             │ Linear Probe     │ Fine-tuning          │
├─────────────────────┼──────────────────┼──────────────────────┤
│ Backbone            │ FROZEN           │ Trainable            │
│ Params trainabili   │ ~200K (solo MLP) │ ~300M (tutto)        │
│ Training time       │ Molto veloce     │ Lento                │
│ Cosa misura         │ Qualità features │ Performance ottimale │
└─────────────────────┴──────────────────┴──────────────────────┘

ARCHITETTURA:
    Audio → WavLM-Base (FROZEN) → Hidden States [B,T,768] → MLP → Logits [B,T,V]
                                     ↓
                                 Frame-level (NO pooling!)
                                     ↓
                                 CTC Loss per sequenze

PERCHÉ FRAME-LEVEL (NO GAP)?
- Il Global Average Pooling rimuove l'informazione temporale
- Per calcolare PER/CER servono predizioni frame-by-frame
- Solo così possiamo usare lo stesso benchmark degli altri modelli

Uso:
    python scripts/train_baseline_mlp.py --epochs 10 --batch-size 8
    python scripts/train_baseline_mlp.py --output-dir outputs/baseline_mlp

Valutazione:
    python scripts/05_evaluate_speechocean.py --model-path outputs/baseline_mlp
"""

import argparse
import sys
import os
import warnings
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import evaluate
from tqdm import tqdm
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.models.wavlm import WavLMModel, WavLMConfig
import shutil

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# DRIVE BACKUP CALLBACK
# =============================================================================

class DriveBackupCallback(TrainerCallback):
    """
    Callback che copia i checkpoint su Drive dopo ogni salvataggio.
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
        if not self.backup_dir:
            return
        
        # Skip se output_dir è già su Drive (evita copia su se stesso)
        if self.env == 'colab' and '/drive/' in str(args.output_dir):
            return  # Già su Drive, niente da fare
        
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            os.makedirs(self.backup_dir, exist_ok=True)
            model_name = Path(args.output_dir).name
            backup_path = Path(self.backup_dir) / model_name / checkpoint_dir.name
            
            # Evita copia su se stessa
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


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# MODELLO BASELINE MLP (FRAME-LEVEL)
# =============================================================================

class BaselineMLPForCTC(nn.Module):
    """
    Baseline MLP con WavLM frozen + Frame-level output per CTC.
    
    ARCHITETTURA:
        WavLM (frozen) → Hidden [B, T, 768] → MLP → Logits [B, T, vocab]
    
    L'MLP viene applicato a OGNI FRAME della sequenza, permettendo
    l'output di una sequenza di fonemi compatibile con CTC loss.
    
    Args:
        vocab_size: Numero di token nel vocabolario (fonemi + speciali)
        model_name: Nome del modello WavLM pre-trained
        hidden_dim: Dimensione hidden dell'MLP (default: 256)
        dropout: Probabilità dropout (default: 0.1)
        
    Attributes:
        wavlm: Backbone WavLM frozen
        mlp: Frame-level classifier
        
    Example:
        >>> model = BaselineMLPForCTC(vocab_size=45)
        >>> audio = torch.randn(2, 16000)  # batch=2, 1 secondo
        >>> output = model(audio)
        >>> print(output['logits'].shape)  # [2, T, 45]
    """
    
    def __init__(
        self,
        vocab_size: int,
        model_name: str = "microsoft/wavlm-base",
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # ---------------------------------------------------------------------
        # BACKBONE: WavLM-Base FROZEN
        # Nessun parametro del backbone viene aggiornato (Linear Probing puro)
        # ---------------------------------------------------------------------
        print(f"[Baseline] Caricamento backbone: {model_name}")
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        # FREEZE COMPLETO: tutto il backbone è congelato
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        # Dimensione hidden del modello
        hidden_size = self.wavlm.config.hidden_size  # 768 per wavlm-base
        
        backbone_params = sum(p.numel() for p in self.wavlm.parameters())
        print(f"[Baseline] Backbone frozen: {backbone_params / 1e6:.1f}M params (0 trainable)")
        
        # ---------------------------------------------------------------------
        # MLP HEAD (Frame-level)
        # Applicato a ogni frame della sequenza: [B, T, 768] → [B, T, vocab]
        # ---------------------------------------------------------------------
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )
        
        trainable_params = sum(p.numel() for p in self.mlp.parameters())
        print(f"[Baseline] MLP trainable: {trainable_params:,} params")
        print(f"[Baseline] Hidden dim: {hidden_dim}, Vocab size: {vocab_size}")
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass frame-level con CTC loss.
        
        Args:
            input_values: Audio waveform [batch, samples]
            attention_mask: Maschera per padding [batch, samples]
            labels: Target token IDs [batch, label_len], -100 per padding
            
        Returns:
            Dict con 'logits' [batch, frames, vocab] e opzionalmente 'loss'
        """
        # ---------------------------------------------------------------------
        # STEP 1: Estrai feature con WavLM (SENZA GRADIENTI)
        # Il backbone è completamente frozen
        # ---------------------------------------------------------------------
        with torch.no_grad():
            outputs = self.wavlm(
                input_values=input_values,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state  # [B, T, 768]
        
        # ---------------------------------------------------------------------
        # STEP 2: FRAME-LEVEL MLP
        # L'MLP viene applicato a ogni frame della sequenza
        # [B, T, 768] → [B, T, vocab_size]
        # 
        # NOTA: Nessun pooling! Manteniamo la dimensione temporale.
        # ---------------------------------------------------------------------
        logits = self.mlp(hidden_states)  # [B, T, vocab]
        
        # ---------------------------------------------------------------------
        # STEP 3: CALCOLA CTC LOSS (se labels fornite)
        # CTC permette l'allineamento automatico input-output
        # ---------------------------------------------------------------------
        loss = None
        if labels is not None:
            loss = self._compute_ctc_loss(logits, labels, attention_mask)
        
        return {"loss": loss, "logits": logits}
    
    def _compute_ctc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Calcola CTC loss per training.
        
        CTC (Connectionist Temporal Classification) permette di:
        - Non richiedere allineamento frame-fonema
        - Gestire sequenze di output più corte dell'input
        - Usare un token "blank" per frame senza output
        """
        # Log probabilities per CTC
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        log_probs = log_probs.transpose(0, 1)  # [T, B, V] per CTC
        
        # Lunghezze input (numero di frame)
        batch_size = logits.size(0)
        if attention_mask is not None:
            # Calcola lunghezze reali considerando il downsampling del feature encoder
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(dim=-1).long()
            )
        else:
            input_lengths = torch.full(
                (batch_size,), logits.size(1), dtype=torch.long, device=logits.device
            )
        
        # Lunghezze target (escludendo padding -100)
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(dim=-1)
        
        # Sostituisci -100 con 0 per evitare errori
        labels_for_ctc = labels.masked_fill(labels < 0, 0)
        
        # CTC Loss
        ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        
        loss = ctc_loss(
            log_probs,
            labels_for_ctc,
            input_lengths,
            target_lengths,
        )
        
        return loss
    
    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calcola la lunghezza dell'output dopo il feature encoder CNN.
        
        WavLM usa un CNN encoder che riduce la dimensione temporale.
        Formula: output_length = (input_length - kernel_size) / stride + 1
        Per WavLM-base: downsampling di circa 320x
        """
        # WavLM feature extractor downsampling
        # 7 conv layers, kernel 10,3,3,3,3,2,2 stride 5,2,2,2,2,2,2
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1
        
        for kernel_size, stride in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        
        return input_lengths.long()


# =============================================================================
# DATA COLLATOR PER CTC
# =============================================================================

class DataCollatorCTCWithPadding:
    """Data collator per CTC training con padding dinamico."""
    
    def __init__(self, processor: Wav2Vec2Processor, padding: bool = True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Separa input e labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        # Pad inputs
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
        
        # Replace padding with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        
        return batch


# =============================================================================
# TRAINING
# =============================================================================

def train_baseline_mlp(
    csv_path: str,
    vocab_path: str,
    output_dir: str,
    audio_base_path: str = ".",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    resume: bool = False,
):
    """
    Training Baseline MLP con Hugging Face Trainer.
    
    Args:
        resume: Se True, riprende dall'ultimo checkpoint nella output_dir
    """
    # Processor
    print("[Training] Caricamento processor...")
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    
    vocab_size = len(tokenizer)
    print(f"[Training] Vocab size: {vocab_size}")
    
    # Modello
    model = BaselineMLPForCTC(vocab_size=vocab_size)
    
    # Dataset - usa load_dataset per evitare problemi con PyArrow
    from datasets import load_dataset
    
    print(f"[Training] Caricamento dataset da: {csv_path}")
    dataset = load_dataset("csv", data_files=csv_path, split="train")
    
    print(f"[Training] Samples totali: {len(dataset)}")
    
    # Split
    splits = dataset.train_test_split(test_size=0.1)
    print(f"[Training] Train: {len(splits['train'])}, Val: {len(splits['test'])}")
    print("[Training] Preprocessing dataset (può richiedere alcuni minuti)...")
    
    # Preprocess con gestione path
    base_path = Path(audio_base_path)
    
    def preprocess(batch):
        # Fix path e carica audio
        audio_path = str(batch.get("audio_path", "")).replace("\\", "/")
        full_path = base_path / audio_path
        
        try:
            import librosa
            audio, sr = librosa.load(full_path, sr=16000)
        except Exception as e:
            # Audio fallback (silenzio breve)
            audio = np.zeros(16000, dtype=np.float32)
        
        inputs = processor(audio, sampling_rate=16000, return_tensors=None)
        batch["input_values"] = inputs.input_values[0]
        
        ipa = batch.get("ipa_clean", "")
        if not ipa or (isinstance(ipa, float) and np.isnan(ipa)):
            ipa = ""
        batch["labels"] = tokenizer(str(ipa)).input_ids
        
        return batch
    
    # Usa num_proc per parallelizzare e mostra barra progresso
    print("\n🔄 Preprocessing TRAIN set...")
    train_dataset = splits["train"].map(
        preprocess, 
        remove_columns=splits["train"].column_names,
        desc="Train preprocessing",
        num_proc=1,  # Può essere aumentato su macchine multi-core
    )
    
    print("\n🔄 Preprocessing VAL set...")
    val_dataset = splits["test"].map(
        preprocess, 
        remove_columns=splits["test"].column_names,
        desc="Val preprocessing",
        num_proc=1,
    )
    print("✓ Preprocessing completato!\n")
    
    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # Metric
    cer_metric = evaluate.load("cer")
    
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer, "per": cer}
    
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        learning_rate=learning_rate,
        warmup_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
    )
    
    # Custom Trainer per gestire modello non-standard
    class BaselineTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
    
    trainer = BaselineTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        callbacks=[DriveBackupCallback()],
    )
    
    print("\n" + "="*60)
    print("TRAINING BASELINE MLP (LINEAR PROBE)")
    print("="*60)
    print("Backbone: WavLM-base (FROZEN)")
    print("Head: MLP frame-level")
    print("Loss: CTC")
    print("="*60)
    
    # Cerca checkpoint esistenti - RESUME AUTOMATICO
    checkpoint = None
    checkpoints = list(Path(output_dir).glob("checkpoint-*"))
    if checkpoints:
        # Ordina per numero checkpoint e prendi l'ultimo
        checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        checkpoint = str(checkpoints[-1])
        print(f"\n🔄 Trovato checkpoint! Resume da: {checkpoint}")
    else:
        print("\n📝 Nessun checkpoint trovato. Inizio training da zero.")
    
    # Train con gestione errore checkpoint
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
    except AttributeError as e:
        if "scaler" in str(e) or "NoneType" in str(e):
            print(f"\n⚠️ Errore caricamento checkpoint (fp16 mismatch): {e}")
            print("   Riprovo training da zero...")
            trainer.train(resume_from_checkpoint=None)
        else:
            raise
    
    # Save
    final_path = Path(output_dir) / "final_baseline_mlp"
    final_path.mkdir(parents=True, exist_ok=True)
    
    # Salva pesi MLP
    torch.save(model.state_dict(), final_path / "pytorch_model.bin")
    
    # Salva config custom
    config = {
        "model_type": "baseline_mlp_ctc",
        "backbone": "microsoft/wavlm-base",
        "vocab_size": vocab_size,
        "hidden_dim": 256,
        "dropout": 0.1,
        "architectures": ["BaselineMLPForCTC"],
    }
    with open(final_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Salva processor
    processor.save_pretrained(final_path)
    
    print("\n" + "="*60)
    print(f"✓ Training completato!")
    print(f"  Modello salvato in: {final_path}")
    print("="*60)
    print("\nPer valutare su SpeechOcean762:")
    print(f"  python scripts/05_evaluate_speechocean.py --model-path {final_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Training Baseline MLP (Linear Probe) con CTC"
    )
    parser.add_argument(
        "--csv-path",
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
        default="outputs/baseline_mlp",
        help="Directory output"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Numero epoche"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Riprendi training dall'ultimo checkpoint"
    )
    
    args = parser.parse_args()
    
    train_baseline_mlp(
        csv_path=args.csv_path,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        audio_base_path=args.audio_base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

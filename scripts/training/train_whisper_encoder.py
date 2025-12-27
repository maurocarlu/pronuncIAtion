#!/usr/bin/env python3
"""
Training script per Whisper Encoder Only con CTC.

=============================================================================
MOTIVAZIONE SCIENTIFICA
=============================================================================

Whisper è un modello encoder-decoder addestrato per ASR multilingua.
Qui usiamo SOLO l'encoder, aggiungendo una testa CTC custom.

Questo test verifica:
- Se le rappresentazioni encoder di Whisper sono utili per phoneme recognition
- Come si confronta con WavLM/Wav2Vec2 (architettura diversa)

IMPORTANTE: Whisper usa Log-Mel Spectrograms (non raw waveform)!
- WhisperFeatureExtractor genera spectra 80-mel
- L'encoder processa questi spectral frames

MODELLO: openai/whisper-small
- 12 Transformer layers, 768 hidden
- ~244M parametri (encoder only)
- Pre-trained su 680k ore audio multilingua

Uso:
    python scripts/training/train_whisper_encoder.py --epochs 10
"""

import argparse
import sys
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import numpy as np
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    Wav2Vec2CTCTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import shutil

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# DRIVE BACKUP CALLBACK
# =============================================================================

class DriveBackupCallback(TrainerCallback):
    """Copia checkpoint su Drive dopo ogni salvataggio."""
    
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
                    print(f"\n💾 Checkpoint copiato: {backup_path}")
                elif self.env == 'kaggle':
                    zip_path = Path(self.backup_dir) / f"{model_name}_checkpoint-{state.global_step}"
                    shutil.make_archive(str(zip_path), 'zip', checkpoint_dir)
                    print(f"\n💾 Checkpoint: {zip_path}.zip")
            except Exception as e:
                print(f"\n⚠️ Backup fallito: {e}")


# =============================================================================
# WHISPER ENCODER + CTC HEAD
# =============================================================================

class WhisperEncoderForCTC(nn.Module):
    """
    Whisper Encoder con CTC head per phoneme recognition.
    
    Architecture:
        Audio → Mel Spectrogram → Whisper Encoder → Linear → CTC Loss
    """
    
    def __init__(self, vocab_size: int, model_name: str = "openai/whisper-small"):
        super().__init__()
        
        # Load Whisper encoder
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder
        
        # CTC head
        hidden_size = self.encoder.config.d_model  # 768 for small
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Freeze encoder layers (keep last 4 trainable)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last 4 layers
        for layer in self.encoder.layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"   Encoder frozen, last 4 layers trainable")
    
    def forward(self, input_features, labels=None, **kwargs):
        """
        Args:
            input_features: Log-mel spectrogram [batch, mel_bins, time]
            labels: Token IDs for CTC loss
        
        Returns:
            Dict with 'loss' and 'logits'
        """
        # Encoder forward
        encoder_outputs = self.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state  # [batch, time, hidden]
        
        # CTC head
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)  # [batch, time, vocab]
        
        loss = None
        if labels is not None:
            # CTC Loss
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # [time, batch, vocab]
            
            input_lengths = torch.full(
                (logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device
            )
            
            # Compute label lengths
            label_mask = labels != -100
            target_lengths = label_mask.sum(dim=-1)
            labels_for_ctc = labels.clone()
            labels_for_ctc[labels_for_ctc == -100] = 0
            
            loss = nn.functional.ctc_loss(
                log_probs, labels_for_ctc, input_lengths, target_lengths,
                blank=0, zero_infinity=True
            )
        
        return {"loss": loss, "logits": logits}


# =============================================================================
# DATA COLLATOR
# =============================================================================

class DataCollatorWhisperCTC:
    """Data collator per Whisper + CTC."""
    
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad input features (mel spectrograms)
        # Convert to numpy if stored as list (happens after dataset.map)
        input_features = []
        for f in features:
            feat = f["input_features"]
            if isinstance(feat, list):
                feat = np.array(feat)
            input_features.append(feat)
        
        # Stack and ensure correct shape
        max_len = max(f.shape[-1] for f in input_features)
        padded_inputs = []
        for f in input_features:
            if f.shape[-1] < max_len:
                pad_width = max_len - f.shape[-1]
                f = np.pad(f, ((0, 0), (0, pad_width)), mode='constant')
            padded_inputs.append(f)
        
        batch_input = torch.tensor(np.stack(padded_inputs), dtype=torch.float32)
        
        # Pad labels
        labels = [f["labels"] for f in features]
        max_label_len = max(len(l) for l in labels)
        padded_labels = []
        for l in labels:
            padded = l + [-100] * (max_label_len - len(l))
            padded_labels.append(padded)
        
        batch_labels = torch.tensor(padded_labels, dtype=torch.long)
        
        return {
            "input_features": batch_input,
            "labels": batch_labels,
        }


# =============================================================================
# CUSTOM TRAINER
# =============================================================================

class WhisperCTCTrainer(Trainer):
    """Custom trainer per Whisper CTC."""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# TRAINING
# =============================================================================

def train_whisper_encoder(
    csv_path: str,
    vocab_path: str,
    output_dir: str,
    audio_base_path: str = ".",
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    resume: bool = False,
):
    """Training Whisper Encoder con CTC."""
    
    print("=" * 60)
    print("TRAINING WHISPER ENCODER + CTC")
    print("=" * 60)
    
    # Setup
    print("\n📦 Setup...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    
    vocab_size = len(tokenizer)
    print(f"   Vocab size: {vocab_size}")
    
    # Load model
    print("\n📦 Caricamento Whisper Encoder...")
    model = WhisperEncoderForCTC(vocab_size=vocab_size)
    
    # Enable gradient checkpointing
    model.encoder.gradient_checkpointing_enable()
    print("   ✓ Gradient checkpointing abilitato")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parametri totali: {total_params/1e6:.1f}M")
    print(f"   Parametri trainabili: {trainable_params/1e6:.1f}M")
    
    # Load dataset with memory optimization
    print(f"\n📥 Caricamento dataset: {csv_path}")
    
    # Limit samples on Kaggle to avoid OOM
    max_samples = None
    if '/kaggle' in os.getcwd():
        max_samples = 10000  # Limita su Kaggle
        print(f"   ⚠️ Kaggle mode: limiting to {max_samples} samples")
    
    ds = load_dataset("csv", data_files=csv_path, keep_in_memory=False)["train"]
    
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))
    
    # Fix paths (in batches to save memory)
    def fix_audio_path(example):
        path = example["audio_path"].replace("\\", "/")
        if not os.path.isabs(path):
            path = os.path.join(audio_base_path, path)
        example["audio_path"] = path
        return example
    
    ds = ds.map(fix_audio_path, num_proc=1, keep_in_memory=False)
    
    # Filter existing files (memory efficient)
    valid_indices = []
    for i, example in enumerate(ds):
        if Path(example["audio_path"]).exists():
            valid_indices.append(i)
    ds = ds.select(valid_indices)
    print(f"   Samples: {len(ds)}")
    
    # Split BEFORE loading audio (saves memory)
    if "split" in ds.column_names:
        train_indices = [i for i, x in enumerate(ds) if x["split"] == "train"]
        val_indices = [i for i, x in enumerate(ds) if x["split"] == "validation"]
        train_ds = ds.select(train_indices)
        val_ds = ds.select(val_indices)
    else:
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        val_ds = split["test"]
    
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Preprocess - Extract mel spectrograms with MANUAL audio loading (avoids torchcodec)
    import librosa
    
    def preprocess(batch):
        # Load audio manually with librosa
        audio, sr = librosa.load(batch["audio_path"], sr=16000)
        mel = feature_extractor(
            audio, sampling_rate=16000, return_tensors="np"
        ).input_features[0]
        batch["input_features"] = mel
        batch["labels"] = tokenizer(batch["ipa_clean"]).input_ids
        return batch
    
    print("\n🔄 Extracting mel spectrograms...")
    # Keep only needed columns
    cols_to_remove = [c for c in train_ds.column_names if c not in ["audio_path", "ipa_clean"]]
    train_ds = train_ds.remove_columns(cols_to_remove)
    val_ds = val_ds.remove_columns(cols_to_remove)
    
    train_ds = train_ds.map(
        preprocess, 
        remove_columns=train_ds.column_names, 
        keep_in_memory=False,
        writer_batch_size=100
    )
    val_ds = val_ds.map(
        preprocess, 
        remove_columns=val_ds.column_names, 
        keep_in_memory=False,
        writer_batch_size=100
    )
    
    # Metrics
    cer_metric = evaluate.load("cer")
    
    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        valid = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not valid:
            return {"cer": 1.0}
        preds, labels = zip(*valid)
        cer = cer_metric.compute(predictions=preds, references=labels)
        return {"cer": cer}
    
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=100,
        dataloader_num_workers=0,
        report_to="none",
    )
    
    # Trainer
    data_collator = DataCollatorWhisperCTC(feature_extractor, tokenizer)
    
    trainer = WhisperCTCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[DriveBackupCallback()],
    )
    
    # Resume
    checkpoint = None
    if resume:
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            checkpoint = str(checkpoints[-1])
            print(f"\n🔄 Resume da: {checkpoint}")
    
    # Train
    print("\n🚀 Avvio training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final
    final_path = Path(output_dir) / "final_model_whisper"
    os.makedirs(final_path, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), final_path / "pytorch_model.bin")
    
    # Save config
    import json
    config = {
        "model_type": "whisper_encoder_ctc",
        "vocab_size": vocab_size,
        "base_model": "openai/whisper-small",
    }
    with open(final_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(str(final_path))
    feature_extractor.save_pretrained(str(final_path))
    
    print(f"\n✓ Modello salvato: {final_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Whisper Encoder + CTC")
    parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--audio-base", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/whisper_encoder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()
    
    train_whisper_encoder(
        csv_path=args.data_csv,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        audio_base_path=args.audio_base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume=args.resume,
    )
    
    print("\n" + "=" * 60)
    print("✓ Training Whisper Encoder completato!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training script per Wav2Vec2 Large.

=============================================================================
MOTIVAZIONE SCIENTIFICA
=============================================================================

Wav2Vec2 è il predecessore di WavLM, con pre-training contrastivo standard
(senza denoising). Questo esperimento verifica se il denoising di WavLM
è davvero superiore per il task di pronuncia.

MODELLO: facebook/wav2vec2-large-960h-lv60-self
- 24 Transformer layers, 1024 hidden dimension
- ~300M parametri
- Pre-trained su LibriSpeech 960h

Uso:
    python scripts/training/train_wav2vec2.py --epochs 10
    python scripts/training/train_wav2vec2.py --data-csv data/processed/combined_augmented.csv
"""

import argparse
import sys
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import numpy as np
import pandas as pd
import yaml
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import shutil

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# DRIVE BACKUP CALLBACK
# =============================================================================

class DriveBackupCallback(TrainerCallback):
    """Copia checkpoint su Drive dopo ogni salvataggio. Mantiene solo gli ultimi 2."""
    
    def __init__(self, backup_dir: str = None, keep_last: int = 2):
        self.backup_dir = backup_dir
        self.keep_last = keep_last
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
    
    def _cleanup_old_backups(self, model_name: str):
        """Elimina vecchi backup, mantiene solo gli ultimi N."""
        if not self.backup_dir:
            return
        
        import glob
        import re
        
        if self.env == 'kaggle':
            pattern = f"{self.backup_dir}/{model_name}_checkpoint-*.zip"
            backups = glob.glob(pattern)
            
            if len(backups) > self.keep_last:
                # Ordina per step number (dal più vecchio al più recente)
                def get_step(path):
                    match = re.search(r'checkpoint-(\d+)', path)
                    return int(match.group(1)) if match else 0
                
                backups.sort(key=get_step)
                # Elimina i più vecchi
                to_delete = backups[:-self.keep_last]
                for old_backup in to_delete:
                    try:
                        os.remove(old_backup)
                        print(f"   🗑️ Deleted old: {Path(old_backup).name}")
                    except Exception as e:
                        print(f"   ⚠️ Delete failed: {e}")
        
        elif self.env == 'colab':
            backup_model_dir = Path(self.backup_dir) / model_name
            if backup_model_dir.exists():
                checkpoints = list(backup_model_dir.glob("checkpoint-*"))
                if len(checkpoints) > self.keep_last:
                    def get_step(path):
                        match = re.search(r'checkpoint-(\d+)', str(path))
                        return int(match.group(1)) if match else 0
                    
                    checkpoints.sort(key=get_step)
                    to_delete = checkpoints[:-self.keep_last]
                    for old_ckpt in to_delete:
                        try:
                            shutil.rmtree(old_ckpt)
                            print(f"   🗑️ Deleted old: {old_ckpt.name}")
                        except Exception as e:
                            print(f"   ⚠️ Delete failed: {e}")
    
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
                    print(f"\n💾 Checkpoint copiato su Drive: {backup_path}")
                elif self.env == 'kaggle':
                    zip_path = Path(self.backup_dir) / f"{model_name}_checkpoint-{state.global_step}"
                    shutil.make_archive(str(zip_path), 'zip', checkpoint_dir)
                    print(f"\n💾 Checkpoint compresso: {zip_path}.zip")
                
                # Cleanup old backups
                self._cleanup_old_backups(model_name)
                
            except Exception as e:
                print(f"\n⚠️ Backup fallito: {e}")


# =============================================================================
# DATA COLLATOR
# =============================================================================

class DataCollatorCTCWithPadding:
    """Data collator per CTC training."""
    
    def __init__(self, processor: Wav2Vec2Processor, padding: bool = True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        
        batch["labels"] = labels
        return batch


# =============================================================================
# TRAINING
# =============================================================================

def train_wav2vec2(
    csv_path: str,
    vocab_path: str,
    output_dir: str,
    audio_base_path: str = ".",
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-5,  # LOWERED from 3e-5 to prevent CTC collapse
    resume: bool = False,
):
    """Training Wav2Vec2 Large con CTC."""
    
    print("=" * 60)
    print("TRAINING WAV2VEC2 LARGE")
    print("=" * 60)
    
    # Setup processor
    print("\n📦 Setup processor...")
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token=None,  # CRITICAL: Disable auto BOS token
        eos_token=None,  # CRITICAL: Disable auto EOS token
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
    print(f"   Vocab size: {vocab_size}")
    
    # [DEBUG] Verify Blank Token
    print(f"\n🔍 DEBUG: Tokenizer Checks")
    print(f"   Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   Unk Token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"   Word Delim: {tokenizer.word_delimiter_token} (ID: {tokenizer.word_delimiter_token_id})")
    
    # [DEBUG] Verify Vocab.json usage
    import json
    with open(vocab_path, 'r') as f:
        vocab_file = json.load(f)
    print(f"   Vocab.json size: {len(vocab_file)}")
    if len(vocab_file) != vocab_size:
        print(f"⚠️ WARNING: Vocab size mismatch! Json: {len(vocab_file)} vs Tokenizer: {vocab_size}")

    
    # Load model
    print("\n📦 Caricamento modello Wav2Vec2-Large...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-960h-lv60-self",
        vocab_size=vocab_size,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,  # CRITICAL: prevents NaN on edge cases
        pad_token_id=tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )
    
    # [DEBUG] Check Model Config
    print(f"\n🔍 DEBUG: Model Config")
    print(f"   Model Vocab Size: {model.config.vocab_size}")
    if model.config.vocab_size != vocab_size:
        print(f"⚠️ CRITICAL: Model vocab size mismatch! {model.config.vocab_size} != {vocab_size}")
    print(f"   Pad Token ID (used as CTC Blank): {model.config.pad_token_id}")

    
    # Freeze feature encoder
    model.freeze_feature_encoder()
    print("   ✓ Feature encoder congelato")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    print("   ✓ Gradient checkpointing abilitato")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parametri totali: {total_params/1e6:.1f}M")
    print(f"   Parametri trainabili: {trainable_params/1e6:.1f}M")
    
    # Load dataset
    print(f"\n📥 Caricamento dataset: {csv_path}")
    ds = load_dataset("csv", data_files=csv_path)["train"]
    
    # Fix paths
    def fix_audio_path(example):
        path = example["audio_path"].replace("\\", "/")
        if not os.path.isabs(path):
            path = os.path.join(audio_base_path, path)
        example["audio_path"] = path
        return example
    
    ds = ds.map(fix_audio_path, num_proc=1)
    ds = ds.filter(lambda x: Path(x["audio_path"]).exists(), num_proc=1)
    print(f"   Samples: {len(ds)}")
    
    # Split
    if "split" in ds.column_names:
        train_ds = ds.filter(lambda x: x["split"] == "train")
        val_ds = ds.filter(lambda x: x["split"] == "validation")
    else:
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        val_ds = split["test"]
    
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Preprocess with MANUAL audio loading (avoids torchcodec issues on Kaggle)
    import librosa
    
    def preprocess(batch):
        # Load audio manually with librosa
        audio, sr = librosa.load(batch["audio_path"], sr=16000)
        
        inputs = processor(
            audio, sampling_rate=16000, return_tensors=None
        )
        batch["input_values"] = inputs.input_values[0]
        labels = processor.tokenizer(batch["ipa_clean"]).input_ids
        
        # CRITICAL: CTC requires input_length > label_length
        # Wav2Vec2 reduces time by ~320x, so input_frames = len(audio) / 320
        input_frames = len(batch["input_values"]) // 320
        if len(labels) > input_frames:
            # Truncate labels if too long
            labels = labels[:input_frames]
        
        batch["labels"] = labels
        batch["input_length"] = len(batch["input_values"])
        batch["label_length"] = len(labels)
        return batch
    
    print("\n🔄 Preprocessing...")
    # Remove audio-related columns before preprocessing
    cols_to_remove = [c for c in train_ds.column_names if c not in ["audio_path", "ipa_clean"]]
    train_ds = train_ds.remove_columns(cols_to_remove)
    val_ds = val_ds.remove_columns(cols_to_remove)
    
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names, num_proc=1)
    val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names, num_proc=1)
    
    # Filter samples where labels are too long
    train_ds = train_ds.filter(lambda x: x["label_length"] > 0 and x["label_length"] < x["input_length"] // 320)
    val_ds = val_ds.filter(lambda x: x["label_length"] > 0 and x["label_length"] < x["input_length"] // 320)
    print(f"   Samples after filtering: Train={len(train_ds)}, Val={len(val_ds)}")
    
    # Metrics
    cer_metric = evaluate.load("cer")
    
    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        valid = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not valid:
            return {"cer": 1.0}
        preds, labels = zip(*valid)
        cer = cer_metric.compute(predictions=preds, references=labels)
        
        # [DEBUG] Log first prediction
        print(f"\n📝 DEBUG Prediction (CER: {cer:.2%})")
        print(f"   Ref:  /{labels[0]}/")
        print(f"   Pred: /{preds[0]}/")
        raw_pred_ids = pred_ids[0]
        # Show top 20 predicted IDs
        print(f"   Raw IDs: {raw_pred_ids[:20]}...")
        if not preds[0].strip():
            print("⚠️ WARNING: Empty Prediction string! Possible gradient collapse.")
            
        return {"cer": cer}
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=1000,  # Increased warmup
        weight_decay=0.01,
        fp16=False,  # DISABLED - can cause NaN with CTC
        bf16=torch.cuda.is_bf16_supported(),  # Use bf16 if available
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=50,
        dataloader_num_workers=0,
        group_by_length=True,  # Group similar lengths for efficiency
        gradient_checkpointing=True,
        report_to="none",
        max_grad_norm=1.0,  # Clip gradients
    )
    
    # Trainer
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=processor,
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
    final_path = Path(output_dir) / "final_model_wav2vec2"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\n✓ Modello salvato: {final_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 Large")
    parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--audio-base", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/wav2vec2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)  # Standard for Wav2Vec2 fine-tuning
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()
    
    train_wav2vec2(
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
    print("✓ Training Wav2Vec2 completato!")
    print("=" * 60)


if __name__ == "__main__":
    main()

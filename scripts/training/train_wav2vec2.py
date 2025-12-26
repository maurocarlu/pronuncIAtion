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
                    print(f"\n💾 Checkpoint copiato su Drive: {backup_path}")
                elif self.env == 'kaggle':
                    zip_path = Path(self.backup_dir) / f"{model_name}_checkpoint-{state.global_step}"
                    shutil.make_archive(str(zip_path), 'zip', checkpoint_dir)
                    print(f"\n💾 Checkpoint compresso: {zip_path}.zip")
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
    learning_rate: float = 3e-5,
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
    
    # Load model
    print("\n📦 Caricamento modello Wav2Vec2-Large...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-960h-lv60-self",
        vocab_size=vocab_size,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )
    
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
    
    # Cast audio
    ds = ds.cast_column("audio_path", Audio(sampling_rate=16000))
    ds = ds.rename_column("audio_path", "audio")
    
    # Split
    if "split" in ds.column_names:
        train_ds = ds.filter(lambda x: x["split"] == "train")
        val_ds = ds.filter(lambda x: x["split"] == "validation")
    else:
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        val_ds = split["test"]
    
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Preprocess
    def preprocess(batch):
        audio = batch["audio"]["array"]
        batch["input_values"] = processor(
            audio, sampling_rate=16000, return_tensors=None
        ).input_values[0]
        batch["labels"] = processor.tokenizer(batch["ipa_clean"]).input_ids
        return batch
    
    print("\n🔄 Preprocessing...")
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names, num_proc=1)
    val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names, num_proc=1)
    
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
        group_by_length=False,
        gradient_checkpointing=True,
        report_to="none",
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
    parser.add_argument("--learning-rate", type=float, default=3e-5)
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

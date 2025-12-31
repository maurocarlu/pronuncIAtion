#!/usr/bin/env python3
"""
Training script per Wav2Vec2-BERT 2.0 con CTC.

=============================================================================
MOTIVAZIONE SCIENTIFICA
=============================================================================

W2V-BERT 2.0 combina la masked language modeling di BERT con il
contrastive learning di Wav2Vec2, ottenendo rappresentazioni audio
più ricche.

MODELLO: facebook/w2v-bert-2.0
- 24 Transformer layers, 1024 hidden dimension
- ~600M parametri
- SOTA su LibriSpeech

Uso:
    python scripts/training/train_w2v2_bert.py --epochs 10
"""

import argparse
import sys
import os
import warnings
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2BertForCTC,
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
    """Copia checkpoint su Drive. Mantiene solo gli ultimi 2."""
    
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
            self.backup_dir = None
    
    def _cleanup_old_backups(self, model_name: str):
        if not self.backup_dir:
            return
        import glob, re
        if self.env == 'kaggle':
            pattern = f"{self.backup_dir}/{model_name}_checkpoint-*.zip"
            backups = glob.glob(pattern)
            if len(backups) > self.keep_last:
                def get_step(p): 
                    m = re.search(r'checkpoint-(\d+)', p)
                    return int(m.group(1)) if m else 0
                backups.sort(key=get_step)
                for old in backups[:-self.keep_last]:
                    try: os.remove(old)
                    except: pass
        elif self.env == 'colab':
            import re
            backup_dir = Path(self.backup_dir) / model_name
            if backup_dir.exists():
                checkpoints = list(backup_dir.glob("checkpoint-*"))
                if len(checkpoints) > self.keep_last:
                    def get_step(p):
                        m = re.search(r'checkpoint-(\d+)', str(p))
                        return int(m.group(1)) if m else 0
                    checkpoints.sort(key=get_step)
                    for old in checkpoints[:-self.keep_last]:
                        try: shutil.rmtree(old)
                        except: pass
    
    def on_save(self, args, state, control, **kwargs):
        if not self.backup_dir:
            return
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            os.makedirs(self.backup_dir, exist_ok=True)
            model_name = Path(args.output_dir).name
            try:
                if self.env == 'colab':
                    backup_path = Path(self.backup_dir) / model_name / checkpoint_dir.name
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    if backup_path.exists(): shutil.rmtree(backup_path)
                    shutil.copytree(checkpoint_dir, backup_path)
                    print(f"\n💾 Backup: {backup_path}")
                elif self.env == 'kaggle':
                    zip_path = Path(self.backup_dir) / f"{model_name}_checkpoint-{state.global_step}"
                    shutil.make_archive(str(zip_path), 'zip', checkpoint_dir)
                    print(f"\n💾 Backup: {zip_path}.zip")
                self._cleanup_old_backups(model_name)
            except Exception as e:
                print(f"\n⚠️ Backup failed: {e}")


# =============================================================================
# DATA COLLATOR
# =============================================================================

class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Debug: check keys in first feature
        if features and hasattr(self, '_debug_printed') is False:
            self._debug_printed = True
            print(f"   [DEBUG] Feature keys: {list(features[0].keys())}")
        
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        batch = self.processor.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        return batch


# =============================================================================
# TRAINING
# =============================================================================

def train_w2v2_bert(
    csv_path: str,
    vocab_path: str,
    output_dir: str,
    audio_base_path: str = ".",
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 5e-6,  # Very conservative to prevent CTC collapse
    resume: bool = False,
):
    """Training Wav2Vec2-BERT 2.0 con CTC."""
    
    print("=" * 60)
    print("TRAINING WAV2VEC2-BERT 2.0")
    print("=" * 60)
    
    # Setup processor
    print("\n📦 Setup processor...")
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token=None,
        eos_token=None,
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
    print("\n📦 Caricamento Wav2Vec2-BERT 2.0...")
    model = Wav2Vec2BertForCTC.from_pretrained(
        "facebook/w2v-bert-2.0",
        vocab_size=vocab_size,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )
    
    # Freeze feature projection (Wav2Vec2-BERT doesn't have freeze_feature_encoder)
    if hasattr(model, 'wav2vec2_bert'):
        for param in model.wav2vec2_bert.feature_projection.parameters():
            param.requires_grad = False
        print("   ✓ Feature projection frozen")
    else:
        print("   ⚠️ Could not find wav2vec2_bert module to freeze")
    
    model.gradient_checkpointing_enable()
    print("   ✓ Gradient checkpointing enabled")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params/1e6:.1f}M")
    print(f"   Trainable: {trainable/1e6:.1f}M")
    
    # Load dataset
    print(f"\n📥 Loading dataset: {csv_path}")
    ds = load_dataset("csv", data_files=csv_path)["train"]
    
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
        train_ds, val_ds = split["train"], split["test"]
    
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Preprocess
    import librosa
    
    def preprocess(batch):
        audio, sr = librosa.load(batch["audio_path"], sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors=None)
        input_values = inputs.input_values[0]
        labels = processor.tokenizer(batch["ipa_clean"]).input_ids
        
        input_frames = len(input_values) // 320
        if len(labels) > input_frames:
            labels = labels[:input_frames]
        
        # Return a new dict - required when using remove_columns with map()
        return {
            "input_values": input_values,
            "labels": labels,
            "input_length": len(input_values),
            "label_length": len(labels),
        }
    
    print("\n🔄 Preprocessing...")
    cols = [c for c in train_ds.column_names if c not in ["audio_path", "ipa_clean"]]
    train_ds = train_ds.remove_columns(cols)
    val_ds = val_ds.remove_columns(cols)
    
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names, num_proc=1)
    val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names, num_proc=1)
    
    # Debug: print columns after preprocessing
    print(f"   Dataset columns after preprocess: {train_ds.column_names}")
    
    train_ds = train_ds.filter(lambda x: x["label_length"] > 0 and x["label_length"] < x["input_length"] // 320)
    val_ds = val_ds.filter(lambda x: x["label_length"] > 0 and x["label_length"] < x["input_length"] // 320)
    print(f"   After filter: Train={len(train_ds)}, Val={len(val_ds)}")
    
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
        return {"cer": cer_metric.compute(predictions=preds, references=labels)}
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=2000,  # Extended warmup to prevent CTC collapse
        weight_decay=0.001,  # Lower weight decay
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=False,  # Disabled - can cause instability with CTC
        bf16=torch.cuda.is_bf16_supported(),  # Use bf16 if available
        dataloader_num_workers=0,
        group_by_length=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorCTCWithPadding(processor),
        compute_metrics=compute_metrics,
        callbacks=[DriveBackupCallback()],
    )
    
    # Resume
    checkpoint = None
    if resume:
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            checkpoint = str(checkpoints[-1])
            print(f"\n🔄 Resuming from: {checkpoint}")
    
    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final
    final_path = Path(output_dir) / "final_model"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\n✓ Model saved: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Wav2Vec2-BERT 2.0")
    parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--audio-base", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/w2v2_bert")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)  # Conservative LR to prevent CTC collapse
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()
    
    train_w2v2_bert(
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
    print("✓ Wav2Vec2-BERT training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

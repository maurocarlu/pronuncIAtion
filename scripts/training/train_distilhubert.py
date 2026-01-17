#!/usr/bin/env python3
"""Training script per DistilHuBERT con CTC su input raw waveform.

Checkpoint: ntu-spml/distilhubert

Requisiti implementati:
- Classe modello: HubertForCTC (NO Wav2Vec2ForCTC)
- Vocab custom: data/processed/vocab.json
- Tokenizer: Wav2Vec2CTCTokenizer con bos_token=None, eos_token=None
- ignore_mismatched_sizes=True + re-init lm_head (std=0.02, bias=0)
- Input: audio raw 16kHz, normalizzato (zero-mean, unit-variance)
- Training: fp16=True, gradient_checkpointing=True
- Hyperparams: lr=3e-5, warmup_ratio=0.1, epochs=10, batch_size=4, grad_accum=4
- Monitoring: PredictionMonitorCallback ogni 100 step
- Backup: DriveBackupCallback (Colab/Kaggle)
- Metric: CER locale (no evaluate.download)

Uso:
    python scripts/training/train_distilhubert.py --epochs 10 --output-dir outputs/distilhubert
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

import soundfile as sf
import librosa

from datasets import load_dataset

from transformers import (
    HubertForCTC,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEFAULT_CHECKPOINT = "ntu-spml/distilhubert"


def _resolve_hf_token(hf_token: Optional[str]) -> Optional[str]:
    if hf_token:
        return hf_token
    return os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")


def _reinit_lm_head(head: nn.Module, std: float = 0.02) -> None:
    if hasattr(head, "weight") and getattr(head, "weight") is not None:
        nn.init.normal_(head.weight, mean=0.0, std=std)
    if hasattr(head, "bias") and getattr(head, "bias") is not None:
        nn.init.zeros_(head.bias)


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            insertions = prev[j] + 1
            deletions = cur[j - 1] + 1
            substitutions = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(insertions, deletions, substitutions))
        prev = cur
    return prev[-1]


def _compute_cer(predictions: List[str], references: List[str]) -> float:
    """CER locale (edit distance / #chars)."""
    if not references:
        return 0.0
    if len(predictions) != len(references):
        n = min(len(predictions), len(references))
        predictions = predictions[:n]
        references = references[:n]

    edits = 0
    chars = 0
    for p, r in zip(predictions, references):
        edits += _levenshtein_distance(p, r)
        chars += len(r)
    return float(edits) / float(max(1, chars))


class DriveBackupCallback(TrainerCallback):
    """Copia checkpoint su Drive dopo ogni salvataggio.

    - Colab: copia la cartella checkpoint in Drive
    - Kaggle: crea uno ZIP scaricabile dall'Output panel
    """

    def __init__(self, backup_dir: str | None = None):
        self.backup_dir = backup_dir
        if "/content" in os.getcwd() or "COLAB_GPU" in os.environ:
            self.env = "colab"
            if not backup_dir:
                self.backup_dir = "/content/drive/MyDrive/phoneme_checkpoints"
        elif "/kaggle" in os.getcwd():
            self.env = "kaggle"
            if not backup_dir:
                self.backup_dir = "/kaggle/working/drive_backup"
        else:
            self.env = "local"
            if not backup_dir:
                self.backup_dir = None

    def on_save(self, args, state, control, **kwargs):
        if not self.backup_dir:
            return

        # Skip se output_dir è già su Drive (evita copia su se stesso)
        if self.env == "colab" and "/drive/" in str(args.output_dir):
            return

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return

        os.makedirs(self.backup_dir, exist_ok=True)
        model_name = Path(args.output_dir).name

        try:
            if self.env == "colab":
                backup_path = Path(self.backup_dir) / model_name / checkpoint_dir.name
                if checkpoint_dir.resolve() == backup_path.resolve():
                    return
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.copytree(checkpoint_dir, backup_path)
                print(f"\n💾 Checkpoint copiato su Drive: {backup_path}")

            elif self.env == "kaggle":
                zip_path = Path(self.backup_dir) / f"{model_name}_checkpoint-{state.global_step}"
                shutil.make_archive(str(zip_path), "zip", checkpoint_dir)
                print(f"\n💾 Checkpoint compresso: {zip_path}.zip")
                print("   (Scarica da Output panel)")
        except Exception as e:
            print(f"\n⚠️ Backup fallito: {e}")


class PredictionMonitorCallback(TrainerCallback):
    """Stampa predizione esempio ogni N step per monitorare blank collapse."""

    def __init__(self, processor: Wav2Vec2Processor, eval_dataset, print_every: int = 100):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.print_every = print_every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step <= 0 or state.global_step % self.print_every != 0:
            return
        if model is None:
            return

        try:
            model.eval()
            sample = self.eval_dataset[0]
            input_values = torch.tensor([sample["input_values"]], device=model.device)

            with torch.no_grad():
                use_amp = torch.cuda.is_available()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)[0]
            pred_str = self.processor.decode(pred_ids)

            label_ids = [i for i in sample["labels"] if i != -100]
            label_str = self.processor.decode(label_ids, group_tokens=False)

            print(f"\n📊 [Step {state.global_step}] Sample Prediction:")
            print(f"   Target: {label_str[:80]}{'...' if len(label_str) > 80 else ''}")
            print(f"   Pred:   {pred_str[:80]}{'...' if len(pred_str) > 80 else ''}")

            if len(pred_str.strip()) == 0:
                print("   ⚠️ WARNING: Empty prediction - possible blank collapse!")
        except Exception as e:
            print(f"\n⚠️ Prediction monitor error: {e}")
        finally:
            try:
                model.train()
            except Exception:
                pass


class DataCollatorCTCWithPadding:
    """Data collator per CTC training con padding dinamico."""

    def __init__(self, processor: Wav2Vec2Processor, padding: str = "longest"):
        self.processor = processor
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch


def train_distilhubert(
    csv_path: str,
    vocab_path: str,
    output_dir: str,
    audio_base_path: str = ".",
    checkpoint: str = DEFAULT_CHECKPOINT,
    hf_token: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 3e-5,
    warmup_ratio: float = 0.1,
    warmup_steps: int = 100,
    freeze_feature_encoder: bool = True,
    seed: int = 42,
    resume: bool = False,
    force_download: bool = False,
    keep_in_memory: bool = True,
):
    print("=" * 60)
    print(f"TRAINING DISTILHUBERT ({checkpoint}) - CTC")
    print("=" * 60)

    hf_token = _resolve_hf_token(hf_token)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token=None,
        eos_token=None,
    )
    print(f"   Vocab size: {len(tokenizer)}")

    # Normalizzazione: usiamo quella del feature extractor (coerente con altri script)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    print("\n📦 Loading DistilHuBERT (HubertForCTC)...")
    model = HubertForCTC.from_pretrained(
        checkpoint,
        token=hf_token,
        force_download=force_download,
        ignore_mismatched_sizes=True,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
    )

    if hasattr(model, "lm_head"):
        _reinit_lm_head(model.lm_head, std=0.02)

    if freeze_feature_encoder:
        try:
            model.freeze_feature_encoder()
            print("   ✓ Feature encoder frozen")
        except Exception as e:
            print(f"   ⚠️ Impossibile congelare feature encoder: {e}")

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Params totali: {total_params/1e6:.1f}M | trainabili: {trainable_params/1e6:.1f}M")

    # Dataset
    print(f"\n📥 Loading dataset: {csv_path}")
    ds = load_dataset("csv", data_files=csv_path)["train"]

    base_path = Path(audio_base_path)

    def fix_audio_path(example):
        path = str(example.get("audio_path", "")).replace("\\\\", "/")
        if not os.path.isabs(path):
            path = str((base_path / path).resolve())
        example["audio_path"] = path
        return example

    ds = ds.map(fix_audio_path, num_proc=1)
    ds = ds.filter(lambda x: Path(x["audio_path"]).exists(), num_proc=1)
    print(f"   Samples: {len(ds)}")

    if "split" in ds.column_names:
        train_ds = ds.filter(lambda x: x["split"] == "train")
        val_ds = ds.filter(lambda x: x["split"] == "validation")
    else:
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds, val_ds = split["train"], split["test"]

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")

    def preprocess(batch):
        # Load audio
        audio_path = batch["audio_path"]
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = processor(audio, sampling_rate=16000, return_tensors=None)
        batch["input_values"] = inputs.input_values[0]

        ipa = batch.get("ipa_clean", "")
        if ipa is None or (isinstance(ipa, float) and np.isnan(ipa)):
            ipa = ""
        labels = tokenizer(str(ipa)).input_ids
        batch["labels"] = labels
        batch["label_length"] = len(labels)
        batch["input_length"] = len(batch["input_values"])
        return batch

    cols = [c for c in train_ds.column_names if c not in ["audio_path", "ipa_clean"]]
    train_ds = train_ds.remove_columns(cols)
    val_ds = val_ds.remove_columns(cols)

    print("\n🔄 Preprocessing...")
    train_ds = train_ds.map(
        preprocess,
        remove_columns=train_ds.column_names,
        num_proc=1,
        load_from_cache_file=False,
        keep_in_memory=keep_in_memory,
    )
    val_ds = val_ds.map(
        preprocess,
        remove_columns=val_ds.column_names,
        num_proc=1,
        load_from_cache_file=False,
        keep_in_memory=keep_in_memory,
    )

    # Filtro CTC di base: elimina label vuote e casi palesemente impossibili.
    # Per raw waveform usiamo una stima conservativa: ~320 samples per frame CTC.
    train_ds = train_ds.filter(
        lambda x: x.get("label_length", 0) > 0 and x.get("input_length", 0) > (x.get("label_length", 0) * 320),
        load_from_cache_file=False,
    )
    val_ds = val_ds.filter(
        lambda x: x.get("label_length", 0) > 0 and x.get("input_length", 0) > (x.get("label_length", 0) * 320),
        load_from_cache_file=False,
    )

    train_ds.set_format(type=None, columns=["input_values", "labels"])
    val_ds.set_format(type=None, columns=["input_values", "labels"])

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids)

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        cer = _compute_cer(pred_str, label_str)
        return {"cer": cer, "per": cer}

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=int(warmup_steps),
        warmup_ratio=(float(warmup_ratio) if int(warmup_steps) == 0 else 0.0),
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=bool(torch.cuda.is_available()),
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
        seed=int(seed),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorCTCWithPadding(processor),
        compute_metrics=compute_metrics,
        callbacks=[
            PredictionMonitorCallback(processor, val_ds, print_every=100),
            DriveBackupCallback(),
        ],
        processing_class=processor,
    )

    checkpoint_path = None
    if resume:
        checkpoints = list(Path(output_dir).glob("checkpoint-*") )
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda p: int(p.name.split("-")[1]))
            checkpoint_path = str(checkpoints[-1])
            print(f"\n🔄 Resume da checkpoint: {checkpoint_path}")

    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint_path)

    final_path = Path(output_dir) / "final_model_distilhubert"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\n✓ Model saved: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DistilHuBERT (HubertForCTC) for phoneme recognition")
    parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--audio-base", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/distilhubert")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hf-token", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps (se > 0 ha priorità su --warmup-ratio)",
    )

    parser.add_argument(
        "--no-freeze-feature-encoder",
        action="store_false",
        dest="freeze_feature_encoder",
        default=True,
        help="NON congelare il feature encoder",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-keep-in-memory", action="store_true")

    args = parser.parse_args()

    train_distilhubert(
        csv_path=args.data_csv,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        audio_base_path=args.audio_base,
        checkpoint=args.checkpoint,
        hf_token=args.hf_token,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        freeze_feature_encoder=args.freeze_feature_encoder,
        seed=args.seed,
        resume=args.resume,
        force_download=args.force_download,
        keep_in_memory=(not args.no_keep_in_memory),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# NOTE: M-CTC-T richiede una versione recente di transformers. Se vedi errori su MctctProcessor/MctctForCTC:
#   pip install --upgrade transformers
"""Training script per M-CTC-T (Meta) con CTC su input Mel Spectrogram.

Checkpoint: facebook/mctct-large

Linee guida:
- Vocab custom: data/processed/vocab.json
- Tokenizer: bos_token=None, eos_token=None
- Inizializzazione: re-init lm_head (std=0.02) + ignore_mismatched_sizes=True
- Stabilità CTC: ctc_zero_infinity=True
- Memoria: fp16=True, gradient_checkpointing=True
- 4-bit: auto se VRAM <16GB; in 4-bit facciamo linear probing (solo lm_head trainabile)
- Hyperparams: learning_rate=3e-5, warmup_ratio=0.1, gradient_accumulation_steps=4
- Monitoring: PredictionMonitorCallback ogni 100 step

Uso:
    python scripts/training/train_mctct.py --epochs 10 --output-dir outputs/mctct_large
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCTC,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


CHECKPOINT = "facebook/mctct-large"


def _get_vram_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)


def _auto_use_4bit(auto_4bit: bool, force_4bit: bool) -> bool:
    if force_4bit:
        return True
    if not auto_4bit:
        return False
    vram_gb = _get_vram_gb()
    if vram_gb is None:
        return False
    return vram_gb < 16.0


def _extract_input_features(processed: Any) -> Any:
    if isinstance(processed, dict):
        for key in ("input_features", "features"):
            if key in processed:
                feats = processed[key]
                # Alcuni processor ritornano una lista batchata: [B, ...]. Qui usiamo B=1.
                if isinstance(feats, (list, tuple)) and len(feats) == 1:
                    return feats[0]
                if isinstance(feats, np.ndarray) and feats.ndim >= 3 and feats.shape[0] == 1:
                    return feats[0]
                return feats
        raise KeyError(f"Processor output keys non riconosciute: {list(processed.keys())}")
    raise TypeError(f"Processor output non dict: {type(processed)}")


class PredictionMonitorCallback(TrainerCallback):
    """Stampa predizione esempio ogni N step per monitorare blank collapse."""

    def __init__(self, tokenizer: Wav2Vec2CTCTokenizer, eval_dataset, print_every: int = 100):
        self.tokenizer = tokenizer
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
            input_features = torch.tensor([sample["input_features"]], device=next(model.parameters()).device)

            with torch.no_grad():
                out = model(input_features=input_features)
                logits = out.logits if hasattr(out, "logits") else out["logits"]

            pred_ids = torch.argmax(logits, dim=-1)[0]
            pred_str = self.tokenizer.decode(pred_ids)

            label_ids = [i for i in sample["labels"] if i != -100]
            label_str = self.tokenizer.decode(label_ids)

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
    def __init__(self, processor, padding: str = "longest"):
        self.processor = processor
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
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


def train_mctct(
    csv_path: str,
    vocab_path: str,
    output_dir: str,
    audio_base_path: str = ".",
    epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 3e-5,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 4,
    resume: bool = False,
    auto_4bit: bool = True,
    use_4bit: bool = False,
    force_download: bool = False,
):
    print("=" * 60)
    print("TRAINING M-CTC-T (facebook/mctct-large) - CTC")
    print("=" * 60)

    # Custom tokenizer
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token=None,
        eos_token=None,
    )
    vocab_size = len(tokenizer)
    print(f"   Vocab size: {vocab_size}")

    # Processor/model classes: in some transformers versions these are exposed as MCTCT* (uppercase).
    # To be robust across environments, we try a few imports and then fall back to Auto*.
    MctctProcessor = None
    MctctForCTC = None
    try:
        from transformers import MCTCTProcessor as MctctProcessor  # type: ignore
        from transformers import MCTCTForCTC as MctctForCTC  # type: ignore
    except Exception:
        try:
            from transformers import MctctProcessor as MctctProcessor  # type: ignore
            from transformers import MctctForCTC as MctctForCTC  # type: ignore
        except Exception:
            MctctProcessor = None
            MctctForCTC = None

    if MctctProcessor is not None:
        base_processor = MctctProcessor.from_pretrained(CHECKPOINT, force_download=force_download)
        try:
            processor = MctctProcessor(feature_extractor=base_processor.feature_extractor, tokenizer=tokenizer)
        except TypeError:
            processor = base_processor
            processor.tokenizer = tokenizer
        model_loader = MctctForCTC.from_pretrained if MctctForCTC is not None else AutoModelForCTC.from_pretrained
    else:
        # Fallback: use feature extractor + custom tokenizer. This avoids relying on
        # top-level exports (MctctProcessor) that may be missing.
        feature_extractor = AutoFeatureExtractor.from_pretrained(CHECKPOINT, force_download=force_download)

        class _Processor:
            def __init__(self, feature_extractor, tokenizer):
                self.feature_extractor = feature_extractor
                self.tokenizer = tokenizer

            def __call__(self, *args, **kwargs):
                return self.feature_extractor(*args, **kwargs)

        processor = _Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        model_loader = AutoModelForCTC.from_pretrained

    want_4bit = _auto_use_4bit(auto_4bit=auto_4bit, force_4bit=use_4bit)
    vram_gb = _get_vram_gb()
    if vram_gb is not None:
        print(f"   GPU VRAM: {vram_gb:.1f} GB")
    print(f"   4-bit quantization: {'ON' if want_4bit else 'OFF'}")

    model_kwargs: Dict[str, Any] = dict(
        vocab_size=vocab_size,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )

    if want_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            print("   ⚠️ bitsandbytes non disponibile: disabilito 4-bit.")
            want_4bit = False

    if want_4bit:
        print("\n📦 Loading M-CTC-T in 4-bit (NF4) - frozen backbone + train CTC head...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = model_loader(
            CHECKPOINT,
            quantization_config=bnb_config,
            force_download=force_download,
            **model_kwargs,
        )
        for p in model.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad_(True)
        nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(model.lm_head.bias)
    else:
        print("\n📦 Loading M-CTC-T in fp16...")
        model = model_loader(
            CHECKPOINT,
            force_download=force_download,
            **model_kwargs,
        )
        nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(model.lm_head.bias)

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Params trainabili: {trainable/1e6:.3f}M / Totali: {total/1e6:.1f}M")

    # Dataset
    print(f"\n📥 Loading dataset: {csv_path}")
    ds = load_dataset("csv", data_files=csv_path)["train"]

    def fix_audio_path(example):
        path = example["audio_path"].replace("\\\\", "/")
        if not os.path.isabs(path):
            path = os.path.join(audio_base_path, path)
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

    import librosa

    def preprocess(batch):
        audio, _ = librosa.load(batch["audio_path"], sr=16000)
        processed = processor(audio, sampling_rate=16000, return_tensors=None)
        feats = _extract_input_features(processed)

        # Typical shape: [T, 80] or [80, T] depending on extractor; keep as-is and let pad handle
        feats = np.array(feats, dtype=np.float32)
        feats = feats.tolist()

        labels = tokenizer(batch["ipa_clean"]).input_ids
        return {"input_features": feats, "labels": labels}

    cols = [c for c in train_ds.column_names if c not in ["audio_path", "ipa_clean"]]
    train_ds = train_ds.remove_columns(cols)
    val_ds = val_ds.remove_columns(cols)

    print("\n🔄 Preprocessing...")
    train_ds = train_ds.map(
        preprocess,
        remove_columns=train_ds.column_names,
        num_proc=1,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    val_ds = val_ds.map(
        preprocess,
        remove_columns=val_ds.column_names,
        num_proc=1,
        load_from_cache_file=False,
        keep_in_memory=True,
    )

    train_ds.set_format(type=None, columns=["input_features", "labels"])
    val_ds.set_format(type=None, columns=["input_features", "labels"])

    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids)
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        dataloader_num_workers=0,
        group_by_length=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorCTCWithPadding(processor),
        compute_metrics=compute_metrics,
        callbacks=[PredictionMonitorCallback(tokenizer, val_ds, print_every=100)],
    )

    checkpoint = None
    if resume:
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            checkpoint = str(checkpoints[-1])
            print(f"\n🔄 Resuming from: {checkpoint}")

    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    final_path = Path(output_dir) / "final_model"
    trainer.save_model(str(final_path))
    try:
        processor.save_pretrained(str(final_path))
    except Exception:
        tokenizer.save_pretrained(str(final_path))
    print(f"\n✓ Model saved: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train M-CTC-T Large (CTC, Mel)")
    parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--audio-base", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/mctct_large")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-4bit", action="store_true", default=True)
    parser.add_argument("--use-4bit", action="store_true", help="Force 4-bit (overrides auto)")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of the HuggingFace checkpoint (fixes corrupted cache issues).",
    )

    args = parser.parse_args()

    train_mctct(
        csv_path=args.data_csv,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        audio_base_path=args.audio_base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume=args.resume,
        auto_4bit=args.auto_4bit,
        use_4bit=args.use_4bit,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# NOTE: M-CTC-T richiede una versione recente di transformers. Se vedi errori su MctctProcessor/MctctForCTC:
#   pip install --upgrade transformers
"""Training script per M-CTC-T (Meta) con CTC su input Mel Spectrogram.

Checkpoint: speechbrain/m-ctc-t-large

Linee guida:
- Vocab custom: data/processed/vocab.json
- Tokenizer: bos_token=None, eos_token=None
- Inizializzazione: re-init lm_head (std=0.02) + ignore_mismatched_sizes=True
- Stabilità CTC: ctc_zero_infinity=True
- Memoria: fp16=True, gradient_checkpointing=True
- 4-bit: auto se VRAM <16GB; in 4-bit facciamo linear probing (solo lm_head trainabile)
- Hyperparams: learning_rate=3e-5, warmup_ratio=0.1, gradient_accumulation_steps=4
- Monitoring: PredictionMonitorCallback ogni 100 step
- Validazione/checkpoint: a fine epoca (CER locale)

Uso:
    python scripts/training/train_mctct.py --epochs 10 --output-dir outputs/mctct_large
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import MCTCTForCTC, MCTCTProcessor
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
            ins = cur[j - 1] + 1
            del_ = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, del_, sub))
        prev = cur
    return prev[-1]


def _compute_cer(predictions: List[str], references: List[str]) -> float:
    """Character Error Rate (CER). Usa jiwer se disponibile, altrimenti Levenshtein."""
    try:
        import jiwer

        return float(jiwer.cer(references, predictions))
    except Exception:
        pass

    if not references:
        return 1.0
    if len(predictions) != len(references):
        n = min(len(predictions), len(references))
        predictions = predictions[:n]
        references = references[:n]

    edits = 0
    chars = 0
    for p, r in zip(predictions, references):
        edits += _levenshtein_distance(r, p)
        chars += len(r)
    return float(edits) / float(max(1, chars))


DEFAULT_CHECKPOINT = "speechbrain/m-ctc-t-large"


def _resolve_hf_token(hf_token: Optional[str]) -> Optional[str]:
    if hf_token:
        return hf_token
    return os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")


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
    # transformers può ritornare BatchFeature (dict-like) invece di dict puro.
    if isinstance(processed, Mapping) or hasattr(processed, "keys"):
        for key in ("input_features", "features"):
            if key in processed:
                feats = processed[key]
                # Alcuni processor ritornano una lista batchata: [B, ...]. Qui usiamo B=1.
                if isinstance(feats, (list, tuple)) and len(feats) == 1:
                    return feats[0]
                if isinstance(feats, np.ndarray) and feats.ndim >= 3 and feats.shape[0] == 1:
                    return feats[0]
                return feats
        try:
            keys = list(processed.keys())
        except Exception:
            keys = []
        raise KeyError(f"Processor output keys non riconosciute: {keys}")
    raise TypeError(f"Processor output non dict-like: {type(processed)}")


def _get_ctc_head(model: nn.Module) -> nn.Module:
    """Ritorna la head CTC del modello (compatibile con modelli che usano lm_head o ctc_head)."""
    if hasattr(model, "lm_head"):
        return getattr(model, "lm_head")
    if hasattr(model, "ctc_head"):
        return getattr(model, "ctc_head")
    raise AttributeError("Il modello non espone né 'lm_head' né 'ctc_head'.")


def _reinit_ctc_head(head: nn.Module, std: float = 0.02) -> None:
    """Re-inizializza pesi/bias della CTC head in modo robusto."""
    if hasattr(head, "weight") and getattr(head, "weight") is not None:
        nn.init.normal_(head.weight, mean=0.0, std=std)
    if hasattr(head, "bias") and getattr(head, "bias") is not None:
        nn.init.zeros_(head.bias)


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
    checkpoint: str = DEFAULT_CHECKPOINT,
    hf_token: Optional[str] = None,
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
    print(f"TRAINING M-CTC-T ({checkpoint}) - CTC")
    print("=" * 60)

    hf_token = _resolve_hf_token(hf_token)

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

    # Classi specifiche (NO AutoProcessor/AutoModel): MCTCTProcessor + MCTCTForCTC.
    try:
        base_processor = MCTCTProcessor.from_pretrained(
            checkpoint,
            force_download=force_download,
            token=hf_token,
        )
    except Exception as e:
        msg = (
            f"Impossibile scaricare/leggere il checkpoint '{checkpoint}'.\n"
            "Se è un repo gated/privato, devi autenticarti su HuggingFace (HUGGINGFACE_HUB_TOKEN o HF_TOKEN).\n"
        )
        raise OSError(msg) from e

    # Manteniamo tokenizer custom sul vocab IPA: lo assegnamo al processor.tokenizer.
    try:
        processor = MCTCTProcessor(feature_extractor=base_processor.feature_extractor, tokenizer=tokenizer)
    except TypeError:
        processor = base_processor
        processor.tokenizer = tokenizer

    model_loader = MCTCTForCTC.from_pretrained

    want_4bit = _auto_use_4bit(auto_4bit=auto_4bit, force_4bit=use_4bit)
    vram_gb = _get_vram_gb()
    if vram_gb is not None:
        print(f"   GPU VRAM: {vram_gb:.1f} GB")
    print(f"   4-bit quantization: {'ON' if want_4bit else 'OFF'}")

    # IMPORTANT: Transformers Trainer (versioni recenti) blocca il training su modelli puramente quantizzati.
    # Questo script usa Trainer (non loop manuale), quindi disabilitiamo 4-bit per evitare crash.
    if want_4bit:
        print("   ⚠️ Trainer non supporta fine-tuning su modelli 4-bit puri: disabilito 4-bit e carico in fp16.")
        want_4bit = False

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
            checkpoint,
            quantization_config=bnb_config,
            force_download=force_download,
            token=hf_token,
            **model_kwargs,
        )
        for p in model.parameters():
            p.requires_grad = False
        ctc_head = _get_ctc_head(model)
        ctc_head.requires_grad_(True)
        _reinit_ctc_head(ctc_head, std=0.02)
    else:
        print("\n📦 Loading M-CTC-T in fp16...")
        model = model_loader(
            checkpoint,
            force_download=force_download,
            token=hf_token,
            **model_kwargs,
        )
        ctc_head = _get_ctc_head(model)
        _reinit_ctc_head(ctc_head, std=0.02)

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

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids)
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
        valid = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not valid:
            return {"cer": 1.0}
        preds, labels = zip(*valid)
        return {"cer": _compute_cer(predictions=list(preds), references=list(labels))}

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
        eval_strategy="epoch",
        save_strategy="epoch",
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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="HuggingFace model id (es. speechbrain/m-ctc-t-large).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (in alternativa usa env HUGGINGFACE_HUB_TOKEN o HF_TOKEN).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-4bit", dest="auto_4bit", action="store_true", default=True)
    parser.add_argument("--no-auto-4bit", dest="auto_4bit", action="store_false", help="Disable auto 4-bit")
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
        checkpoint=args.checkpoint,
        hf_token=args.hf_token,
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

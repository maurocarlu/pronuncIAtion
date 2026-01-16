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

# Helps reduce CUDA memory fragmentation. Must be set before the first CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import MCTCTForCTC, MCTCTProcessor
from transformers import (
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


def _configure_torch_runtime() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        # PyTorch 2.x only
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


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

    def __init__(
        self,
        tokenizer: Wav2Vec2CTCTokenizer,
        eval_dataset,
        feature_extractor,
        print_every: int = 100,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.feature_extractor = feature_extractor
        self.print_every = print_every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step <= 0 or state.global_step % self.print_every != 0:
            return
        if model is None:
            return

        try:
            model.eval()
            sample = self.eval_dataset[0]

            feats = np.asarray(sample["input_features"], dtype=np.float32)
            padded = self.feature_extractor.pad(
                [{"input_features": feats}],
                padding="longest",
                return_tensors="pt",
            )
            input_features = padded["input_features"].to(next(model.parameters()).device)

            with torch.no_grad():
                out = model(input_features=input_features)
                logits = out.logits if hasattr(out, "logits") else out["logits"]

            pred_ids = torch.argmax(logits, dim=-1)[0]
            pred_str = self.tokenizer.decode(pred_ids)

            try:
                blank_id = int(self.tokenizer.pad_token_id)
                blank_ratio = float((pred_ids == blank_id).float().mean().item())
            except Exception:
                blank_ratio = None

            label_ids = [i for i in sample["labels"] if i != -100]
            label_str = self.tokenizer.decode(label_ids)

            print(f"\n📊 [Step {state.global_step}] Sample Prediction:")
            print(f"   Target: {label_str[:80]}{'...' if len(label_str) > 80 else ''}")
            print(f"   Pred:   {pred_str[:80]}{'...' if len(pred_str) > 80 else ''}")
            if len(pred_str.strip()) == 0:
                print("   ⚠️ WARNING: Empty prediction - possible blank collapse!")
            if blank_ratio is not None:
                print(f"   Blank ratio: {blank_ratio:.3f}")
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
        try:
            self.feature_size = getattr(self.processor.feature_extractor, "feature_size", None)
            if self.feature_size is None:
                self.feature_size = getattr(self.processor.feature_extractor, "n_mels", None)
        except Exception:
            self.feature_size = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = []
        for f in features:
            feats = np.asarray(f["input_features"], dtype=np.float32)
            if feats.ndim == 2 and self.feature_size is not None:
                # Pad si aspetta [time, feature_size]. Se arrivano [feature_size, time], trasponiamo.
                if feats.shape[0] == self.feature_size and feats.shape[1] != self.feature_size:
                    feats = feats.T
            input_features.append({"input_features": feats})
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
    eval_batch_size: Optional[int] = None,
    learning_rate: float = 3e-5,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 4,
    group_by_length: bool = True,
    max_audio_seconds: Optional[float] = None,
    long_audio_policy: str = "truncate",
    keep_in_memory: bool = True,
    resume: bool = False,
    force_download: bool = False,
):
    print("=" * 60)
    print(f"TRAINING M-CTC-T ({checkpoint}) - CTC")
    print("=" * 60)

    _configure_torch_runtime()

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

    vram_gb = _get_vram_gb()
    if vram_gb is not None:
        print(f"   GPU VRAM: {vram_gb:.1f} GB")

    model_loader = MCTCTForCTC.from_pretrained

    model_kwargs: Dict[str, Any] = dict(
        vocab_size=vocab_size,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )

    eval_batch_size = int(eval_batch_size) if eval_batch_size is not None else int(batch_size)
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size deve essere > 0")

    if max_audio_seconds is not None and max_audio_seconds <= 0:
        raise ValueError("max_audio_seconds deve essere > 0")
    if long_audio_policy not in {"truncate", "drop"}:
        raise ValueError("long_audio_policy deve essere 'truncate' o 'drop'")

    use_bf16 = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

    # Regola importante:
    # - se usi fp16 nel Trainer, NON caricare i pesi del modello in fp16, altrimenti puoi ottenere
    #   grads fp16 e GradScaler/Accelerate può fallire con: "Attempting to unscale FP16 gradients".
    # - su A100 (Kaggle) bf16 è supportato e spesso più stabile (senza GradScaler).
    if use_bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    print("\n📦 Loading M-CTC-T...")
    model = model_loader(
        checkpoint,
        force_download=force_download,
        token=hf_token,
        torch_dtype=torch_dtype,
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

    max_len_samples: Optional[int] = None
    if max_audio_seconds is not None:
        max_len_samples = int(float(max_audio_seconds) * 16000)

    feature_size = None
    try:
        feature_size = getattr(processor.feature_extractor, "feature_size", None)
        if feature_size is None:
            feature_size = getattr(processor.feature_extractor, "n_mels", None)
    except Exception:
        feature_size = None

    def preprocess(batch):
        audio, _ = librosa.load(batch["audio_path"], sr=16000)
        if max_len_samples is not None and len(audio) > max_len_samples:
            if long_audio_policy == "truncate":
                audio = audio[:max_len_samples]
            else:
                return {"input_features": [], "labels": [], "input_length": 0}
        # return_tensors="np" aiuta a mantenere shape consistente tra versioni di transformers
        processed = processor(audio, sampling_rate=16000, return_tensors="np")
        feats = _extract_input_features(processed)

        # Normalizza orientamento: feature_extractor.pad si aspetta [time, feature_size]
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 3 and feats.shape[0] == 1:
            feats = feats[0]
        if feats.ndim == 2 and feature_size is not None:
            # Se arriva [feature_size, time], trasponi in [time, feature_size]
            if feats.shape[0] == feature_size and feats.shape[1] != feature_size:
                feats = feats.T
        feats = feats.tolist()

        labels = tokenizer(batch["ipa_clean"]).input_ids
        input_length = int(len(feats)) if isinstance(feats, list) and feats and isinstance(feats[0], list) else 0
        return {"input_features": feats, "labels": labels, "input_length": input_length}

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

    train_ds = train_ds.filter(lambda x: x.get("input_length", 0) > 0, load_from_cache_file=False)
    val_ds = val_ds.filter(lambda x: x.get("input_length", 0) > 0, load_from_cache_file=False)

    train_ds.set_format(type=None, columns=["input_features", "labels", "input_length"])
    val_ds.set_format(type=None, columns=["input_features", "labels", "input_length"])

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
        per_device_eval_batch_size=eval_batch_size,
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
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=0,
        group_by_length=group_by_length,
        length_column_name="input_length",
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
        callbacks=[PredictionMonitorCallback(tokenizer, val_ds, processor.feature_extractor, print_every=100)],
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
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size per eval (default: uguale a --batch-size). Riducilo se vai in OOM durante eval/save.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)

    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=None,
        help="Cap sulla durata audio (secondi). Utile su GPU 16GB per ridurre OOM (truncate/drop).",
    )
    parser.add_argument(
        "--long-audio-policy",
        type=str,
        default="truncate",
        choices=["truncate", "drop"],
        help="Cosa fare con clip oltre --max-audio-seconds: truncate (default) o drop.",
    )

    parser.add_argument(
        "--no-group-by-length",
        action="store_true",
        help="Disabilita bucketing per lunghezza (può aumentare padding e peggiorare memoria/stabilità).",
    )
    parser.add_argument(
        "--no-keep-in-memory",
        action="store_true",
        help="Non tenere il dataset preprocessato in RAM (consigliato se RAM limitata su Kaggle).",
    )
    parser.add_argument("--resume", action="store_true")
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
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        group_by_length=(not args.no_group_by_length),
        max_audio_seconds=args.max_audio_seconds,
        long_audio_policy=args.long_audio_policy,
        keep_in_memory=(not args.no_keep_in_memory),
        resume=args.resume,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()

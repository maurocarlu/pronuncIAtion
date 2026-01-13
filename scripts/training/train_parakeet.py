#!/usr/bin/env python3
"""Training script per Parakeet-CTC 1.1B (FastConformer-CTC) con CTC.

Checkpoint: nvidia/parakeet-ctc-1.1b

Linee guida implementate:
- Vocab custom: data/processed/vocab.json
- Tokenizer: bos_token=None, eos_token=None
- Classi specifiche: ParakeetProcessor + ParakeetForCTC (NO Auto*)
- Inizializzazione: re-init lm_head (std=0.02) + ignore_mismatched_sizes=True
- Stabilità CTC: ctc_zero_infinity=True
- Memoria (CRITICO): carico obbligatoriamente in 4-bit (BitsAndBytesConfig) e faccio linear probing:
  backbone congelato (requires_grad=False) + train solo lm_head reinizializzata.
- Hyperparams benchmark: lr=3e-5, warmup_ratio=0.1, fp16=True, gradient_accumulation_steps=4
- Monitoring: PredictionMonitorCallback ogni 100 step

Uso:
    python scripts/training/train_parakeet.py --epochs 10 --output-dir outputs/parakeet_ctc_1p1b
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import ParakeetForCTC, ParakeetProcessor
from transformers import (
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
)
from transformers.optimization import get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEFAULT_CHECKPOINT = "nvidia/parakeet-ctc-1.1b"


def _resolve_hf_token(hf_token: Optional[str]) -> Optional[str]:
    if hf_token:
        return hf_token
    return os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")


def _get_vram_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)


def _extract_first_key(processed: Any, keys: Tuple[str, ...]) -> Any:
    if isinstance(processed, dict):
        for key in keys:
            if key in processed:
                value = processed[key]
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    return value[0]
                return value
        raise KeyError(f"Processor output keys non riconosciute: {list(processed.keys())}")

    # transformers processors tipicamente ritornano BatchEncoding con attributi
    for key in keys:
        if hasattr(processed, key):
            value = getattr(processed, key)
            if isinstance(value, (list, tuple)) and len(value) == 1:
                return value[0]
            return value

    raise TypeError(f"Processor output non compatibile: {type(processed)}")


def _detect_model_input_key(processor: ParakeetProcessor) -> str:
    dummy_audio = np.zeros(16000, dtype=np.float32)
    processed = processor(dummy_audio, sampling_rate=16000, return_tensors=None)

    # Ordine preferito: waveform raw (input_values) oppure feature (input_features)
    for key in ("input_values", "input_features"):
        try:
            _ = _extract_first_key(processed, (key,))
            return key
        except Exception:
            continue

    if isinstance(processed, dict):
        raise RuntimeError(f"Impossibile determinare la chiave input dal processor: {list(processed.keys())}")
    raise RuntimeError("Impossibile determinare la chiave input dal ParakeetProcessor")


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
        processor: ParakeetProcessor,
        eval_dataset,
        input_key: str,
        dataset_input_key: str,
        print_every: int = 100,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.input_key = input_key
        self.dataset_input_key = dataset_input_key
        self.print_every = print_every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step <= 0 or state.global_step % self.print_every != 0:
            return
        if model is None:
            return

        try:
            model.eval()
            sample = self.eval_dataset[0]

            device = next(model.parameters()).device
            if self.input_key == "input_features":
                audio = np.asarray(sample[self.dataset_input_key], dtype=np.float32)
                feats = self.processor.feature_extractor(
                    [audio],
                    sampling_rate=16000,
                    padding=True,
                    return_tensors="pt",
                )
                x = feats["input_features"].to(device=device, dtype=torch.float16)
            else:
                x = torch.tensor([sample[self.dataset_input_key]], device=device)

            with torch.no_grad():
                out = model(**{self.input_key: x})
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
    def __init__(self, processor: ParakeetProcessor, input_key: str, padding: str = "longest"):
        self.processor = processor
        self.input_key = input_key
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.input_key == "input_features":
            audio = [np.asarray(f["input_values"], dtype=np.float32) for f in features]
            batch = self.processor.feature_extractor(
                audio,
                sampling_rate=16000,
                padding=self.padding,
                return_tensors="pt",
            )
            # Parakeet (subsampling conv) spesso è in fp16 quando caricato in 4-bit; evita mismatch input(float32)/bias(float16)
            if "input_features" in batch:
                batch["input_features"] = batch["input_features"].to(dtype=torch.float16)
        else:
            input_features = [{self.input_key: f[self.input_key]} for f in features]
            batch = self.processor.feature_extractor.pad(
                input_features, padding=self.padding, return_tensors="pt"
            )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch


def train_parakeet(
    csv_path: str,
    vocab_path: str,
    output_dir: str,
    audio_base_path: str = ".",
    checkpoint: str = DEFAULT_CHECKPOINT,
    hf_token: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    precompute_features: bool = False,
    learning_rate: float = 3e-5,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 4,
    resume: bool = False,
    force_download: bool = False,
):
    print("=" * 60)
    print(f"TRAINING PARAKEET-CTC 1.1B ({checkpoint}) - CTC")
    print("=" * 60)

    hf_token = _resolve_hf_token(hf_token)

    # Tokenizer custom (IPA labels)
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

    # Processor specifico Parakeet + tokenizer custom
    try:
        base_processor = ParakeetProcessor.from_pretrained(
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

    try:
        processor = ParakeetProcessor(feature_extractor=base_processor.feature_extractor, tokenizer=tokenizer)
    except TypeError:
        processor = base_processor
        processor.tokenizer = tokenizer

    input_key = _detect_model_input_key(processor)
    print(f"   Model input key: {input_key}")

    # Per Parakeet il processor produce spesso input_features (mel). Precomputarle e serializzarle
    # come liste annidate è molto lento. Default: calcolo feature on-the-fly nel collator.
    dataset_input_key = input_key
    if input_key == "input_features" and not precompute_features:
        dataset_input_key = "input_values"
        print("   Feature extraction: on-the-fly (salvo solo input_values nel dataset)")

    vram_gb = _get_vram_gb()
    if vram_gb is not None:
        print(f"   GPU VRAM: {vram_gb:.1f} GB")

    # 4-bit OBBLIGATORIO per 1.1B
    try:
        import bitsandbytes  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "bitsandbytes non disponibile ma è richiesto per caricare Parakeet 1.1B in 4-bit. "
            "Installa con: pip install bitsandbytes"
        ) from e

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_kwargs: Dict[str, Any] = dict(
        vocab_size=vocab_size,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )

    print("\n📦 Loading Parakeet in 4-bit (NF4) - frozen backbone + train CTC head...")
    model = ParakeetForCTC.from_pretrained(
        checkpoint,
        quantization_config=bnb_config,
        device_map="auto",
        force_download=force_download,
        token=hf_token,
        **model_kwargs,
    )

    # Linear probing: congela tutto, allena solo lm_head reinizializzata
    for p in model.parameters():
        p.requires_grad = False

    ctc_head = _get_ctc_head(model)
    ctc_head.requires_grad_(True)
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

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples deve essere > 0")
        n_train = min(max_samples, len(train_ds))
        n_val = min(max_samples, len(val_ds))
        train_ds = train_ds.select(range(n_train))
        val_ds = val_ds.select(range(n_val))
        print(f"   Subsample attivo: Train={n_train} | Val={n_val}")

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)}")

    import librosa

    def preprocess(batch):
        audio, _ = librosa.load(batch["audio_path"], sr=16000)
        if dataset_input_key == "input_values":
            x = audio.tolist()
        else:
            processed = processor(audio, sampling_rate=16000, return_tensors=None)
            x = _extract_first_key(processed, (input_key,))
            if hasattr(x, "tolist"):
                x = x.tolist()

        labels = tokenizer(batch["ipa_clean"]).input_ids

        # Heuristic per evitare crash CTC se labels troppo lunghe
        if input_key == "input_values":
            approx_frames = int(len(x) // 320)
            if len(labels) > approx_frames:
                labels = labels[:approx_frames]
        elif input_key == "input_features":
            # tipicamente [T, F]
            try:
                t = len(x)
                if len(labels) > t:
                    labels = labels[:t]
            except Exception:
                pass

        return {
            dataset_input_key: x,
            "labels": labels,
            "input_length": len(x) if hasattr(x, "__len__") else 0,
            "label_length": len(labels),
        }

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

    train_ds = train_ds.filter(
        lambda x: x["label_length"] > 0,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    val_ds = val_ds.filter(
        lambda x: x["label_length"] > 0,
        load_from_cache_file=False,
        keep_in_memory=True,
    )

    train_ds.set_format(type=None, columns=[dataset_input_key, "labels"])
    val_ds.set_format(type=None, columns=[dataset_input_key, "labels"])

    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids)
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
        valid = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not valid:
            return {"cer": 1.0}
        preds, labels2 = zip(*valid)
        return {"cer": cer_metric.compute(predictions=preds, references=labels2)}

    # NOTE: Transformers Trainer (versioni recenti) blocca il training su modelli puramente quantizzati.
    # Qui facciamo un training loop PyTorch che allena SOLO la ctc_head (linear probing) mantenendo 4-bit sul backbone.

    os.makedirs(output_dir, exist_ok=True)
    device = next(ctc_head.parameters()).device

    collator = DataCollatorCTCWithPadding(processor, input_key=input_key)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    steps_per_epoch = max(1, int(np.ceil(len(train_loader) / max(1, gradient_accumulation_steps))))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def _decode_batch(logits: torch.Tensor) -> List[str]:
        pred_ids = torch.argmax(logits, dim=-1)
        pred_ids = pred_ids.detach().cpu().numpy()
        return tokenizer.batch_decode(pred_ids)

    def _eval_cer() -> float:
        model.eval()
        preds: List[str] = []
        refs: List[str] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                if "input_features" in batch and isinstance(batch["input_features"], torch.Tensor):
                    batch["input_features"] = batch["input_features"].to(dtype=torch.float16)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                    out = model(**batch)
                    logits = out.logits

                pred_str = _decode_batch(logits)
                label_ids = batch["labels"].detach().cpu().numpy()
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                label_str = tokenizer.batch_decode(label_ids, group_tokens=False)

                for p, r in zip(pred_str, label_str):
                    if r.strip():
                        preds.append(p)
                        refs.append(r)

        if not refs:
            return 1.0
        return float(cer_metric.compute(predictions=preds, references=refs))

    def _print_sample(step: int) -> None:
        try:
            model.eval()
            sample = val_ds[0]
            if input_key == "input_features":
                audio = np.asarray(sample[dataset_input_key], dtype=np.float32)
                feats = processor.feature_extractor([audio], sampling_rate=16000, padding=True, return_tensors="pt")
                x = feats["input_features"].to(device=device, dtype=torch.float16)
                out = model(input_features=x)
            else:
                x = torch.tensor([sample[dataset_input_key]], device=device)
                out = model(input_values=x)
            logits = out.logits
            pred = tokenizer.decode(torch.argmax(logits, dim=-1)[0])
            target_ids = [i for i in sample["labels"] if i != -100]
            target = tokenizer.decode(target_ids)
            print(f"\n📊 [Step {step}] Sample Prediction:")
            print(f"   Target: {target[:80]}{'...' if len(target) > 80 else ''}")
            print(f"   Pred:   {pred[:80]}{'...' if len(pred) > 80 else ''}")
        except Exception as e:
            print(f"\n⚠️ Prediction monitor error: {e}")
        finally:
            model.train()

    print("\n🚀 Starting training (manual loop)...")
    model.train()
    global_step = 0
    best_cer = None

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_loader, start=1):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if "input_features" in batch and isinstance(batch["input_features"], torch.Tensor):
                batch["input_features"] = batch["input_features"].to(dtype=torch.float16)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                out = model(**batch)
                loss = out.loss

            loss_to_backprop = loss / max(1, gradient_accumulation_steps)
            scaler.scale(loss_to_backprop).backward()
            running_loss += float(loss.detach().cpu())

            if step_idx % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ctc_head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = running_loss / max(1, 50 * gradient_accumulation_steps)
                    running_loss = 0.0
                    print(f"[Epoch {epoch}/{epochs}] step={global_step}/{total_steps} loss={avg_loss:.4f}")

                if global_step % 100 == 0:
                    _print_sample(global_step)

        cer = _eval_cer()
        print(f"\n✅ Epoch {epoch} CER: {cer:.4f}")

        if best_cer is None or cer < best_cer:
            best_cer = cer
            best_dir = Path(output_dir) / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            try:
                processor.save_pretrained(str(best_dir))
            except Exception:
                tokenizer.save_pretrained(str(best_dir))
            print(f"✓ Best model saved: {best_dir}")

    final_path = Path(output_dir) / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_path))
    try:
        processor.save_pretrained(str(final_path))
    except Exception:
        tokenizer.save_pretrained(str(final_path))
    print(f"\n✓ Model saved: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Parakeet-CTC 1.1B (CTC)")
    parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--audio-base", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/parakeet_ctc_1p1b")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="HuggingFace model id (es. nvidia/parakeet-ctc-1.1b).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (in alternativa usa env HUGGINGFACE_HUB_TOKEN o HF_TOKEN).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limita il numero di sample (train e val) per sanity run veloci. Esempio: --max-samples 200",
    )
    parser.add_argument(
        "--precompute-features",
        action="store_true",
        help="Precalcola e salva input_features nel dataset (più lento nel preprocessing, ma evita recompute ad ogni epoca).",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of the HuggingFace checkpoint (fixes corrupted cache issues).",
    )

    args = parser.parse_args()

    train_parakeet(
        csv_path=args.data_csv,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        audio_base_path=args.audio_base,
        checkpoint=args.checkpoint,
        hf_token=args.hf_token,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        precompute_features=args.precompute_features,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume=args.resume,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Training script per XLS-R 1B (Raw Waveform) con CTC.

Checkpoint: facebook/wav2vec2-xls-r-1b

Linee guida:
- Vocab custom: data/processed/vocab.json
- Tokenizer: bos_token=None, eos_token=None
- Memoria: fp16=True, gradient_checkpointing=True
- Hyperparams: learning_rate=3e-5, warmup_ratio=0.1, gradient_accumulation_steps=4
- Stabilità CTC: ctc_zero_infinity=True + re-init lm_head (fp16 mode)
- Monitoring: PredictionMonitorCallback ogni 100 step

Uso:
    python scripts/training/train_xlsr_1b.py --epochs 10 --output-dir outputs/xlsr_1b
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Helps reduce CUDA memory fragmentation. Must be set before the first CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCTC,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEFAULT_CHECKPOINT = "facebook/wav2vec2-xls-r-1b"


def _get_vram_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _compute_cer(predictions: List[str], references: List[str]) -> float:
    """CER locale per evitare download/lock di evaluate su Kaggle."""
    try:
        import jiwer

        return float(jiwer.cer(references, predictions))
    except Exception:
        total_edits = 0
        total_chars = 0
        for ref, hyp in zip(references, predictions):
            ref = "" if ref is None else str(ref)
            hyp = "" if hyp is None else str(hyp)
            if len(ref) == 0:
                continue
            total_edits += _levenshtein_distance(ref, hyp)
            total_chars += len(ref)
        return float(total_edits / max(1, total_chars))


def _prepare_model_for_kbit_training_audio(model: nn.Module) -> nn.Module:
    """Fallback for PEFT QLoRA on audio models without input embeddings.

    Newer PEFT versions call `model.enable_input_require_grads()`, which relies on
    `get_input_embeddings()` and fails for Wav2Vec2-style models.

    We replicate the useful parts:
    - cast LayerNorms to fp32 for stability
    - ensure a tensor entering the encoder requires grad (needed by checkpoint)
    """
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.to(torch.float32)

    base = getattr(model, "wav2vec2", None)
    feature_projection = getattr(base, "feature_projection", None) if base is not None else None
    if feature_projection is not None:

        def _make_outputs_require_grads(_module, _inputs, output):
            # IMPORTANT: do NOT call requires_grad_(True) on the output.
            # That would make it a leaf tensor requiring grad, and Wav2Vec2 does
            # in-place ops during SpecAugment (masking), which would crash:
            # "a leaf Variable that requires grad is being used in an in-place operation".
            # Instead, return a non-leaf tensor that requires grad.
            if torch.is_tensor(output):
                dummy = output.new_zeros((), requires_grad=True)
                return output + dummy * 0.0
            if isinstance(output, (tuple, list)):
                new_out = []
                for out in output:
                    if torch.is_tensor(out):
                        dummy = out.new_zeros((), requires_grad=True)
                        new_out.append(out + dummy * 0.0)
                    else:
                        new_out.append(out)
                return tuple(new_out) if isinstance(output, tuple) else new_out
            return None

        try:
            model._require_grads_hook = feature_projection.register_forward_hook(_make_outputs_require_grads)
        except Exception:
            pass

    return model


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
                logits = model(input_values).logits

            pred_ids = torch.argmax(logits, dim=-1)[0]
            pred_str = self.processor.decode(pred_ids)

            blank_id = getattr(getattr(model, "config", None), "pad_token_id", 0)
            try:
                blank_ratio = float((pred_ids == blank_id).float().mean().item())
            except Exception:
                blank_ratio = float("nan")

            label_ids = [i for i in sample["labels"] if i != -100]
            label_str = self.processor.decode(label_ids)

            print(f"\n📊 [Step {state.global_step}] Sample Prediction:")
            print(f"   Target: {label_str[:80]}{'...' if len(label_str) > 80 else ''}")
            print(f"   Pred:   {pred_str[:80]}{'...' if len(pred_str) > 80 else ''}")
            if len(pred_str.strip()) == 0:
                print(f"   ⚠️ WARNING: Empty prediction (blank_ratio={blank_ratio:.3f})")
                if blank_ratio == blank_ratio and blank_ratio > 0.98:
                    print("   ⚠️ Likely blank collapse (almost all blank token).")
        except Exception as e:
            print(f"\n⚠️ Prediction monitor error: {e}")
        finally:
            try:
                model.train()
            except Exception:
                pass


class DataCollatorCTCWithPadding:
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


def train_xlsr_1b(
    csv_path: str,
    vocab_path: str,
    output_dir: str,
    audio_base_path: str = ".",
    epochs: int = 10,
    batch_size: int = 2,
    eval_batch_size: Optional[int] = None,
    max_samples: Optional[int] = None,
    learning_rate: float = 3e-5,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 4,
    max_audio_seconds: Optional[float] = 12.0,
    truncate_audio: bool = True,
    group_by_length: bool = True,
    freeze_backbone: bool = False,
    optim: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    resume: bool = False,
    force_download: bool = False,
):
    print("=" * 60)
    print("TRAINING XLS-R 1B - CTC")
    print("=" * 60)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token=None,
        eos_token=None,
    )

    # Allineato a train_data2vec2.py: feature extractor dal checkpoint
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        DEFAULT_CHECKPOINT,
        force_download=force_download,
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    vocab_size = len(tokenizer)
    print(f"   Vocab size: {vocab_size}")

    vram_gb = _get_vram_gb()
    if vram_gb is not None:
        print(f"   GPU VRAM: {vram_gb:.1f} GB")
        if vram_gb <= 16.5 and (not load_in_4bit and not load_in_8bit) and (not freeze_backbone):
            print(
                "\n⚠️  Nota VRAM: XLS-R 1B full fine-tuning su ~16GB spesso va in OOM.\n"
                "   Consigliato: --load-in-4bit (QLoRA) oppure --freeze-backbone (linear probe).\n"
                "   Inoltre: --batch-size 1 --eval-batch-size 1 --max-audio-seconds 6-12"
            )

    model_kwargs: Dict[str, Any] = dict(
        vocab_size=vocab_size,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )

    if load_in_4bit and load_in_8bit:
        raise ValueError("Scegli una sola modalità: --load-in-4bit oppure --load-in-8bit")

    quantization_config = None
    use_qlora = bool(load_in_4bit or load_in_8bit)
    if use_qlora:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError(
                "Quantizzazione richiesta ma Transformers non espone BitsAndBytesConfig. "
                "Su Kaggle installa/aggiorna transformers e bitsandbytes."
            ) from e

        compute_dtype = torch.float16
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        print(
            "\n📦 Loading XLS-R 1B with quantization "
            + ("(4-bit NF4, QLoRA)" if load_in_4bit else "(8-bit, QLoRA)")
            + "..."
        )
        model = AutoModelForCTC.from_pretrained(
            DEFAULT_CHECKPOINT,
            force_download=force_download,
            quantization_config=quantization_config,
            device_map="auto",
            **model_kwargs,
        )
    else:
        print("\n📦 Loading XLS-R 1B in fp16...")
        model = AutoModelForCTC.from_pretrained(
            DEFAULT_CHECKPOINT,
            force_download=force_download,
            **model_kwargs,
        )

    try:
        model.config.use_cache = False
    except Exception:
        pass

    # TF32 can slightly reduce memory pressure / speed up matmuls on Ampere+.
    # Safe no-op on older GPUs.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    model.freeze_feature_encoder()
    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = ("lm_head" in name)
    try:
        nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(model.lm_head.bias)
    except Exception:
        pass

    if use_qlora:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except Exception as e:
            raise RuntimeError(
                "Quantizzazione attiva: serve anche peft per fare QLoRA. "
                "Su Kaggle: pip install -q peft bitsandbytes"
            ) from e

        try:
            model = prepare_model_for_kbit_training(model)
        except NotImplementedError:
            # Wav2Vec2-style models don't implement get_input_embeddings()
            model = _prepare_model_for_kbit_training_audio(model)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            # NOTE: non impostiamo task_type. Su PEFT recenti evita wrapper che possono
            # inoltrare keyword non compatibili (es. input_ids) ai modelli Wav2Vec2ForCTC.
            modules_to_save=["lm_head"],
        )
        model = get_peft_model(model, lora_config)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    print(f"   Total params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    print(
        f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M"
    )

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

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples deve essere > 0")
        n_train = min(max_samples, len(train_ds))
        n_val = min(max_samples, len(val_ds))
        train_ds = train_ds.select(range(n_train))
        val_ds = val_ds.select(range(n_val))
        print(f"   Subsample attivo: Train={n_train} | Val={n_val}")

    import librosa

    def preprocess(batch):
        audio, _ = librosa.load(batch["audio_path"], sr=16000)

        if max_audio_seconds is not None and max_audio_seconds > 0:
            duration_s = float(len(audio)) / 16000.0
            if duration_s > max_audio_seconds:
                if truncate_audio:
                    audio = audio[: int(max_audio_seconds * 16000)]
                else:
                    return {
                        "input_values": [],
                        "labels": [],
                        "input_length": 0,
                        "label_length": 0,
                    }

        inputs = processor(audio, sampling_rate=16000, return_tensors=None)
        input_values = inputs.input_values[0]
        if hasattr(input_values, "tolist"):
            input_values = input_values.tolist()

        labels = processor.tokenizer(batch["ipa_clean"]).input_ids
        input_frames = len(input_values) // 320
        if len(labels) > input_frames:
            labels = labels[:input_frames]

        return {
            "input_values": input_values,
            "labels": labels,
            "input_length": len(input_values),
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
        keep_in_memory=False,
        desc="Preprocessing train",
    )
    val_ds = val_ds.map(
        preprocess,
        remove_columns=val_ds.column_names,
        num_proc=1,
        load_from_cache_file=False,
        keep_in_memory=False,
        desc="Preprocessing val",
    )

    train_ds = train_ds.filter(
        lambda x: x["input_length"] > 0
        and x["label_length"] > 0
        and x["label_length"] < x["input_length"] // 320,
        load_from_cache_file=False,
        keep_in_memory=False,
    )
    val_ds = val_ds.filter(
        lambda x: x["input_length"] > 0
        and x["label_length"] > 0
        and x["label_length"] < x["input_length"] // 320,
        load_from_cache_file=False,
        keep_in_memory=False,
    )

    train_ds.set_format(type=None, columns=["input_values", "labels", "input_length"])
    val_ds.set_format(type=None, columns=["input_values", "labels", "input_length"])

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        valid = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not valid:
            return {"cer": 1.0}
        preds, labels = zip(*valid)
        return {"cer": _compute_cer(predictions=list(preds), references=list(labels))}

    # Optimizer choice matters a lot for memory on 1B models.
    if optim is None:
        if use_qlora:
            optim = "paged_adamw_8bit"
        elif vram_gb is not None and vram_gb <= 16.5:
            optim = "adafactor"
        else:
            optim = "adamw_torch"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=(eval_batch_size or batch_size),
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
        group_by_length=group_by_length,
        length_column_name="input_length",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        optim=optim,
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
        callbacks=[PredictionMonitorCallback(processor, val_ds, print_every=100)],
    )

    checkpoint = None
    if resume:
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            checkpoint = str(checkpoints[-1])
            print(f"\n🔄 Resuming from: {checkpoint}")

    print("\n🚀 Starting training...")
    print("   (Nota: a fine epoca Trainer lancia evaluation + save; può sembrare un 'blocco' per 1-3 minuti)")
    trainer.train(resume_from_checkpoint=checkpoint)

    final_path = Path(output_dir) / "final_model"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\n✓ Model saved: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train XLS-R 1B (CTC)")
    parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--audio-base", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/xlsr_1b")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size per eval. Default: uguale a --batch-size (consigliato: 1 per 1B).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limita il numero di sample (train e val) per sanity run veloci. Esempio: --max-samples 200",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=12.0,
        help="Durata massima audio in secondi. Default 12s per evitare OOM sui 1B. Metti 0 per disabilitare.",
    )
    parser.add_argument(
        "--no-truncate-audio",
        action="store_true",
        help="Se impostato, i sample oltre --max-audio-seconds vengono scartati invece che troncati.",
    )
    parser.add_argument(
        "--no-group-by-length",
        action="store_true",
        help="Disabilita bucketing per lunghezza (può aumentare padding e memoria).",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Carica il modello quantizzato 8-bit (richiede bitsandbytes). Per training usa QLoRA.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Carica il modello quantizzato 4-bit NF4 (richiede bitsandbytes). Per training usa QLoRA.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Linear probe: congela tutto tranne lm_head (utile quando non puoi fare full fine-tuning).",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Override optimizer Trainer (es. adafactor, adamw_torch, paged_adamw_8bit). Default: auto (paged_adamw_8bit per QLoRA).",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download del checkpoint HF (utile se cache corrotta su Kaggle).",
    )

    args = parser.parse_args()

    max_audio_seconds = args.max_audio_seconds
    if max_audio_seconds is not None and max_audio_seconds <= 0:
        max_audio_seconds = None

    train_xlsr_1b(
        csv_path=args.data_csv,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        audio_base_path=args.audio_base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_samples=args.max_samples,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_audio_seconds=max_audio_seconds,
        truncate_audio=(not args.no_truncate_audio),
        group_by_length=(not args.no_group_by_length),
        freeze_backbone=args.freeze_backbone,
        optim=args.optim,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        resume=args.resume,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Training script per Wav2Vec2 Phoneme (CTC) su SpeechOcean762.

=============================================================================
MOTIVAZIONE SCIENTIFICA
=============================================================================
Test di un modello con inizializzazione specifica per il dominio dei fonemi
invece che SSL generico.

Specifiche:
- Backbone: facebook/wav2vec2-xlsr-53-espeak-cv-ft
- Model class: Wav2Vec2ForCTC (architettura identica a Wav2Vec2Phoneme)
- Tokenizer: Wav2Vec2PhonemeCTCTokenizer (fallback controllato a Wav2Vec2CTCTokenizer)
- Vocab: usa data/processed/vocab.json (custom)
- do_phonemize=False (ipa_clean già fonetico)
- Audio: raw waveform 16kHz

Training:
- learning_rate=3e-5
- warmup_ratio=0.1
- fp16=True
- gradient_checkpointing=True
- gradient_accumulation_steps=4
- Reinit lm_head per evitare CTC blank collapse

Uso:
    python scripts/training/train_wav2vec2_phoneme.py \
        --data-csv data/processed/combined_augmented.csv \
        --vocab-path data/processed/vocab.json \
        --output-dir outputs/wav2vec2_phoneme
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.normalize_ipa import IPANormalizer


# =============================================================================
# CALLBACKS
# =============================================================================


class PredictionMonitorCallback(TrainerCallback):
    """Stampa predizione esempio ogni N step per monitorare convergenza/collapse."""

    def __init__(self, processor: Wav2Vec2Processor, eval_dataset, print_every: int = 100):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.print_every = print_every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.print_every != 0 or state.global_step <= 0:
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

            label_ids = [i for i in sample["labels"] if i != -100]
            label_str = self.processor.decode(label_ids)

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


# =============================================================================
# DATA COLLATOR
# =============================================================================


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

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        return batch


# =============================================================================
# TOKENIZER / PROCESSOR
# =============================================================================


def _load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_processor(vocab_path: str) -> Wav2Vec2Processor:
    """Crea processor usando vocab custom.

    Preferisce Wav2Vec2PhonemeCTCTokenizer se disponibile.
    Imposta bos/eos a None se non presenti nel vocab.json.
    """

    vocab = _load_vocab(vocab_path)
    has_bos = "[BOS]" in vocab or "<s>" in vocab or "bos" in vocab
    has_eos = "[EOS]" in vocab or "</s>" in vocab or "eos" in vocab

    bos_token = None if not has_bos else None
    eos_token = None if not has_eos else None

    tokenizer = None
    tokenizer_init_error: Optional[Exception] = None

    try:
        from transformers import Wav2Vec2PhonemeCTCTokenizer  # type: ignore

        try:
            tokenizer = Wav2Vec2PhonemeCTCTokenizer(
                vocab_path,
                unk_token="[UNK]",
                pad_token="[PAD]",
                word_delimiter_token="|",
                do_phonemize=False,
                bos_token=bos_token,
                eos_token=eos_token,
            )
        except TypeError:
            # Alcune versioni non supportano do_phonemize nel tokenizer
            tokenizer = Wav2Vec2PhonemeCTCTokenizer(
                vocab_path,
                unk_token="[UNK]",
                pad_token="[PAD]",
                word_delimiter_token="|",
                bos_token=bos_token,
                eos_token=eos_token,
            )

        print("✓ Using Wav2Vec2PhonemeCTCTokenizer")

    except Exception as e:
        tokenizer_init_error = e

    if tokenizer is None:
        from transformers import Wav2Vec2CTCTokenizer

        print(
            "⚠️ Wav2Vec2PhonemeCTCTokenizer non disponibile; "
            "fallback a Wav2Vec2CTCTokenizer.\n"
            f"   (details: {type(tokenizer_init_error).__name__}: {tokenizer_init_error})"
        )

        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
            bos_token=bos_token,
            eos_token=eos_token,
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

    # Verifica tecnica: se non presenti nel vocab, BOS/EOS devono essere None
    if getattr(processor.tokenizer, "bos_token", None) is not None and not has_bos:
        processor.tokenizer.bos_token = None
        processor.tokenizer.bos_token_id = None
    if getattr(processor.tokenizer, "eos_token", None) is not None and not has_eos:
        processor.tokenizer.eos_token = None
        processor.tokenizer.eos_token_id = None

    return processor


# =============================================================================
# TRAINING
# =============================================================================


def train_wav2vec2_phoneme(
    csv_path: str,
    vocab_path: str,
    output_dir: str,
    audio_base_path: str = ".",
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 3e-5,
    resume: bool = False,
    freeze_feature_encoder: bool = True,
):
    print("=" * 60)
    print("TRAINING WAV2VEC2 PHONEME (CTC)")
    print("=" * 60)

    print("\n📦 Setup processor with custom IPA vocab...")
    processor = build_processor(vocab_path)
    vocab_size = len(processor.tokenizer)
    print(f"   Custom IPA vocab size: {vocab_size}")

    print("\n📦 Loading model: facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        vocab_size=vocab_size,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )

    if freeze_feature_encoder:
        model.freeze_feature_encoder()
        print("   ✓ Feature encoder frozen")

    # CRITICAL: Reinitialize lm_head to prevent blank collapse
    nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
    nn.init.zeros_(model.lm_head.bias)
    print("   ✓ lm_head reinitialized for custom IPA vocab")

    model.gradient_checkpointing_enable()
    print("   ✓ Gradient checkpointing enabled")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params/1e6:.1f}M")
    print(f"   Trainable: {trainable/1e6:.1f}M")

    # Dataset
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

    # Cast audio (raw waveform 16kHz)
    ds = ds.cast_column("audio_path", Audio(sampling_rate=16000))
    ds = ds.rename_column("audio_path", "audio")

    # Split
    if "split" in ds.column_names:
        train_ds = ds.filter(lambda x: x["split"] == "train")
        val_ds = ds.filter(lambda x: x["split"] == "validation")
        test_ds = ds.filter(lambda x: x["split"] == "test")
    else:
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds, val_ds = split["train"], split["test"]
        test_ds = split["test"]

    dataset = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })
    print(f"   Split: train={len(dataset['train'])}, val={len(dataset['validation'])}, test={len(dataset['test'])}")

    # Normalizzazione IPA coerente (strict, già usata in build_combined/speechocean)
    ipa_normalizer = IPANormalizer(mode="strict")

    def preprocess(batch):
        audio = batch["audio"]["array"]
        inputs = processor(audio, sampling_rate=16000, return_tensors=None)
        input_values = inputs.input_values[0]

        ipa_clean = batch.get("ipa_clean", "")
        ipa_clean = ipa_normalizer.normalize(ipa_clean)

        labels = processor.tokenizer(ipa_clean).input_ids

        return {
            "input_values": input_values,
            "labels": labels,
        }

    print("\n🔄 Preprocessing...")
    dataset = dataset.map(
        preprocess,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )

    # Metrics: PER ~ CER sui simboli IPA
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        valid = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not valid:
            return {"per": 1.0}

        preds, labels = zip(*valid)
        per = cer_metric.compute(predictions=preds, references=labels)
        return {"per": per}

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        logging_steps=100,
        dataloader_num_workers=0,
        group_by_length=False,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorCTCWithPadding(processor),
        compute_metrics=compute_metrics,
        callbacks=[PredictionMonitorCallback(processor, dataset["validation"], print_every=100)],
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
    processor.save_pretrained(str(final_path))

    print(f"\n✓ Model saved: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 Phoneme (CTC)")
    parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--audio-base", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/wav2vec2_phoneme")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-freeze-feature-encoder", action="store_true")

    args = parser.parse_args()

    train_wav2vec2_phoneme(
        csv_path=args.data_csv,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        audio_base_path=args.audio_base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume=args.resume,
        freeze_feature_encoder=not args.no_freeze_feature_encoder,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Training script per MMS-1B (Raw Waveform) con CTC.

Checkpoint: facebook/mms-1b-all

Linee guida implementate:
- Vocab custom: data/processed/vocab.json
- Tokenizer: bos_token=None, eos_token=None
- Stabilità CTC: ctc_zero_infinity=True + re-init lm_head (fp16 mode)
- Memoria: fp16=True, gradient_checkpointing=True
- Hyperparams: lr=3e-5, warmup_ratio=0.1, grad_accum=4
- Monitoring: sample prediction ogni 100 step

Uso:
	python scripts/training/train_mms_1b.py --epochs 10 --output-dir outputs/mms_1b
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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

DEFAULT_CHECKPOINT = "facebook/mms-1b-all"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def _get_vram_gb() -> Optional[float]:
	if not torch.cuda.is_available():
		return None
	props = torch.cuda.get_device_properties(0)
	return props.total_memory / (1024**3)


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


def train_mms_1b(
	csv_path: str,
	vocab_path: str,
	output_dir: str,
	audio_base_path: str = ".",
	epochs: int = 10,
	batch_size: int = 4,
	max_samples: Optional[int] = None,
	learning_rate: float = 3e-5,
	warmup_ratio: float = 0.1,
	gradient_accumulation_steps: int = 4,
	resume: bool = False,
	force_download: bool = False,
):
	print("=" * 60)
	print(f"TRAINING MMS-1B ({DEFAULT_CHECKPOINT}) - CTC")
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

	model_kwargs: Dict[str, Any] = dict(
		vocab_size=vocab_size,
		ctc_loss_reduction="mean",
		ctc_zero_infinity=True,
		pad_token_id=tokenizer.pad_token_id,
		ignore_mismatched_sizes=True,
	)

	print("\n📦 Loading MMS-1B in fp16...")
	model = AutoModelForCTC.from_pretrained(
		DEFAULT_CHECKPOINT,
		force_download=force_download,
		**model_kwargs,
	)
	model.freeze_feature_encoder()
	nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
	nn.init.zeros_(model.lm_head.bias)

	try:
		model.gradient_checkpointing_enable()
	except Exception:
		pass

	print(f"   Total params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

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
		lambda x: x["label_length"] > 0 and x["label_length"] < x["input_length"] // 320,
		load_from_cache_file=False,
		keep_in_memory=False,
	)
	val_ds = val_ds.filter(
		lambda x: x["label_length"] > 0 and x["label_length"] < x["input_length"] // 320,
		load_from_cache_file=False,
		keep_in_memory=False,
	)

	train_ds.set_format(type=None, columns=["input_values", "labels"])
	val_ds.set_format(type=None, columns=["input_values", "labels"])

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
	trainer.train(resume_from_checkpoint=checkpoint)

	final_path = Path(output_dir) / "final_model"
	trainer.save_model(str(final_path))
	processor.save_pretrained(str(final_path))
	print(f"\n✓ Model saved: {final_path}")


def main():
	parser = argparse.ArgumentParser(description="Train MMS-1B (CTC)")
	parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
	parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
	parser.add_argument("--audio-base", type=str, default=".")
	parser.add_argument("--output-dir", type=str, default="outputs/mms_1b")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="Limita il numero di sample (train e val) per sanity run veloci. Esempio: --max-samples 200",
	)
	parser.add_argument("--learning-rate", type=float, default=3e-5)
	parser.add_argument("--warmup-ratio", type=float, default=0.1)
	parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument(
		"--force-download",
		action="store_true",
		help="Force re-download del checkpoint HF (utile se cache corrotta).",
	)

	args = parser.parse_args()

	train_mms_1b(
		csv_path=args.data_csv,
		vocab_path=args.vocab_path,
		output_dir=args.output_dir,
		audio_base_path=args.audio_base,
		epochs=args.epochs,
		batch_size=args.batch_size,
		max_samples=args.max_samples,
		learning_rate=args.learning_rate,
		warmup_ratio=args.warmup_ratio,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		resume=args.resume,
		force_download=args.force_download,
	)


if __name__ == "__main__":
	main()
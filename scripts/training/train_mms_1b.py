#!/usr/bin/env python3
"""Training script per MMS-1B (Raw Waveform) con CTC.

Checkpoint: facebook/mms-1b-all

Linee guida implementate:
- Vocab custom: data/processed/vocab.json
- Tokenizer: bos_token=None, eos_token=None
- Stabilità CTC: ctc_zero_infinity=True + re-init lm_head (fp16 mode)
- Memoria: fp16=True, gradient_checkpointing=True
- 4-bit quantization: auto se VRAM <16GB (BitsAndBytesConfig). In 4-bit il backbone viene congelato e si allena solo la CTC head.
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
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
	BitsAndBytesConfig,
	Wav2Vec2CTCTokenizer,
	Wav2Vec2FeatureExtractor,
	Wav2Vec2ForCTC,
	Wav2Vec2Processor,
)
from transformers.optimization import get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))




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
	learning_rate: float = 3e-5,
	warmup_ratio: float = 0.1,
	gradient_accumulation_steps: int = 4,
	resume: bool = False,
	auto_4bit: bool = True,
	use_4bit: bool = False,
):
	print("=" * 60)
	print("TRAINING MMS-1B (facebook/mms-1b-all) - CTC")
	print("=" * 60)

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

	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
	vocab_size = len(tokenizer)
	print(f"   Vocab size: {vocab_size}")

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
			print("   ⚠️ bitsandbytes non disponibile: disabilito 4-bit (rischio OOM su <16GB).")
			want_4bit = False

	if want_4bit:
		print("\n📦 Loading MMS-1B in 4-bit (NF4) - frozen backbone + train CTC head...")
		bnb_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.float16,
		)
		model = Wav2Vec2ForCTC.from_pretrained(
			"facebook/mms-1b-all",
			quantization_config=bnb_config,
			**model_kwargs,
		)
		for p in model.parameters():
			p.requires_grad = False
		# Train solo la CTC head: tienila in FP32 per evitare crash AMP ("Attempting to unscale FP16 gradients")
		model.lm_head.requires_grad_(True)
		model.lm_head.to(dtype=torch.float32)
		nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
		nn.init.zeros_(model.lm_head.bias)
	else:
		print("\n📦 Loading MMS-1B in fp16...")
		model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", **model_kwargs)
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
		lambda x: x["label_length"] > 0 and x["label_length"] < x["input_length"] // 320,
		load_from_cache_file=False,
		keep_in_memory=True,
	)
	val_ds = val_ds.filter(
		lambda x: x["label_length"] > 0 and x["label_length"] < x["input_length"] // 320,
		load_from_cache_file=False,
		keep_in_memory=True,
	)

	train_ds.set_format(type=None, columns=["input_values", "labels"])
	val_ds.set_format(type=None, columns=["input_values", "labels"])

	# NOTE: Transformers Trainer (versioni recenti) blocca il training su modelli puramente quantizzati (4-bit).
	# Per evitare dipendenze PEFT/LoRA e mantenere linear probing, usiamo un training loop PyTorch manuale.
	if resume:
		print("   ⚠️ --resume non supportato nel training loop manuale (per ora).")

	os.makedirs(output_dir, exist_ok=True)

	collator = DataCollatorCTCWithPadding(processor)
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

	if want_4bit:
		# Device: usa la head (trainabile) come riferimento
		device = next(model.lm_head.parameters()).device
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device)
		if torch.cuda.is_available():
			model.half()
			# Head in FP32 (stabilità + evita crash GradScaler)
			model.lm_head.to(dtype=torch.float32)

	optimizer = torch.optim.AdamW(
		[p for p in model.parameters() if p.requires_grad],
		lr=learning_rate,
		weight_decay=0.01,
	)

	steps_per_epoch = max(1, int(np.ceil(len(train_loader) / max(1, gradient_accumulation_steps))))
	total_steps = epochs * steps_per_epoch
	warmup_steps = int(total_steps * warmup_ratio)
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=warmup_steps,
		num_training_steps=total_steps,
	)
	scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

	def _decode_batch(logits: torch.Tensor) -> List[str]:
		pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
		return processor.batch_decode(pred_ids)

	def _eval_cer() -> float:
		model.eval()
		preds: List[str] = []
		refs: List[str] = []
		with torch.no_grad():
			for batch in val_loader:
				batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
				with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
					out = model(**batch)
					logits = out.logits

				pred_str = _decode_batch(logits)
				label_ids = batch["labels"].detach().cpu().numpy()
				label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
				label_str = processor.batch_decode(label_ids, group_tokens=False)

				for p, r in zip(pred_str, label_str):
					if r.strip():
						preds.append(p)
						refs.append(r)

		if not refs:
			return 1.0
		return _compute_cer(predictions=preds, references=refs)

	def _print_sample(step: int) -> None:
		try:
			model.eval()
			sample = val_ds[0]
			x = torch.tensor([sample["input_values"]], device=device)
			with torch.no_grad():
				with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
					logits = model(input_values=x).logits
			pred = processor.decode(torch.argmax(logits, dim=-1)[0])
			target_ids = [i for i in sample["labels"] if i != -100]
			target = processor.decode(target_ids)
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
			with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
				out = model(**batch)
				loss = out.loss

			loss_to_backprop = loss / max(1, gradient_accumulation_steps)
			scaler.scale(loss_to_backprop).backward()
			running_loss += float(loss.detach().cpu())

			if step_idx % gradient_accumulation_steps == 0:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.lm_head.parameters(), 1.0)
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
			processor.save_pretrained(str(best_dir))
			print(f"✓ Best model saved: {best_dir}")

	final_path = Path(output_dir) / "final_model"
	final_path.mkdir(parents=True, exist_ok=True)
	model.save_pretrained(str(final_path))
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
	parser.add_argument("--learning-rate", type=float, default=3e-5)
	parser.add_argument("--warmup-ratio", type=float, default=0.1)
	parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--auto-4bit", dest="auto_4bit", action="store_true", default=True)
	parser.add_argument("--no-auto-4bit", dest="auto_4bit", action="store_false", help="Disable auto 4-bit")
	parser.add_argument("--use-4bit", action="store_true", help="Force 4-bit (overrides auto)")

	args = parser.parse_args()

	train_mms_1b(
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
	)


if __name__ == "__main__":
	main()
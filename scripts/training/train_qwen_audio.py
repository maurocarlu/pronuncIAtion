#!/usr/bin/env python3
"""
Training script per Qwen2-Audio Encoder con CTC.

=============================================================================
MOTIVAZIONE SCIENTIFICA
=============================================================================

Qwen2-Audio è un modello multimodale SOTA che include un potente audio encoder.
Estraiamo solo la parte audio_tower e aggiungiamo una testa CTC.

Questo test verifica:
- Se encoder multimodali generalizzano a phoneme recognition
- La potenza di rappresentazioni audio da modelli 7B

IMPORTANTE:
- Usiamo quantizzazione 4-bit con bitsandbytes per ridurre VRAM
- Gradient checkpointing per backpropagation efficiente
- Solo l'audio encoder, non il decoder LLM

MODELLO: Qwen/Qwen2-Audio-7B
- Audio encoder: ~1B parametri
- Quantizzazione: 4-bit (BNB)
- VRAM stimata: ~5-6GB

Uso:
    python scripts/training/train_qwen_audio.py --epochs 10
"""

import argparse
import sys
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import numpy as np
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
)
import shutil

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# DRIVE BACKUP CALLBACK
# =============================================================================

class DriveBackupCallback(TrainerCallback):
    """Copia checkpoint su Drive dopo ogni salvataggio."""
    
    def __init__(self, backup_dir: str = None):
        self.backup_dir = backup_dir
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
            if not backup_dir:
                self.backup_dir = None
    
    def on_save(self, args, state, control, **kwargs):
        if not self.backup_dir:
            return
        if self.env == 'colab' and '/drive/' in str(args.output_dir):
            return
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            os.makedirs(self.backup_dir, exist_ok=True)
            model_name = Path(args.output_dir).name
            backup_path = Path(self.backup_dir) / model_name / checkpoint_dir.name
            if checkpoint_dir.resolve() == backup_path.resolve():
                return
            try:
                if self.env == 'colab':
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.copytree(checkpoint_dir, backup_path)
                    print(f"\n💾 Checkpoint: {backup_path}")
                elif self.env == 'kaggle':
                    zip_path = Path(self.backup_dir) / f"{model_name}_checkpoint-{state.global_step}"
                    shutil.make_archive(str(zip_path), 'zip', checkpoint_dir)
                    print(f"\n💾 Checkpoint: {zip_path}.zip")
            except Exception as e:
                print(f"\n⚠️ Backup fallito: {e}")


# =============================================================================
# QWEN2-AUDIO ENCODER + CTC
# =============================================================================

class Qwen2AudioEncoderForCTC(nn.Module):
    """
    Qwen2-Audio Encoder con CTC head.
    
    Estrae solo l'audio encoder dal modello multimodale,
    aggiungendo una testa per CTC loss su fonemi.
    
    Usa quantizzazione 4-bit per ridurre VRAM.
    """
    
    def __init__(self, vocab_size: int, device: str = "cuda"):
        super().__init__()
        
        print("   📦 Caricamento Qwen2-Audio (4-bit)...")
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load Qwen2-Audio
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            trust_remote_code=True,
        )
        
        qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Extract only audio encoder
        self.audio_encoder = qwen_model.audio_tower
        
        # Freeze encoder
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        
        # CTC head (trainable)
        hidden_size = 1280  # Qwen2-Audio encoder hidden size
        self.ctc_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, vocab_size),
        ).to(device)
        
        # For gradient checkpointing
        self._device = device
        
        print("   ✓ Audio encoder estratto")
        print("   ✓ Quantization 4-bit attiva")
    
    def forward(self, input_features, labels=None, **kwargs):
        """
        Args:
            input_features: Audio mel features
            labels: Target phoneme IDs
        
        Returns:
            Dict with 'loss' and 'logits'
        """
        # Encode audio
        with torch.no_grad():
            audio_outputs = self.audio_encoder(input_features)
            hidden_states = audio_outputs.last_hidden_state  # [batch, time, hidden]
        
        # CTC head
        logits = self.ctc_head(hidden_states.to(self._device))  # [batch, time, vocab]
        
        loss = None
        if labels is not None:
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)
            
            input_lengths = torch.full(
                (logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device
            )
            
            label_mask = labels != -100
            target_lengths = label_mask.sum(dim=-1)
            labels_for_ctc = labels.clone()
            labels_for_ctc[labels_for_ctc == -100] = 0
            
            loss = nn.functional.ctc_loss(
                log_probs, labels_for_ctc, input_lengths, target_lengths,
                blank=0, zero_infinity=True
            )
        
        return {"loss": loss, "logits": logits}


# =============================================================================
# ALTERNATIVE: Simpler Audio Encoder (if Qwen fails)
# =============================================================================

class SimpleAudioEncoderForCTC(nn.Module):
    """
    Fallback: Encoder audio semplice se Qwen2-Audio non è disponibile.
    
    Usa Whisper encoder come alternativa.
    """
    
    def __init__(self, vocab_size: int):
        super().__init__()
        
        from transformers import WhisperModel
        
        print("   📦 Fallback: usando Whisper encoder...")
        whisper = WhisperModel.from_pretrained("openai/whisper-small")
        self.encoder = whisper.encoder
        
        # Freeze most layers
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 layers
        for layer in self.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        hidden_size = 768
        self.ctc_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, vocab_size),
        )
    
    def forward(self, input_features, labels=None, **kwargs):
        encoder_outputs = self.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state
        logits = self.ctc_head(hidden_states)
        
        loss = None
        if labels is not None:
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)
            
            input_lengths = torch.full(
                (logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device
            )
            
            label_mask = labels != -100
            target_lengths = label_mask.sum(dim=-1)
            labels_for_ctc = labels.clone()
            labels_for_ctc[labels_for_ctc == -100] = 0
            
            loss = nn.functional.ctc_loss(
                log_probs, labels_for_ctc, input_lengths, target_lengths,
                blank=0, zero_infinity=True
            )
        
        return {"loss": loss, "logits": logits}


# =============================================================================
# DATA COLLATOR
# =============================================================================

class DataCollatorQwenCTC:
    """Data collator per Qwen2-Audio + CTC."""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad mel features
        input_features = [f["input_features"] for f in features]
        
        max_len = max(f.shape[-1] for f in input_features)
        padded = []
        for f in input_features:
            if f.shape[-1] < max_len:
                pad_width = max_len - f.shape[-1]
                f = np.pad(f, ((0, 0), (0, pad_width)), mode='constant')
            padded.append(f)
        
        batch_input = torch.tensor(np.stack(padded), dtype=torch.float32)
        
        # Pad labels
        labels = [f["labels"] for f in features]
        max_label_len = max(len(l) for l in labels)
        padded_labels = [l + [-100] * (max_label_len - len(l)) for l in labels]
        batch_labels = torch.tensor(padded_labels, dtype=torch.long)
        
        return {
            "input_features": batch_input,
            "labels": batch_labels,
        }


# =============================================================================
# CUSTOM TRAINER
# =============================================================================

class Qwen2CTCTrainer(Trainer):
    """Custom trainer per Qwen2-Audio CTC."""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# TRAINING
# =============================================================================

def train_qwen_audio(
    csv_path: str,
    vocab_path: str,
    output_dir: str,
    audio_base_path: str = ".",
    epochs: int = 10,
    batch_size: int = 2,  # Small batch for 7B model
    learning_rate: float = 1e-4,
    resume: bool = False,
    use_fallback: bool = False,
):
    """Training Qwen2-Audio encoder con CTC."""
    
    print("=" * 60)
    print("TRAINING QWEN2-AUDIO ENCODER (4-bit)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ Device: {device}")
    
    # Check bitsandbytes
    try:
        import bitsandbytes
        print("   ✓ bitsandbytes disponibile")
    except ImportError:
        print("   ⚠️ bitsandbytes non installato! Installa con:")
        print("      pip install bitsandbytes")
        use_fallback = True
    
    # Setup tokenizer
    print("\n📦 Setup...")
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    vocab_size = len(tokenizer)
    print(f"   Vocab size: {vocab_size}")
    
    # Load model
    print("\n📦 Caricamento modello...")
    
    if use_fallback:
        model = SimpleAudioEncoderForCTC(vocab_size)
        model = model.to(device)
        # Use Whisper feature extractor
        from transformers import WhisperFeatureExtractor
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    else:
        try:
            model = Qwen2AudioEncoderForCTC(vocab_size, device=device)
            feature_extractor = model.processor.feature_extractor
        except Exception as e:
            print(f"   ⚠️ Errore caricamento Qwen2-Audio: {e}")
            print("   📦 Usando fallback (Whisper encoder)...")
            model = SimpleAudioEncoderForCTC(vocab_size)
            model = model.to(device)
            from transformers import WhisperFeatureExtractor
            feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    
    # Enable gradient checkpointing
    if hasattr(model, 'audio_encoder') and hasattr(model.audio_encoder, 'gradient_checkpointing_enable'):
        model.audio_encoder.gradient_checkpointing_enable()
        print("   ✓ Gradient checkpointing abilitato")
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()
        print("   ✓ Gradient checkpointing abilitato")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parametri trainabili: {trainable_params/1e6:.1f}M")
    
    # Load dataset
    print(f"\n📥 Caricamento dataset: {csv_path}")
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
    
    ds = ds.cast_column("audio_path", Audio(sampling_rate=16000))
    ds = ds.rename_column("audio_path", "audio")
    
    # Split
    if "split" in ds.column_names:
        train_ds = ds.filter(lambda x: x["split"] == "train")
        val_ds = ds.filter(lambda x: x["split"] == "validation")
    else:
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        val_ds = split["test"]
    
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Preprocess
    def preprocess(batch):
        audio = batch["audio"]["array"]
        mel = feature_extractor(
            audio, sampling_rate=16000, return_tensors="np"
        ).input_features[0]
        batch["input_features"] = mel
        batch["labels"] = tokenizer(batch["ipa_clean"]).input_ids
        return batch
    
    print("\n🔄 Extracting features...")
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names, num_proc=1)
    val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names, num_proc=1)
    
    # Metrics
    cer_metric = evaluate.load("cer")
    
    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        valid = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not valid:
            return {"cer": 1.0}
        preds, labels = zip(*valid)
        cer = cer_metric.compute(predictions=preds, references=labels)
        return {"cer": cer}
    
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Higher for small batch
        learning_rate=learning_rate,
        warmup_steps=300,
        weight_decay=0.01,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=50,
        dataloader_num_workers=0,
        report_to="none",
    )
    
    # Trainer
    data_collator = DataCollatorQwenCTC(None)
    
    trainer = Qwen2CTCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[DriveBackupCallback()],
    )
    
    # Resume
    checkpoint = None
    if resume:
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            checkpoint = str(checkpoints[-1])
            print(f"\n🔄 Resume da: {checkpoint}")
    
    # Train
    print("\n🚀 Avvio training...")
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save final
    final_path = Path(output_dir) / "final_model_qwen_audio"
    os.makedirs(final_path, exist_ok=True)
    
    # Save CTC head only (encoder is quantized)
    torch.save(model.ctc_head.state_dict(), final_path / "ctc_head.bin")
    
    import json
    config = {
        "model_type": "qwen2_audio_ctc",
        "vocab_size": vocab_size,
        "base_model": "Qwen/Qwen2-Audio-7B-Instruct",
        "quantization": "4bit",
    }
    with open(final_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    tokenizer.save_pretrained(str(final_path))
    
    print(f"\n✓ Modello salvato: {final_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Qwen2-Audio Encoder + CTC")
    parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--audio-base", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/qwen_audio")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-fallback", action="store_true", help="Use Whisper encoder instead")
    
    args = parser.parse_args()
    
    train_qwen_audio(
        csv_path=args.data_csv,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        audio_base_path=args.audio_base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume=args.resume,
        use_fallback=args.use_fallback,
    )
    
    print("\n" + "=" * 60)
    print("✓ Training Qwen2-Audio completato!")
    print("=" * 60)


if __name__ == "__main__":
    main()

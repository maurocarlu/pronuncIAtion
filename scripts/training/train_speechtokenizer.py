#!/usr/bin/env python3
"""
Training script per SpeechTokenizer - Approccio Discreto.

=============================================================================
MOTIVAZIONE SCIENTIFICA
=============================================================================

SpeechTokenizer converte l'audio in token discreti usando Residual Vector
Quantization (RVQ). Questo approccio è radicalmente diverso dagli altri
modelli che usano rappresentazioni continue.

Esperimento:
- Estraiamo gli indici discreti dal PRIMO layer RVQ
- Usiamo un classificatore (embedding + transformer + linear) sui token
- Verifichiamo se le unità discrete catturano info fonetiche

MODELLO: fnlp/SpeechTokenizer
- Encoder + RVQ layers
- Ogni frame audio → sequenza di codici discreti

Uso:
    python scripts/training/train_speechtokenizer.py --epochs 10
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
# DISCRETE TOKEN CLASSIFIER
# =============================================================================

class DiscreteTokenClassifier(nn.Module):
    """
    Classificatore per token discreti da SpeechTokenizer.
    
    Architecture:
        RVQ Codes → Embedding → Transformer → Linear → CTC Loss
    
    I codici RVQ del primo layer (semantic layer) vengono embedded
    e processati da un leggero transformer per la classificazione.
    """
    
    def __init__(
        self,
        vocab_size: int,           # Output phonemes
        codebook_size: int = 1024, # SpeechTokenizer codebook
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Embedding for discrete codes
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, 2048, embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CTC head
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids, labels=None, **kwargs):
        """
        Args:
            input_ids: Discrete token indices [batch, seq_len]
            labels: Target phoneme IDs for CTC
        
        Returns:
            Dict with 'loss' and 'logits'
        """
        seq_len = input_ids.size(1)
        
        # Embed tokens
        x = self.embedding(input_ids)  # [batch, seq, embed]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer
        x = self.transformer(x)  # [batch, seq, embed]
        
        # CTC head
        x = self.dropout(x)
        logits = self.lm_head(x)  # [batch, seq, vocab]
        
        loss = None
        if labels is not None:
            # CTC Loss
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # [time, batch, vocab]
            
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
# SPEECHTOKENIZER EXTRACTOR
# =============================================================================

class SpeechTokenizerExtractor:
    """Wrapper per estrarre codici discreti da SpeechTokenizer."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carica SpeechTokenizer."""
        try:
            from speechtokenizer import SpeechTokenizer
            
            # Download config e checkpoint
            config_path = "speechtokenizer_hubert_avg/config.json"
            ckpt_path = "speechtokenizer_hubert_avg/SpeechTokenizer.pt"
            
            # Se non esistono, scarica
            if not Path(config_path).exists():
                print("   📥 Downloading SpeechTokenizer...")
                import gdown
                os.makedirs("speechtokenizer_hubert_avg", exist_ok=True)
                
                # Download from HuggingFace
                from huggingface_hub import hf_hub_download
                config_path = hf_hub_download(
                    repo_id="fnlp/SpeechTokenizer",
                    filename="speechtokenizer_hubert_avg/config.json",
                    local_dir="."
                )
                ckpt_path = hf_hub_download(
                    repo_id="fnlp/SpeechTokenizer",
                    filename="speechtokenizer_hubert_avg/SpeechTokenizer.pt",
                    local_dir="."
                )
            
            self.model = SpeechTokenizer.load_from_checkpoint(
                config_path, ckpt_path
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print("   ✓ SpeechTokenizer caricato")
            
        except ImportError:
            print("   ⚠️ SpeechTokenizer non installato!")
            print("   Installa con: pip install speechtokenizer")
            raise
    
    @torch.no_grad()
    def extract_codes(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Estrae codici discreti dal primo layer RVQ.
        
        Args:
            audio: Audio waveform [samples]
            sr: Sample rate (16kHz expected)
        
        Returns:
            Array of discrete codes [time_steps]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        # Encode
        codes = self.model.encode(audio_tensor)  # [n_q, batch, time]
        
        # Take first RVQ layer (semantic layer)
        first_layer_codes = codes[0, 0, :].cpu().numpy()
        
        return first_layer_codes


# =============================================================================
# DATA COLLATOR
# =============================================================================

class DataCollatorDiscreteCTC:
    """Data collator per discrete tokens + CTC."""
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Pad input codes
        input_ids = [f["input_ids"] for f in features]
        max_len = max(len(ids) for ids in input_ids)
        
        padded_inputs = []
        for ids in input_ids:
            padded = list(ids) + [self.pad_token_id] * (max_len - len(ids))
            padded_inputs.append(padded)
        
        batch_inputs = torch.tensor(padded_inputs, dtype=torch.long)
        
        # Pad labels
        labels = [f["labels"] for f in features]
        max_label_len = max(len(l) for l in labels)
        
        padded_labels = []
        for l in labels:
            padded = list(l) + [-100] * (max_label_len - len(l))
            padded_labels.append(padded)
        
        batch_labels = torch.tensor(padded_labels, dtype=torch.long)
        
        return {
            "input_ids": batch_inputs,
            "labels": batch_labels,
        }


# =============================================================================
# CUSTOM TRAINER
# =============================================================================

class DiscreteCTCTrainer(Trainer):
    """Custom trainer per discrete token + CTC."""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# TRAINING
# =============================================================================

def train_speechtokenizer(
    csv_path: str,
    vocab_path: str,
    output_dir: str,
    audio_base_path: str = ".",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    resume: bool = False,
):
    """Training classifier su discrete speech tokens."""
    
    print("=" * 60)
    print("TRAINING SPEECHTOKENIZER (Discrete Codes)")
    print("=" * 60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️ Device: {device}")
    
    # Setup tokenizer
    print("\n📦 Setup...")
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
    
    # Load SpeechTokenizer
    print("\n📦 Caricamento SpeechTokenizer...")
    speech_tokenizer = SpeechTokenizerExtractor(device=device)
    
    # Load classifier model
    print("\n📦 Creazione classificatore...")
    model = DiscreteTokenClassifier(
        vocab_size=vocab_size,
        codebook_size=1024,  # SpeechTokenizer default
        embed_dim=256,
        num_heads=4,
        num_layers=2,
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parametri classificatore: {total_params/1e6:.1f}M")
    
    # Load dataset
    print(f"\n📥 Caricamento dataset: {csv_path}")
    ds = load_dataset("csv", data_files=csv_path)["train"]
    
    # Fix paths
    def fix_audio_path(example):
        path = example["audio_path"].replace("\\", "/")
        if not os.path.isabs(path):
            path = os.path.join(audio_base_path, path)
        example["audio_path"] = path
        return example
    
    ds = ds.map(fix_audio_path, num_proc=1)
    ds = ds.filter(lambda x: Path(x["audio_path"]).exists(), num_proc=1)
    print(f"   Samples: {len(ds)}")
    
    # Cast audio
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
    
    # Preprocess - Extract discrete codes
    def preprocess(batch):
        audio = batch["audio"]["array"]
        
        # Extract discrete codes
        try:
            codes = speech_tokenizer.extract_codes(audio)
            batch["input_ids"] = codes.tolist()
        except Exception as e:
            # Fallback: random codes (will be filtered)
            batch["input_ids"] = [0]
        
        batch["labels"] = tokenizer(batch["ipa_clean"]).input_ids
        return batch
    
    print("\n🔄 Extracting discrete codes (questo può richiedere tempo)...")
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names, num_proc=1)
    val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names, num_proc=1)
    
    # Filter failed extractions
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 1)
    val_ds = val_ds.filter(lambda x: len(x["input_ids"]) > 1)
    
    print(f"   After filtering: Train={len(train_ds)}, Val={len(val_ds)}")
    
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
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=200,
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
    data_collator = DataCollatorDiscreteCTC()
    
    trainer = DiscreteCTCTrainer(
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
    final_path = Path(output_dir) / "final_model_speechtokenizer"
    os.makedirs(final_path, exist_ok=True)
    
    torch.save(model.state_dict(), final_path / "pytorch_model.bin")
    
    import json
    config = {
        "model_type": "speechtokenizer_discrete_ctc",
        "vocab_size": vocab_size,
        "codebook_size": 1024,
        "embed_dim": 256,
        "num_heads": 4,
        "num_layers": 2,
    }
    with open(final_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    tokenizer.save_pretrained(str(final_path))
    
    print(f"\n✓ Modello salvato: {final_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SpeechTokenizer Discrete Classifier")
    parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--audio-base", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="outputs/speechtokenizer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()
    
    train_speechtokenizer(
        csv_path=args.data_csv,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        audio_base_path=args.audio_base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume=args.resume,
    )
    
    print("\n" + "=" * 60)
    print("✓ Training SpeechTokenizer completato!")
    print("=" * 60)


if __name__ == "__main__":
    main()

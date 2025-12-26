#!/usr/bin/env python3
"""
Training script per HuBERT Large.

=============================================================================
ARCHITETTURA E MOTIVAZIONE SCIENTIFICA
=============================================================================

HuBERT (Hidden-Unit BERT) è un modello self-supervised per speech che usa
una strategia di pre-training diversa da WavLM:

DIFFERENZE CHIAVE:
┌─────────────────────┬──────────────────┬──────────────────────┐
│ Aspetto             │ WavLM            │ HuBERT               │
├─────────────────────┼──────────────────┼──────────────────────┤
│ Pre-training        │ Contrastive +    │ Masked Prediction    │
│                     │ Denoising        │ (K-means targets)    │
├─────────────────────┼──────────────────┼──────────────────────┤
│ Target durante      │ Frame audio      │ Cluster ID (discrete)│
│ pre-training        │ (continuo)       │                      │
├─────────────────────┼──────────────────┼──────────────────────┤
│ Robustezza rumore   │ Alta (denoising) │ Moderata             │
├─────────────────────┼──────────────────┼──────────────────────┤
│ Performance ASR     │ SOTA             │ Molto buona          │
└─────────────────────┴──────────────────┴──────────────────────┘

PERCHÉ TESTARE HUBERT?
1. Confronto SOTA: Verificare se WavLM è effettivamente superiore
2. Pre-training diverso: HuBERT usa target discreti (k-means)
3. Richiesta supervisore: Benchmark richiesto esplicitamente

MODELLO USATO:
- facebook/hubert-large-ls960-ft: Fine-tuned su LibriSpeech 960h
- 24 Transformer layers, 1024 hidden dimension
- ~300M parametri

TRAINING:
- Fine-tuning standard con CTC loss
- Freeze feature encoder per stabilità
- 5 epoche (come richiesto)

Uso:
    python scripts/train_hubert.py --epochs 5
    python scripts/train_hubert.py --config configs/training_config.yaml
"""

import argparse
import sys
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
import pandas as pd
import yaml
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    HubertForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import shutil


# =============================================================================
# DRIVE BACKUP CALLBACK
# =============================================================================

class DriveBackupCallback(TrainerCallback):
    """
    Callback che copia i checkpoint su Drive dopo ogni salvataggio.
    Funziona su Colab (Drive montato) o crea ZIP su Kaggle.
    """
    
    def __init__(self, backup_dir: str = None):
        self.backup_dir = backup_dir
        # Rileva ambiente
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
                self.backup_dir = None  # No backup on local
    
    def on_save(self, args, state, control, **kwargs):
        """Chiamato dopo ogni salvataggio checkpoint."""
        if not self.backup_dir:
            return
        
        # Skip se output_dir è già su Drive (evita copia su se stesso)
        if self.env == 'colab' and '/drive/' in str(args.output_dir):
            return  # Già su Drive, niente da fare
        
        # Trova l'ultimo checkpoint salvato
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        
        if checkpoint_dir.exists():
            # Crea directory backup se non esiste
            os.makedirs(self.backup_dir, exist_ok=True)
            
            # Nome del modello dalla directory
            model_name = Path(args.output_dir).name
            backup_path = Path(self.backup_dir) / model_name / checkpoint_dir.name
            
            # Evita copia su se stessa
            if checkpoint_dir.resolve() == backup_path.resolve():
                return
            
            try:
                if self.env == 'colab':
                    # Crea parent directory
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    # Colab: copia diretta su Drive montato
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.copytree(checkpoint_dir, backup_path)
                    print(f"\n💾 Checkpoint copiato su Drive: {backup_path}")
                    
                elif self.env == 'kaggle':
                    # Kaggle: crea ZIP per download
                    zip_path = Path(self.backup_dir) / f"{model_name}_checkpoint-{state.global_step}"
                    shutil.make_archive(str(zip_path), 'zip', checkpoint_dir)
                    print(f"\n💾 Checkpoint compresso: {zip_path}.zip")
                    print("   (Scarica da Output panel)")
                    
            except Exception as e:
                print(f"\n⚠️ Backup fallito: {e}")

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# DATA COLLATOR
# =============================================================================

class DataCollatorCTCWithPadding:
    """
    Data collator per CTC training.
    
    Raggruppa batch di samples e applica padding appropriato
    sia agli input audio che alle labels.
    
    NOTA IMPORTANTE:
    - Labels vengono paddati con -100 (ignorato da CTC loss)
    - Input audio vengono paddati con 0.0
    """
    
    def __init__(self, processor: Wav2Vec2Processor, padding: bool = True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        # Separa input e labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Pad labels
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Replace padding with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        
        return batch


# =============================================================================
# TRAINING CLASS
# =============================================================================

class HuBERTTrainer:
    """
    Trainer per HuBERT Large con CTC loss.
    
    Questo trainer implementa il fine-tuning di HuBERT per phoneme recognition
    usando la stessa pipeline degli altri modelli (WavLM, XLS-R).
    
    Args:
        config: Dizionario di configurazione
        
    Attributes:
        model: HubertForCTC model
        processor: Wav2Vec2Processor
        config: Configuration dict
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[HuBERT] Device: {self.device}")
        
        # Carica processor e modello
        self._setup_processor()
        self._setup_model()
    
    def _setup_processor(self):
        """Inizializza tokenizer e processor."""
        vocab_path = self.config["data"]["vocab_path"]
        
        print(f"[HuBERT] Caricamento vocab da: {vocab_path}")
        
        # Tokenizer condiviso con altri modelli
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
        
        # Feature extractor per HuBERT
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
        )
        
        print(f"[HuBERT] Vocab size: {len(self.tokenizer)}")
    
    def _setup_model(self):
        """
        Carica HuBERT Large e configura per CTC.
        
        NOTA: Usiamo HubertForCTC che ha già il CTC head integrato.
        La configurazione è simile a Wav2Vec2ForCTC.
        """
        model_name = self.config["model"]["name"]
        
        print(f"[HuBERT] Caricamento modello: {model_name}")
        
        self.model = HubertForCTC.from_pretrained(
            model_name,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            ctc_loss_reduction="mean",
            pad_token_id=self.tokenizer.pad_token_id,
            vocab_size=len(self.tokenizer),
            ignore_mismatched_sizes=True,  # Necessario per vocab diverso
        )
        
        # Freeze feature encoder per stabilità
        # I layer CNN sono già pre-trained e funzionano bene
        if self.config["model"].get("freeze_feature_encoder", True):
            self.model.freeze_feature_encoder()
            print("[HuBERT] Feature encoder frozen")
        
        # Conta parametri
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[HuBERT] Parametri totali: {total_params / 1e6:.1f}M")
        print(f"[HuBERT] Parametri trainabili: {trainable_params / 1e6:.1f}M")
    
    def load_and_prepare_dataset(self):
        """
        Carica e preprocessa il dataset.
        
        Usa lo stesso formato CSV degli altri script per consistenza.
        """
        csv_path = self.config["data"]["csv_path"]
        audio_base = self.config["data"]["audio_base_path"]
        
        print(f"[HuBERT] Caricamento dataset da: {csv_path}")
        
        # Carica CSV con load_dataset per evitare problemi PyArrow
        dataset = load_dataset("csv", data_files=csv_path, split="train")
        
        print(f"[HuBERT] Samples totali: {len(dataset)}")
        
        # Split
        val_size = self.config["data"].get("val_size", 0.1)
        splits = dataset.train_test_split(test_size=val_size)
        
        print(f"[HuBERT] Train: {len(splits['train'])}, Val: {len(splits['test'])}")
        
        # Preprocess function con gestione path
        base_path = Path(audio_base)
        
        def preprocess(batch):
            # Fix path e carica audio
            audio_path = str(batch.get("audio_path", "")).replace("\\", "/")
            full_path = base_path / audio_path
            
            try:
                import librosa
                audio, sr = librosa.load(full_path, sr=16000)
            except Exception as e:
                # Audio fallback (silenzio breve)
                audio = np.zeros(16000, dtype=np.float32)
            
            # Process audio
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors=None,
            )
            batch["input_values"] = inputs.input_values[0]
            
            # Tokenize IPA
            ipa = batch.get("ipa_clean", "")
            if not ipa or (isinstance(ipa, float) and np.isnan(ipa)):
                ipa = ""
            batch["labels"] = self.tokenizer(str(ipa)).input_ids
            
            return batch
        
        # Apply preprocessing con barra progresso
        print("\n🔄 Preprocessing TRAIN set...")
        self.train_dataset = splits["train"].map(
            preprocess,
            remove_columns=splits["train"].column_names,
            desc="Train preprocessing",
            num_proc=1,
        )
        
        print("\n🔄 Preprocessing VAL set...")
        self.val_dataset = splits["test"].map(
            preprocess,
            remove_columns=splits["test"].column_names,
            desc="Val preprocessing",
            num_proc=1,
        )
        print("✓ Preprocessing completato!\n")
    
    def train(self):
        """
        Esegue il training con Hugging Face Trainer.
        """
        output_dir = self.config["training"]["output_dir"]
        
        # Data collator
        data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True,
        )
        
        # Metric
        cer_metric = evaluate.load("cer")
        
        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)
            
            # Decode
            pred_str = self.processor.batch_decode(pred_ids)
            
            # Get labels (replace -100)
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
            label_str = self.processor.batch_decode(label_ids, group_tokens=False)
            
            # Compute CER (= PER for phonemes)
            cer = cer_metric.compute(predictions=pred_str, references=label_str)
            
            return {"cer": cer, "per": cer}
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            group_by_length=False,
            per_device_train_batch_size=self.config["training"].get("batch_size", 4),
            per_device_eval_batch_size=self.config["training"].get("batch_size", 4),
            gradient_accumulation_steps=self.config["training"].get("gradient_accumulation", 4),
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.config["training"].get("epochs", 5),
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            learning_rate=self.config["training"].get("learning_rate", 3e-5),
            warmup_steps=self.config["training"].get("warmup_steps", 100),
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            report_to="none",
        )
        
        # Trainer con callback per backup su Drive
        trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.processor,
            callbacks=[DriveBackupCallback()],
        )
        
        print("\n" + "="*60)
        print("TRAINING HUBERT LARGE")
        print("="*60)
        
        # Cerca checkpoint esistenti - RESUME AUTOMATICO
        checkpoint = None
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            # Ordina per numero checkpoint e prendi l'ultimo
            checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            checkpoint = str(checkpoints[-1])
            print(f"\n🔄 Trovato checkpoint! Resume da: {checkpoint}")
        else:
            print("\n📝 Nessun checkpoint trovato. Inizio training da zero.")
        
        # Train con gestione errore scaler (mismatch fp16)
        try:
            trainer.train(resume_from_checkpoint=checkpoint)
        except AttributeError as e:
            if "scaler" in str(e) or "NoneType" in str(e):
                print(f"\n⚠️ Errore scaler fp16: {e}")
                # Rimuovi solo il file scaler.pt, non tutto il checkpoint
                if checkpoint:
                    scaler_file = Path(checkpoint) / "scaler.pt"
                    if scaler_file.exists():
                        scaler_file.unlink()
                        print(f"   Rimosso {scaler_file}, riprovo...")
                        trainer.train(resume_from_checkpoint=checkpoint)
                    else:
                        print("   Riprovo senza checkpoint...")
                        trainer.train(resume_from_checkpoint=None)
                else:
                    trainer.train(resume_from_checkpoint=None)
            else:
                raise
        
        # Save final model
        final_path = Path(output_dir) / "final_model_hubert"
        trainer.save_model(str(final_path))
        self.processor.save_pretrained(str(final_path))
        
        print("\n" + "="*60)
        print(f"✓ Training completato!")
        print(f"  Modello salvato in: {final_path}")
        print("="*60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Training HuBERT Large per phoneme recognition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path al config YAML"
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default="data/processed/combined_augmented.csv",
        help="Path al CSV training"
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="data/processed/vocab.json",
        help="Path al vocab.json"
    )
    parser.add_argument(
        "--audio-base",
        type=str,
        default=".",
        help="Base path per audio files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/hubert_large",
        help="Directory output"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Numero epoche"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (ridotto per memoria)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Riprendi training dall'ultimo checkpoint"
    )
    
    args = parser.parse_args()
    
    # Build config
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model": {
                "name": "facebook/hubert-large-ls960-ft",
                "freeze_feature_encoder": True,
            },
            "data": {
                "csv_path": args.data_csv,
                "vocab_path": args.vocab_path,
                "audio_base_path": args.audio_base,
                "val_size": 0.1,
            },
            "training": {
                "output_dir": args.output_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "gradient_accumulation": 4,
                "warmup_steps": 100,
            },
        }
    
    # Aggiungi flag resume
    config["resume"] = args.resume
    
    # Train
    trainer = HuBERTTrainer(config)
    trainer.load_and_prepare_dataset()
    trainer.train()


if __name__ == "__main__":
    main()

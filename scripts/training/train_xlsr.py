#!/usr/bin/env python3
"""
Training script per XLS-R (wav2vec2-xls-r-300m) - Modello Ausiliario.

XLS-R è un modello wav2vec2 multilingua addestrato su 128 lingue.
Viene usato come secondo modello nell'ensemble per massimizzare
la diversità e migliorare la generalizzazione.

Motivazione scientifica:
- XLS-R ha visto molte più lingue durante il pre-training
- Cattura variazioni fonetiche cross-lingua
- Complementa WavLM (focalizzato su inglese) nell'ensemble
- La diversità tra modelli migliora le performance dell'ensemble

IMPORTANTE: Usa lo stesso vocab.json di WavLM per allineare gli output.

Uso:
    python scripts/train_xlsr.py --config configs/training_config.yaml
    python scripts/train_xlsr.py --data-csv data/processed/combined_augmented.csv
"""

import argparse
import sys
import os
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import shutil

warnings.filterwarnings("ignore")

# Add project root to path
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
        
        # Skip se output_dir è già su Drive (evita copia su se stesso)
        if self.env == 'colab' and '/drive/' in str(args.output_dir):
            return  # Già su Drive, niente da fare
        
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            os.makedirs(self.backup_dir, exist_ok=True)
            model_name = Path(args.output_dir).name
            backup_path = Path(self.backup_dir) / model_name / checkpoint_dir.name
            
            # Evita copia su se stessa
            if checkpoint_dir.resolve() == backup_path.resolve():
                return
            
            try:
                if self.env == 'colab':
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.copytree(checkpoint_dir, backup_path)
                    print(f"\n💾 Checkpoint copiato su Drive: {backup_path}")
                elif self.env == 'kaggle':
                    zip_path = Path(self.backup_dir) / f"{model_name}_checkpoint-{state.global_step}"
                    shutil.make_archive(str(zip_path), 'zip', checkpoint_dir)
                    print(f"\n💾 Checkpoint compresso: {zip_path}.zip")
            except Exception as e:
                print(f"\n⚠️ Backup fallito: {e}")



# =============================================================================
# DATA COLLATOR
# =============================================================================

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data Collator per CTC training con padding dinamico.
    
    Identico a quello usato per WavLM per garantire consistenza.
    Gestisce il padding di input audio e labels.
    
    Note:
        A differenza di WavLM, XLS-R potrebbe avere lunghezze
        di output leggermente diverse. Il padding garantisce
        che i batch siano allineati.
    
    Args:
        processor: Wav2Vec2Processor per processing
        padding: Tipo di padding ("longest" o "max_length")
    """
    processor: Wav2Vec2Processor
    padding: str = "longest"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch con padding.
        
        Args:
            features: Lista di dict con "input_values" e "labels"
            
        Returns:
            Batch dict con tensori paddati
        """
        # Separa input e labels
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        # Pad input audio
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Maschera padding con -100 per CTC loss
        # -100 è il valore speciale che PyTorch ignora nel calcolo della loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100
        )
        
        batch["labels"] = labels
        return batch


# =============================================================================
# XLS-R TRAINER
# =============================================================================

class XLSRTrainer:
    """
    Trainer per XLS-R (wav2vec2-xls-r-300m).
    
    Differenze chiave rispetto a WavLM:
    1. Modello multilingua (128 lingue vs inglese-focused)
    2. Architettura wav2vec2 (simile ma diversa da WavLM)
    3. Più grande (300M parametri vs 94M per WavLM base)
    
    CRITICO: Usa lo stesso vocab.json di WavLM per garantire
    che gli output siano allineati per il Late Fusion.
    
    Attributes:
        config: Configurazione training da YAML
        model: Wav2Vec2ForCTC con XLS-R backbone
        processor: Wav2Vec2Processor
        trainer: HuggingFace Trainer
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza trainer XLS-R.
        
        Args:
            config: Dict con configurazione training
        """
        self.config = config
        self.model = None
        self.processor = None
        self.trainer = None
        
        # XLS-R richiede più VRAM - stampa warning
        print("⚠️  XLS-R (300M) richiede ~12GB VRAM")
        print("    Considera batch_size=4 su GPU con <16GB")
        
        self._set_seed(config.get("seed", 42))
    
    def _set_seed(self, seed: int) -> None:
        """Imposta seed per riproducibilità."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def setup_processor(self) -> None:
        """
        Crea processor da vocab.json.
        
        IMPORTANTE: Usa LO STESSO vocab.json di WavLM.
        Questo è CRUCIALE per l'ensemble - entrambi i modelli
        devono predire sullo stesso spazio di token.
        
        Differenza con WavLM Processor:
        - Feature extractor è simile (stessa normalizzazione audio)
        - Tokenizer DEVE essere identico (stesso vocab)
        """
        vocab_path = self.config["data"]["vocab_path"]
        
        print(f"[XLS-R] Caricamento vocab: {vocab_path}")
        print(f"[XLS-R] NOTA: Stesso vocab di WavLM per allineamento output")
        
        # ---------------------------------------------------------------------
        # TOKENIZER: Identico a WavLM
        # Usa lo stesso vocab.json per garantire mapping fonema→ID identico
        # ---------------------------------------------------------------------
        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
        
        # ---------------------------------------------------------------------
        # FEATURE EXTRACTOR: Configurazione wav2vec2
        # Stessa configurazione audio: 16kHz, normalizzazione
        # ---------------------------------------------------------------------
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        
        self.processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )
        
        print(f"[XLS-R] Vocab size: {len(tokenizer)}")
    
    def setup_model(self) -> None:
        """
        Inizializza XLS-R per CTC.
        
        Usa facebook/wav2vec2-xls-r-300m come backbone.
        La CTC head viene inizializzata per il nostro vocabolario IPA.
        
        Differenze architetturali con WavLM:
        - XLS-R: 24 layer transformer (vs 12 per WavLM base)
        - XLS-R: 1024 hidden size (vs 768 per WavLM base)
        - XLS-R: Pre-training multilingua (vs inglese per WavLM)
        """
        # Modello XLS-R 300M
        model_name = "facebook/wav2vec2-xls-r-300m"
        vocab_size = len(self.processor.tokenizer)
        
        print(f"[XLS-R] Inizializzazione modello: {model_name}")
        print(f"[XLS-R] Vocab size: {vocab_size}")
        
        # ---------------------------------------------------------------------
        # CARICA MODELLO
        # Wav2Vec2ForCTC aggiunge automaticamente la CTC head
        # Configuriamo per il nostro vocabolario
        # ---------------------------------------------------------------------
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            vocab_size=vocab_size,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            # Ignora mismatch di dimensioni per la lm_head
            # (verrà re-inizializzata per il nostro vocab)
            ignore_mismatched_sizes=True,
        )
        
        # ---------------------------------------------------------------------
        # CONGELA FEATURE ENCODER
        # Come per WavLM, congeliamo l'encoder CNN per stabilità
        # ---------------------------------------------------------------------
        if self.config["model"].get("freeze_feature_encoder", True):
            self.model.freeze_feature_encoder()
            print("[XLS-R] Feature encoder congelato")
        
        # Info modello
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[XLS-R] Parametri totali: {total_params/1e6:.1f}M")
        print(f"[XLS-R] Parametri trainabili: {trainable_params/1e6:.1f}M")
    
    def load_and_prepare_dataset(self, csv_path: str):
        """
        Carica e preprocessa dataset.
        
        IMPORTANTE: Usa lo stesso dataset di WavLM per coerenza.
        Il preprocessing è identico a quello di WavLM.
        
        Args:
            csv_path: Path al CSV (stesso di WavLM)
            
        Returns:
            DatasetDict con split train/validation/test
        """
        from datasets import DatasetDict
        
        print(f"[XLS-R] Caricamento dataset: {csv_path}")
        
        # Carica CSV
        ds = load_dataset("csv", data_files=csv_path)["train"]
        
        # ---------------------------------------------------------------------
        # FIX PATHS: Converti Windows paths a Unix e filtra file mancanti
        # Questo è necessario quando il dataset è stato creato su Windows
        # ma il training avviene su Linux/Colab
        # ---------------------------------------------------------------------
        def fix_audio_path(example):
            """Converte path Windows in path Unix."""
            path = example["audio_path"]
            # Converti backslash in forward slash
            path = path.replace("\\", "/")
            example["audio_path"] = path
            return example
        
        print("[XLS-R] Fixing audio paths (Windows -> Unix)...")
        ds = ds.map(fix_audio_path, num_proc=1)
        
        # Filtra file che non esistono
        def file_exists(example):
            path = example["audio_path"]
            return Path(path).exists()
        
        initial_count = len(ds)
        print("[XLS-R] Filtering missing audio files...")
        ds = ds.filter(file_exists, num_proc=1)
        final_count = len(ds)
        
        if final_count < initial_count:
            print(f"[XLS-R] ⚠️ Rimossi {initial_count - final_count} file mancanti")
            print(f"[XLS-R] Dataset finale: {final_count} samples")
        
        # Carica audio a 16kHz
        ds = ds.cast_column(
            "audio_path",
            Audio(sampling_rate=16000)
        )
        ds = ds.rename_column("audio_path", "audio")
        
        # Split dataset
        if "split" in ds.column_names:
            train_ds = ds.filter(lambda x: x["split"] == "train")
            val_ds = ds.filter(lambda x: x["split"] == "validation")
            test_ds = ds.filter(lambda x: x["split"] == "test")
        else:
            val_size = self.config["data"].get("val_size", 0.05)
            test_size = self.config["data"].get("test_size", 0.05)
            total_eval = val_size + test_size
            
            split1 = ds.train_test_split(test_size=total_eval, seed=42)
            val_ratio = val_size / total_eval
            split2 = split1["test"].train_test_split(test_size=(1-val_ratio), seed=42)
            
            train_ds = split1["train"]
            val_ds = split2["train"]
            test_ds = split2["test"]
        
        dataset = DatasetDict({
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        })
        
        print(f"[XLS-R] Split: train={len(dataset['train'])}, "
              f"val={len(dataset['validation'])}, test={len(dataset['test'])}")
        
        # ---------------------------------------------------------------------
        # PREPROCESSING
        # Identico a WavLM: estrae features audio e converte IPA in token IDs
        # ---------------------------------------------------------------------
        def preprocess(batch):
            """Preprocessa batch: audio -> features, IPA -> token IDs."""
            audio = batch["audio"]["array"]
            
            # Feature extractor: normalizza audio
            batch["input_values"] = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors=None,
            ).input_values[0]
            
            # Tokenizer: converte IPA in token IDs
            batch["labels"] = self.processor.tokenizer(
                batch["ipa_clean"]
            ).input_ids
            
            return batch
        
        print("[XLS-R] Preprocessing dataset...")
        dataset = dataset.map(
            preprocess,
            remove_columns=dataset["train"].column_names,
            num_proc=1,
        )
        
        return dataset
    
    def train(self, dataset, resume: bool = False) -> None:
        """
        Avvia training XLS-R.
        
        NOTA: XLS-R è più grande di WavLM, quindi:
        - Usa batch size più piccolo se necessario
        - Training più lento (~2x rispetto a WavLM base)
        
        Args:
            dataset: DatasetDict preprocessato
            resume: Se True, riprende dall'ultimo checkpoint
        """
        training_config = self.config["training"]
        
        # Override output dir per XLS-R (solo se necessario)
        output_dir = training_config.get("output_dir", "outputs")
        # Non aggiungere /xlsr se il path già contiene 'xlsr'
        if "xlsr" not in output_dir.lower():
            if "weighted" in output_dir or "wavlm" in output_dir.lower():
                output_dir = output_dir.replace("wavlm", "xlsr").replace("weighted", "xlsr")
            else:
                output_dir = str(Path(output_dir) / "xlsr")
        
        # Metriche
        cer_metric = evaluate.load("cer")
        
        def compute_metrics(pred):
            """Calcola PER (Phoneme Error Rate)."""
            pred_ids = np.argmax(pred.predictions, axis=-1)
            pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
            
            pred_str = self.processor.batch_decode(pred_ids)
            label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)
            
            valid = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
            if not valid:
                return {"per": 1.0}
            
            preds, labels = zip(*valid)
            per = cer_metric.compute(predictions=preds, references=labels)
            return {"per": per}
        
        # ---------------------------------------------------------------------
        # TRAINING ARGUMENTS
        # Adattati per XLS-R (più grande, richiede più VRAM)
        # ---------------------------------------------------------------------
        
        # Riduci batch size per XLS-R (300M vs 94M parametri)
        batch_size = training_config.get("per_device_train_batch_size", 8)
        if batch_size > 4:
            print(f"[XLS-R] Warning: batch_size={batch_size} potrebbe causare OOM")
            print(f"[XLS-R] Consigliato: batch_size=4 per XLS-R 300M")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get("num_train_epochs", 10),
            per_device_train_batch_size=min(batch_size, 4),  # Max 4 per XLS-R
            per_device_eval_batch_size=min(batch_size, 4),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=training_config.get("learning_rate", 3e-5),
            warmup_steps=training_config.get("warmup_steps", 500),
            weight_decay=training_config.get("weight_decay", 0.01),
            fp16=training_config.get("fp16", True),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,  # Meno checkpoint (modello grande)
            load_best_model_at_end=True,
            metric_for_best_model="per",
            greater_is_better=False,
            logging_steps=100,
            dataloader_num_workers=0,
            # group_by_length non funziona con input_values (audio)
            group_by_length=False,
            # XLS-R specifico: gradient checkpointing per risparmiare VRAM
            gradient_checkpointing=True,
            # Disabilita wandb/tensorboard
            report_to="none",
        )
        
        # Data collator
        data_collator = DataCollatorCTCWithPadding(processor=self.processor)
        
        # Trainer con callback per backup su Drive
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=[DriveBackupCallback()],
        )
        
        print(f"[XLS-R] Output dir: {output_dir}")
        
        # Trova checkpoint per resume
        resume_from_checkpoint = None
        if resume:
            output_path = Path(output_dir)
            if output_path.exists():
                checkpoints = sorted([
                    d for d in output_path.iterdir() 
                    if d.is_dir() and d.name.startswith("checkpoint-")
                ])
                if checkpoints:
                    resume_from_checkpoint = str(checkpoints[-1])
                    print(f"[XLS-R] Ripresa da checkpoint: {resume_from_checkpoint}")
                else:
                    print("[XLS-R] Nessun checkpoint trovato, training da zero")
        
        print("[XLS-R] Avvio training...")
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Salva modello finale
        final_path = Path(output_dir) / "final_model_xlsr"
        self.trainer.save_model(str(final_path))
        self.processor.save_pretrained(str(final_path))
        print(f"[XLS-R] Modello salvato: {final_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train XLS-R (wav2vec2-xls-r-300m) per ensemble"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path al file config YAML"
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=None,
        help="Override path dataset CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    args = parser.parse_args()
    
    # Carica config
    print("=" * 60)
    print("TRAINING XLS-R (wav2vec2-xls-r-300m)")
    print("=" * 60)
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override
    if args.data_csv:
        config["data"]["csv_path"] = args.data_csv
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    
    print(f"Config: {args.config}")
    print(f"Dataset: {config['data']['csv_path']}")
    print(f"Resume: {args.resume}")
    
    # Training
    trainer = XLSRTrainer(config)
    trainer.setup_processor()
    trainer.setup_model()
    dataset = trainer.load_and_prepare_dataset(config["data"]["csv_path"])
    trainer.train(dataset, resume=args.resume)
    
    print("\n" + "=" * 60)
    print("✓ Training XLS-R completato!")
    print("=" * 60)
    print("\nProssimo passo: Late Fusion")
    print("  python scripts/evaluate_fusion.py \\")
    print("      --model-a outputs/final_model_weighted \\")
    print("      --model-b outputs/xlsr/final_model_xlsr")


if __name__ == "__main__":
    main()

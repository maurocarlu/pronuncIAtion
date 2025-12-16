"""
Modulo principale per training del modello.
"""

# Silenzia log prima degli import
import os
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DATASETS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import logging
for logger_name in ["transformers", "datasets", "tensorflow", "absl", "urllib3", "filelock", "huggingface_hub"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Utility per sopprimere stderr durante import pesanti
class SuppressOutput:
    """Sopprimi stdout/stderr temporaneamente."""
    def __init__(self, suppress_stdout=False, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self.null = None
        self.old_stdout = None
        self.old_stderr = None
        
    def __enter__(self):
        self.null = open(os.devnull, 'w')
        if self.suppress_stdout:
            self.old_stdout = sys.stdout
            sys.stdout = self.null
        if self.suppress_stderr:
            self.old_stderr = sys.stderr
            sys.stderr = self.null
        return self
        
    def __exit__(self, *args):
        if self.old_stdout:
            sys.stdout = self.old_stdout
        if self.old_stderr:
            sys.stderr = self.old_stderr
        self.null.close()

import torch
import numpy as np
import random
import evaluate
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    WavLMForCTC,
    TrainingArguments,
    Trainer,
)

from .dataset import DataCollatorCTCWithPadding, prepare_dataset_function


@dataclass
class TrainingConfig:
    """Configurazione training."""
    seed: int = 42
    model_name: str = "microsoft/wavlm-large"
    freeze_feature_encoder: bool = True
    
    csv_path: str = "data/processed/phonemeref_processed.csv"
    vocab_path: str = "data/processed/vocab.json"
    audio_base_path: str = "data/raw/phonemeref_data"
    val_size: float = 0.05   # Validation split
    test_size: float = 0.05  # Test split (90/5/5 default)
    sampling_rate: int = 16000
    
    output_dir: str = "outputs/wavlm-phoneme-recognizer"
    num_train_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.01
    fp16: bool = True
    bf16: bool = False
    
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 2
    
    # Strategia eval/save: "steps" o "epoch"
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    
    # Ottimizzazioni DataLoader
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_prefetch_factor: Optional[int] = None
    
    # Ottimizzazioni Training
    group_by_length: bool = False
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = False
    disable_tqdm: bool = False
    max_grad_norm: float = 1.0  # Gradient clipping per evitare esplosione
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Carica configurazione da YAML."""
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Flatten nested config
        flat_config = {}
        flat_config['seed'] = config.get('seed', 42)
        
        if 'model' in config:
            flat_config['model_name'] = config['model'].get('name', cls.model_name)
            flat_config['freeze_feature_encoder'] = config['model'].get('freeze_feature_encoder', True)
        
        if 'data' in config:
            flat_config['csv_path'] = config['data'].get('csv_path', cls.csv_path)
            flat_config['vocab_path'] = config['data'].get('vocab_path', cls.vocab_path)
            flat_config['audio_base_path'] = config['data'].get('audio_base_path', cls.audio_base_path)
            flat_config['val_size'] = config['data'].get('val_size', cls.val_size)
            flat_config['test_size'] = config['data'].get('test_size', cls.test_size)
            flat_config['sampling_rate'] = config['data'].get('sampling_rate', cls.sampling_rate)
        
        if 'training' in config:
            t = config['training']
            flat_config['output_dir'] = t.get('output_dir', cls.output_dir)
            flat_config['num_train_epochs'] = t.get('num_train_epochs', cls.num_train_epochs)
            flat_config['batch_size'] = t.get('per_device_train_batch_size', cls.batch_size)
            flat_config['gradient_accumulation_steps'] = t.get('gradient_accumulation_steps', cls.gradient_accumulation_steps)
            flat_config['learning_rate'] = t.get('learning_rate', cls.learning_rate)
            flat_config['warmup_steps'] = t.get('warmup_steps', cls.warmup_steps)
            flat_config['weight_decay'] = t.get('weight_decay', cls.weight_decay)
            flat_config['fp16'] = t.get('fp16', cls.fp16)
            flat_config['bf16'] = t.get('bf16', cls.bf16)
            flat_config['eval_steps'] = t.get('eval_steps', cls.eval_steps)
            flat_config['save_steps'] = t.get('save_steps', cls.save_steps)
            flat_config['logging_steps'] = t.get('logging_steps', cls.logging_steps)
            flat_config['save_total_limit'] = t.get('save_total_limit', cls.save_total_limit)
            flat_config['eval_strategy'] = t.get('eval_strategy', cls.eval_strategy)
            flat_config['save_strategy'] = t.get('save_strategy', cls.save_strategy)
            
            # Ottimizzazioni DataLoader
            flat_config['dataloader_num_workers'] = t.get('dataloader_num_workers', cls.dataloader_num_workers)
            flat_config['dataloader_pin_memory'] = t.get('dataloader_pin_memory', cls.dataloader_pin_memory)
            flat_config['dataloader_prefetch_factor'] = t.get('dataloader_prefetch_factor', cls.dataloader_prefetch_factor)
            
            # Ottimizzazioni Training
            flat_config['group_by_length'] = t.get('group_by_length', cls.group_by_length)
            flat_config['optim'] = t.get('optim', cls.optim)
            flat_config['gradient_checkpointing'] = t.get('gradient_checkpointing', cls.gradient_checkpointing)
            flat_config['disable_tqdm'] = t.get('disable_tqdm', cls.disable_tqdm)
            flat_config['max_grad_norm'] = t.get('max_grad_norm', cls.max_grad_norm)
        
        return cls(**flat_config)


class PhonemeTrainer:
    """Trainer per modello di riconoscimento fonemico."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.processor: Optional[Wav2Vec2Processor] = None
        self.model: Optional[WavLMForCTC] = None
        self.trainer: Optional[Trainer] = None
        
        self._set_seed()
    
    def _set_seed(self) -> None:
        """Imposta seed per riproducibilità."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def setup_processor(self) -> Wav2Vec2Processor:
        """Configura tokenizer e feature extractor."""
        print(">>> Setup Processor...")
        
        tokenizer = Wav2Vec2CTCTokenizer(
            self.config.vocab_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|"
        )
        
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.config.sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False
        )
        
        self.processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
        
        return self.processor
    
    def load_dataset(self):
        """Carica e prepara dataset."""
        print(">>> Caricamento dataset...")
        
        dataset = load_dataset("csv", data_files=self.config.csv_path)
        dataset = dataset["train"]
        
        # Rimuovi colonne non necessarie
        columns_to_remove = ["id", "word", "split", "ipa", "source", "is_correct", "error_type"]
        existing_columns = [c for c in columns_to_remove if c in dataset.column_names]
        if existing_columns:
            dataset = dataset.remove_columns(existing_columns)
        
        # Correggi percorsi audio
        audio_base = Path(self.config.audio_base_path).resolve()
        
        def fix_audio_path(example):
            audio_path = example["audio_path"]
            path_obj = Path(audio_path)
            
            # Se e' gia un path assoluto e il file esiste, usalo direttamente
            if path_obj.is_absolute():
                if path_obj.exists():
                    example["audio_path"] = str(path_obj)
                    return example
            
            # Prova il path cosi com'e
            if path_obj.exists():
                example["audio_path"] = str(path_obj.resolve())
                return example
            
            # Prova aggiungendo il base path
            full_path = audio_base / audio_path
            if full_path.exists():
                example["audio_path"] = str(full_path)
                return example
            
            # Prova rimuovendo eventuali prefissi duplicati
            # es. "data/raw/phonemeref_data/data/raw/..." -> correggi
            audio_path_str = str(audio_path)
            if "phonemeref_data" in audio_path_str:
                # Estrai solo la parte dopo phonemeref_data
                parts = audio_path_str.split("phonemeref_data")
                if len(parts) > 1:
                    relative_part = parts[-1].lstrip("/\\")
                    check_path = audio_base / relative_part
                    if check_path.exists():
                        example["audio_path"] = str(check_path)
                        return example
            
            # Fallback: usa il path originale con base
            example["audio_path"] = str(audio_base / audio_path)
            return example
        
        dataset = dataset.map(fix_audio_path)
        
        # Split train/val/test (90/5/5 or custom)
        # First split: separate test set
        total_eval_size = self.config.val_size + self.config.test_size
        split1 = dataset.train_test_split(test_size=total_eval_size, seed=self.config.seed)
        
        # Second split: separate val from test
        val_ratio = self.config.val_size / total_eval_size
        split2 = split1["test"].train_test_split(test_size=(1 - val_ratio), seed=self.config.seed)
        
        # Combine into final dataset dict
        from datasets import DatasetDict
        dataset = DatasetDict({
            "train": split1["train"],
            "validation": split2["train"],
            "test": split2["test"]
        })
        
        print(f">>> Split: train={len(dataset['train'])}, val={len(dataset['validation'])}, test={len(dataset['test'])}")
        
        # Cast audio column
        dataset = dataset.cast_column(
            "audio_path", 
            Audio(sampling_rate=self.config.sampling_rate)
        )
        dataset = dataset.rename_column("audio_path", "audio")
        
        # Preprocessing
        print(">>> Preprocessing dataset...")
        prepare_fn = prepare_dataset_function(self.processor)
        
        # Usa più processi per preprocessing (accelera su Colab)
        num_proc = min(4, self.config.dataloader_num_workers) if self.config.dataloader_num_workers > 0 else 1
        
        encoded_dataset = dataset.map(
            prepare_fn,
            remove_columns=dataset["train"].column_names,
            num_proc=num_proc
        )
        
        return encoded_dataset
    
    def setup_model(self) -> WavLMForCTC:
        """Carica e configura modello."""
        print(f">>> Caricamento modello {self.config.model_name}...")
        
        # Sopprimi stderr durante il download/caricamento del modello
        with SuppressOutput(suppress_stderr=True):
            self.model = WavLMForCTC.from_pretrained(
                self.config.model_name,
                ctc_loss_reduction="mean",
                pad_token_id=self.processor.tokenizer.pad_token_id,
                vocab_size=len(self.processor.tokenizer),
            )
        
        if self.config.freeze_feature_encoder:
            self.model.freeze_feature_encoder()
            print(">>> Feature encoder congelato")
        
        return self.model
    
    def get_compute_metrics(self):
        """Restituisce funzione per calcolo metriche."""
        cer_metric = evaluate.load("cer")
        processor = self.processor
        
        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)
            
            # Sostituisci -100 con pad_token_id
            pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
            
            # Decodifica
            pred_str = processor.batch_decode(pred_ids)
            label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
            
            # Calcola PER (usiamo CER come proxy)
            per = cer_metric.compute(predictions=pred_str, references=label_str)
            
            return {"per": per}
        
        return compute_metrics
    
    def train(self, resume_from_checkpoint: bool = False) -> None:
        """Esegue training completo."""
        # Setup
        self.setup_processor()
        encoded_dataset = self.load_dataset()
        self.setup_model()
        
        # Data collator
        data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            
            # Batch & Accumulation
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Ottimizzazioni DataLoader (per T4/GPU)
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_prefetch_factor=self.config.dataloader_prefetch_factor if self.config.dataloader_num_workers > 0 else None,
            
            # Training efficiency
            group_by_length=self.config.group_by_length,
            optim=self.config.optim,
            
            # Schedule
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            
            # Mixed precision
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            
            # Gradient clipping
            max_grad_norm=self.config.max_grad_norm,
            
            # Evaluation & Saving Strategy
            # "epoch" = valuta/salva alla fine di ogni epoca
            # "steps" = valuta/salva ogni N steps
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            eval_steps=self.config.eval_steps if self.config.eval_strategy == "steps" else None,
            save_steps=self.config.save_steps if self.config.save_strategy == "steps" else None,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="per",
            greater_is_better=False,
            
            # Output control
            disable_tqdm=self.config.disable_tqdm,
            log_level="error",  # Solo errori, TQDM mostra il progresso
            logging_first_step=False,
            
            # Misc
            push_to_hub=False,
            report_to="none",
            seed=self.config.seed,
        )
        
        # Gradient checkpointing (se richiesto)
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=self.get_compute_metrics(),
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],  # Use validation set for eval during training
            processing_class=self.processor,
        )
        
        # Train
        print(">>> Avvio Training...")
        
        # Gestione resume da checkpoint
        checkpoint_path = None
        should_train = True
        
        if resume_from_checkpoint:
            # Cerca l'ultimo checkpoint nella output_dir
            output_dir = Path(self.config.output_dir)
            if output_dir.exists():
                checkpoints = sorted([
                    d for d in output_dir.iterdir() 
                    if d.is_dir() and d.name.startswith("checkpoint-")
                ], key=lambda x: int(x.name.split("-")[1]))
                
                if checkpoints:
                    checkpoint_path = str(checkpoints[-1])
                    print(f">>> Ripresa da checkpoint: {checkpoint_path}")
                    
                    # Verifica se training già completato
                    import json
                    state_file = Path(checkpoint_path) / "trainer_state.json"
                    if state_file.exists():
                        with open(state_file) as f:
                            state = json.load(f)
                        current_epoch = state.get('epoch', 0)
                        target_epochs = self.config.num_train_epochs
                        
                        if current_epoch >= target_epochs:
                            print(f"⚠️ Training già completato (epoch {current_epoch}/{target_epochs})")
                            print(f"   Per continuare, aumenta 'num_train_epochs' nella config")
                            should_train = False
                else:
                    print(">>> Nessun checkpoint trovato, inizio da zero")
            else:
                print(">>> Output dir non esiste, inizio da zero")
        
        # Esegui training solo se necessario
        train_result = None
        if should_train:
            train_result = self.trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            print(">>> Skip training (già completato)")
        
        # Save finale solo se training è avvenuto O se non esiste già
        final_path = f"{self.config.output_dir}/final_model"
        final_model_exists = Path(final_path).exists()
        
        if should_train and train_result is not None:
            # Training appena completato, salva
            print(f">>> Salvataggio modello: {final_path}")
            self.trainer.save_model(final_path)
            self.processor.save_pretrained(final_path)
            print("✅ Modello salvato con successo!")
        elif not final_model_exists and checkpoint_path:
            # Training skippato ma final_model non esiste, copia dall'ultimo checkpoint
            print(f">>> Creazione final_model da checkpoint...")
            import shutil
            shutil.copytree(checkpoint_path, final_path, dirs_exist_ok=True)
            print(f"✅ Final model creato da {Path(checkpoint_path).name}")
        elif final_model_exists:
            print(f"ℹ️ Final model già esistente: {final_path}")
        
        print("Training completato!")

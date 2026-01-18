#!/usr/bin/env python3
"""
==============================================================================
MULTI-MODEL FUSION - Supporta Tutte le Combinazioni di Modelli
==============================================================================

Questo script implementa fusione flessibile per qualsiasi combinazione di modelli:
- 2 modelli (coppie): HuBERT+WavLM, HuBERT+Base, WavLM+Base
- 3 modelli (tripla): HuBERT + WavLM Weighted + WavLM Base

TECNICHE SUPPORTATE:
-------------------
1. EARLY FUSION: Concatenazione feature → CTC head
2. LATE FUSION: Combinazione logits con pesi α
3. GATED FUSION: Gate apprendibile per fusione dinamica

COMBINAZIONI MODELLI:
--------------------
- 2 modelli: 3 coppie possibili
- 3 modelli: 1 tripla

ARCHITETTURA TRIPLA FUSIONE:
---------------------------
    Audio → Model_A → h_a ─┐
          → Model_B → h_b ─┼→ concat (3072D) → CTC Head    [Early]
          → Model_C → h_c ─┘
                           
          → softmax([g_a, g_b, g_c]) → weighted sum         [Gated]

USO:
----
    # Early Fusion con 2 modelli (HuBERT + WavLM)
    python scripts/training/train_multi_fusion.py \
        --model-a outputs/backup/hubert/final_model \
        --model-b outputs/backup/wavlm_weighted/final_model \
        --fusion-type early \
        --epochs 5

    # Gated Fusion con 3 modelli
    python scripts/training/train_multi_fusion.py \
        --model-a outputs/backup/hubert/final_model \
        --model-b outputs/backup/wavlm_weighted/final_model \
        --model-c outputs/backup/wavlm_base/final_model \
        --fusion-type gated \
        --epochs 5

    # Late Fusion Grid Search con 2 modelli
    python scripts/training/train_multi_fusion.py \
        --model-a outputs/backup/hubert/final_model \
        --model-b outputs/backup/wavlm_weighted/final_model \
        --fusion-type late \
        --weight-grid

Autore: DeepLearning-Phoneme Project
Riferimenti: docs/FUSION_TECHNIQUES.md
"""

import argparse
import sys
import os
import warnings
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    HubertModel,
    HubertForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.models.wavlm import WavLMModel, WavLMForCTC
import shutil
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# MODEL LOADER - Carica qualsiasi modello audio
# =============================================================================

def load_audio_encoder(model_path: str, model_type: str = "auto") -> Tuple[nn.Module, int]:
    """
    Carica un encoder audio da checkpoint.
    
    Supporta:
    - HuBERT (base, large, ForCTC)
    - WavLM (base, large, ForCTC)
    - Rilevamento automatico tipo modello
    
    Args:
        model_path: Path al modello (HuggingFace o locale)
        model_type: "hubert", "wavlm", o "auto" per rilevamento automatico
        
    Returns:
        Tuple (encoder, hidden_size)
    """
    print(f"📦 Loading encoder from: {model_path}")
    
    # Rileva tipo modello se auto
    if model_type == "auto":
        model_path_lower = model_path.lower()
        if "hubert" in model_path_lower:
            model_type = "hubert"
        elif "wavlm" in model_path_lower:
            model_type = "wavlm"
        else:
            # Prova a caricare config
            try:
                config_path = Path(model_path) / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    if "hubert" in config.get("model_type", "").lower():
                        model_type = "hubert"
                    else:
                        model_type = "wavlm"
                else:
                    model_type = "wavlm"  # Default
            except:
                model_type = "wavlm"
    
    # Carica modello
    if model_type == "hubert":
        try:
            model_full = HubertForCTC.from_pretrained(model_path, torch_dtype=torch.float16)
            encoder = model_full.hubert
            print(f"   ✓ HuBERT: Loaded from ForCTC (encoder extracted)")
        except:
            encoder = HubertModel.from_pretrained(model_path, torch_dtype=torch.float16)
            print(f"   ✓ HuBERT: Loaded as base Model")
    else:  # wavlm
        try:
            model_full = WavLMForCTC.from_pretrained(model_path, torch_dtype=torch.float16)
            encoder = model_full.wavlm
            print(f"   ✓ WavLM: Loaded from ForCTC (encoder extracted)")
        except:
            encoder = WavLMModel.from_pretrained(
                model_path, 
                output_hidden_states=True,
                torch_dtype=torch.float16
            )
            print(f"   ✓ WavLM: Loaded as base Model")
    
    hidden_size = encoder.config.hidden_size
    print(f"   Hidden size: {hidden_size}")
    
    return encoder, hidden_size


# =============================================================================
# MULTI-MODEL EARLY FUSION
# =============================================================================

class MultiModelEarlyFusion(nn.Module):
    """
    Early Fusion flessibile per 2 o 3 modelli.
    
    Concatena le rappresentazioni di tutti i modelli e passa attraverso CTC head.
    
    Args:
        model_paths: Lista di path ai modelli (2 o 3)
        vocab_size: Dimensione vocabolario
        dropout_rate: Dropout prima di CTC head
    """
    
    def __init__(
        self,
        model_paths: List[str],
        vocab_size: int = 43,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        assert len(model_paths) in [2, 3], "Supportato solo 2 o 3 modelli"
        
        print(f"\n{'='*60}")
        print(f"MULTI-MODEL EARLY FUSION ({len(model_paths)} modelli)")
        print(f"{'='*60}")
        
        # Carica tutti i modelli
        self.encoders = nn.ModuleList()
        total_hidden = 0
        
        for i, path in enumerate(model_paths):
            encoder, hidden_size = load_audio_encoder(path)
            self.encoders.append(encoder)
            total_hidden += hidden_size
            
            # Freeze encoder
            for param in encoder.parameters():
                param.requires_grad = False
        
        print(f"\n📊 Total concatenated dimension: {total_hidden}D")
        
        # CTC Head
        self.dropout = nn.Dropout(dropout_rate)
        self.ctc_head = nn.Linear(total_hidden, vocab_size)
        
        # Init CTC head
        nn.init.normal_(self.ctc_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.ctc_head.bias)
        
        self.num_models = len(model_paths)
        
        # Stats
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   Trainable params: {trainable/1e3:.1f}K")
        print(f"{'='*60}\n")
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass con concatenazione di tutti gli encoder."""
        
        # Converti dtype
        target_dtype = next(self.encoders[0].parameters()).dtype
        if input_values.dtype != target_dtype:
            input_values = input_values.to(target_dtype)
        
        # Forward tutti gli encoder (frozen)
        hidden_states = []
        with torch.no_grad():
            for encoder in self.encoders:
                output = encoder(input_values, attention_mask=attention_mask)
                hidden_states.append(output.last_hidden_state.clone())
        
        # Allinea temporalmente
        min_len = min(h.size(1) for h in hidden_states)
        hidden_states = [h[:, :min_len, :] for h in hidden_states]
        
        # Concatena
        combined = torch.cat(hidden_states, dim=-1)
        
        # CTC head
        logits = self.ctc_head(self.dropout(combined))
        
        # Loss
        loss = None
        if labels is not None:
            loss = self._compute_ctc_loss(logits, labels)
        
        return {"logits": logits, "loss": loss}
    
    def _compute_ctc_loss(self, logits, labels):
        """Compute CTC loss."""
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        batch_size = logits.size(0)
        input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long, device=logits.device)
        
        labels_no_pad = labels.clone()
        labels_no_pad[labels_no_pad == -100] = 0
        target_lengths = (labels != -100).sum(dim=-1)
        
        return F.ctc_loss(log_probs, labels_no_pad, input_lengths, target_lengths, 
                          blank=0, reduction="mean", zero_infinity=True)


# =============================================================================
# MULTI-MODEL GATED FUSION
# =============================================================================

class MultiModelGatedFusion(nn.Module):
    """
    Gated Fusion flessibile per 2 o 3 modelli.
    
    Per 2 modelli:
        gate = sigmoid(W · [h_a, h_b])
        h_fused = gate * h_a + (1-gate) * h_b
    
    Per 3 modelli:
        gates = softmax(W · [h_a, h_b, h_c])  # [batch, time, 3]
        h_fused = gates[0]*h_a + gates[1]*h_b + gates[2]*h_c
    
    Args:
        model_paths: Lista di path ai modelli (2 o 3)
        vocab_size: Dimensione vocabolario
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        model_paths: List[str],
        vocab_size: int = 43,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        
        assert len(model_paths) in [2, 3], "Supportato solo 2 o 3 modelli"
        
        print(f"\n{'='*60}")
        print(f"MULTI-MODEL GATED FUSION ({len(model_paths)} modelli)")
        print(f"{'='*60}")
        
        # Carica tutti i modelli
        self.encoders = nn.ModuleList()
        hidden_sizes = []
        
        for i, path in enumerate(model_paths):
            encoder, hidden_size = load_audio_encoder(path)
            self.encoders.append(encoder)
            hidden_sizes.append(hidden_size)
            
            # Freeze
            for param in encoder.parameters():
                param.requires_grad = False
        
        self.num_models = len(model_paths)
        self.hidden_size = hidden_sizes[0]  # Assumiamo tutti stessa size (1024)
        total_input = sum(hidden_sizes)
        
        # Gate network
        if self.num_models == 2:
            # Per 2 modelli: output singolo [0,1]
            self.gate_network = nn.Linear(total_input, 1)
        else:
            # Per 3 modelli: output 3 (softmax)
            self.gate_network = nn.Linear(total_input, 3)
        
        nn.init.xavier_uniform_(self.gate_network.weight)
        nn.init.zeros_(self.gate_network.bias)
        
        print(f"\n🚪 Gate Network: {total_input}D → {self.num_models if self.num_models == 3 else 1} gates")
        
        # CTC Head
        self.dropout = nn.Dropout(dropout_rate)
        self.ctc_head = nn.Linear(self.hidden_size, vocab_size)
        
        nn.init.normal_(self.ctc_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.ctc_head.bias)
        
        # Stats
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   Trainable params: {trainable/1e3:.1f}K")
        print(f"{'='*60}\n")
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass con gating dinamico."""
        
        target_dtype = next(self.encoders[0].parameters()).dtype
        if input_values.dtype != target_dtype:
            input_values = input_values.to(target_dtype)
        
        # Forward tutti gli encoder
        hidden_states = []
        with torch.no_grad():
            for encoder in self.encoders:
                output = encoder(input_values, attention_mask=attention_mask)
                hidden_states.append(output.last_hidden_state.clone())
        
        # Allinea
        min_len = min(h.size(1) for h in hidden_states)
        hidden_states = [h[:, :min_len, :] for h in hidden_states]
        
        # Calcola gate
        gate_input = torch.cat(hidden_states, dim=-1)
        
        if self.num_models == 2:
            # Sigmoid per 2 modelli
            gate = torch.sigmoid(self.gate_network(gate_input))  # [B, T, 1]
            h_fused = gate * hidden_states[0] + (1 - gate) * hidden_states[1]
            gate_values = gate
        else:
            # Softmax per 3 modelli
            gates = F.softmax(self.gate_network(gate_input), dim=-1)  # [B, T, 3]
            h_fused = (gates[:, :, 0:1] * hidden_states[0] + 
                       gates[:, :, 1:2] * hidden_states[1] + 
                       gates[:, :, 2:3] * hidden_states[2])
            gate_values = gates
        
        # CTC head
        logits = self.ctc_head(self.dropout(h_fused))
        
        # Loss
        loss = None
        if labels is not None:
            loss = self._compute_ctc_loss(logits, labels)
        
        return {"logits": logits, "loss": loss, "gate_values": gate_values}
    
    def _compute_ctc_loss(self, logits, labels):
        """Compute CTC loss."""
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        batch_size = logits.size(0)
        input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long, device=logits.device)
        
        labels_no_pad = labels.clone()
        labels_no_pad[labels_no_pad == -100] = 0
        target_lengths = (labels != -100).sum(dim=-1)
        
        return F.ctc_loss(log_probs, labels_no_pad, input_lengths, target_lengths,
                          blank=0, reduction="mean", zero_infinity=True)
    
    def get_gate_statistics(self) -> str:
        """Restituisce descrizione interpretazione gate."""
        if self.num_models == 2:
            return "gate≈1 → Model A, gate≈0 → Model B"
        else:
            return "gates softmax tra Model A, B, C"


# =============================================================================
# MULTI-MODEL LATE FUSION (per evaluation)
# =============================================================================

class MultiModelLateFusion:
    """
    Late Fusion per valutazione (no training).
    
    Combina logits di 2 o 3 modelli con pesi alpha.
    
    Args:
        model_paths: Lista di path modelli CTC completi (con head)
        weights: Lista di pesi (deve sommare a 1)
    """
    
    def __init__(self, model_paths: List[str], weights: List[float] = None):
        assert len(model_paths) in [2, 3]
        
        self.num_models = len(model_paths)
        
        if weights is None:
            # Default: pesi uguali
            weights = [1.0 / self.num_models] * self.num_models
        
        assert len(weights) == self.num_models
        assert abs(sum(weights) - 1.0) < 0.01, "Pesi devono sommare a 1"
        
        self.weights = weights
        
        print(f"\n{'='*60}")
        print(f"MULTI-MODEL LATE FUSION ({self.num_models} modelli)")
        print(f"Weights: {weights}")
        print(f"{'='*60}")
        
        # Carica modelli CTC completi
        self.models = []
        self.processors = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for path in model_paths:
            path_lower = path.lower()
            if "hubert" in path_lower:
                model = HubertForCTC.from_pretrained(path).to(device).eval()
            else:
                model = WavLMForCTC.from_pretrained(path).to(device).eval()
            self.models.append(model)
            
            processor = Wav2Vec2Processor.from_pretrained(path)
            self.processors.append(processor)
        
        self.device = device
        print(f"{'='*60}\n")
    
    def predict(self, audio_array: np.ndarray, sampling_rate: int = 16000) -> Dict:
        """
        Predizione con Late Fusion.
        
        Args:
            audio_array: Audio numpy array
            sampling_rate: Sample rate
            
        Returns:
            Dict con logits_fused, prediction, confidences
        """
        all_logits = []
        
        for model, processor in zip(self.models, self.processors):
            inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits  # [1, T, vocab]
            all_logits.append(logits)
        
        # Allinea e fonde
        min_len = min(l.size(1) for l in all_logits)
        all_logits = [l[:, :min_len, :] for l in all_logits]
        
        # Fusione pesata
        fused = sum(w * l for w, l in zip(self.weights, all_logits))
        
        # Decode
        pred_ids = torch.argmax(fused, dim=-1)
        prediction = self.processors[0].decode(pred_ids[0])
        
        return {
            "logits_fused": fused,
            "prediction": prediction,
            "weights": self.weights,
        }


# =============================================================================
# DATA COLLATOR
# =============================================================================

class DataCollatorCTCWithPadding:
    """Data Collator per CTC training con padding dinamico."""
    
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        
        return batch


# =============================================================================
# TRAINER
# =============================================================================

class MultiModelTrainer(Trainer):
    """Custom Trainer per Multi-Model Fusion."""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(
            input_values=inputs["input_values"],
            attention_mask=inputs.get("attention_mask"),
            labels=labels,
        )
        loss = outputs["loss"]
        
        if return_outputs:
            return loss, outputs
        return loss


# =============================================================================
# TRAINING WRAPPER
# =============================================================================

class MultiModelTrainingWrapper:
    """Wrapper per training multi-model fusion."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MultiModel] Device: {self.device}")
        
        self._setup_processor()
        self._setup_model()
    
    def _setup_processor(self):
        """Setup processor."""
        vocab_path = self.config["data"]["vocab_path"]
        
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=True
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor, tokenizer=self.tokenizer
        )
        print(f"[MultiModel] Vocab size: {len(self.tokenizer)}")
    
    def _setup_model(self):
        """Setup model."""
        model_paths = self.config["model"]["model_paths"]
        fusion_type = self.config["model"]["fusion_type"]
        
        if fusion_type == "early":
            self.model = MultiModelEarlyFusion(
                model_paths=model_paths,
                vocab_size=len(self.tokenizer),
                dropout_rate=self.config["model"].get("dropout", 0.1),
            )
        elif fusion_type == "gated":
            self.model = MultiModelGatedFusion(
                model_paths=model_paths,
                vocab_size=len(self.tokenizer),
                dropout_rate=self.config["model"].get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Fusion type {fusion_type} non supportato per training")
        
        # Gradient checkpointing
        if self.config["training"].get("gradient_checkpointing", True):
            for encoder in self.model.encoders:
                if hasattr(encoder, 'gradient_checkpointing_enable'):
                    encoder.gradient_checkpointing_enable()
    
    def load_and_prepare_dataset(self):
        """Carica e preprocessa dataset."""
        csv_path = self.config["data"]["csv_path"]
        audio_base = self.config["data"]["audio_base_path"]
        
        print(f"[MultiModel] Caricamento dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        
        val_size = self.config["data"].get("val_size", 0.1)
        val_count = int(len(df) * val_size)
        
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = df.iloc[val_count:]
        val_df = df.iloc[:val_count]
        
        print(f"[MultiModel] Train: {len(train_df)}, Val: {len(val_df)}")
        
        base_path = Path(audio_base)
        
        def preprocess_dataframe(dataframe, desc):
            import librosa
            processed = []
            for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=desc):
                audio_path = str(row.get("audio_path", "")).replace("\\", "/")
                full_path = base_path / audio_path
                try:
                    audio, sr = librosa.load(full_path, sr=16000)
                except:
                    audio = np.zeros(16000, dtype=np.float32)
                
                inputs = self.processor(audio, sampling_rate=16000, return_tensors=None)
                ipa = row.get("ipa_clean", "")
                if pd.isna(ipa):
                    ipa = ""
                labels = self.tokenizer(str(ipa)).input_ids
                
                processed.append({"input_values": inputs.input_values[0], "labels": labels})
            return processed
        
        print("\n🔄 Preprocessing...")
        train_data = preprocess_dataframe(train_df, "Train")
        val_data = preprocess_dataframe(val_df, "Val")
        
        self.train_dataset = Dataset.from_list(train_data)
        self.val_dataset = Dataset.from_list(val_data)
    
    def train(self):
        """Esegue training."""
        output_dir = self.config["training"]["output_dir"]
        
        data_collator = DataCollatorCTCWithPadding(self.processor)
        cer_metric = evaluate.load("cer")
        
        def compute_metrics(pred):
            pred_logits = pred.predictions
            if isinstance(pred_logits, dict):
                pred_logits = pred_logits["logits"]
            pred_ids = np.argmax(pred_logits, axis=-1)
            pred_str = self.processor.batch_decode(pred_ids)
            
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
            label_str = self.processor.batch_decode(label_ids, group_tokens=False)
            
            cer = cer_metric.compute(predictions=pred_str, references=label_str)
            return {"cer": cer, "per": cer}
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.config["training"].get("batch_size", 2),
            per_device_eval_batch_size=self.config["training"].get("batch_size", 2),
            gradient_accumulation_steps=self.config["training"].get("gradient_accumulation", 8),
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.config["training"].get("epochs", 5),
            fp16=True,
            logging_steps=25,
            learning_rate=self.config["training"].get("learning_rate", 1e-4),
            warmup_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            report_to="none",
        )
        
        trainer = MultiModelTrainer(
            model=self.model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            processing_class=self.processor,
        )
        
        fusion_type = self.config["model"]["fusion_type"]
        num_models = len(self.config["model"]["model_paths"])
        
        print(f"\n{'='*60}")
        print(f"TRAINING MULTI-MODEL {fusion_type.upper()} FUSION ({num_models} models)")
        print(f"{'='*60}")
        
        # Resume
        checkpoint = None
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
            checkpoint = str(checkpoints[-1])
            print(f"🔄 Resume da: {checkpoint}")
        
        trainer.train(resume_from_checkpoint=checkpoint)
        
        # Salva
        final_path = Path(output_dir) / f"final_model_{fusion_type}_{num_models}way"
        trainer.save_model(str(final_path))
        self.processor.save_pretrained(str(final_path))
        
        # Config
        config_save = {
            "model_type": f"multi_{fusion_type}_fusion",
            "num_models": num_models,
            "model_paths": self.config["model"]["model_paths"],
        }
        with open(final_path / "config.json", "w") as f:
            json.dump(config_save, f, indent=2)
        
        print(f"\n✓ Modello salvato: {final_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Model Fusion Training")
    
    # Modelli
    parser.add_argument("--model-a", type=str, required=True, help="Path primo modello")
    parser.add_argument("--model-b", type=str, required=True, help="Path secondo modello")
    parser.add_argument("--model-c", type=str, default=None, help="Path terzo modello (opzionale)")
    
    # Fusion type
    parser.add_argument("--fusion-type", type=str, choices=["early", "gated", "late"], 
                        default="early", help="Tipo fusione")
    
    # Late fusion weights (per late fusion)
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Pesi per Late Fusion (devono sommare a 1)")
    parser.add_argument("--weight-grid", action="store_true",
                        help="Grid search pesi per Late Fusion")
    
    # Data
    parser.add_argument("--data-csv", type=str, default="data/processed/combined_augmented.csv")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--audio-base", type=str, default="")
    
    # Training
    parser.add_argument("--output-dir", type=str, default="outputs/multi_fusion")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Costruisci lista modelli
    model_paths = [args.model_a, args.model_b]
    if args.model_c:
        model_paths.append(args.model_c)
    
    print(f"\n📦 Modelli: {len(model_paths)}")
    for i, p in enumerate(model_paths):
        print(f"   {chr(65+i)}: {p}")
    
    # Late fusion (evaluation only)
    if args.fusion_type == "late":
        if args.weight_grid:
            # Grid search
            if len(model_paths) == 2:
                weight_sets = [[0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]
            else:
                weight_sets = [
                    [0.33, 0.33, 0.34],
                    [0.5, 0.25, 0.25],
                    [0.25, 0.5, 0.25],
                    [0.25, 0.25, 0.5],
                ]
            
            print(f"\n🔍 Weight Grid Search:")
            for weights in weight_sets:
                print(f"   Testing weights: {weights}")
                fusion = MultiModelLateFusion(model_paths, weights)
                # TODO: run evaluation
        else:
            weights = args.weights
            fusion = MultiModelLateFusion(model_paths, weights)
            print("Late Fusion pronto per evaluation")
        return
    
    # Training (early o gated)
    config = {
        "data": {
            "csv_path": args.data_csv,
            "vocab_path": args.vocab_path,
            "audio_base_path": args.audio_base,
            "val_size": 0.1,
        },
        "model": {
            "model_paths": model_paths,
            "fusion_type": args.fusion_type,
            "dropout": 0.1,
        },
        "training": {
            "output_dir": args.output_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gradient_accumulation": 8,
            "gradient_checkpointing": True,
        },
    }
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    trainer = MultiModelTrainingWrapper(config)
    trainer.load_and_prepare_dataset()
    trainer.train()


if __name__ == "__main__":
    main()

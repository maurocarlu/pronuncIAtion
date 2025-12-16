#!/usr/bin/env python3
"""
Training script per WavLM con Weighted Layer Sum.

Questa architettura SOTA estrae feature acustiche più ricche combinando
tutti i 12 hidden states del Transformer con pesi apprendibili,
invece di usare solo l'ultimo layer.

Motivazione scientifica:
- Layer bassi: catturano informazioni acustiche (formanti, pitch)
- Layer alti: catturano informazioni fonetiche/semantiche
- La somma pesata permette al modello di imparare automaticamente
  quale combinazione è ottimale per il task di riconoscimento fonemico.

Uso:
    python scripts/train_weighted.py --config configs/training_config.yaml
    python scripts/train_weighted.py --data-csv data/processed/combined_augmented.csv
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
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
)
from transformers.models.wavlm import WavLMModel

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# WEIGHTED WAVLM MODEL
# =============================================================================

class WavLMWithWeightedLayers(nn.Module):
    """
    WavLM con somma pesata dei layer per CTC.
    
    Questa architettura implementa una strategia SOTA chiamata "Weighted Layer Sum"
    dove gli hidden states di tutti i layer Transformer vengono combinati con
    pesi apprendibili invece di usare solo l'output dell'ultimo layer.
    
    Formula matematica:
        weighted_output = Σ(w_i * h_i) for i in [0, num_layers]
        dove w_i = softmax(raw_weights)[i]
    
    Attributes:
        wavlm: Modello WavLM large (microsoft/wavlm-large)
        layer_weights: Pesi apprendibili per ogni layer (nn.Parameter)
        dropout: Dropout layer per regolarizzazione
        lm_head: Linear layer per proiezione sul vocabolario CTC
        
    Args:
        model_name: Nome o path del modello WavLM pretrained
        vocab_size: Dimensione del vocabolario per CTC output
        dropout_rate: Probabilità dropout (default: 0.1)
        freeze_feature_encoder: Se True, congela l'encoder CNN (default: True)
    
    Example:
        >>> model = WavLMWithWeightedLayers(
        ...     model_name="microsoft/wavlm-large",
        ...     vocab_size=75
        ... )
        >>> inputs = torch.randn(2, 16000)  # batch=2, 1 secondo @ 16kHz
        >>> logits = model(inputs)
        >>> print(logits.shape)  # [2, T, 75]
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        vocab_size: int = 75,
        dropout_rate: float = 0.1,
        freeze_feature_encoder: bool = True,
    ):
        super().__init__()
        
        # Carica WavLM base (NON WavLMForCTC, vogliamo i hidden states)
        self.wavlm = WavLMModel.from_pretrained(
            model_name,
            output_hidden_states=True,  # CRITICO: abilita output di tutti i layer
        )
        
        # Congela feature encoder (CNN) per stabilità training
        if freeze_feature_encoder:
            self._freeze_feature_encoder()
        
        # Numero di layer transformer in WavLM (12 per base, 24 per large)
        # +1 perché includiamo anche l'output del feature encoder (layer 0)
        self.num_layers = self.wavlm.config.num_hidden_layers + 1
        
        # ---------------------------------------------------------------------
        # PESI APPRENDIBILI PER WEIGHTED SUM
        # ---------------------------------------------------------------------
        # Inizializza pesi uniformi (1/num_layers ciascuno dopo softmax)
        # nn.Parameter li rende parte del grafo computazionale
        self.layer_weights = nn.Parameter(
            torch.zeros(self.num_layers, dtype=torch.float32)
        )
        
        # Dimensione hidden state (768 per base, 1024 per large)
        hidden_size = self.wavlm.config.hidden_size
        
        # Dropout + Linear head per CTC
        self.dropout = nn.Dropout(dropout_rate)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Log info
        print(f"[WavLMWeighted] Layers: {self.num_layers}")
        print(f"[WavLMWeighted] Hidden size: {hidden_size}")
        print(f"[WavLMWeighted] Vocab size: {vocab_size}")
    
    def _freeze_feature_encoder(self) -> None:
        """Congela l'encoder CNN di WavLM per stabilità del training."""
        for param in self.wavlm.feature_extractor.parameters():
            param.requires_grad = False
        print("[WavLMWeighted] Feature encoder congelato")
    
    def _get_normalized_weights(self) -> torch.Tensor:
        """
        Normalizza i pesi dei layer con softmax.
        
        La softmax garantisce che:
        1. I pesi siano tutti positivi
        2. La somma dei pesi sia 1.0
        3. I gradienti fluiscano correttamente durante backprop
        
        Returns:
            Tensor di shape [num_layers] con pesi normalizzati
        """
        return F.softmax(self.layer_weights, dim=0)
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass con weighted layer sum.
        
        Args:
            input_values: Audio waveform tensor [batch, samples]
            attention_mask: Maschera per padding [batch, samples] (opzionale)
            labels: Token IDs per CTC loss [batch, label_len] (opzionale)
            
        Returns:
            Dict con:
                - logits: Output logits [batch, time, vocab_size]
                - loss: CTC loss se labels fornite (opzionale)
        """
        # ---------------------------------------------------------------------
        # STEP 1: FORWARD THROUGH WAVLM
        # Output include tutti i hidden states di ogni layer
        # ---------------------------------------------------------------------
        outputs = self.wavlm(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # hidden_states è una tupla di (num_layers + 1) tensori
        # Ogni tensore ha shape [batch, time, hidden_size]
        hidden_states = outputs.hidden_states
        
        # ---------------------------------------------------------------------
        # STEP 2: WEIGHTED SUM DEI LAYER
        #
        # Formula: weighted = Σ(softmax(w_i) * h_i)
        #
        # Stack tutti gli hidden states: [num_layers, batch, time, hidden]
        # Moltiplica per pesi normalizzati e somma lungo la dimensione dei layer
        # ---------------------------------------------------------------------
        
        # Ottieni pesi normalizzati [num_layers]
        weights = self._get_normalized_weights()
        
        # Stack hidden states: [num_layers, batch, time, hidden_size]
        stacked = torch.stack(hidden_states, dim=0)
        
        # Reshape weights per broadcasting: [num_layers, 1, 1, 1]
        weights = weights.view(-1, 1, 1, 1)
        
        # Weighted sum: [batch, time, hidden_size]
        # Moltiplica ogni layer per il suo peso e somma
        weighted_output = (stacked * weights).sum(dim=0)
        
        # ---------------------------------------------------------------------
        # STEP 3: DROPOUT + LINEAR HEAD
        # ---------------------------------------------------------------------
        weighted_output = self.dropout(weighted_output)
        logits = self.lm_head(weighted_output)  # [batch, time, vocab_size]
        
        # ---------------------------------------------------------------------
        # STEP 4: CALCOLA CTC LOSS (se labels fornite)
        # ---------------------------------------------------------------------
        loss = None
        if labels is not None:
            loss = self._compute_ctc_loss(logits, labels, attention_mask)
        
        return {"loss": loss, "logits": logits}
    
    def _compute_ctc_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Calcola CTC loss per training.
        
        Args:
            logits: Output del modello [batch, time, vocab]
            labels: Target labels [batch, label_len], -100 per padding
            attention_mask: Maschera input [batch, time]
            
        Returns:
            CTC loss scalare
        """
        # Log-softmax per CTC
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Transpose per CTC: [time, batch, vocab]
        log_probs = log_probs.transpose(0, 1)
        
        # Calcola lunghezze input
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=-1)
        else:
            input_lengths = torch.full(
                (logits.size(0),), logits.size(1), dtype=torch.long
            )
        
        # Calcola lunghezze label (ignora -100)
        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(dim=-1)
        flattened_labels = labels.masked_select(labels_mask)
        
        # CTC Loss
        loss = F.ctc_loss(
            log_probs,
            flattened_labels,
            input_lengths,
            target_lengths,
            blank=0,  # PAD token come blank
            zero_infinity=True,
        )
        
        return loss
    
    def get_layer_weights_info(self) -> Dict[str, float]:
        """
        Restituisce i pesi normalizzati per analisi.
        
        Utile per interpretare quale layer contribuisce maggiormente
        alle predizioni del modello.
        
        Returns:
            Dict con nome layer e peso normalizzato
        """
        weights = self._get_normalized_weights().detach().cpu().numpy()
        return {f"layer_{i}": float(w) for i, w in enumerate(weights)}


# =============================================================================
# DATA COLLATOR
# =============================================================================

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data Collator per CTC training con padding dinamico.
    
    Gestisce il padding di input audio e labels, mascherando
    le posizioni di padding con -100 per ignorarle nella loss CTC.
    
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
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100
        )
        
        batch["labels"] = labels
        return batch


# =============================================================================
# TRAINER
# =============================================================================

class WeightedWavLMTrainer:
    """
    Trainer per WavLMWithWeightedLayers.
    
    Gestisce il ciclo completo di training:
    1. Setup processor e modello
    2. Caricamento e preprocessing dataset
    3. Training con HuggingFace Trainer
    4. Salvataggio modello finale
    
    Attributes:
        config: Configurazione training da YAML
        model: WavLMWithWeightedLayers
        processor: Wav2Vec2Processor
        trainer: HuggingFace Trainer
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza trainer.
        
        Args:
            config: Dict con configurazione training (da YAML)
        """
        self.config = config
        self.model = None
        self.processor = None
        self.trainer = None
        
        # Set random seed per riproducibilità
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
        
        Il processor combina:
        - FeatureExtractor: normalizza audio waveform
        - Tokenizer: converte fonemi IPA in token IDs
        """
        vocab_path = self.config["data"]["vocab_path"]
        
        print(f"[Trainer] Caricamento vocab: {vocab_path}")
        
        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
        
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
        
        print(f"[Trainer] Vocab size: {len(tokenizer)}")
    
    def setup_model(self) -> None:
        """
        Inizializza WavLMWithWeightedLayers.
        """
        model_name = self.config["model"]["name"]
        vocab_size = len(self.processor.tokenizer)
        
        print(f"[Trainer] Inizializzazione modello: {model_name}")
        
        self.model = WavLMWithWeightedLayers(
            model_name=model_name,
            vocab_size=vocab_size,
            dropout_rate=0.1,
            freeze_feature_encoder=self.config["model"].get(
                "freeze_feature_encoder", True
            ),
        )
    
    def load_and_prepare_dataset(self, csv_path: str):
        """
        Carica e preprocessa dataset.
        
        Args:
            csv_path: Path al CSV con colonne [audio_path, ipa_clean, split]
            
        Returns:
            DatasetDict con split train/validation/test
        """
        from datasets import DatasetDict
        
        print(f"[Trainer] Caricamento dataset: {csv_path}")
        
        # Carica CSV
        ds = load_dataset("csv", data_files=csv_path)["train"]
        
        # Carica audio
        ds = ds.cast_column(
            "audio_path",
            Audio(sampling_rate=16000)
        )
        ds = ds.rename_column("audio_path", "audio")
        
        # Split dataset
        val_size = self.config["data"].get("val_size", 0.05)
        test_size = self.config["data"].get("test_size", 0.05)
        
        if "split" in ds.column_names:
            # Usa split esistente
            train_ds = ds.filter(lambda x: x["split"] == "train")
            val_ds = ds.filter(lambda x: x["split"] == "validation")
            test_ds = ds.filter(lambda x: x["split"] == "test")
        else:
            # Crea split
            total_eval = val_size + test_size
            split1 = ds.train_test_split(test_size=total_eval, seed=42)
            val_ratio = val_size / total_eval
            split2 = split1["test"].train_test_split(
                test_size=(1 - val_ratio), seed=42
            )
            train_ds = split1["train"]
            val_ds = split2["train"]
            test_ds = split2["test"]
        
        dataset = DatasetDict({
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        })
        
        print(f"[Trainer] Split: train={len(dataset['train'])}, "
              f"val={len(dataset['validation'])}, test={len(dataset['test'])}")
        
        # Preprocess
        def preprocess(batch):
            """Preprocessa batch: audio -> features, IPA -> token IDs."""
            audio = batch["audio"]["array"]
            batch["input_values"] = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors=None,
            ).input_values[0]
            batch["labels"] = self.processor.tokenizer(
                batch["ipa_clean"]
            ).input_ids
            return batch
        
        print("[Trainer] Preprocessing dataset...")
        dataset = dataset.map(
            preprocess,
            remove_columns=dataset["train"].column_names,
            num_proc=1,
        )
        
        return dataset
    
    def train(self, dataset) -> None:
        """
        Avvia training.
        
        Args:
            dataset: DatasetDict preprocessato
        """
        training_config = self.config["training"]
        output_dir = training_config["output_dir"]
        
        # Metriche
        cer_metric = evaluate.load("cer")
        
        def compute_metrics(pred):
            """Calcola PER (Phoneme Error Rate)."""
            pred_ids = np.argmax(pred.predictions, axis=-1)
            pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
            pred_str = self.processor.batch_decode(pred_ids)
            label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)
            
            # Filtra stringhe vuote
            valid = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
            if not valid:
                return {"per": 1.0}
            preds, labels = zip(*valid)
            per = cer_metric.compute(predictions=preds, references=labels)
            return {"per": per}
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get("num_train_epochs", 10),
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 8),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 2),
            learning_rate=training_config.get("learning_rate", 3e-5),
            warmup_steps=training_config.get("warmup_steps", 500),
            weight_decay=training_config.get("weight_decay", 0.01),
            fp16=training_config.get("fp16", True),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="per",
            greater_is_better=False,
            logging_steps=100,
            dataloader_num_workers=0,
            group_by_length=training_config.get("group_by_length", True),
        )
        
        # Data collator
        data_collator = DataCollatorCTCWithPadding(processor=self.processor)
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
        
        print("[Trainer] Avvio training...")
        self.trainer.train()
        
        # Salva modello finale
        final_path = Path(output_dir) / "final_model_weighted"
        self.model.save_pretrained(final_path)
        self.processor.save_pretrained(final_path)
        print(f"[Trainer] Modello salvato: {final_path}")
        
        # Log layer weights finali
        print("\n[Trainer] Layer weights finali:")
        weights = self.model.get_layer_weights_info()
        for layer, weight in weights.items():
            print(f"  {layer}: {weight:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train WavLM con Weighted Layer Sum"
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
    
    args = parser.parse_args()
    
    # Carica config
    print("=" * 60)
    print("TRAINING WAVLM WEIGHTED LAYERS")
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
    print(f"Output: {config['training']['output_dir']}")
    
    # Training
    trainer = WeightedWavLMTrainer(config)
    trainer.setup_processor()
    trainer.setup_model()
    dataset = trainer.load_and_prepare_dataset(config["data"]["csv_path"])
    trainer.train(dataset)
    
    print("\n" + "=" * 60)
    print("✓ Training completato!")
    print("=" * 60)


if __name__ == "__main__":
    main()

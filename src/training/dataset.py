"""
Dataset e DataCollator per training CTC.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Union, Any
from transformers import Wav2Vec2Processor


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator per CTC che gestisce padding dinamico.
    
    Attributes:
        processor: Wav2Vec2Processor per padding
        padding: Tipo di padding (True, 'longest', 'max_length')
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, 
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate batch di features con padding.
        
        Args:
            features: Lista di dict con 'input_values' e 'labels'
            
        Returns:
            Batch dict con tensori paddati
        """
        # Separa input e labels
        input_features = [
            {"input_values": feature["input_values"]} 
            for feature in features
        ]
        label_features = [
            {"input_ids": feature["labels"]} 
            for feature in features
        ]

        # Pad audio
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

        # Sostituisci padding con -100 per ignorarlo nella loss CTC
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), 
            -100
        )
        batch["labels"] = labels
        
        return batch


def prepare_dataset_function(processor: Wav2Vec2Processor):
    """
    Factory per funzione di preprocessing dataset.
    
    Args:
        processor: Wav2Vec2Processor configurato
        
    Returns:
        Funzione di preprocessing per dataset.map()
    """
    def prepare_dataset(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch["audio"]
        
        # Processa audio -> array normalizzato
        batch["input_values"] = processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        
        # Processa testo IPA -> token IDs
        batch["labels"] = processor.tokenizer(
            batch["ipa_clean"]
        ).input_ids
        
        return batch
    
    return prepare_dataset

"""
Modulo per inferenza del modello.
"""

import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
from transformers import (
    Wav2Vec2Processor,
    WavLMForCTC,
)


class PhonemePredictor:
    """Predittore per trascrizione fonetica da audio."""
    
    def __init__(
        self, 
        model_path: str,
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: Path al modello salvato
            device: Device da usare ('cuda', 'cpu', o None per auto)
        """
        self.model_path = Path(model_path)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Carica modello e processor."""
        print(f">>> Caricamento modello da {self.model_path}...")
        
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        self.model = WavLMForCTC.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Modello caricato su {self.device}")
    
    def load_audio(
        self, 
        audio_path: str, 
        target_sr: int = 16000
    ) -> np.ndarray:
        """
        Carica e preprocessa file audio.
        
        Args:
            audio_path: Path al file audio
            target_sr: Sample rate target
            
        Returns:
            Array audio normalizzato
        """
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio
    
    def predict(
        self, 
        audio: Union[str, np.ndarray],
        return_logits: bool = False
    ) -> Union[str, tuple]:
        """
        Predice trascrizione fonetica.
        
        Args:
            audio: Path audio o array numpy
            return_logits: Se True, restituisce anche logits
            
        Returns:
            Stringa IPA (e logits se richiesto)
        """
        # Carica audio se path
        if isinstance(audio, str):
            audio_array = self.load_audio(audio)
        else:
            audio_array = audio
        
        # Preprocessing
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inferenza
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Decodifica
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]
        
        if return_logits:
            return transcription, logits.cpu().numpy()
        return transcription
    
    def predict_batch(
        self, 
        audio_paths: List[str]
    ) -> List[str]:
        """
        Predice batch di file audio.
        
        Args:
            audio_paths: Lista path audio
            
        Returns:
            Lista trascrizioni IPA
        """
        results = []
        for path in audio_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                print(f"⚠️  Errore su {path}: {e}")
                results.append("")
        return results
    
    def interactive_mode(self) -> None:
        """Modalità interattiva per testing."""
        print("\n" + "="*50)
        print("MODALITÀ INTERATTIVA")
        print("Inserisci path audio o 'quit' per uscire")
        print("="*50 + "\n")
        
        while True:
            try:
                path = input("Audio path > ").strip()
                
                if path.lower() in ('quit', 'exit', 'q'):
                    print("Arrivederci!")
                    break
                
                if not Path(path).exists():
                    print(f"❌ File non trovato: {path}")
                    continue
                
                result = self.predict(path)
                print(f"📝 IPA: /{result}/\n")
                
            except KeyboardInterrupt:
                print("\nArrivederci!")
                break
            except Exception as e:
                print(f"❌ Errore: {e}")

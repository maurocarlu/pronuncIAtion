"""
Modulo per preprocessing dataset e creazione vocabolario.

Utilizza il modulo normalize_ipa per normalizzazione IPA consistente.
"""

import json
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Tuple, Dict, Optional

from .normalize_ipa import IPANormalizer, normalize_for_training


class PhonemePreprocessor:
    """Preprocessa dataset IPA e crea vocabolario per CTC."""
    
    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[UNK]": 1,
        "|": 2  # Word delimiter
    }
    
    def __init__(
        self,
        min_freq: int = 1,
        normalize_mode: Optional[str] = None,
    ):
        """
        Args:
            min_freq: Frequenza minima per includere un carattere nel vocab
            normalize_mode: Modalità di normalizzazione IPA:
                - None: Solo rimuove delimitatori (comportamento originale)
                - 'training': Normalizzazione per training (rimuove stress)
                - 'strict': Normalizzazione completa
        """
        self.min_freq = min_freq
        self.normalize_mode = normalize_mode
        self.vocab: Dict[str, int] = {}
        self.char_counts: Counter = Counter()
        
        # Inizializza normalizzatore se richiesto
        if normalize_mode:
            self.normalizer = IPANormalizer(mode=normalize_mode)
        else:
            self.normalizer = None
    
    def clean_ipa(self, text: str) -> str:
        """
        Pulisce stringa IPA rimuovendo delimitatori.
        Se normalize_mode è impostato, applica anche normalizzazione.
        """
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if text.startswith('/'):
            text = text[1:]
        if text.endswith('/'):
            text = text[:-1]
        text = text.strip()
        
        # Applica normalizzazione se configurata
        if self.normalizer:
            text = self.normalizer.normalize(text)
        
        return text

    
    def process_dataframe(self, df: pd.DataFrame, ipa_column: str = "ipa") -> pd.DataFrame:
        """
        Processa DataFrame aggiungendo colonna IPA pulita.
        
        Args:
            df: DataFrame con colonna IPA
            ipa_column: Nome colonna contenente IPA
            
        Returns:
            DataFrame con colonna 'ipa_clean' aggiunta
        """
        df = df.copy()
        df['ipa_clean'] = df[ipa_column].apply(self.clean_ipa)
        
        # Rimuovi righe vuote
        initial_count = len(df)
        df = df[df['ipa_clean'] != ""]
        removed = initial_count - len(df)
        
        if removed > 0:
            print(f"⚠️  Rimosse {removed} righe con IPA vuoto")
        
        return df
    
    def add_split_column(
        self,
        df: pd.DataFrame,
        val_size: float = 0.05,
        test_size: float = 0.05,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Aggiunge colonna 'split' al DataFrame se non presente.
        
        Args:
            df: DataFrame da processare
            val_size: Proporzione validation set
            test_size: Proporzione test set
            seed: Random seed
            
        Returns:
            DataFrame con colonna 'split'
        """
        import numpy as np
        
        if 'split' in df.columns:
            print("✓ Colonna 'split' già presente")
            return df
        
        np.random.seed(seed)
        
        n = len(df)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        n_train = n - n_test - n_val
        
        splits = ['train'] * n_train + ['validation'] * n_val + ['test'] * n_test
        np.random.shuffle(splits)
        
        df = df.copy()
        df['split'] = splits
        
        print(f"✓ Creata colonna 'split': train={n_train}, val={n_val}, test={n_test}")
        
        return df
    
    def build_vocab(self, df: pd.DataFrame, ipa_column: str = "ipa_clean") -> Dict[str, int]:
        """
        Costruisce vocabolario da DataFrame.
        
        Args:
            df: DataFrame processato
            ipa_column: Colonna con IPA pulito
            
        Returns:
            Dizionario vocabolario {char: id}
        """
        # Conta tutti i caratteri
        all_text = "".join(df[ipa_column].tolist())
        self.char_counts = Counter(all_text)
        
        # Filtra per frequenza minima
        valid_chars = sorted([
            char for char, count in self.char_counts.items() 
            if count >= self.min_freq
        ])
        
        # Costruisci vocab con special tokens
        self.vocab = self.SPECIAL_TOKENS.copy()
        for i, char in enumerate(valid_chars):
            self.vocab[char] = i + len(self.SPECIAL_TOKENS)
        
        return self.vocab
    
    def save_vocab(self, path: str) -> None:
        """Salva vocabolario su file JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"✓ Vocabolario salvato: {path} ({len(self.vocab)} simboli)")
    
    def get_stats(self) -> Dict:
        """Restituisce statistiche sul vocabolario."""
        return {
            "total_unique_chars": len(self.char_counts),
            "vocab_size": len(self.vocab),
            "most_common": self.char_counts.most_common(20),
            "rare_chars": [
                (char, count) for char, count in self.char_counts.items() 
                if count < 5
            ]
        }
    
    def print_stats(self) -> None:
        """Stampa statistiche vocabolario."""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("STATISTICHE VOCABOLARIO")
        print("="*50)
        print(f"Caratteri unici trovati: {stats['total_unique_chars']}")
        print(f"Dimensione vocabolario: {stats['vocab_size']}")
        
        print("\nTop 20 caratteri più frequenti:")
        for char, count in stats['most_common']:
            print(f"  '{char}': {count}")
        
        if stats['rare_chars']:
            print(f"\nCaratteri rari (freq < 5): {len(stats['rare_chars'])}")
            for char, count in stats['rare_chars'][:10]:
                print(f"  '{char}': {count}")


def preprocess_dataset(
    input_csv: str,
    output_csv: str,
    output_vocab: str,
    min_freq: int = 1
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Funzione completa di preprocessing.
    
    Args:
        input_csv: Path CSV input
        output_csv: Path CSV output
        output_vocab: Path vocab JSON
        min_freq: Frequenza minima caratteri
        
    Returns:
        Tuple (DataFrame processato, vocabolario)
    """
    print(f"📂 Caricamento: {input_csv}")
    df = pd.read_csv(input_csv)
    
    preprocessor = PhonemePreprocessor(min_freq=min_freq)
    
    # Processa
    df = preprocessor.process_dataframe(df)
    vocab = preprocessor.build_vocab(df)
    
    # Salva
    df.to_csv(output_csv, index=False)
    print(f"✓ CSV salvato: {output_csv}")
    
    preprocessor.save_vocab(output_vocab)
    preprocessor.print_stats()
    
    return df, vocab

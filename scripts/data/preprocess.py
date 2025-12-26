"""
Script per preprocessing dataset e creazione vocabolario.

Supporta normalizzazione IPA opzionale per garantire consistenza
tra training e evaluation.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessor import PhonemePreprocessor


def preprocess_dataset_with_options(
    input_csv: str,
    output_csv: str,
    output_vocab: str,
    min_freq: int = 1,
    normalize_mode: str = None,
    add_split: bool = False,
    val_size: float = 0.05,
    test_size: float = 0.05,
):
    """
    Funzione di preprocessing con supporto normalizzazione.
    
    Args:
        input_csv: Path CSV input
        output_csv: Path CSV output
        output_vocab: Path vocab JSON
        min_freq: Frequenza minima caratteri
        normalize_mode: Modalità normalizzazione ('strict', 'training', None)
        add_split: Se True, aggiunge colonna split train/val/test
        val_size: Proporzione validation set
        test_size: Proporzione test set
    """
    import pandas as pd
    
    print(f"📂 Caricamento: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Crea preprocessor con normalizzazione opzionale
    preprocessor = PhonemePreprocessor(
        min_freq=min_freq,
        normalize_mode=normalize_mode
    )
    
    if normalize_mode:
        print(f"🔧 Normalizzazione IPA: mode='{normalize_mode}'")
    
    # Processa
    df = preprocessor.process_dataframe(df)
    
    # Aggiungi split se richiesto
    if add_split:
        df = preprocessor.add_split_column(df, val_size=val_size, test_size=test_size)
    
    # Build vocab
    vocab = preprocessor.build_vocab(df)
    
    # Salva
    df.to_csv(output_csv, index=False)
    print(f"✓ CSV salvato: {output_csv}")
    
    preprocessor.save_vocab(output_vocab)
    preprocessor.print_stats()
    
    return df, vocab


def main():
    parser = argparse.ArgumentParser(description="Preprocess phoneme dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/phonemeref_metadata.csv",
        help="Input CSV file"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/processed/phonemeref_processed.csv",
        help="Output processed CSV"
    )
    parser.add_argument(
        "--output-vocab",
        type=str,
        default="data/processed/vocab.json",
        help="Output vocabulary JSON"
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum character frequency for vocab"
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["strict", "training", "none"],
        default="none",
        help="IPA normalization mode: 'strict' (remove stress, normalize all), "
             "'training' (remove stress only), 'none' (no normalization)"
    )
    parser.add_argument(
        "--add-split",
        action="store_true",
        help="Add train/val/test split column (90/5/5 default)"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.05,
        help="Validation set proportion (default: 0.05)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.05,
        help="Test set proportion (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("PREPROCESSING DATASET")
    print("=" * 50)
    
    # Crea directory output se non esiste
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Determina normalize_mode
    normalize_mode = None if args.normalize == "none" else args.normalize
    
    # Processa
    preprocess_dataset_with_options(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_vocab=args.output_vocab,
        min_freq=args.min_freq,
        normalize_mode=normalize_mode,
        add_split=args.add_split,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    
    print("\n✓ Preprocessing completato!")


if __name__ == "__main__":
    main()


"""
Script per preprocessing dataset e creazione vocabolario.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import preprocess_dataset


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
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🔧 PREPROCESSING DATASET")
    print("=" * 50)
    
    # Crea directory output se non esiste
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Processa
    preprocess_dataset(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_vocab=args.output_vocab,
        min_freq=args.min_freq
    )
    
    print("\n✓ Preprocessing completato!")


if __name__ == "__main__":
    main()

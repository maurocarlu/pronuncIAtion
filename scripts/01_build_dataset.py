"""
Script per costruzione dataset da file JSON e audio scaricati.
Unisce i metadati in un unico CSV.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent.parent))


def build_phonemeref_csv(
    data_dir: str = "data/raw/phonemeref_data",
    output_file: str = "data/processed/phonemeref_metadata.csv",
    seed: int = 42
) -> pd.DataFrame:
    """
    Costruisce CSV dai file JSON e audio.
    
    Args:
        data_dir: Directory con audio/ e json/
        output_file: Path CSV output
        seed: Seed per split random
        
    Returns:
        DataFrame con metadati
    """
    data_path = Path(data_dir)
    json_dir = data_path / "json"
    audio_dir = data_path / "audio"
    
    if not json_dir.exists():
        raise FileNotFoundError(f"Directory JSON non trovata: {json_dir}")
    
    records = []
    json_files = list(json_dir.glob("*.json"))
    
    print(f"📂 Trovati {len(json_files)} file JSON")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            word = data.get("word", json_file.stem)
            # Il campo può essere 'ipa' o 'phonetics'
            ipa = data.get("ipa", data.get("phonetics", ""))
            
            # Trova file audio corrispondente
            audio_subdir = audio_dir / word
            audio_file = None
            
            if audio_subdir.exists():
                for ext in ['.mp3', '.wav', '.ogg']:
                    candidates = list(audio_subdir.glob(f"*{ext}"))
                    if candidates:
                        audio_file = candidates[0]
                        break
            
            if audio_file and ipa:
                # Path relativo dalla root del progetto (data/raw/phonemeref_data/audio/...)
                # Questo garantisce portabilità Windows -> Colab
                relative_audio = f"data/raw/phonemeref_data/{audio_file.relative_to(data_path)}"
                relative_audio = relative_audio.replace("\\", "/")
                
                records.append({
                    "id": len(records),
                    "word": word,
                    "ipa": ipa,
                    "audio_path": relative_audio
                })
                
        except Exception as e:
            print(f"⚠️  Errore su {json_file.name}: {e}")
    
    if not records:
        raise ValueError("Nessun record valido trovato!")
    
    df = pd.DataFrame(records)
    
    # Salva
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Dataset creato: {output_file}")
    print(f"  Totale record: {len(df)}")
    print(f"  Parole uniche: {df['word'].nunique()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Build phoneme dataset CSV")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/phonemeref_data",
        help="Directory with audio/ and json/ subdirs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/phonemeref_metadata.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🔧 BUILD DATASET")
    print("=" * 50)
    
    build_phonemeref_csv(
        data_dir=args.data_dir,
        output_file=args.output,
        seed=args.seed
    )
    
    print("\n✓ Completato!")


if __name__ == "__main__":
    main()

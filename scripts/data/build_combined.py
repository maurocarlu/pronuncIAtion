#!/usr/bin/env python3
"""
Script per integrare samples SpeechOcean762 (alta qualità) nel training.

Questo script:
1. Scarica SpeechOcean762
2. Filtra samples con score >= threshold (default: 8)
3. Converte ARPABET → IPA usando il nostro modulo
4. Estrae e salva gli audio
5. Crea CSV compatibile con il training
6. Combina con il dataset WordReference esistente
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.normalize_ipa import (
    arpa_to_ipa,
    IPANormalizer,
)


def extract_phones_from_words(words_list: list) -> str:
    """Estrae e converte tutti i fonemi dalla lista di parole."""
    all_phones_ipa = []
    for word_info in words_list:
        phones = word_info.get("phones", [])
        for p in phones:
            ipa = arpa_to_ipa(p, use_corrected=True)
            if ipa:
                all_phones_ipa.append(ipa)
    return "".join(all_phones_ipa)


def build_speechocean_dataset(
    output_dir: str = "data/speechocean",
    output_csv: str = "data/processed/speechocean_processed.csv",
    min_score: int = 8,
    split: str = "train",
    max_samples: int = None,
):
    """
    Estrae samples SpeechOcean762 e crea dataset per training.
    
    Args:
        output_dir: Directory per salvare audio files
        output_csv: Path CSV output
        min_score: Score minimo per includere sample (1-10)
        split: Split da usare ('train' o 'test')
        max_samples: Limite samples (None = tutti)
    """
    print("=" * 60)
    print(f"ESTRAZIONE SPEECHOCEAN762 (score >= {min_score})")
    print("=" * 60)
    
    # Setup
    output_path = Path(output_dir)
    audio_dir = output_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    normalizer = IPANormalizer(mode='strict')
    
    # Carica dataset
    print(f"\n1. Caricamento SpeechOcean762 ({split})...")
    ds = load_dataset("mispeech/speechocean762", split=split, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"   Totale samples: {len(ds)}")
    
    # Filtra per score
    print(f"\n2. Filtro per score >= {min_score}...")
    ds_filtered = ds.filter(lambda x: x["accuracy"] >= min_score)
    print(f"   Samples dopo filtro: {len(ds_filtered)}")
    
    if max_samples:
        ds_filtered = ds_filtered.select(range(min(max_samples, len(ds_filtered))))
        print(f"   Limitato a: {len(ds_filtered)}")
    
    # Processa samples
    print(f"\n3. Estrazione audio e conversione IPA...")
    records = []
    errors = 0
    
    for idx, sample in enumerate(tqdm(ds_filtered, desc="Processing")):
        try:
            # Converti ARPABET → IPA
            ipa_raw = extract_phones_from_words(sample["words"])
            
            if not ipa_raw:
                errors += 1
                continue
            
            # Normalizza IPA
            ipa_clean = normalizer.normalize(ipa_raw)
            
            if not ipa_clean:
                errors += 1
                continue
            
            # Salva audio
            audio_array = sample["audio"]["array"]
            audio_sr = sample["audio"]["sampling_rate"]
            
            # Nome file unico
            filename = f"so_{split}_{idx:05d}.wav"
            audio_path = audio_dir / filename
            
            sf.write(audio_path, audio_array, audio_sr)
            
            # Record
            records.append({
                "id": f"speechocean_{split}_{idx}",
                "word": sample["text"],  # Frase completa
                "ipa": ipa_raw,
                "ipa_clean": ipa_clean,
                "audio_path": str(audio_path),
                "source": "speechocean762",
                "score": sample["accuracy"],
                "age": sample.get("age", None),
            })
            
        except Exception as e:
            errors += 1
            continue
    
    print(f"   Processati: {len(records)}")
    print(f"   Errori: {errors}")
    
    # Crea DataFrame
    df = pd.DataFrame(records)
    
    # Salva
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n4. Salvato: {output_csv}")
    print(f"   Samples: {len(df)}")
    
    return df


def combine_datasets(
    wordref_csv: str = "data/processed/phonemeref_processed.csv",
    speechocean_csv: str = "data/processed/speechocean_processed.csv",
    output_csv: str = "data/processed/combined_dataset.csv",
    val_size: float = 0.05,
    test_size: float = 0.05,
    seed: int = 42,
):
    """
    Combina WordReference e SpeechOcean in un unico dataset con split.
    
    Args:
        wordref_csv: Path CSV WordReference
        speechocean_csv: Path CSV SpeechOcean
        output_csv: Path CSV output combinato
        val_size: Proporzione validation set
        test_size: Proporzione test set
        seed: Random seed
    """
    print("\n" + "=" * 60)
    print("COMBINAZIONE DATASETS")
    print("=" * 60)
    
    np.random.seed(seed)
    
    # Carica WordReference
    print("\n1. Caricamento WordReference...")
    df_wr = pd.read_csv(wordref_csv)
    df_wr['source'] = 'wordreference'
    print(f"   Samples: {len(df_wr)}")
    
    # Carica SpeechOcean (se esiste)
    df_so = None
    if Path(speechocean_csv).exists():
        print("\n2. Caricamento SpeechOcean...")
        df_so = pd.read_csv(speechocean_csv)
        print(f"   Samples: {len(df_so)}")
    else:
        print("\n2. SpeechOcean non trovato, uso solo WordReference")
    
    # Combina
    if df_so is not None:
        # Assicura colonne comuni
        common_cols = ['audio_path', 'word', 'ipa', 'ipa_clean', 'source']
        
        df_wr_sub = df_wr[[c for c in common_cols if c in df_wr.columns]].copy()
        df_so_sub = df_so[[c for c in common_cols if c in df_so.columns]].copy()
        
        df = pd.concat([df_wr_sub, df_so_sub], ignore_index=True)
    else:
        df = df_wr.copy()
    
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Crea split
    print("\n3. Creazione split train/val/test...")
    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val
    
    splits = ['train'] * n_train + ['validation'] * n_val + ['test'] * n_test
    np.random.shuffle(splits)
    df['split'] = splits
    
    print(f"   Train: {n_train}")
    print(f"   Validation: {n_val}")
    print(f"   Test: {n_test}")
    
    # Statistiche per source
    print("\n4. Distribuzione per source:")
    for source in df['source'].unique():
        count = len(df[df['source'] == source])
        pct = 100 * count / len(df)
        print(f"   {source}: {count} ({pct:.1f}%)")
    
    # Salva
    df.to_csv(output_csv, index=False)
    print(f"\n5. Salvato: {output_csv}")
    print(f"   Totale samples: {len(df)}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Integra SpeechOcean762 nel training dataset"
    )
    parser.add_argument(
        "--min-score", type=int, default=8,
        help="Score minimo per includere (default: 8)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples da estrarre (default: tutti)"
    )
    parser.add_argument(
        "--wordref-csv", type=str,
        default="data/processed/phonemeref_processed.csv",
        help="Path CSV WordReference"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/processed/combined_dataset.csv",
        help="Path CSV output"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Salta estrazione SpeechOcean (usa CSV esistente)"
    )
    
    args = parser.parse_args()
    
    # Step 1: Estrai SpeechOcean
    if not args.skip_extract:
        build_speechocean_dataset(
            min_score=args.min_score,
            max_samples=args.max_samples,
        )
    
    # Step 2: Combina datasets
    combine_datasets(
        wordref_csv=args.wordref_csv,
        output_csv=args.output,
    )
    
    print("\n" + "=" * 60)
    print("COMPLETATO!")
    print("=" * 60)
    print(f"\nProssimi passi:")
    print(f"  1. Rigenera vocab (opzionale):")
    print(f"     python scripts/02_preprocess.py --input {args.output}")
    print(f"  2. Avvia training:")
    print(f"     python scripts/03_train.py --data-csv {args.output}")


if __name__ == "__main__":
    main()

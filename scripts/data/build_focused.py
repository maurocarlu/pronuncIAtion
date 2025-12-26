#!/usr/bin/env python3
"""
Script per creare dataset augmentato focalizzato su SpeechOcean.

Strategia:
- WordReference: 1 variante augmentata (già performa bene)
- SpeechOcean: 3 varianti augmentate (più varietà per generalizzazione)

Trasformazioni:
- Pitch Shift: voci diverse
- Time Stretch: velocità parlato
- Noise: ambienti rumorosi
- Gain: volumi diversi
- Filtri: qualità microfono
"""

import argparse
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class FocusedAugmentationConfig:
    """Configurazione per augmentation focalizzata."""
    # Input
    combined_csv: str = "data/processed/combined_dataset.csv"
    
    # Output
    output_dir: str = "data/augmented_focused"
    output_csv: str = "data/processed/combined_augmented.csv"
    
    # Varianti per source
    wordref_variants: int = 1   # 1 variante per WordRef
    speechocean_variants: int = 3  # 3 varianti per SpeechOcean
    
    sample_rate: int = 16000
    seed: int = 42
    
    # Augmentation params
    pitch_shift_range: tuple = (-5, 5)
    pitch_probability: float = 0.7
    
    time_stretch_range: tuple = (0.85, 1.15)
    time_stretch_probability: float = 0.5
    
    noise_amplitude_range: tuple = (0.005, 0.025)
    noise_probability: float = 0.6
    
    gain_db_range: tuple = (-6, 6)
    gain_probability: float = 0.4


class FocusedAugmenter:
    """Augmenter con trasformazioni multiple."""
    
    def __init__(self, config: FocusedAugmentationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup audiomentations pipeline."""
        try:
            from audiomentations import (
                Compose,
                PitchShift,
                TimeStretch,
                AddGaussianNoise,
                Gain,
                Normalize,
            )
            
            self.pipeline = Compose([
                PitchShift(
                    min_semitones=self.config.pitch_shift_range[0],
                    max_semitones=self.config.pitch_shift_range[1],
                    p=self.config.pitch_probability
                ),
                TimeStretch(
                    min_rate=self.config.time_stretch_range[0],
                    max_rate=self.config.time_stretch_range[1],
                    p=self.config.time_stretch_probability
                ),
                AddGaussianNoise(
                    min_amplitude=self.config.noise_amplitude_range[0],
                    max_amplitude=self.config.noise_amplitude_range[1],
                    p=self.config.noise_probability
                ),
                Gain(
                    min_gain_db=self.config.gain_db_range[0],
                    max_gain_db=self.config.gain_db_range[1],
                    p=self.config.gain_probability
                ),
                Normalize(p=1.0),
            ])
            self.use_audiomentations = True
            print("✓ Usando audiomentations (avanzato)")
            
        except ImportError:
            self.use_audiomentations = False
            print("⚠ audiomentations non disponibile, uso librosa fallback")
    
    def augment(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Applica augmentation."""
        if self.use_audiomentations:
            return self.pipeline(samples=audio, sample_rate=sr)
        else:
            # Fallback librosa
            augmented = audio.copy()
            
            # Pitch shift
            if self.rng.random() < self.config.pitch_probability:
                n_steps = self.rng.uniform(*self.config.pitch_shift_range)
                augmented = librosa.effects.pitch_shift(
                    augmented, sr=sr, n_steps=n_steps
                )
            
            # Time stretch
            if self.rng.random() < self.config.time_stretch_probability:
                rate = self.rng.uniform(*self.config.time_stretch_range)
                augmented = librosa.effects.time_stretch(augmented, rate=rate)
            
            # Noise
            if self.rng.random() < self.config.noise_probability:
                noise_amp = self.rng.uniform(*self.config.noise_amplitude_range)
                noise = self.rng.normal(0, noise_amp, len(augmented))
                augmented = augmented + noise
            
            # Normalize
            if np.max(np.abs(augmented)) > 0:
                augmented = augmented / np.max(np.abs(augmented)) * 0.95
            
            return augmented.astype(np.float32)


def build_focused_augmented_dataset(config: FocusedAugmentationConfig):
    """Costruisce dataset augmentato focalizzato su SpeechOcean."""
    
    print("=" * 60)
    print("AUGMENTATION FOCALIZZATA")
    print("=" * 60)
    print(f"  WordReference varianti: {config.wordref_variants}")
    print(f"  SpeechOcean varianti:   {config.speechocean_variants}")
    
    # Carica dataset combinato
    print(f"\n1. Caricamento: {config.combined_csv}")
    df = pd.read_csv(config.combined_csv)
    
    # Conta per source
    sources = df['source'].value_counts()
    print(f"   Trovati:")
    for s, c in sources.items():
        print(f"   - {s}: {c}")
    
    # Setup
    augmenter = FocusedAugmenter(config)
    output_dir = Path(config.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Process
    print(f"\n2. Generazione varianti augmentate...")
    augmented_records = []
    errors = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        audio_path = Path(row['audio_path'])
        source = row.get('source', 'wordreference')
        
        # Determina numero varianti
        if source == 'speechocean762':
            num_variants = config.speechocean_variants
        else:
            num_variants = config.wordref_variants
        
        # Skip se non ci sono varianti da generare
        if num_variants == 0:
            continue
        
        # Carica audio
        try:
            if not audio_path.exists():
                errors += 1
                continue
            
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            # Genera varianti
            for var_idx in range(num_variants):
                # Reseed per varietà
                augmenter.rng = np.random.default_rng(
                    config.seed + hash(f"{audio_path.stem}_{var_idx}") % 100000
                )
                
                augmented = augmenter.augment(audio, sr)
                
                # Salva
                out_filename = f"{audio_path.stem}_aug{var_idx+1}.wav"
                out_path = audio_dir / out_filename
                sf.write(out_path, augmented, sr)
                
                # Record
                augmented_records.append({
                    'audio_path': str(out_path),
                    'word': row.get('word', ''),
                    'ipa': row.get('ipa', ''),
                    'ipa_clean': row.get('ipa_clean', ''),
                    'source': f"{source}_aug",
                    'split': row.get('split', 'train'),
                })
                
        except Exception as e:
            errors += 1
            continue
    
    # Crea DataFrame augmentato
    aug_df = pd.DataFrame(augmented_records)
    print(f"   Generati: {len(aug_df)} samples augmentati")
    if errors > 0:
        print(f"   Errori: {errors}")
    
    # Combina originale + augmentato
    print(f"\n3. Assemblaggio dataset finale...")
    
    # Aggiungi source al df originale se manca
    if 'source' not in df.columns:
        df['source'] = 'wordreference'
    
    final_df = pd.concat([df, aug_df], ignore_index=True)
    
    # Shuffle
    final_df = final_df.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    
    # Statistiche
    print(f"\n4. Statistiche finali:")
    print(f"   Originali: {len(df)}")
    print(f"   Augmentati: {len(aug_df)}")
    print(f"   TOTALE: {len(final_df)}")
    
    print(f"\n   Per source:")
    for s, c in final_df['source'].value_counts().items():
        pct = 100 * c / len(final_df)
        print(f"   - {s}: {c} ({pct:.1f}%)")
    
    print(f"\n   Per split:")
    for s, c in final_df['split'].value_counts().items():
        print(f"   - {s}: {c}")
    
    # Salva
    final_df.to_csv(config.output_csv, index=False)
    print(f"\n5. Salvato: {config.output_csv}")
    
    return final_df


def main():
    parser = argparse.ArgumentParser(
        description="Crea dataset augmentato focalizzato su SpeechOcean"
    )
    parser.add_argument(
        "--input", type=str,
        default="data/processed/combined_dataset.csv",
        help="CSV dataset combinato"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/processed/combined_augmented.csv",
        help="CSV output"
    )
    parser.add_argument(
        "--wordref-variants", type=int, default=1,
        help="Varianti per WordReference (default: 1)"
    )
    parser.add_argument(
        "--speechocean-variants", type=int, default=3,
        help="Varianti per SpeechOcean (default: 3)"
    )
    
    args = parser.parse_args()
    
    config = FocusedAugmentationConfig(
        combined_csv=args.input,
        output_csv=args.output,
        wordref_variants=args.wordref_variants,
        speechocean_variants=args.speechocean_variants,
    )
    
    build_focused_augmented_dataset(config)
    
    print("\n" + "=" * 60)
    print("COMPLETATO!")
    print("=" * 60)
    print(f"\nProssimo passo:")
    print(f"  Carica su Google Drive e usa colab_train_augmented.ipynb")


if __name__ == "__main__":
    main()

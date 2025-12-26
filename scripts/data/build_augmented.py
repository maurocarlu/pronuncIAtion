#!/usr/bin/env python3
"""
Script per augmentation dataset fonetico.

Trasformazioni acustiche per aumentare varietà del dataset:
- Pitch Shift: simula voci diverse (bambini, adulti, generi)
- Time Stretch: velocità di parlato diverse
- Rumore: ambienti diversi (ufficio, strada, casa)
- Riverbero: stanze diverse
- Filtri: qualità microfono diverse

Uso:
    python scripts/build_augmented_dataset.py --input data/processed/phonemeref_processed.csv
    python scripts/build_augmented_dataset.py --num-variants 3  # più varianti
"""

import argparse
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# CONFIGURAZIONE
# =============================================================================

@dataclass
class AugmentationConfig:
    """Configurazione per augmentation acustica."""
    # Paths
    input_csv: str = "data/processed/phonemeref_processed.csv"
    audio_base_path: str = "data/raw/phonemeref_data"
    output_dir: str = "data/augmented"
    output_csv: str = "data/processed/phonemeref_augmented.csv"
    
    # Numero varianti per sample
    num_variants: int = 2
    sample_rate: int = 16000
    
    # === PITCH SHIFT (simula voci diverse) ===
    # Range ampio per coprire bambini (-6) e voci profonde (+4)
    pitch_shift_semitones: tuple = (-6, 6)
    pitch_probability: float = 0.7
    
    # === TIME STRETCH (velocità parlato) ===
    # 0.85 = più lento, 1.15 = più veloce
    time_stretch_range: tuple = (0.85, 1.15)
    time_stretch_probability: float = 0.5
    
    # === RUMORE AMBIENTE ===
    noise_amplitude_range: tuple = (0.002, 0.02)
    noise_probability: float = 0.6
    
    # === GAIN (volume) ===
    gain_db_range: tuple = (-8, 8)
    gain_probability: float = 0.5
    
    # === FILTRI (qualità audio) ===
    # Low pass: simula telefono/microfono scarso
    lowpass_cutoff_range: tuple = (3000, 7500)
    lowpass_probability: float = 0.3
    
    # High pass: rimuove rumble basse frequenze
    highpass_cutoff_range: tuple = (50, 200)
    highpass_probability: float = 0.2
    
    # === CLIPPING (distorsione leggera) ===
    clipping_probability: float = 0.1
    clipping_threshold: tuple = (0.7, 0.95)
    
    # Processing
    seed: int = 42


# =============================================================================
# AUGMENTER ACUSTICO
# =============================================================================

class AcousticAugmenter:
    """
    Augmenter acustico avanzato per simulare condizioni audio realistiche.
    
    Trasformazioni disponibili:
    - Pitch Shift: voci bambini/adulti/generi
    - Time Stretch: velocità parlato
    - Rumore Gaussiano: ambiente rumoroso
    - Gain: volumi diversi
    - Low/High Pass Filter: qualità microfono
    - Clipping: distorsione leggera
    """
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.use_audiomentations = self._setup_audiomentations()
        
    def _setup_audiomentations(self) -> bool:
        """Setup audiomentations con pipeline avanzata."""
        try:
            from audiomentations import (
                Compose,
                OneOf,
                PitchShift,
                TimeStretch,
                AddGaussianNoise,
                Gain,
                LowPassFilter,
                HighPassFilter,
                ClippingDistortion,
                Normalize,
            )
            
            # Pipeline principale con trasformazioni varie
            self.augment_pipeline = Compose([
                # Pitch shift - molto comune
                PitchShift(
                    min_semitones=self.config.pitch_shift_semitones[0],
                    max_semitones=self.config.pitch_shift_semitones[1],
                    p=self.config.pitch_probability
                ),
                
                # Time stretch
                TimeStretch(
                    min_rate=self.config.time_stretch_range[0],
                    max_rate=self.config.time_stretch_range[1],
                    p=self.config.time_stretch_probability
                ),
                
                # Rumore ambiente
                AddGaussianNoise(
                    min_amplitude=self.config.noise_amplitude_range[0],
                    max_amplitude=self.config.noise_amplitude_range[1],
                    p=self.config.noise_probability
                ),
                
                # Gain (volume)
                Gain(
                    min_gain_db=self.config.gain_db_range[0],
                    max_gain_db=self.config.gain_db_range[1],
                    p=self.config.gain_probability
                ),
                
                # Filtri (uno dei due, non entrambi)
                OneOf([
                    LowPassFilter(
                        min_cutoff_freq=self.config.lowpass_cutoff_range[0],
                        max_cutoff_freq=self.config.lowpass_cutoff_range[1],
                    ),
                    HighPassFilter(
                        min_cutoff_freq=self.config.highpass_cutoff_range[0],
                        max_cutoff_freq=self.config.highpass_cutoff_range[1],
                    ),
                ], p=self.config.lowpass_probability + self.config.highpass_probability),
                
                # Clipping leggero (simula registrazione saturata)
                ClippingDistortion(
                    min_percentile_threshold=int(self.config.clipping_threshold[0] * 100),
                    max_percentile_threshold=int(self.config.clipping_threshold[1] * 100),
                    p=self.config.clipping_probability
                ),
                
                # Normalizza sempre alla fine
                Normalize(p=1.0),
            ])
            
            print("[OK] audiomentations configurato (pipeline avanzata)")
            return True
            
        except ImportError as e:
            print(f"[WARN] audiomentations non disponibile ({e}), uso fallback librosa")
            return False
        except Exception as e:
            print(f"[WARN] Errore setup audiomentations ({e}), uso fallback librosa")
            return False
    
    def _augment_with_librosa(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Fallback augmentation usando librosa (meno opzioni)."""
        augmented = audio.copy()
        
        # Pitch shift
        if self.rng.random() < self.config.pitch_probability:
            n_steps = self.rng.uniform(*self.config.pitch_shift_semitones)
            augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)
        
        # Time stretch
        if self.rng.random() < self.config.time_stretch_probability:
            rate = self.rng.uniform(*self.config.time_stretch_range)
            augmented = librosa.effects.time_stretch(augmented, rate=rate)
        
        # Rumore
        if self.rng.random() < self.config.noise_probability:
            noise_amp = self.rng.uniform(*self.config.noise_amplitude_range)
            noise = self.rng.normal(0, noise_amp, len(augmented))
            augmented = augmented + noise.astype(augmented.dtype)
        
        # Gain
        if self.rng.random() < self.config.gain_probability:
            gain_db = self.rng.uniform(*self.config.gain_db_range)
            gain_linear = 10 ** (gain_db / 20)
            augmented = augmented * gain_linear
        
        # Normalize
        max_val = np.max(np.abs(augmented))
        if max_val > 0:
            augmented = augmented / max_val * 0.95
        
        return augmented.astype(np.float32)
    
    def augment(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Applica augmentation casuale."""
        if self.use_audiomentations:
            return self.augment_pipeline(samples=audio, sample_rate=sr)
        else:
            return self._augment_with_librosa(audio, sr)
    
    def generate_variants(
        self, 
        audio_path: Path, 
        output_dir: Path,
        num_variants: int = 2
    ) -> List[Path]:
        """
        Genera varianti augmentate di un file audio.
        
        Args:
            audio_path: Path audio originale
            output_dir: Directory output
            num_variants: Numero varianti da generare
            
        Returns:
            Lista path varianti generate
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            variants = []
            stem = audio_path.stem
            
            for i in range(num_variants):
                # Reseed per ogni variante (diversa augmentation)
                self.rng = np.random.default_rng(self.config.seed + hash(f"{stem}_{i}") % 10000)
                
                augmented = self.augment(audio, sr)
                variant_path = output_dir / f"{stem}_aug{i+1}.wav"
                sf.write(variant_path, augmented, sr)
                variants.append(variant_path)
            
            return variants
            
        except Exception as e:
            # Silenzioso per non spammare
            return []


# =============================================================================
# DATASET BUILDER
# =============================================================================

class DatasetBuilder:
    """Costruisce dataset augmentato."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.augmenter = AcousticAugmenter(config)
        
        self.output_dir = Path(config.output_dir)
        self.audio_output_dir = self.output_dir / "acoustic"
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Crea directories output."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_output_dir.mkdir(exist_ok=True)
    
    def load_dataset(self) -> pd.DataFrame:
        """Carica dataset originale."""
        print(f"[INFO] Caricamento: {self.config.input_csv}")
        df = pd.read_csv(self.config.input_csv)
        
        # Aggiungi colonne per tracking
        df['source'] = 'original'
        df['is_correct'] = True
        
        print(f"       {len(df):,} samples originali")
        return df
    
    def process_augmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera varianti augmentate per tutti i samples.
        
        Args:
            df: DataFrame originale
            
        Returns:
            DataFrame con samples augmentati
        """
        print("\n" + "="*60)
        print("ACOUSTIC AUGMENTATION")
        print(f"Generazione {self.config.num_variants} varianti per sample...")
        print("="*60)
        
        augmented_samples = []
        audio_base = Path(self.config.audio_base_path)
        errors = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
            audio_path = Path(row['audio_path'])
            
            # Gestisci path relativi e assoluti
            if not audio_path.is_absolute():
                audio_path = audio_base / audio_path
            
            if not audio_path.exists():
                # Prova path alternativo (gestisci NaN in word)
                word = row.get('word', '')
                if pd.isna(word):
                    word = ''
                alt_path = audio_base / "audio" / str(word) / audio_path.name
                if alt_path.exists():
                    audio_path = alt_path
                else:
                    errors += 1
                    continue
            
            variants = self.augmenter.generate_variants(
                audio_path,
                self.audio_output_dir,
                num_variants=self.config.num_variants
            )
            
            for variant_path in variants:
                # Usa path relativo per portabilità (Windows -> Colab)
                relative_path = str(variant_path).replace('\\', '/')
                augmented_samples.append({
                    'audio_path': relative_path,
                    'word': row.get('word', ''),
                    'ipa': row.get('ipa', ''),
                    'ipa_clean': row.get('ipa_clean', row.get('ipa', '')),
                    'split': row.get('split', 'train'),  # Eredita split da originale
                    'source': 'augmented',
                    'is_correct': True
                })
        
        aug_df = pd.DataFrame(augmented_samples)
        print(f"[OK] Generati {len(aug_df):,} samples augmentati")
        if errors > 0:
            print(f"[WARN] {errors} file audio non trovati (saltati)")
        
        return aug_df
    
    def assemble_dataset(
        self,
        original_df: pd.DataFrame,
        augmented_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Assembla dataset finale."""
        print("\n" + "="*60)
        print("ASSEMBLAGGIO DATASET")
        print("="*60)
        
        # Unisci
        final_df = pd.concat([original_df, augmented_df], ignore_index=True)
        
        # Rimuovi duplicati
        initial_count = len(final_df)
        final_df = final_df.drop_duplicates(subset=['audio_path'], keep='first')
        if initial_count - len(final_df) > 0:
            print(f"[INFO] Rimossi {initial_count - len(final_df)} duplicati")
        
        # Rimuovi entry con IPA vuoto
        final_df = final_df[final_df['ipa_clean'].notna() & (final_df['ipa_clean'] != '')]
        
        # Shuffle
        final_df = final_df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
        
        print(f"[OK] Dataset finale: {len(final_df):,} samples")
        
        return final_df
    
    def print_statistics(self, df: pd.DataFrame):
        """Stampa statistiche dataset."""
        print("\n" + "="*60)
        print("STATISTICHE DATASET FINALE")
        print("="*60)
        
        print(f"\nTotale samples: {len(df):,}")
        
        print("\nPer sorgente:")
        for source, count in df['source'].value_counts().items():
            pct = count / len(df) * 100
            print(f"   - {source}: {count:,} ({pct:.1f}%)")
        
        # Stima durata
        avg_duration = 1.5  # secondi
        total_hours = len(df) * avg_duration / 3600
        print(f"\nDurata stimata: {total_hours:.1f} ore")
        
        # Parole uniche
        unique_words = df['word'].nunique()
        print(f"Parole uniche: {unique_words:,}")
        
        print("="*60)
    
    def run(self) -> pd.DataFrame:
        """Esegue pipeline completa."""
        print("\n" + "="*60)
        print("AVVIO AUGMENTATION PIPELINE")
        print("="*60)
        
        # 1. Carica originale
        original_df = self.load_dataset()
        
        # 2. Genera augmentation
        augmented_df = self.process_augmentation(original_df)
        
        # 3. Assembla
        final_df = self.assemble_dataset(original_df, augmented_df)
        
        # 4. Salva
        output_path = Path(self.config.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"\n[OK] Dataset salvato: {output_path}")
        
        # 5. Statistiche
        self.print_statistics(final_df)
        
        return final_df


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build augmented phoneme dataset (acoustic augmentation only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Augmentation standard (2 varianti)
  python scripts/build_augmented_dataset.py

  # Più varianti
  python scripts/build_augmented_dataset.py --num-variants 3

  # Custom input/output
  python scripts/build_augmented_dataset.py \\
      --input data/processed/phonemeref_processed.csv \\
      --output data/processed/phonemeref_augmented.csv
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/processed/phonemeref_processed.csv",
        help="Input CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/processed/phonemeref_augmented.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/augmented",
        help="Directory for augmented audio files"
    )
    parser.add_argument(
        "--audio-base",
        type=str,
        default="data/raw/phonemeref_data",
        help="Base path for original audio files"
    )
    parser.add_argument(
        "--num-variants", "-n",
        type=int,
        default=2,
        help="Number of augmented variants per sample (default: 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    
    config = AugmentationConfig(
        input_csv=args.input,
        output_csv=args.output,
        output_dir=args.output_dir,
        audio_base_path=args.audio_base,
        num_variants=args.num_variants,
        seed=args.seed,
    )
    
    builder = DatasetBuilder(config)
    builder.run()
    
    print("\n[OK] Augmentation completata!")


if __name__ == "__main__":
    main()

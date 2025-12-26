"""
Script per valutazione e inferenza del modello.
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import evaluate
from datasets import load_dataset, Audio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.predictor import PhonemePredictor


def evaluate_test_set(
    model_path: str,
    csv_path: str = "data/processed/phonemeref_processed.csv",
    audio_base_path: str = "data/raw/phonemeref_data",
    test_size: float = 0.1,
    seed: int = 42
):
    """Valuta modello sul test set."""
    from transformers import Wav2Vec2Processor, WavLMForCTC
    
    print("=" * 60)
    print("🔬 VALUTAZIONE MODELLO SUL TEST SET")
    print("=" * 60)
    
    # Carica modello
    print(f"\n📦 Caricamento modello da {model_path}...")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = WavLMForCTC.from_pretrained(model_path)
    model.eval()
    print("✓ Modello caricato!")
    
    # Carica dataset
    print(f"\n📊 Caricamento dataset da {csv_path}...")
    dataset = load_dataset("csv", data_files=csv_path)
    dataset = dataset["train"]
    
    # Rimuovi colonne non necessarie
    cols_to_remove = [c for c in ["id", "word", "ipa"] if c in dataset.column_names]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    
    # Filtra solo test set
    if "split" in dataset.column_names:
        # Prova a usare la colonna split
        test_dataset = dataset.filter(lambda x: x["split"] == "test")
        
        if len(test_dataset) > 0:
            print(f"✓ Trovata colonna 'split', uso {len(test_dataset)} esempi di test")
            # Rimuovi colonne extra
            cols_to_remove = [c for c in ["split", "source", "is_correct"] if c in test_dataset.column_names]
            if cols_to_remove:
                test_dataset = test_dataset.remove_columns(cols_to_remove)
        else:
            # Nessun sample "test" - usa originali e fai split
            print("⚠ Nessun sample 'test' trovato, uso split casuale su originali")
            if "source" in dataset.column_names:
                original_dataset = dataset.filter(lambda x: x["source"] == "original")
                print(f"   Trovati {len(original_dataset)} sample originali")
            else:
                original_dataset = dataset
            
            # Rimuovi colonne extra prima dello split
            cols_to_remove = [c for c in ["split", "source", "is_correct"] if c in original_dataset.column_names]
            if cols_to_remove:
                original_dataset = original_dataset.remove_columns(cols_to_remove)
            
            split_data = original_dataset.train_test_split(test_size=test_size, seed=seed)
            test_dataset = split_data["test"]
            print(f"   Creato test set: {len(test_dataset)} esempi")
    else:
        # Fallback: split casuale
        print("⚠ Colonna 'split' non trovata, uso split casuale")
        split_data = dataset.train_test_split(test_size=test_size, seed=seed)
        test_dataset = split_data["test"]
    
    # Cast audio
    test_dataset = test_dataset.cast_column("audio_path", Audio(sampling_rate=16000))
    test_dataset = test_dataset.rename_column("audio_path", "audio")
    
    print(f"✓ Test set caricato: {len(test_dataset)} esempi")
    
    # Metrica
    cer_metric = evaluate.load("cer")
    
    # Predizione
    def predict(batch):
        audio_arrays = [audio["array"] for audio in batch["audio"]]
        
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        batch["predicted"] = processor.batch_decode(predicted_ids)
        batch["reference"] = batch["ipa_clean"]
        
        return batch
    
    print("\n🔄 Esecuzione inferenza sul test set...")
    results = test_dataset.map(predict, batched=True, batch_size=8)
    
    # Calcola metriche
    predictions = results["predicted"]
    references = results["reference"]
    per = cer_metric.compute(predictions=predictions, references=references)
    
    # Risultati
    print("\n" + "=" * 60)
    print("📈 RISULTATI")
    print("=" * 60)
    print(f"\n🎯 Phoneme Error Rate (PER): {per*100:.2f}%")
    print(f"   Accuratezza: {(1-per)*100:.2f}%")
    
    # Esempi
    print("\n" + "=" * 60)
    print("📝 ESEMPI DI PREDIZIONI")
    print("=" * 60)
    
    num_examples = min(10, len(results))
    indices = np.random.choice(len(results), num_examples, replace=False)
    
    for i, idx in enumerate(indices, 1):
        example = results[int(idx)]
        print(f"\n--- Esempio {i} ---")
        print(f"Ground Truth: /{example['reference']}/")
        print(f"Predizione:   /{example['predicted']}/")
        
        single_per = cer_metric.compute(
            predictions=[example['predicted']],
            references=[example['reference']]
        )
        print(f"PER: {single_per*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("✓ Valutazione completata!")
    print("=" * 60)
    
    return per


def main():
    parser = argparse.ArgumentParser(description="Evaluate phoneme recognition model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/wavlm-phoneme-recognizer/final_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/processed/phonemeref_processed.csv",
        help="Path to test CSV"
    )
    parser.add_argument(
        "--audio-base",
        type=str,
        default="data/raw/phonemeref_data",
        help="Base path for audio files"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Single audio file for inference"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive inference mode"
    )
    
    args = parser.parse_args()
    
    # Modalità
    if args.interactive:
        predictor = PhonemePredictor(args.model_path)
        predictor.interactive_mode()
    elif args.audio:
        predictor = PhonemePredictor(args.model_path)
        result = predictor.predict(args.audio)
        print(f"\n📝 Trascrizione IPA: /{result}/")
    else:
        evaluate_test_set(
            model_path=args.model_path,
            csv_path=args.test_csv,
            audio_base_path=args.audio_base
        )


if __name__ == "__main__":
    main()

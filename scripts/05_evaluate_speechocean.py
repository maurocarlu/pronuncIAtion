"""
Script per valutazione su SpeechOcean762 (speaker non-nativi).

Utilizza il modulo centralizzato di normalizzazione IPA per garantire
consistenza tra training e evaluation.
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import evaluate
from datasets import load_dataset, Audio

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importa il modulo di normalizzazione centralizzato
from src.data.normalize_ipa import (
    IPANormalizer,
    normalize_for_evaluation,
    arpa_to_ipa,
    get_corrected_arpa_to_ipa_mapping,
)


def extract_phones_from_words(words_list: list) -> str:
    """
    Estrae e converte tutti i fonemi dalla lista di parole.
    Usa la mappatura ARPA→IPA corretta (EH→ɛ).
    """
    all_phones_ipa = []
    for word_info in words_list:
        phones = word_info.get("phones", [])
        for p in phones:
            ipa = arpa_to_ipa(p, use_corrected=True)
            if ipa:
                all_phones_ipa.append(ipa)
    return "".join(all_phones_ipa)


def evaluate_speechocean(model_path: str, verbose: bool = True):
    """Valuta modello su SpeechOcean762."""
    from transformers import Wav2Vec2Processor, WavLMForCTC
    
    # Inizializza normalizzatore
    normalizer = IPANormalizer(mode='strict')
    
    print("=" * 60)
    print("🌊 VALUTAZIONE SU SPEECHOCEAN762 (Non-Native Speakers)")
    print("=" * 60)
    print("\n📋 Configurazione normalizzazione:")
    print(f"   Mode: {normalizer.mode}")
    print(f"   Normalize e→ɛ: {normalizer.normalize_e}")
    print(f"   Remove stress: {normalizer.remove_stress}")
    
    # Carica modello
    print("\n📦 Caricamento modello...")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = WavLMForCTC.from_pretrained(model_path)
    model.eval()
    print("✓ Modello caricato!")
    
    # Carica dataset
    print("\n📥 Scaricamento SpeechOcean762...")
    ds = load_dataset("mispeech/speechocean762", split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"✓ Caricati {len(ds)} esempi")
    
    # Preprocessing
    def prepare_example(example):
        example["reference_ipa"] = extract_phones_from_words(example["words"])
        return example
    
    print("\n🔄 Conversione fonemi ARPABET → IPA (mappatura corretta)...")
    ds = ds.map(prepare_example)
    ds = ds.filter(lambda x: len(x["reference_ipa"]) > 0)
    print(f"✓ Esempi validi: {len(ds)}")
    
    # Predizione
    def predict(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        batch["predicted_ipa"] = processor.batch_decode(predicted_ids)
        return batch
    
    print("\n🔄 Esecuzione inferenza...")
    results = ds.map(predict, batched=True, batch_size=4)
    
    # Metriche
    print("\n📊 Calcolo metriche...")
    cer_metric = evaluate.load("cer")
    
    # Applica normalizzazione IPA usando il modulo centralizzato
    print("🧹 Normalizzazione stringhe IPA...")
    
    clean_preds = [normalizer.normalize(p) for p in results["predicted_ipa"]]
    clean_refs = [normalizer.normalize(r) for r in results["reference_ipa"]]
    
    # Debug: check first few examples before filtering
    print("\n🔍 Debug - Prime 5 predizioni:")
    for i in range(min(5, len(results))):
        print(f"  [{i}] pred='{results['predicted_ipa'][i][:50]}...' -> clean='{clean_preds[i][:50] if clean_preds[i] else 'EMPTY'}'")
        print(f"      ref='{results['reference_ipa'][i][:50]}' -> clean='{clean_refs[i][:50] if clean_refs[i] else 'EMPTY'}'")
    
    # Filtra stringhe vuote
    valid_preds = []
    valid_refs = []
    orig_preds = []
    orig_refs = []
    
    for pred, ref, orig_p, orig_r in zip(
        clean_preds, clean_refs, 
        results["predicted_ipa"], results["reference_ipa"]
    ):
        if pred and ref:
            valid_preds.append(pred)
            valid_refs.append(ref)
            orig_preds.append(orig_p)
            orig_refs.append(orig_r)
    
    print(f"\n📊 Statistiche filtro:")
    print(f"   Totale esempi: {len(results)}")
    print(f"   Predizioni vuote: {sum(1 for p in clean_preds if not p)}")
    print(f"   Riferimenti vuoti: {sum(1 for r in clean_refs if not r)}")
    print(f"   Coppie valide: {len(valid_preds)}")
    
    # Handle empty results
    if len(valid_preds) == 0:
        print("\n❌ ERRORE: Nessuna coppia valida trovata!")
        print("   Possibili cause:")
        print("   - Il modello produce predizioni vuote")
        print("   - La normalizzazione è troppo aggressiva")
        print("\n   Debug: mostra esempi raw...")
        for i in range(min(5, len(results))):
            print(f"   Pred raw: '{results['predicted_ipa'][i]}'")
        return 1.0  # Return 100% error
    
    # Calcola PER normalizzato
    per = cer_metric.compute(predictions=valid_preds, references=valid_refs)
    
    # Calcola anche PER non-normalizzato per confronto
    per_raw = cer_metric.compute(predictions=orig_preds, references=orig_refs)
    
    # Debug: mostra esempi di normalizzazione
    if verbose:
        print("\n🔍 Esempi post-normalizzazione:")
        for i in range(min(3, len(orig_refs))):
            print(f"\n--- Esempio {i+1} ---")
            print(f"Ref Orig:   '{orig_refs[i]}'")
            print(f"Ref Clean:  '{valid_refs[i]}'")
            print(f"Pred Orig:  '{orig_preds[i]}'")
            print(f"Pred Clean: '{valid_preds[i]}'")
    
    # Risultati
    print("\n" + "=" * 60)
    print("📈 RISULTATI SU SPEECHOCEAN762")
    print("=" * 60)
    print(f"\n🎯 Phoneme Error Rate (PER):")
    print(f"   - RAW (senza normalizzazione): {per_raw*100:.2f}%")
    print(f"   - NORMALIZZATO:                {per*100:.2f}%")
    print(f"\n✨ Accuratezza Normalizzata: {(1-per)*100:.2f}%")
    print(f"   Esempi valutati: {len(valid_preds)}")
    
    # Analisi per fasce di score
    print("\n" + "=" * 60)
    print("📊 ANALISI PER QUALITÀ PRONUNCIA (Score Umano)")
    print("=" * 60)
    
    score_buckets = {
        "Bassa (1-4)": [],
        "Media (5-7)": [],
        "Alta (8-10)": []
    }
    
    for i in range(len(results)):
        score = results[i]["accuracy"]
        pred = normalizer.normalize(results[i]["predicted_ipa"])
        ref = normalizer.normalize(results[i]["reference_ipa"])
        
        if not pred or not ref:
            continue
        
        single_per = cer_metric.compute(predictions=[pred], references=[ref])
        
        if score <= 4:
            score_buckets["Bassa (1-4)"].append(single_per)
        elif score <= 7:
            score_buckets["Media (5-7)"].append(single_per)
        else:
            score_buckets["Alta (8-10)"].append(single_per)
    
    for bucket_name, pers in score_buckets.items():
        if pers:
            avg_per = np.mean(pers) * 100
            print(f"  {bucket_name}: PER = {avg_per:.2f}% (n={len(pers)})")
    
    # Esempi
    if verbose:
        print("\n" + "=" * 60)
        print("📝 ESEMPI DI PREDIZIONI")
        print("=" * 60)
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(results), min(10, len(results)), replace=False)
        
        for i, idx in enumerate(sample_indices, 1):
            ex = results[int(idx)]
            pred_clean = normalizer.normalize(ex['predicted_ipa'])
            ref_clean = normalizer.normalize(ex['reference_ipa'])
            
            print(f"\n--- Esempio {i} (Score: {ex['accuracy']}/10, Età: {ex['age']}) ---")
            print(f"Testo:      {ex['text']}")
            print(f"Ref (IPA):  /{ex['reference_ipa']}/")
            print(f"Ref Clean:  /{ref_clean}/")
            print(f"Pred:       /{ex['predicted_ipa']}/")
            print(f"Pred Clean: /{pred_clean}/")
            
            if pred_clean and ref_clean:
                single_per = cer_metric.compute(
                    predictions=[pred_clean],
                    references=[ref_clean]
                )
                print(f"PER:        {single_per*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("✓ Valutazione completata!")
    print("=" * 60)
    
    return per


def main():
    parser = argparse.ArgumentParser(description="Evaluate on SpeechOcean762")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/final_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity"
    )
    
    args = parser.parse_args()
    evaluate_speechocean(args.model_path, verbose=not args.quiet)


if __name__ == "__main__":
    main()

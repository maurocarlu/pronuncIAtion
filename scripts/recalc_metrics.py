#!/usr/bin/env python3
"""
Ricalcolo metriche con normalizzazione IPA.
Questo script applica una pulizia aggressiva alle trascrizioni IPA
per ricalcolare il PER in modo più equo.
"""

import re
import json
import argparse
from pathlib import Path

import evaluate


def normalize_ipa(ipa_text: str) -> str:
    """
    Normalizzazione aggressiva delle stringhe IPA.
    
    Rimuove:
    - Accenti primari (ˈ) e secondari (ˌ)
    - Indicatori di lunghezza (ː)
    - Separatori di sillabe (.)
    - Spazi extra
    
    Args:
        ipa_text: Stringa IPA da normalizzare
        
    Returns:
        Stringa IPA pulita
    """
    if not isinstance(ipa_text, str):
        return ""
    
    # 1. Rimuovi accenti primari (ˈ) e secondari (ˌ)
    text = re.sub(r"[ˈˌ]", "", ipa_text)
    
    # 2. Rimuovi indicatori di lunghezza (ː)
    text = re.sub(r"[ː]", "", text)
    
    # 3. Rimuovi separatori di sillabe (.) e spazi extra
    text = re.sub(r"[.\s]", "", text)
    
    # 4. Rimuovi delimitatori di parole (|)
    text = re.sub(r"\|", "", text)
    
    return text


def load_results(results_path: Path) -> dict:
    """Carica risultati da file JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def recalculate_metrics(results: dict, verbose: bool = True) -> dict:
    """
    Ricalcola le metriche con normalizzazione IPA.
    
    Args:
        results: Dizionario con 'predicted_ipa' e 'reference_ipa'
        verbose: Se True, stampa informazioni di debug
        
    Returns:
        Dizionario con metriche ricalcolate
    """
    predictions = results.get("predicted_ipa", results.get("predictions", []))
    references = results.get("reference_ipa", results.get("references", []))
    
    if not predictions or not references:
        raise ValueError("Risultati non contengono predictions o references!")
    
    print(f"📊 Campioni totali: {len(predictions)}")
    
    # Normalizzazione
    print("🧹 Normalizzazione stringhe IPA in corso...")
    clean_preds = [normalize_ipa(p) for p in predictions]
    clean_refs = [normalize_ipa(r) for r in references]
    
    # Filtra stringhe vuote
    final_preds = []
    final_refs = []
    skipped = 0
    
    for p, r in zip(clean_preds, clean_refs):
        if r:  # Mantieni solo se la reference esiste
            final_preds.append(p)
            final_refs.append(r)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"⚠️  Saltati {skipped} campioni con reference vuota")
    
    print(f"✅ Campioni validi: {len(final_preds)}")
    
    # Calcola PER
    cer_metric = evaluate.load("cer")
    per = cer_metric.compute(predictions=final_preds, references=final_refs)
    
    # Calcola anche WER per confronto
    wer_metric = evaluate.load("wer")
    # Per WER, separa i fonemi con spazi
    wer_preds = [" ".join(list(p)) for p in final_preds]
    wer_refs = [" ".join(list(r)) for r in final_refs]
    wer = wer_metric.compute(predictions=wer_preds, references=wer_refs)
    
    print(f"\n{'='*50}")
    print(f"✨ PER Normalizzato: {per*100:.2f}%")
    print(f"   Accuratezza:      {(1-per)*100:.2f}%")
    print(f"   WER (per fonema): {wer*100:.2f}%")
    print(f"{'='*50}")
    
    # Debug visivo
    if verbose and len(predictions) > 0:
        print("\n🔍 Esempi post-pulizia:")
        for i in range(min(3, len(predictions))):
            print(f"\n--- Esempio {i+1} ---")
            print(f"Ref Orig:   '{references[i]}'")
            print(f"Ref Clean:  '{clean_refs[i]}'")
            print(f"Pred Orig:  '{predictions[i]}'")
            print(f"Pred Clean: '{clean_preds[i]}'")
    
    return {
        "per_normalized": per,
        "accuracy_normalized": 1 - per,
        "wer_normalized": wer,
        "total_samples": len(final_preds),
        "skipped_samples": skipped
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ricalcola metriche PER con normalizzazione IPA"
    )
    parser.add_argument(
        "--results",
        type=str,
        help="Path al file JSON con risultati (predictions e references)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path per salvare metriche ricalcolate"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disabilita output verboso"
    )
    
    args = parser.parse_args()
    
    if args.results:
        # Carica da file
        results_path = Path(args.results)
        if not results_path.exists():
            print(f"❌ File non trovato: {results_path}")
            return
        
        print(f"📂 Caricamento risultati da: {results_path}")
        results = load_results(results_path)
    else:
        # Modalità demo con dati di esempio
        print("⚠️  Nessun file specificato. Uso dati di esempio...")
        results = {
            "predicted_ipa": [
                "ˈhɛ.loʊ",
                "ˈwɜːld",
                "ˈtɛst.ɪŋ"
            ],
            "reference_ipa": [
                "hɛloʊ",
                "wɜːld",
                "tɛstɪŋ"
            ]
        }
    
    # Ricalcola metriche
    metrics = recalculate_metrics(results, verbose=not args.quiet)
    
    # Salva se richiesto
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Metriche salvate in: {output_path}")


if __name__ == "__main__":
    main()

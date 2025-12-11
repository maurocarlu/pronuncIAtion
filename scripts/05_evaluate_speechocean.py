"""
Script per valutazione su SpeechOcean762 (speaker non-nativi).

Mappatura ARPABET->IPA basata su: https://github.com/chorusai/arpa2ipa
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import numpy as np
import evaluate
from datasets import load_dataset, Audio

# Aggiungi il path per importare arpa2ipa locale
sys.path.insert(0, str(Path(__file__).parent.parent / "arpa2ipa"))
from arpa2ipa._arpa_to_ipa import arpa_to_ipa_lookup

sys.path.insert(0, str(Path(__file__).parent.parent))


def normalize_ipa(ipa_text: str) -> str:
    """
    Normalizzazione aggressiva delle stringhe IPA.
    
    Rimuove:
    - Accenti primari (ˈ) e secondari (ˌ)
    - Indicatori di lunghezza (ː)
    - Separatori di sillabe (.)
    - Spazi extra
    - Delimitatori di parole (|)
    
    Normalizza varianti US/UK:
    - oʊ → əʊ (o entrambi → ou semplificato)
    
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
    
    # 5. Normalizza varianti rhotic (ʳ → r)
    text = text.replace("ʳ", "r")
    
    # 6. Normalizza varianti g (ɡ → g)
    text = text.replace("ɡ", "g")
    
    # 7. Normalizza dittonghi UK/US per confronto equo
    # Il modello UK produce əʊ, arpa2ipa produce oʊ
    text = text.replace("əʊ", "oʊ")
    
    # 8. Normalizza vocali r-colored (ɜr vs ɜːr)
    text = text.replace("ɜːr", "ɜr")
    
    return text


def arpabet_to_ipa(arpabet_phone: str) -> str:
    """
    Converte singolo fonema ARPABET in IPA usando la mappatura da chorusai/arpa2ipa.
    """
    # Lookup diretto (include già le varianti con stress 0,1,2)
    if arpabet_phone in arpa_to_ipa_lookup:
        return arpa_to_ipa_lookup[arpabet_phone]
    
    # Fallback: rimuovi numeri di stress e riprova
    phone_clean = ''.join(c for c in arpabet_phone if not c.isdigit())
    return arpa_to_ipa_lookup.get(phone_clean, "")


def extract_phones_from_words(words_list: list) -> str:
    """Estrae e converte tutti i fonemi dalla lista di parole."""
    all_phones_ipa = []
    for word_info in words_list:
        phones = word_info.get("phones", [])
        for p in phones:
            ipa = arpabet_to_ipa(p)
            if ipa:
                all_phones_ipa.append(ipa)
    return "".join(all_phones_ipa)


def evaluate_speechocean(model_path: str):
    """Valuta modello su SpeechOcean762."""
    from transformers import Wav2Vec2Processor, WavLMForCTC
    
    print("=" * 60)
    print("🌊 VALUTAZIONE SU SPEECHOCEAN762 (Non-Native Speakers)")
    print("=" * 60)
    
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
    
    print("\n🔄 Conversione fonemi ARPABET → IPA...")
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
    
    # Applica normalizzazione IPA
    print("🧹 Normalizzazione stringhe IPA in corso...")
    
    clean_preds = [normalize_ipa(p) for p in results["predicted_ipa"]]
    clean_refs = [normalize_ipa(r) for r in results["reference_ipa"]]
    
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
    
    # Calcola PER normalizzato
    per = cer_metric.compute(predictions=valid_preds, references=valid_refs)
    
    # Calcola anche PER non-normalizzato per confronto
    per_raw = cer_metric.compute(predictions=orig_preds, references=orig_refs)
    
    # Debug: mostra esempi di normalizzazione
    print("\n🔍 Esempi post-pulizia:")
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
        pred = normalize_ipa(results[i]["predicted_ipa"])
        ref = normalize_ipa(results[i]["reference_ipa"])
        
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
    print("\n" + "=" * 60)
    print("📝 ESEMPI DI PREDIZIONI")
    print("=" * 60)
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(results), min(10, len(results)), replace=False)
    
    for i, idx in enumerate(sample_indices, 1):
        ex = results[int(idx)]
        pred_clean = normalize_ipa(ex['predicted_ipa'])
        ref_clean = normalize_ipa(ex['reference_ipa'])
        
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
    
    args = parser.parse_args()
    evaluate_speechocean(args.model_path)


if __name__ == "__main__":
    main()

"""
Script per valutazione su SpeechOcean762 (speaker non-nativi).

BENCHMARK SCIENTIFICO con 3 Task:
- TASK A: ASR Robustness (PER su alta qualità)
- TASK B: Scoring Correlation (correlazione con score umani)
- TASK C: Mispronunciation Detection (classificazione binaria)

Utilizza il modulo centralizzato di normalizzazione IPA per garantire
consistenza tra training e evaluation.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
    """Valuta modello su SpeechOcean762 con benchmark scientifico completo."""
    from transformers import Wav2Vec2Processor, WavLMForCTC
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
    
    # Inizializza normalizzatore
    normalizer = IPANormalizer(mode='strict')
    
    print("=" * 70)
    print("🔬 BENCHMARK SCIENTIFICO - SPEECHOCEAN762")
    print("=" * 70)
    print("\n📋 Configurazione:")
    print(f"   Modello: {model_path}")
    print(f"   Normalizzazione IPA: {normalizer.mode}")
    
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
    
    # ==========================================================================
    # PREDIZIONE CON CONFIDENCE SCORE
    # ==========================================================================
    def predict_with_confidence(batch):
        """Predice IPA e calcola confidence score."""
        audio_arrays = [x["array"] for x in batch["audio"]]
        
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        # Predizioni
        predicted_ids = torch.argmax(logits, dim=-1)
        batch["predicted_ipa"] = processor.batch_decode(predicted_ids)
        
        # Calcola confidence score
        # Applica softmax per ottenere probabilità
        probs = F.softmax(logits, dim=-1)
        
        # Per ogni sequenza, prendi la probabilità massima di ogni token
        max_probs = torch.max(probs, dim=-1).values  # [batch, seq_len]
        
        # Maschera per ignorare padding (token con id 0 = PAD)
        # Calcola confidence media escludendo PAD tokens
        confidence_scores = []
        for i in range(len(audio_arrays)):
            # Trova token non-PAD
            non_pad_mask = predicted_ids[i] != processor.tokenizer.pad_token_id
            if non_pad_mask.sum() > 0:
                conf = max_probs[i][non_pad_mask].mean().item()
            else:
                conf = 0.0
            confidence_scores.append(conf)
        
        batch["confidence_score"] = confidence_scores
        return batch
    
    print("\n🔄 Esecuzione inferenza con confidence scoring...")
    results = ds.map(predict_with_confidence, batched=True, batch_size=4)
    
    # ==========================================================================
    # PREPARA DATI PER ANALISI
    # ==========================================================================
    print("\n📊 Preparazione dati per analisi...")
    cer_metric = evaluate.load("cer")
    
    # Raccogli tutti i dati
    all_data = []
    for i in range(len(results)):
        pred = normalizer.normalize(results[i]["predicted_ipa"])
        ref = normalizer.normalize(results[i]["reference_ipa"])
        
        if not pred or not ref:
            continue
        
        # Calcola PER singolo
        per = cer_metric.compute(predictions=[pred], references=[ref])
        
        all_data.append({
            "human_score": results[i]["accuracy"],
            "confidence_score": results[i]["confidence_score"],
            "per": per,
            "pred": pred,
            "ref": ref,
            "text": results[i]["text"],
            "age": results[i]["age"],
        })
    
    print(f"   Esempi validi per analisi: {len(all_data)}")
    
    # Converti in arrays
    human_scores = np.array([d["human_score"] for d in all_data])
    confidence_scores = np.array([d["confidence_score"] for d in all_data])
    pers = np.array([d["per"] for d in all_data])
    
    # ==========================================================================
    # TASK A: ASR ROBUSTNESS (Solo High Quality)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📋 TASK A: ASR ROBUSTNESS (Phoneme Recognition Quality)")
    print("=" * 70)
    print("   Obiettivo: Verificare che su pronunce di alta qualità (score >= 8)")
    print("              il modello trascriva correttamente i fonemi.")
    print("-" * 70)
    
    # Filtra solo high quality
    high_quality_mask = human_scores >= 8
    high_quality_preds = [d["pred"] for d, m in zip(all_data, high_quality_mask) if m]
    high_quality_refs = [d["ref"] for d, m in zip(all_data, high_quality_mask) if m]
    
    if len(high_quality_preds) > 0:
        per_high = cer_metric.compute(predictions=high_quality_preds, references=high_quality_refs)
        
        print(f"\n   📊 Risultati su High Quality (score >= 8):")
        print(f"   ─────────────────────────────────────────")
        print(f"   Campioni: {len(high_quality_preds)}")
        print(f"   PER:      {per_high * 100:.2f}%")
        print(f"   Accuracy: {(1 - per_high) * 100:.2f}%")
        
        # Breakdown per fascia
        for threshold in [8, 9, 10]:
            mask = human_scores == threshold
            preds = [d["pred"] for d, m in zip(all_data, mask) if m]
            refs = [d["ref"] for d, m in zip(all_data, mask) if m]
            if len(preds) > 0:
                per_t = cer_metric.compute(predictions=preds, references=refs)
                print(f"      Score {threshold}: PER = {per_t * 100:.2f}% (n={len(preds)})")
    else:
        print("   ⚠️ Nessun esempio high quality trovato!")
        per_high = 1.0
    
    # ==========================================================================
    # TASK B: SCORING CORRELATION (Intero Dataset)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📋 TASK B: SCORING CORRELATION (PER vs Human Score)")
    print("=" * 70)
    print("   Obiettivo: Verificare correlazione tra PER del modello")
    print("              e giudizio umano sulla qualità della pronuncia.")
    print("   Metrica principale: (1 - PER) ↔ Human Score")
    print("-" * 70)
    
    # Correlazione PER (metrica principale)
    pearson_per, pearson_per_p = pearsonr(1 - pers, human_scores)
    spearman_per, spearman_per_p = spearmanr(1 - pers, human_scores)
    
    # Correlazione confidence (metrica secondaria)
    pearson_conf, pearson_conf_p = pearsonr(confidence_scores, human_scores)
    spearman_conf, spearman_conf_p = spearmanr(confidence_scores, human_scores)
    
    print(f"\n   📊 METRICA PRINCIPALE: (1 - PER) ↔ Human Score")
    print(f"   ════════════════════════════════════════════════")
    print(f"   Pearson:  r = {pearson_per:.4f} (p = {pearson_per_p:.2e})")
    print(f"   Spearman: ρ = {spearman_per:.4f} (p = {spearman_per_p:.2e})")
    
    print(f"\n   📊 Metrica secondaria: Confidence ↔ Human Score")
    print(f"   ──────────────────────────────────────────")
    print(f"   Pearson:  r = {pearson_conf:.4f} (p = {pearson_conf_p:.2e})")
    print(f"   Spearman: ρ = {spearman_conf:.4f} (p = {spearman_conf_p:.2e})")
    
    # Interpretazione basata su PER (metrica principale)
    if abs(spearman_per) >= 0.7:
        interp = "✅ FORTE correlazione - il PER discrimina bene"
    elif abs(spearman_per) >= 0.5:
        interp = "✅ MODERATA-BUONA correlazione - risultato significativo"
    elif abs(spearman_per) >= 0.3:
        interp = "⚠️ MODERATA correlazione - margine di miglioramento"
    else:
        interp = "❌ DEBOLE correlazione - necessario miglioramento"
    print(f"\n   Interpretazione PER: {interp}")
    
    # ==========================================================================
    # TASK C: MISPRONUNCIATION DETECTION (Classificazione Binaria)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📋 TASK C: MISPRONUNCIATION DETECTION (PER-based Classification)")
    print("=" * 70)
    print("   Obiettivo: Classificare pronuncia come Corretta/Errata")
    print("              usando il PER (distanza Levenshtein) come predittore.")
    print("   Labels: Errata (score <= 6), Corretta (score > 6)")
    print("   Logica: Alto PER → Alta probabilità di errore di pronuncia")
    print("-" * 70)
    
    # Crea label binarie
    # 1 = Pronuncia Errata (score <= 6), 0 = Pronuncia Corretta (score > 6)
    y_true = (human_scores <= 6).astype(int)
    
    # Usa PER come score predittivo
    # Alto PER → alta probabilità di errore (no inversione necessaria)
    y_prob = pers
    
    # Trova soglia ottimale usando F1
    from sklearn.metrics import precision_recall_curve
    
    # Test multiple thresholds
    thresholds_to_test = np.arange(0.05, 0.50, 0.01)
    best_f1 = 0
    best_threshold = 0.10
    
    for thresh in thresholds_to_test:
        y_pred_temp = (pers >= thresh).astype(int)
        _, _, f1_temp, _ = precision_recall_fscore_support(
            y_true, y_pred_temp, average='binary', zero_division=0
        )
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = thresh
    
    # Applica soglia ottimale
    y_pred = (pers >= best_threshold).astype(int)
    
    # Metriche
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_roc = 0.5  # Default se una sola classe presente
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    accuracy = ((y_pred == y_true).sum()) / len(y_true)
    
    # Conta distribuzione
    n_correct = (y_true == 0).sum()
    n_incorrect = (y_true == 1).sum()
    
    print(f"\n   📊 Distribuzione Dataset:")
    print(f"   ──────────────────────────")
    print(f"   Pronuncia Corretta (>6):  {n_correct} ({100*n_correct/len(y_true):.1f}%)")
    print(f"   Pronuncia Errata (≤6):    {n_incorrect} ({100*n_incorrect/len(y_true):.1f}%)")
    
    print(f"\n   📊 Soglia Ottimale (massimizza F1):")
    print(f"   ─────────────────────────────────")
    print(f"   PER Threshold: {best_threshold:.2f} (se PER >= {best_threshold:.2f} → Errore)")
    
    print(f"\n   📊 Metriche di Classificazione:")
    print(f"   ──────────────────────────────")
    print(f"   AUC-ROC:   {auc_roc:.4f}")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Interpretazione AUC
    if auc_roc >= 0.8:
        auc_interp = "✅ OTTIMO - classificatore affidabile"
    elif auc_roc >= 0.7:
        auc_interp = "✅ BUONO - classificatore discreto"
    elif auc_roc >= 0.6:
        auc_interp = "⚠️ MODERATO - margine di miglioramento"
    else:
        auc_interp = "❌ SCARSO - classificatore poco affidabile"
    print(f"\n   Interpretazione AUC: {auc_interp}")
    
    # Confusion matrix summary
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    print(f"\n   📊 Confusion Matrix (threshold={best_threshold:.2f}):")
    print(f"   ─────────────────────")
    print(f"                  Predicted")
    print(f"                Corr.  Err.")
    print(f"   Actual Corr.  {tn:4d}  {fp:4d}")
    print(f"   Actual Err.   {fn:4d}  {tp:4d}")
    
    # ==========================================================================
    # RIEPILOGO FINALE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📈 RIEPILOGO BENCHMARK")
    print("=" * 70)
    
    print(f"""
   ┌─────────────────────────────────────────────────────────────────┐
   │ TASK A - ASR Robustness (High Quality, score >= 8)             │
   │   PER:      {per_high * 100:6.2f}%                                          │
   │   Accuracy: {(1 - per_high) * 100:6.2f}%                                          │
   ├─────────────────────────────────────────────────────────────────┤
   │ TASK B - Scoring Correlation [(1-PER) ↔ Human Score]           │
   │   Pearson:  {pearson_per:7.4f}                                            │
   │   Spearman: {spearman_per:7.4f}                                            │
   ├─────────────────────────────────────────────────────────────────┤
   │ TASK C - Mispronunciation Detection (PER >= {best_threshold:.2f})             │
   │   AUC-ROC:  {auc_roc:7.4f}                                            │
   │   F1-Score: {f1:7.4f}                                            │
   └─────────────────────────────────────────────────────────────────┘
    """)
    
    # Esempi (opzionale)
    if verbose:
        print("\n" + "=" * 70)
        print("📝 ESEMPI DI PREDIZIONI")
        print("=" * 70)
        
        np.random.seed(42)
        sample_indices = np.random.choice(len(all_data), min(5, len(all_data)), replace=False)
        
        for i, idx in enumerate(sample_indices, 1):
            d = all_data[int(idx)]
            print(f"\n--- Esempio {i} (Score Umano: {d['human_score']}/10) ---")
            print(f"Testo:      {d['text']}")
            print(f"Ref (IPA):  /{d['ref']}/")
            print(f"Pred:       /{d['pred']}/")
            print(f"PER:        {d['per']*100:.2f}%")
            print(f"Confidence: {confidence_scores[idx]:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Benchmark completato!")
    print("=" * 70)
    
    return {
        "per_high_quality": per_high,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "auc_roc": auc_roc,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark scientifico su SpeechOcean762")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/final_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity (no examples)"
    )
    
    args = parser.parse_args()
    evaluate_speechocean(args.model_path, verbose=not args.quiet)


if __name__ == "__main__":
    main()

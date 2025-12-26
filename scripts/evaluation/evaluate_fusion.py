#!/usr/bin/env python3
"""
Script di valutazione con Late Fusion (Ensemble WavLM + XLS-R).

Late Fusion combina le predizioni (logits) di due modelli diversi
per ottenere predizioni più robuste e accurate.

Formula Ensemble:
    final_logits = w * logits_A + (1-w) * logits_B
    
dove w è il peso del modello A (default: 0.5 per media semplice).

Questo script:
1. Carica entrambi i modelli (WavLM Weighted + XLS-R)
2. Esegue inferenza su SpeechOcean762
3. Combina i logits con Late Fusion
4. Calcola le metriche del benchmark (PER, Correlazione, AUC-ROC)

Uso:
    python scripts/evaluate_fusion.py \\
        --model-a outputs/final_model_weighted \\
        --model-b outputs/xlsr/final_model_xlsr \\
        --weight 0.6
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
import numpy as np
import evaluate
from datasets import load_dataset, Audio
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.normalize_ipa import (
    IPANormalizer,
    arpa_to_ipa,
)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_and_processor(
    model_path: str,
    model_type: str = "auto"
) -> Tuple[torch.nn.Module, Any]:
    """
    Carica modello e processor da path.
    
    Supporta:
    - WavLMForCTC (standard)
    - WavLMWithWeightedLayers (custom)
    - Wav2Vec2ForCTC (XLS-R)
    
    Args:
        model_path: Path al modello salvato
        model_type: Tipo modello ("wavlm", "xlsr", "weighted", "auto")
        
    Returns:
        Tuple (model, processor)
    """
    from transformers import (
        Wav2Vec2Processor,
        Wav2Vec2ForCTC,
        WavLMForCTC,
        WavLMModel,
    )
    import torch.nn as nn
    import json
    
    model_path = Path(model_path)
    
    print(f"[Fusion] Caricamento: {model_path}")
    
    # Carica processor
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    
    # ---------------------------------------------------------------------
    # DETERMINAZIONE AUTOMATICA TIPO MODELLO
    # Controlla config.json per capire quale classe istanziare
    # ---------------------------------------------------------------------
    config_path = model_path / "config.json"
    is_custom_weighted = False
    is_xlsr = False
    config = {}
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        
        # Controlla se è modello custom WavLMWithWeightedLayers
        if config.get("model_type") == "wavlm_weighted_layers":
            is_custom_weighted = True
        # Controlla se è XLS-R (wav2vec2)
        elif "wav2vec2" in config.get("architectures", [""])[0].lower():
            is_xlsr = True
        elif "xlsr" in str(config.get("_name_or_path", "")).lower():
            is_xlsr = True
    
    # Override da parametro
    if model_type == "xlsr":
        is_xlsr = True
        is_custom_weighted = False
    elif model_type == "weighted":
        is_custom_weighted = True
        is_xlsr = False
    
    # ---------------------------------------------------------------------
    # CARICAMENTO MODELLO
    # ---------------------------------------------------------------------
    if is_custom_weighted:
        print(f"[Fusion] Tipo: WavLMWithWeightedLayers (custom)")
        
        vocab_size = config.get("vocab_size", 45)
        base_model = config.get("base_model", "microsoft/wavlm-large")
        
        # Definisci classe inline (stesso codice di train_weighted.py)
        class WavLMWithWeightedLayers(nn.Module):
            def __init__(self, vocab_size, model_name="microsoft/wavlm-large"):
                super().__init__()
                self.wavlm = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
                self.num_layers = self.wavlm.config.num_hidden_layers + 1
                self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))
                hidden_size = self.wavlm.config.hidden_size
                self.dropout = nn.Dropout(0.1)
                self.lm_head = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, input_values, attention_mask=None, **kwargs):
                outputs = self.wavlm(input_values, attention_mask=attention_mask)
                hidden_states = outputs.hidden_states
                weights = torch.softmax(self.layer_weights, dim=0)
                stacked = torch.stack(hidden_states, dim=0)
                weights_view = weights.view(-1, 1, 1, 1)
                weighted_output = (stacked * weights_view).sum(dim=0)
                weighted_output = self.dropout(weighted_output)
                logits = self.lm_head(weighted_output)
                # Wrap in object-like structure for compatibility
                class Output:
                    pass
                out = Output()
                out.logits = logits
                return out
        
        model = WavLMWithWeightedLayers(vocab_size, base_model)
        
        # Carica pesi
        model_file = model_path / "pytorch_model.bin"
        if model_file.exists():
            state_dict = torch.load(model_file, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"[Fusion] ✓ Pesi caricati da: {model_file}")
        else:
            raise FileNotFoundError(f"pytorch_model.bin non trovato in {model_path}")
            
    elif is_xlsr:
        print(f"[Fusion] Tipo: Wav2Vec2ForCTC (XLS-R)")
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    else:
        print(f"[Fusion] Tipo: WavLMForCTC (standard)")
        model = WavLMForCTC.from_pretrained(model_path)
    
    model.eval()
    
    return model, processor


# =============================================================================
# LATE FUSION
# =============================================================================

class LateFusionEnsemble:
    """
    Ensemble con Late Fusion per combinare logits di due modelli.
    
    Late Fusion combina le predizioni a livello di logits:
        final_logits = w * logits_A + (1-w) * logits_B
    
    Vantaggi:
    - Semplice da implementare
    - Non richiede re-training
    - Peso w può essere ottimizzato su validation set
    
    Attributes:
        model_a: Primo modello (es. WavLM Weighted)
        model_b: Secondo modello (es. XLS-R)
        processor_a: Processor modello A
        processor_b: Processor modello B
        weight: Peso del modello A (0-1)
        device: Device per inferenza (cuda/cpu)
        
    Args:
        model_a_path: Path al modello A
        model_b_path: Path al modello B
        weight: Peso modello A (default: 0.5)
        device: Device (default: auto)
    
    Example:
        >>> ensemble = LateFusionEnsemble(
        ...     "outputs/wavlm_weighted",
        ...     "outputs/xlsr",
        ...     weight=0.6
        ... )
        >>> result = ensemble.predict(audio_array)
        >>> print(result["prediction"])
    """
    
    def __init__(
        self,
        model_a_path: str,
        model_b_path: str,
        weight: float = 0.5,
        device: Optional[str] = None,
    ):
        self.weight = weight
        
        # Determina device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[Fusion] Device: {self.device}")
        print(f"[Fusion] Peso modello A: {weight}")
        print(f"[Fusion] Peso modello B: {1-weight}")
        
        # ---------------------------------------------------------------------
        # CARICAMENTO MODELLI
        # ---------------------------------------------------------------------
        print("\n[Fusion] === Caricamento Modello A ===")
        self.model_a, self.processor_a = load_model_and_processor(model_a_path)
        self.model_a.to(self.device)
        
        print("\n[Fusion] === Caricamento Modello B ===")
        self.model_b, self.processor_b = load_model_and_processor(model_b_path)
        self.model_b.to(self.device)
        
        # ---------------------------------------------------------------------
        # VERIFICA ALLINEAMENTO VOCAB
        # CRITICO: I due modelli devono avere lo stesso vocabolario
        # ---------------------------------------------------------------------
        vocab_a = len(self.processor_a.tokenizer)
        vocab_b = len(self.processor_b.tokenizer)
        
        if vocab_a != vocab_b:
            raise ValueError(
                f"ERRORE: Vocab size mismatch! A={vocab_a}, B={vocab_b}\n"
                f"I modelli devono usare lo stesso vocab.json"
            )
        
        print(f"\n[Fusion] ✓ Vocab allineato: {vocab_a} tokens")
    
    def predict_single(
        self,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
    ) -> Dict[str, Any]:
        """
        Predice fonemi per singolo audio usando Late Fusion.
        
        Args:
            audio_array: Audio waveform numpy array
            sampling_rate: Sample rate (default: 16000)
            
        Returns:
            Dict con:
                - prediction: Stringa IPA predetta
                - logits_a: Logits modello A
                - logits_b: Logits modello B
                - logits_fused: Logits combinati
                - confidence: Score confidenza media
        """
        with torch.no_grad():
            # -----------------------------------------------------------------
            # STEP 1: PREPROCESS AUDIO PER ENTRAMBI I MODELLI
            # -----------------------------------------------------------------
            inputs_a = self.processor_a(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            inputs_b = self.processor_b(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            # -----------------------------------------------------------------
            # STEP 2: INFERENZA SU ENTRAMBI I MODELLI
            # -----------------------------------------------------------------
            outputs_a = self.model_a(**inputs_a)
            outputs_b = self.model_b(**inputs_b)
            
            logits_a = outputs_a.logits  # [1, time, vocab]
            logits_b = outputs_b.logits  # [1, time, vocab]
            
            # -----------------------------------------------------------------
            # STEP 3: ALLINEAMENTO TEMPORALE
            # I due modelli possono avere output di lunghezza diversa
            # a causa di differenze nel downsampling CNN
            # -----------------------------------------------------------------
            min_len = min(logits_a.size(1), logits_b.size(1))
            logits_a = logits_a[:, :min_len, :]
            logits_b = logits_b[:, :min_len, :]
            
            # -----------------------------------------------------------------
            # STEP 4: LATE FUSION
            # Formula: final = w * A + (1-w) * B
            # -----------------------------------------------------------------
            logits_fused = (
                self.weight * logits_a + 
                (1 - self.weight) * logits_b
            )
            
            # -----------------------------------------------------------------
            # STEP 5: DECODIFICA
            # -----------------------------------------------------------------
            predicted_ids = torch.argmax(logits_fused, dim=-1)
            prediction = self.processor_a.decode(predicted_ids[0])
            
            # Calcola confidence score
            probs = F.softmax(logits_fused, dim=-1)
            max_probs = torch.max(probs, dim=-1).values
            non_pad = predicted_ids[0] != self.processor_a.tokenizer.pad_token_id
            if non_pad.sum() > 0:
                confidence = max_probs[0][non_pad].mean().item()
            else:
                confidence = 0.0
        
        return {
            "prediction": prediction,
            "logits_a": logits_a.cpu().numpy(),
            "logits_b": logits_b.cpu().numpy(),
            "logits_fused": logits_fused.cpu().numpy(),
            "confidence": confidence,
        }
    
    def predict_batch(
        self,
        audio_arrays: List[np.ndarray],
        sampling_rate: int = 16000,
    ) -> List[Dict[str, Any]]:
        """
        Predice fonemi per batch di audio.
        
        Args:
            audio_arrays: Lista di audio waveform
            sampling_rate: Sample rate
            
        Returns:
            Lista di result dict
        """
        results = []
        for audio in audio_arrays:
            result = self.predict_single(audio, sampling_rate)
            results.append(result)
        return results


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def extract_phones_from_words(words_list: list) -> str:
    """Converte fonemi ARPABET in IPA."""
    all_phones_ipa = []
    for word_info in words_list:
        phones = word_info.get("phones", [])
        for p in phones:
            ipa = arpa_to_ipa(p, use_corrected=True)
            if ipa:
                all_phones_ipa.append(ipa)
    return "".join(all_phones_ipa)


def run_fusion_benchmark(
    ensemble: LateFusionEnsemble,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Esegue benchmark completo con Late Fusion.
    
    Benchmark include:
    - TASK A: ASR Robustness (PER su high quality)
    - TASK B: Scoring Correlation (correlazione con score umani)
    - TASK C: Mispronunciation Detection (classificazione binaria)
    
    Args:
        ensemble: LateFusionEnsemble inizializzato
        verbose: Se True, stampa esempi
        
    Returns:
        Dict con metriche benchmark
    """
    normalizer = IPANormalizer(mode='strict')
    cer_metric = evaluate.load("cer")
    
    print("\n" + "=" * 70)
    print("🔬 BENCHMARK FUSION - SPEECHOCEAN762")
    print("=" * 70)
    
    # Carica dataset
    print("\n📥 Caricamento SpeechOcean762...")
    ds = load_dataset("mispeech/speechocean762", split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"✓ Caricati {len(ds)} esempi")
    
    # Preprocess references
    print("\n🔄 Preprocessing...")
    references = []
    for sample in ds:
        ref = extract_phones_from_words(sample["words"])
        references.append(ref)
    
    # Inferenza
    print("\n🔄 Inferenza Fusion...")
    from tqdm import tqdm
    
    all_data = []
    
    for i, sample in enumerate(tqdm(ds, desc="Fusion")):
        audio = sample["audio"]["array"]
        
        # Predizione ensemble
        result = ensemble.predict_single(audio)
        
        # Normalizza
        pred = normalizer.normalize(result["prediction"])
        ref = normalizer.normalize(references[i])
        
        if not pred or not ref:
            continue
        
        # Calcola PER singolo
        per = cer_metric.compute(predictions=[pred], references=[ref])
        
        all_data.append({
            "human_score": sample["accuracy"],
            "confidence_score": result["confidence"],
            "per": per,
            "pred": pred,
            "ref": ref,
            "text": sample["text"],
        })
    
    print(f"✓ Esempi validi: {len(all_data)}")
    
    # Converti in arrays
    human_scores = np.array([d["human_score"] for d in all_data])
    confidence_scores = np.array([d["confidence_score"] for d in all_data])
    pers = np.array([d["per"] for d in all_data])
    
    # =========================================================================
    # TASK A: ASR ROBUSTNESS
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 TASK A: ASR ROBUSTNESS (High Quality)")
    print("=" * 70)
    
    high_quality_mask = human_scores >= 8
    hq_preds = [d["pred"] for d, m in zip(all_data, high_quality_mask) if m]
    hq_refs = [d["ref"] for d, m in zip(all_data, high_quality_mask) if m]
    
    per_high = cer_metric.compute(predictions=hq_preds, references=hq_refs)
    
    print(f"\n   Campioni: {len(hq_preds)}")
    print(f"   PER:      {per_high * 100:.2f}%")
    print(f"   Accuracy: {(1 - per_high) * 100:.2f}%")
    
    # =========================================================================
    # TASK B: SCORING CORRELATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 TASK B: SCORING CORRELATION")
    print("=" * 70)
    
    pearson_per, _ = pearsonr(1 - pers, human_scores)
    spearman_per, _ = spearmanr(1 - pers, human_scores)
    
    print(f"\n   📊 (1 - PER) ↔ Human Score:")
    print(f"   Pearson:  r = {pearson_per:.4f}")
    print(f"   Spearman: ρ = {spearman_per:.4f}")
    
    # =========================================================================
    # TASK C: MISPRONUNCIATION DETECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 TASK C: MISPRONUNCIATION DETECTION")
    print("=" * 70)
    
    y_true = (human_scores <= 6).astype(int)
    y_prob = pers
    
    # Trova soglia ottimale
    best_f1 = 0
    best_threshold = 0.1
    
    for thresh in np.arange(0.05, 0.50, 0.01):
        y_pred_temp = (pers >= thresh).astype(int)
        _, _, f1_temp, _ = precision_recall_fscore_support(
            y_true, y_pred_temp, average='binary', zero_division=0
        )
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = thresh
    
    y_pred = (pers >= best_threshold).astype(int)
    
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_roc = 0.5
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    print(f"\n   Threshold: {best_threshold:.2f}")
    print(f"   AUC-ROC:   {auc_roc:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    
    # =========================================================================
    # RIEPILOGO
    # =========================================================================
    print("\n" + "=" * 70)
    print("📈 RIEPILOGO FUSION BENCHMARK")
    print("=" * 70)
    
    print(f"""
   ┌─────────────────────────────────────────────────────────────────┐
   │ ENSEMBLE FUSION (w={ensemble.weight})                                    │
   ├─────────────────────────────────────────────────────────────────┤
   │ TASK A - ASR Robustness (High Quality)                         │
   │   PER:      {per_high * 100:6.2f}%                                          │
   │   Accuracy: {(1 - per_high) * 100:6.2f}%                                          │
   ├─────────────────────────────────────────────────────────────────┤
   │ TASK B - Scoring Correlation                                   │
   │   Pearson:  {pearson_per:7.4f}                                            │
   │   Spearman: {spearman_per:7.4f}                                            │
   ├─────────────────────────────────────────────────────────────────┤
   │ TASK C - Mispronunciation Detection                            │
   │   AUC-ROC:  {auc_roc:7.4f}                                            │
   │   F1-Score: {f1:7.4f}                                            │
   └─────────────────────────────────────────────────────────────────┘
    """)
    
    if verbose:
        print("\n📝 ESEMPI:")
        for i in range(min(3, len(all_data))):
            d = all_data[i]
            print(f"\n--- Esempio {i+1} (Score: {d['human_score']}) ---")
            print(f"Text: {d['text']}")
            print(f"Ref:  /{d['ref']}/")
            print(f"Pred: /{d['pred']}/")
            print(f"PER:  {d['per']*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("✓ Benchmark Fusion completato!")
    print("=" * 70)
    
    return {
        "per_high_quality": per_high,
        "pearson": pearson_per,
        "spearman": spearman_per,
        "auc_roc": auc_roc,
        "f1": f1,
        "weight": ensemble.weight,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Valutazione Late Fusion (Ensemble WavLM + XLS-R)"
    )
    parser.add_argument(
        "--model-a",
        type=str,
        required=True,
        help="Path modello A (es. WavLM Weighted)"
    )
    parser.add_argument(
        "--model-b",
        type=str,
        required=True,
        help="Path modello B (es. XLS-R)"
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=0.5,
        help="Peso modello A (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Riduci output"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LATE FUSION EVALUATION")
    print("=" * 70)
    
    # Crea ensemble
    ensemble = LateFusionEnsemble(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        weight=args.weight,
    )
    
    # Esegui benchmark
    results = run_fusion_benchmark(
        ensemble,
        verbose=not args.quiet,
    )
    
    print("\n✓ Valutazione completata!")


if __name__ == "__main__":
    main()

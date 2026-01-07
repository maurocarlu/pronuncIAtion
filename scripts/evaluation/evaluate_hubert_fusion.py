#!/usr/bin/env python3
"""
Late Fusion "Dream Team" - HuBERT Large + WavLM Weighted Ensemble.

Questo script combina i due top-performer del benchmark:
- HuBERT Large: Best PER (8.84%) - eccellente per trascrizione fonetica
- WavLM Weighted: Best AUC (0.8523) - eccellente per detection errori

La Late Fusion combina le predizioni a livello di logits:
    final_logits = α * logits_HuBERT + (1-α) * logits_WavLM

dove α è il peso del modello HuBERT (default: 0.5 per media semplice).

MOTIVAZIONE SCIENTIFICA:
- HuBERT è pre-training con target discreti (k-means) → focalizzato su unità fonetiche
- WavLM usa denoising pre-training → robusto a variazioni acustiche
- Combinandoli otteniamo: trascrizione precisa + detection robusta

Uso:
    # Test pesi singoli
    python scripts/evaluation/evaluate_hubert_fusion.py \\
        --model-hubert outputs/hubert_large/final_model_hubert \\
        --model-wavlm outputs/final_model_weighted \\
        --weight 0.5

    # Sweep automatico
    python scripts/evaluation/evaluate_hubert_fusion.py \\
        --model-hubert outputs/hubert_large/final_model_hubert \\
        --model-wavlm outputs/final_model_weighted \\
        --weights 0.3 0.5 0.7

Autore: DeepLearning-Phoneme Project
"""

import argparse
import sys
import warnings
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import evaluate
from datasets import load_dataset, Audio
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize_ipa import (
    IPANormalizer,
    arpa_to_ipa,
)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class WavLMWithWeightedLayers(nn.Module):
    """
    WavLM con somma pesata dei layer per CTC.
    
    Implementa la Weighted Layer Sum strategy (SUPERB) dove gli hidden states
    di tutti i layer Transformer sono combinati con pesi apprendibili.
    
    Formula:
        w'_i = Softmax(α_i)
        H_out = Σ w'_i * H_i
    
    Args:
        vocab_size: Dimensione vocabolario IPA
        model_name: Nome modello HuggingFace (default: microsoft/wavlm-large)
    """
    
    def __init__(self, vocab_size: int, model_name: str = "microsoft/wavlm-large"):
        super().__init__()
        from transformers.models.wavlm import WavLMModel
        
        self.wavlm = WavLMModel.from_pretrained(model_name, output_hidden_states=True)
        self.num_layers = self.wavlm.config.num_hidden_layers + 1  # +1 per embedding layer
        self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))
        
        hidden_size = self.wavlm.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass con weighted layer sum.
        
        Args:
            input_values: Audio waveform tensor [batch, samples]
            attention_mask: Maschera per padding (opzionale)
            
        Returns:
            Object con .logits tensor [batch, time, vocab]
        """
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # Tuple di (num_layers) tensori
        
        # Normalizza pesi con softmax
        weights = torch.softmax(self.layer_weights, dim=0)
        
        # Somma pesata
        stacked = torch.stack(hidden_states, dim=0)  # [layers, batch, time, hidden]
        weights_view = weights.view(-1, 1, 1, 1)
        weighted_output = (stacked * weights_view).sum(dim=0)  # [batch, time, hidden]
        
        weighted_output = self.dropout(weighted_output)
        logits = self.lm_head(weighted_output)
        
        # Wrap per compatibilità con HuggingFace
        class Output:
            pass
        out = Output()
        out.logits = logits
        return out


# =============================================================================
# MODEL LOADING UTILITIES
# =============================================================================

def load_hubert_model(model_path: str, device: torch.device) -> Tuple[nn.Module, Any]:
    """
    Carica HuBERT Large fine-tuned per CTC.
    
    Args:
        model_path: Path al modello salvato
        device: Device per inferenza
        
    Returns:
        Tuple (model, processor)
    """
    from transformers import (
        Wav2Vec2Processor,
        HubertForCTC,
    )
    
    model_path = Path(model_path)
    print(f"[HuBERT] Caricamento: {model_path}")
    
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = HubertForCTC.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print(f"[HuBERT] ✓ Vocab size: {len(processor.tokenizer)}")
    
    return model, processor


def load_wavlm_weighted_model(model_path: str, device: torch.device) -> Tuple[nn.Module, Any]:
    """
    Carica WavLM Weighted (custom architecture).
    
    Args:
        model_path: Path al modello salvato
        device: Device per inferenza
        
    Returns:
        Tuple (model, processor)
    """
    from transformers import Wav2Vec2Processor
    
    model_path = Path(model_path)
    print(f"[WavLM Weighted] Caricamento: {model_path}")
    
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    
    # Leggi config per vocab_size
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    vocab_size = config.get("vocab_size", 43)
    base_model = config.get("base_model", "microsoft/wavlm-large")
    
    # Istanzia modello custom
    model = WavLMWithWeightedLayers(vocab_size, base_model)
    
    # Carica pesi
    model_file = model_path / "pytorch_model.bin"
    if model_file.exists():
        state_dict = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[WavLM Weighted] ✓ Pesi caricati")
    else:
        # Prova safetensors
        safetensors_file = model_path / "model.safetensors"
        if safetensors_file.exists():
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_file)
            model.load_state_dict(state_dict)
            print(f"[WavLM Weighted] ✓ Pesi caricati (safetensors)")
        else:
            raise FileNotFoundError(f"Nessun file pesi trovato in {model_path}")
    
    model.to(device)
    model.eval()
    
    print(f"[WavLM Weighted] ✓ Vocab size: {len(processor.tokenizer)}")
    
    return model, processor


# =============================================================================
# LATE FUSION ENSEMBLE
# =============================================================================

class HuBERTWavLMFusion:
    """
    Ensemble Late Fusion tra HuBERT Large e WavLM Weighted.
    
    Combina le predizioni dei due top-performer:
    - HuBERT: Migliore per PER (trascrizione)
    - WavLM Weighted: Migliore per AUC (detection)
    
    Formula:
        final_logits = α * logits_HuBERT + (1-α) * logits_WavLM
    
    Attributes:
        model_hubert: HubertForCTC model
        model_wavlm: WavLMWithWeightedLayers model
        processor: Condiviso (stesso vocab.json)
        weight: Peso HuBERT (α), WavLM = 1-α
        device: Device per inferenza
        
    Example:
        >>> fusion = HuBERTWavLMFusion(
        ...     "outputs/hubert",
        ...     "outputs/wavlm_weighted",
        ...     weight=0.5
        ... )
        >>> result = fusion.predict(audio_array)
        >>> print(result["prediction"])  # IPA string
    """
    
    def __init__(
        self,
        hubert_path: str,
        wavlm_path: str,
        weight: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Inizializza ensemble Late Fusion.
        
        Args:
            hubert_path: Path modello HuBERT Large
            wavlm_path: Path modello WavLM Weighted
            weight: Peso HuBERT (α), range [0, 1]
            device: Device per inferenza (default: auto)
        """
        self.weight = weight
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"\n{'='*60}")
        print("LATE FUSION: HuBERT Large + WavLM Weighted")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Peso HuBERT (α): {weight}")
        print(f"Peso WavLM (1-α): {1-weight}")
        
        # Carica modelli
        print(f"\n--- Caricamento Modelli ---")
        self.model_hubert, self.processor_hubert = load_hubert_model(hubert_path, self.device)
        self.model_wavlm, self.processor_wavlm = load_wavlm_weighted_model(wavlm_path, self.device)
        
        # Verifica allineamento vocab
        vocab_h = len(self.processor_hubert.tokenizer)
        vocab_w = len(self.processor_wavlm.tokenizer)
        
        if vocab_h != vocab_w:
            raise ValueError(
                f"ERRORE: Vocab mismatch! HuBERT={vocab_h}, WavLM={vocab_w}\n"
                f"I modelli devono usare lo stesso vocab.json"
            )
        
        # Usa processor HuBERT come riferimento (stesso vocab)
        self.processor = self.processor_hubert
        
        print(f"\n✓ Ensemble inizializzato!")
        print(f"  Vocab condiviso: {vocab_h} tokens")
        print(f"{'='*60}\n")
    
    def predict(
        self,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
    ) -> Dict[str, Any]:
        """
        Predice fonemi usando Late Fusion.
        
        Args:
            audio_array: Audio waveform numpy array
            sampling_rate: Sample rate (default: 16000)
            
        Returns:
            Dict con:
                - prediction: Stringa IPA
                - logits_hubert: Logits HuBERT
                - logits_wavlm: Logits WavLM
                - logits_fused: Logits combinati
                - confidence: Score confidenza media
        """
        with torch.no_grad():
            # Preprocess audio
            inputs = self.processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            # Inferenza HuBERT
            outputs_h = self.model_hubert(**inputs)
            logits_h = outputs_h.logits
            
            # Inferenza WavLM
            outputs_w = self.model_wavlm(inputs.input_values, attention_mask=inputs.get("attention_mask"))
            logits_w = outputs_w.logits
            
            # Allineamento temporale
            min_len = min(logits_h.size(1), logits_w.size(1))
            logits_h = logits_h[:, :min_len, :]
            logits_w = logits_w[:, :min_len, :]
            
            # Late Fusion
            logits_fused = self.weight * logits_h + (1 - self.weight) * logits_w
            
            # Decodifica
            predicted_ids = torch.argmax(logits_fused, dim=-1)
            prediction = self.processor.decode(predicted_ids[0])
            
            # Confidence
            probs = F.softmax(logits_fused, dim=-1)
            max_probs = torch.max(probs, dim=-1).values
            non_pad = predicted_ids[0] != self.processor.tokenizer.pad_token_id
            
            if non_pad.sum() > 0:
                confidence = max_probs[0][non_pad].mean().item()
            else:
                confidence = 0.0
        
        return {
            "prediction": prediction,
            "logits_hubert": logits_h.cpu().numpy(),
            "logits_wavlm": logits_w.cpu().numpy(),
            "logits_fused": logits_fused.cpu().numpy(),
            "confidence": confidence,
        }


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def extract_phones_from_words(words_list: list) -> str:
    """
    Estrae e converte fonemi da annotazioni SpeechOcean.
    
    Args:
        words_list: Lista di dict con chiave "phones" (ARPABET)
        
    Returns:
        Stringa fonemi IPA concatenati
    """
    all_phones_ipa = []
    for word_info in words_list:
        phones = word_info.get("phones", [])
        for p in phones:
            ipa = arpa_to_ipa(p, use_corrected=True)
            if ipa:
                all_phones_ipa.append(ipa)
    return "".join(all_phones_ipa)


def run_fusion_benchmark(
    fusion: HuBERTWavLMFusion,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Esegue benchmark scientifico completo su SpeechOcean762.
    
    Tasks:
    - TASK A: ASR Robustness - PER su pronuncie high quality (score >= 8)
    - TASK B: Scoring Correlation - correlazione (1-PER) vs human score
    - TASK C: Mispronunciation Detection - AUC-ROC per classificazione errori
    
    Args:
        fusion: HuBERTWavLMFusion inizializzato
        verbose: Se True, stampa esempi
        
    Returns:
        Dict con metriche: per_high_quality, pearson, spearman, auc_roc, f1
    """
    normalizer = IPANormalizer(mode='strict')
    cer_metric = evaluate.load("cer")
    
    print("\n" + "=" * 70)
    print("🔬 BENCHMARK FUSION - HuBERT + WavLM WEIGHTED")
    print("=" * 70)
    
    # Carica dataset
    print("\n📥 Caricamento SpeechOcean762...")
    ds = load_dataset("mispeech/speechocean762", split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"✓ Caricati {len(ds)} esempi")
    
    # Preprocess references
    print("\n🔄 Preprocessing references...")
    references = []
    for sample in ds:
        ref = extract_phones_from_words(sample["words"])
        references.append(ref)
    
    # Inferenza
    print("\n🔄 Inferenza Fusion...")
    all_data = []
    
    for i, sample in enumerate(tqdm(ds, desc="Fusion")):
        audio = sample["audio"]["array"]
        
        # Predizione ensemble
        result = fusion.predict(audio)
        
        # Normalizza
        pred = normalizer.normalize(result["prediction"])
        ref = normalizer.normalize(references[i])
        
        if not pred or not ref:
            continue
        
        # Calcola PER
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
    
    # Arrays
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
    weight = fusion.weight
    print("\n" + "=" * 70)
    print("📈 RIEPILOGO FUSION BENCHMARK")
    print("=" * 70)
    
    print(f"""
   ┌─────────────────────────────────────────────────────────────────┐
   │ LATE FUSION HuBERT + WavLM (α={weight})                            │
   ├─────────────────────────────────────────────────────────────────┤
   │ TASK A - ASR Robustness (High Quality, score >= 8)             │
   │   PER:      {per_high * 100:6.2f}%                                          │
   │   Accuracy: {(1 - per_high) * 100:6.2f}%                                          │
   ├─────────────────────────────────────────────────────────────────┤
   │ TASK B - Scoring Correlation [(1-PER) ↔ Human Score]           │
   │   Pearson:  {pearson_per:7.4f}                                            │
   │   Spearman: {spearman_per:7.4f}                                            │
   ├─────────────────────────────────────────────────────────────────┤
   │ TASK C - Mispronunciation Detection                            │
   │   AUC-ROC:  {auc_roc:7.4f}   {'🏆 NEW RECORD!' if auc_roc > 0.8552 else ''}                                       │
   │   F1-Score: {f1:7.4f}                                            │
   └─────────────────────────────────────────────────────────────────┘
    """)
    
    # Record check
    if auc_roc > 0.8552:
        print("🎉 NUOVO RECORD AUC! Superato il precedente 0.8552")
    
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
        "weight": weight,
    }


def run_weight_sweep(
    hubert_path: str,
    wavlm_path: str,
    weights: List[float],
) -> List[Dict[str, float]]:
    """
    Esegue sweep su diversi pesi α per Late Fusion.
    
    Args:
        hubert_path: Path modello HuBERT
        wavlm_path: Path modello WavLM Weighted
        weights: Lista di pesi α da testare
        
    Returns:
        Lista di risultati benchmark per ogni peso
    """
    all_results = []
    
    print("\n" + "=" * 70)
    print("🔄 WEIGHT SWEEP - LATE FUSION")
    print("=" * 70)
    print(f"Pesi da testare: {weights}")
    
    for weight in weights:
        print(f"\n{'='*70}")
        print(f"📊 TESTING WEIGHT α = {weight}")
        print(f"{'='*70}")
        
        fusion = HuBERTWavLMFusion(
            hubert_path=hubert_path,
            wavlm_path=wavlm_path,
            weight=weight,
        )
        
        results = run_fusion_benchmark(fusion, verbose=False)
        all_results.append(results)
        
        # Cleanup
        del fusion
        torch.cuda.empty_cache()
    
    # Riepilogo comparativo
    print("\n" + "=" * 70)
    print("📈 RIEPILOGO WEIGHT SWEEP")
    print("=" * 70)
    
    print("\n┌─────────┬───────────┬──────────┬──────────┬──────────┐")
    print("│ Weight  │ PER (HQ)  │ Pearson  │ Spearman │ AUC-ROC  │")
    print("├─────────┼───────────┼──────────┼──────────┼──────────┤")
    
    best_auc = max(r["auc_roc"] for r in all_results)
    best_per = min(r["per_high_quality"] for r in all_results)
    
    for r in all_results:
        auc_marker = "🏆" if r["auc_roc"] == best_auc else "  "
        per_marker = "🏆" if r["per_high_quality"] == best_per else "  "
        
        print(f"│  {r['weight']:.1f}    │ {r['per_high_quality']*100:5.2f}% {per_marker} │ {r['pearson']:7.4f}  │ {r['spearman']:7.4f}  │ {r['auc_roc']:.4f} {auc_marker}│")
    
    print("└─────────┴───────────┴──────────┴──────────┴──────────┘")
    
    # Confronto con record
    print(f"\n📊 Confronto con record attuali:")
    print(f"   Record PER (HuBERT alone):       8.84%")
    print(f"   Miglior PER Fusion:              {best_per*100:.2f}%")
    print(f"   Record AUC (WavLM alone):        0.8552")
    print(f"   Miglior AUC Fusion:              {best_auc:.4f}")
    
    if best_auc > 0.8552:
        print(f"\n🎉 NUOVO RECORD AUC!")
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Late Fusion HuBERT + WavLM Weighted per Phoneme Recognition"
    )
    parser.add_argument(
        "--model-hubert",
        type=str,
        required=True,
        help="Path modello HuBERT Large"
    )
    parser.add_argument(
        "--model-wavlm",
        type=str,
        required=True,
        help="Path modello WavLM Weighted"
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=None,
        help="Peso singolo α per HuBERT (0-1)"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Lista pesi α per sweep (es. 0.3 0.5 0.7)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Riduci output"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LATE FUSION EVALUATION - HuBERT + WavLM WEIGHTED")
    print("=" * 70)
    
    # Weight sweep o singolo peso
    if args.weights:
        run_weight_sweep(
            hubert_path=args.model_hubert,
            wavlm_path=args.model_wavlm,
            weights=args.weights,
        )
    elif args.weight is not None:
        fusion = HuBERTWavLMFusion(
            hubert_path=args.model_hubert,
            wavlm_path=args.model_wavlm,
            weight=args.weight,
        )
        run_fusion_benchmark(fusion, verbose=not args.quiet)
    else:
        # Default: sweep standard
        run_weight_sweep(
            hubert_path=args.model_hubert,
            wavlm_path=args.model_wavlm,
            weights=[0.3, 0.5, 0.7],
        )
    
    print("\n✓ Valutazione completata!")


if __name__ == "__main__":
    main()

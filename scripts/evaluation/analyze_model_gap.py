#!/usr/bin/env python3
"""
Analisi Qualitativa degli Errori - HuBERT vs WavLM.

Questo script identifica e analizza i casi in cui i due modelli top-performer
hanno comportamenti divergenti:

CASI ANALIZZATI:
1. "HuBERT vince": Audio dove HuBERT ha PER < 5% ma WavLM ha PER > 20%
   → Indica superiorità di HuBERT nella trascrizione fonetica precisa

2. "WavLM vince": Audio dove WavLM rileva correttamente un errore di pronuncia
   che HuBERT trascrive come corretto
   → Indica superiorità di WavLM nella detection di mispronunciation

OUTPUT:
- Report markdown con trascrizioni IPA a confronto
- Statistiche sui gap tra modelli
- Link ai file audio per ispezione manuale

Uso:
    python scripts/evaluation/analyze_model_gap.py \\
        --model-hubert outputs/hubert_large/final_model_hubert \\
        --model-wavlm outputs/final_model_weighted \\
        --output-report docs/qualitative_analysis_report.md

Autore: DeepLearning-Phoneme Project
"""

import argparse
import sys
import warnings
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import evaluate
from datasets import load_dataset, Audio
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
# MODEL LOADING (reused from evaluate_hubert_fusion.py)
# =============================================================================

class WavLMWithWeightedLayers(nn.Module):
    """WavLM con somma pesata dei layer per CTC."""
    
    def __init__(self, vocab_size: int, model_name: str = "microsoft/wavlm-large"):
        super().__init__()
        from transformers.models.wavlm import WavLMModel
        
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
        
        class Output:
            pass
        out = Output()
        out.logits = logits
        return out


def load_hubert_model(model_path: str, device: torch.device):
    """Carica HuBERT Large."""
    from transformers import Wav2Vec2Processor, HubertForCTC
    
    model_path = Path(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = HubertForCTC.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    return model, processor


def load_wavlm_model(model_path: str, device: torch.device):
    """Carica WavLM Weighted."""
    from transformers import Wav2Vec2Processor
    
    model_path = Path(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    vocab_size = config.get("vocab_size", 43)
    base_model = config.get("base_model", "microsoft/wavlm-large")
    
    model = WavLMWithWeightedLayers(vocab_size, base_model)
    
    model_file = model_path / "pytorch_model.bin"
    if model_file.exists():
        state_dict = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        safetensors_file = model_path / "model.safetensors"
        if safetensors_file.exists():
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_file)
            model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model, processor


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def extract_phones_from_words(words_list: list) -> str:
    """Estrae e converte fonemi ARPABET in IPA."""
    all_phones_ipa = []
    for word_info in words_list:
        phones = word_info.get("phones", [])
        for p in phones:
            ipa = arpa_to_ipa(p, use_corrected=True)
            if ipa:
                all_phones_ipa.append(ipa)
    return "".join(all_phones_ipa)


def predict_with_model(
    model: nn.Module,
    processor,
    audio_array: np.ndarray,
    device: torch.device,
    is_hubert: bool = True,
) -> Tuple[str, float]:
    """
    Esegue predizione con un modello.
    
    Args:
        model: Modello (HuBERT o WavLM)
        processor: Processor per preprocessing
        audio_array: Audio waveform
        device: Device
        is_hubert: True se HuBERT, False se WavLM
        
    Returns:
        Tuple (prediction_ipa, confidence)
    """
    with torch.no_grad():
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(device)
        
        if is_hubert:
            outputs = model(**inputs)
            logits = outputs.logits
        else:
            outputs = model(inputs.input_values, attention_mask=inputs.get("attention_mask"))
            logits = outputs.logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.decode(predicted_ids[0])
        
        # Confidence
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1).values
        non_pad = predicted_ids[0] != processor.tokenizer.pad_token_id
        
        if non_pad.sum() > 0:
            confidence = max_probs[0][non_pad].mean().item()
        else:
            confidence = 0.0
    
    return prediction, confidence


def analyze_model_gap(
    hubert_path: str,
    wavlm_path: str,
    output_report: str,
    max_examples: int = 500,
    hubert_wins_threshold: Tuple[float, float] = (0.05, 0.20),
    wavlm_wins_score_threshold: int = 6,
) -> Dict[str, Any]:
    """
    Analizza il gap tra HuBERT e WavLM su SpeechOcean762.
    
    Args:
        hubert_path: Path modello HuBERT
        wavlm_path: Path modello WavLM Weighted
        output_report: Path report markdown output
        max_examples: Numero massimo esempi da analizzare
        hubert_wins_threshold: (max_hubert_per, min_wavlm_per) per "HuBERT wins"
        wavlm_wins_score_threshold: Score umano threshold per mispronunciation
        
    Returns:
        Dict con statistiche analisi
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 70)
    print("🔬 ANALISI QUALITATIVA - HuBERT vs WavLM")
    print("=" * 70)
    print(f"Device: {device}")
    
    # Carica modelli
    print("\n📦 Caricamento modelli...")
    model_h, proc_h = load_hubert_model(hubert_path, device)
    model_w, proc_w = load_wavlm_model(wavlm_path, device)
    print("✓ Modelli caricati")
    
    # Carica dataset
    print("\n📥 Caricamento SpeechOcean762...")
    ds = load_dataset("mispeech/speechocean762", split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    
    if len(ds) > max_examples:
        ds = ds.select(range(max_examples))
    print(f"✓ Analizzando {len(ds)} esempi")
    
    # Normalizer e metric
    normalizer = IPANormalizer(mode='strict')
    cer_metric = evaluate.load("cer")
    
    # Raccogli risultati
    print("\n🔄 Esecuzione inferenza...")
    all_results = []
    
    for i, sample in enumerate(tqdm(ds, desc="Analisi")):
        audio = sample["audio"]["array"]
        ref_raw = extract_phones_from_words(sample["words"])
        ref = normalizer.normalize(ref_raw)
        
        if not ref:
            continue
        
        # Predizioni
        pred_h, conf_h = predict_with_model(model_h, proc_h, audio, device, is_hubert=True)
        pred_w, conf_w = predict_with_model(model_w, proc_w, audio, device, is_hubert=False)
        
        pred_h_norm = normalizer.normalize(pred_h)
        pred_w_norm = normalizer.normalize(pred_w)
        
        if not pred_h_norm or not pred_w_norm:
            continue
        
        # Calcola PER
        per_h = cer_metric.compute(predictions=[pred_h_norm], references=[ref])
        per_w = cer_metric.compute(predictions=[pred_w_norm], references=[ref])
        
        all_results.append({
            "index": i,
            "text": sample["text"],
            "human_score": sample["accuracy"],
            "ref": ref,
            "pred_hubert": pred_h_norm,
            "pred_wavlm": pred_w_norm,
            "per_hubert": per_h,
            "per_wavlm": per_w,
            "conf_hubert": conf_h,
            "conf_wavlm": conf_w,
            "audio_id": sample.get("id", f"sample_{i}"),
        })
    
    print(f"✓ Analisi completata: {len(all_results)} esempi validi")
    
    # ==========================================================================
    # IDENTIFICA CASI INTERESSANTI
    # ==========================================================================
    
    hubert_wins = []
    wavlm_wins = []
    
    for r in all_results:
        # HuBERT wins: HuBERT PER < 5% AND WavLM PER > 20%
        if r["per_hubert"] < hubert_wins_threshold[0] and r["per_wavlm"] > hubert_wins_threshold[1]:
            hubert_wins.append(r)
        
        # WavLM wins: audio è mispronunciation (score <= 6),
        # WavLM ha PER alto (detecta errore), HuBERT ha PER basso (non detecta)
        if r["human_score"] <= wavlm_wins_score_threshold:
            # Se WavLM "vede" l'errore (PER > 15%) ma HuBERT no (PER < 10%)
            if r["per_wavlm"] > 0.15 and r["per_hubert"] < 0.10:
                wavlm_wins.append(r)
    
    print(f"\n📊 Casi identificati:")
    print(f"   HuBERT vince (PER_h<5%, PER_w>20%): {len(hubert_wins)}")
    print(f"   WavLM vince (detects mispronunciation): {len(wavlm_wins)}")
    
    # ==========================================================================
    # STATISTICHE GENERALI
    # ==========================================================================
    
    pers_h = np.array([r["per_hubert"] for r in all_results])
    pers_w = np.array([r["per_wavlm"] for r in all_results])
    scores = np.array([r["human_score"] for r in all_results])
    
    stats = {
        "total_examples": len(all_results),
        "hubert_wins_count": len(hubert_wins),
        "wavlm_wins_count": len(wavlm_wins),
        "mean_per_hubert": float(np.mean(pers_h)),
        "mean_per_wavlm": float(np.mean(pers_w)),
        "std_per_hubert": float(np.std(pers_h)),
        "std_per_wavlm": float(np.std(pers_w)),
        "hubert_better_count": int((pers_h < pers_w).sum()),
        "wavlm_better_count": int((pers_w < pers_h).sum()),
        "equal_count": int((pers_h == pers_w).sum()),
    }
    
    # ==========================================================================
    # GENERA REPORT MARKDOWN
    # ==========================================================================
    
    print(f"\n📝 Generazione report: {output_report}")
    
    report_lines = [
        "# 🔬 Analisi Qualitativa: HuBERT vs WavLM Weighted",
        "",
        f"*Generato: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "---",
        "",
        "## 📊 Executive Summary",
        "",
        "| Metrica | HuBERT Large | WavLM Weighted |",
        "|---------|--------------|----------------|",
        f"| Mean PER | {stats['mean_per_hubert']*100:.2f}% | {stats['mean_per_wavlm']*100:.2f}% |",
        f"| Std PER | {stats['std_per_hubert']*100:.2f}% | {stats['std_per_wavlm']*100:.2f}% |",
        f"| Wins | {stats['hubert_better_count']} | {stats['wavlm_better_count']} |",
        "",
        f"**Totale esempi analizzati:** {stats['total_examples']}",
        "",
        "---",
        "",
        "## 🏆 Casi \"HuBERT Vince\"",
        "",
        f"Criteri: `PER_HuBERT < {hubert_wins_threshold[0]*100:.0f}%` AND `PER_WavLM > {hubert_wins_threshold[1]*100:.0f}%`",
        "",
        f"**Trovati:** {len(hubert_wins)} casi",
        "",
    ]
    
    if hubert_wins:
        report_lines.extend([
            "| # | Testo | Score | Ref | HuBERT | WavLM | PER_H | PER_W |",
            "|---|-------|-------|-----|--------|-------|-------|-------|",
        ])
        
        for i, r in enumerate(hubert_wins[:10]):  # Max 10 esempi
            report_lines.append(
                f"| {i+1} | {r['text'][:30]}... | {r['human_score']} | "
                f"`/{r['ref'][:20]}/` | `/{r['pred_hubert'][:20]}/` | "
                f"`/{r['pred_wavlm'][:20]}/` | {r['per_hubert']*100:.1f}% | {r['per_wavlm']*100:.1f}% |"
            )
        
        if len(hubert_wins) > 10:
            report_lines.append(f"\n*...e altri {len(hubert_wins)-10} casi*")
    else:
        report_lines.append("*Nessun caso trovato con questi criteri.*")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## 🎯 Casi \"WavLM Vince\" (Detection)",
        "",
        f"Criteri: `Score <= {wavlm_wins_score_threshold}` (mispronunciation) AND `PER_WavLM > 15%` AND `PER_HuBERT < 10%`",
        "",
        f"**Trovati:** {len(wavlm_wins)} casi",
        "",
        "Questi sono casi dove WavLM rileva correttamente un errore di pronuncia",
        "che HuBERT invece trascrive come \"corretto\".",
        "",
    ])
    
    if wavlm_wins:
        report_lines.extend([
            "| # | Testo | Score | Ref | HuBERT | WavLM | PER_H | PER_W |",
            "|---|-------|-------|-----|--------|-------|-------|-------|",
        ])
        
        for i, r in enumerate(wavlm_wins[:10]):
            report_lines.append(
                f"| {i+1} | {r['text'][:30]}... | {r['human_score']} | "
                f"`/{r['ref'][:20]}/` | `/{r['pred_hubert'][:20]}/` | "
                f"`/{r['pred_wavlm'][:20]}/` | {r['per_hubert']*100:.1f}% | {r['per_wavlm']*100:.1f}% |"
            )
        
        if len(wavlm_wins) > 10:
            report_lines.append(f"\n*...e altri {len(wavlm_wins)-10} casi*")
    else:
        report_lines.append("*Nessun caso trovato con questi criteri.*")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## 📈 Distribuzione PER",
        "",
        "### Per fasce di score umano",
        "",
        "| Score Range | N | Mean PER HuBERT | Mean PER WavLM | Δ |",
        "|-------------|---|-----------------|----------------|---|",
    ])
    
    for score_range in [(9, 10), (7, 8), (5, 6), (1, 4)]:
        mask = (scores >= score_range[0]) & (scores <= score_range[1])
        if mask.sum() > 0:
            mean_h = pers_h[mask].mean() * 100
            mean_w = pers_w[mask].mean() * 100
            delta = mean_h - mean_w
            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
            report_lines.append(
                f"| {score_range[0]}-{score_range[1]} | {mask.sum()} | {mean_h:.2f}% | {mean_w:.2f}% | {delta_str} |"
            )
    
    report_lines.extend([
        "",
        "---",
        "",
        "## 🔍 Interpretazione per la Tesi",
        "",
        "### Punti chiave:",
        "",
        "1. **HuBERT eccelle nella trascrizione precisa**",
        "   - Pre-training con target discreti (k-means) → rappresentazioni fonetiche pulite",
        "   - Migliore per PER su pronuncie di alta qualità",
        "",
        "2. **WavLM eccelle nella detection di errori**",
        "   - Pre-training con denoising → robustezza a variazioni acustiche",
        "   - Migliore per AUC-ROC su mispronunciation detection",
        "",
        "3. **Complementarità dei modelli**",
        f"   - HuBERT migliore in {stats['hubert_better_count']}/{stats['total_examples']} casi",
        f"   - WavLM migliore in {stats['wavlm_better_count']}/{stats['total_examples']} casi",
        "   - La fusione può sfruttare i punti di forza di entrambi",
        "",
        "### Raccomandazioni:",
        "",
        "- **Per trascrizione fonetica**: usare HuBERT o Late Fusion con α > 0.5",
        "- **Per scoring/detection**: usare WavLM Weighted o Late Fusion con α < 0.5",
        "- **Per applicazioni generali**: Early Fusion o Late Fusion con α = 0.5",
        "",
        "---",
        "",
        f"*Report generato da `analyze_model_gap.py`*",
    ])
    
    # Scrivi report
    output_path = Path(output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"✓ Report salvato: {output_path}")
    
    # ==========================================================================
    # RIEPILOGO CONSOLE
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("📈 RIEPILOGO ANALISI")
    print("=" * 70)
    
    print(f"""
   ┌─────────────────────────────────────────────────────────────────┐
   │ CONFRONTO HuBERT vs WavLM                                      │
   ├─────────────────────────────────────────────────────────────────┤
   │ Esempi analizzati:    {stats['total_examples']:5d}                                    │
   │ HuBERT migliore:      {stats['hubert_better_count']:5d} ({100*stats['hubert_better_count']/stats['total_examples']:.1f}%)                               │
   │ WavLM migliore:       {stats['wavlm_better_count']:5d} ({100*stats['wavlm_better_count']/stats['total_examples']:.1f}%)                               │
   ├─────────────────────────────────────────────────────────────────┤
   │ Casi "HuBERT vince":  {len(hubert_wins):5d}                                    │
   │ Casi "WavLM vince":   {len(wavlm_wins):5d}                                    │
   └─────────────────────────────────────────────────────────────────┘
    """)
    
    print("=" * 70)
    print("✓ Analisi completata!")
    print("=" * 70)
    
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analisi qualitativa HuBERT vs WavLM per Phoneme Recognition"
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
        "--output-report",
        type=str,
        default="docs/qualitative_analysis_report.md",
        help="Path report markdown output"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=500,
        help="Numero massimo esempi da analizzare"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("QUALITATIVE ANALYSIS - HuBERT vs WavLM")
    print("=" * 70)
    
    analyze_model_gap(
        hubert_path=args.model_hubert,
        wavlm_path=args.model_wavlm,
        output_report=args.output_report,
        max_examples=args.max_examples,
    )
    
    print("\n✓ Analisi completata!")


if __name__ == "__main__":
    main()

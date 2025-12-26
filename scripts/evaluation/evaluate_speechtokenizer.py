#!/usr/bin/env python3
"""
Evaluation script per SpeechTokenizer su SpeechOcean762.

Usa SpeechTokenizer per estrarre codici discreti e il classificatore trainato
per predire fonemi IPA.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, Audio
import evaluate
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.normalize_ipa import IPANormalizer


# =============================================================================
# DISCRETE TOKEN CLASSIFIER (from training)
# =============================================================================

class DiscreteTokenClassifier(nn.Module):
    def __init__(self, vocab_size, codebook_size=1024, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 2048, embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids, **kwargs):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.dropout(x)
        logits = self.lm_head(x)
        return {"logits": logits}


# =============================================================================
# SPEECHTOKENIZER WRAPPER
# =============================================================================

class SpeechTokenizerExtractor:
    """Estrae codici discreti da audio usando SpeechTokenizer."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            from speechtokenizer import SpeechTokenizer
            from huggingface_hub import hf_hub_download
            
            # Download if needed
            config_path = hf_hub_download(
                repo_id="fnlp/SpeechTokenizer",
                filename="speechtokenizer_hubert_avg/config.json",
                local_dir="."
            )
            ckpt_path = hf_hub_download(
                repo_id="fnlp/SpeechTokenizer",
                filename="speechtokenizer_hubert_avg/SpeechTokenizer.pt",
                local_dir="."
            )
            
            self.model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("   ✓ SpeechTokenizer encoder caricato")
            
        except ImportError:
            raise ImportError("SpeechTokenizer non installato! pip install speechtokenizer")
    
    @torch.no_grad()
    def extract_codes(self, audio: np.ndarray) -> np.ndarray:
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        codes = self.model.encode(audio_tensor)
        return codes[0, 0, :].cpu().numpy()


# =============================================================================
# ARPABET TO IPA - Use centralized module
# =============================================================================

from src.data.normalize_ipa import arpa_to_ipa

def extract_phones_from_words(words_list: list) -> str:
    """
    Estrae e converte tutti i fonemi dalla lista di parole.
    Usa la mappatura ARPA→IPA corretta.
    """
    all_phones_ipa = []
    for word_info in words_list:
        phones = word_info.get("phones", [])
        for p in phones:
            ipa = arpa_to_ipa(p, use_corrected=True)
            if ipa:
                all_phones_ipa.append(ipa)
    return "".join(all_phones_ipa)


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_speechtokenizer(model_path: str, device: str = "cuda"):
    """
    Valuta SpeechTokenizer su SpeechOcean762.
    """
    print("=" * 70)
    print("🔬 BENCHMARK SPEECHTOKENIZER - SPEECHOCEAN762")
    print("=" * 70)
    
    normalizer = IPANormalizer(mode='strict')
    
    # Load config
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    vocab_size = config.get("vocab_size", 100)
    codebook_size = config.get("codebook_size", 1024)
    embed_dim = config.get("embed_dim", 256)
    num_heads = config.get("num_heads", 4)
    num_layers = config.get("num_layers", 2)
    
    print(f"\n📋 Configurazione:")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Codebook size: {codebook_size}")
    
    # Load tokenizer for decoding
    from transformers import Wav2Vec2CTCTokenizer
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
    
    print("\n📦 Caricamento modelli...")
    
    # 1. Load SpeechTokenizer encoder
    speech_tokenizer = SpeechTokenizerExtractor(device=device)
    
    # 2. Load classifier
    classifier = DiscreteTokenClassifier(vocab_size, codebook_size, embed_dim, num_heads, num_layers)
    model_file = Path(model_path) / "pytorch_model.bin"
    state_dict = torch.load(model_file, map_location="cpu")
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(device)
    classifier.eval()
    print("   ✓ Classificatore caricato")
    
    # Load dataset
    print("\n📥 Scaricamento SpeechOcean762...")
    ds = load_dataset("mispeech/speechocean762", split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    print(f"   ✓ Caricati {len(ds)} esempi")
    
    # Prepare references
    def prepare_example(example):
        example["reference_ipa"] = extract_phones_from_words(example["words"])
        return example
    
    ds = ds.map(prepare_example)
    ds = ds.filter(lambda x: len(x["reference_ipa"]) > 0)
    print(f"   ✓ Esempi validi: {len(ds)}")
    
    # Evaluate
    print("\n🔄 Esecuzione inferenza...")
    
    results = []
    cer_metric = evaluate.load("cer")
    
    for i, example in enumerate(ds):
        if i % 100 == 0:
            print(f"   Processing {i}/{len(ds)}...")
        
        try:
            audio = example["audio"]["array"]
            
            # 1. Extract discrete codes
            codes = speech_tokenizer.extract_codes(audio)
            
            # 2. Run classifier
            input_ids = torch.tensor(codes, dtype=torch.long).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = classifier(input_ids)
                logits = outputs["logits"]
                pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
            # 3. Decode
            pred_str = tokenizer.decode(pred_ids, skip_special_tokens=True)
            pred_ipa = normalizer.normalize(pred_str)
            ref_ipa = normalizer.normalize(example["reference_ipa"])
            
            # 4. Compute CER
            if ref_ipa:
                per = cer_metric.compute(predictions=[pred_ipa], references=[ref_ipa])
            else:
                per = 1.0
            
            results.append({
                "score": example["accuracy"],
                "per": per,
                "ref": ref_ipa,
                "pred": pred_ipa,
            })
            
        except Exception as e:
            if i < 5:
                print(f"   ⚠️ Errore esempio {i}: {e}")
            continue
    
    print(f"\n   ✓ Valutati {len(results)} esempi")
    
    # Compute metrics
    scores = np.array([r["score"] for r in results])
    pers = np.array([r["per"] for r in results])
    
    # Task A: High quality
    hq_mask = scores >= 8
    hq_per = np.mean(pers[hq_mask]) * 100
    hq_acc = 100 - hq_per
    
    # Task B: Correlation
    pearson_r, pearson_p = stats.pearsonr(1 - pers, scores)
    spearman_r, spearman_p = stats.spearmanr(1 - pers, scores)
    
    # Task C: Detection
    y_true = (scores <= 6).astype(int)
    
    best_f1 = 0
    best_thresh = 0.2
    for thresh in np.arange(0.1, 0.5, 0.02):
        y_pred = (pers >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    y_pred = (pers >= best_thresh).astype(int)
    auc = roc_auc_score(y_true, pers)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 RISULTATI SPEECHTOKENIZER")
    print("=" * 70)
    
    print(f"\n   TASK A - ASR (High Quality, score >= 8):")
    print(f"   ─────────────────────────────────────")
    print(f"   PER:      {hq_per:.2f}%")
    print(f"   Accuracy: {hq_acc:.2f}%")
    
    print(f"\n   TASK B - Scoring Correlation:")
    print(f"   ─────────────────────────────")
    print(f"   Pearson:  {pearson_r:.4f}")
    print(f"   Spearman: {spearman_r:.4f}")
    
    print(f"\n   TASK C - Detection (threshold={best_thresh:.2f}):")
    print(f"   ──────────────────────────────────────")
    print(f"   AUC-ROC:   {auc:.4f}")
    print(f"   F1-Score:  {best_f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Benchmark SpeechTokenizer completato!")
    print("=" * 70)
    
    # Print command to add to Excel
    print(f"\n💡 Comando per aggiungere all'Excel:")
    print(f'python scripts/evaluation/track_benchmark.py --model_name "SpeechTokenizer" '
          f'--architecture "SpeechTokenizer" --training_data "Aug+Comb" '
          f'--per {hq_per:.2f} --accuracy {hq_acc:.2f} '
          f'--pearson {pearson_r:.4f} --spearman {spearman_r:.4f} '
          f'--auc {auc:.4f} --f1 {best_f1:.4f} --recall {recall:.4f} --precision {precision:.4f} '
          f'--threshold {best_thresh:.2f} --notes "Discrete RVQ codes + Transformer classifier"')


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpeechTokenizer on SpeechOcean762")
    parser.add_argument("--model-path", type=str, required=True, help="Path to SpeechTokenizer model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    evaluate_speechtokenizer(args.model_path, device=device)


if __name__ == "__main__":
    main()

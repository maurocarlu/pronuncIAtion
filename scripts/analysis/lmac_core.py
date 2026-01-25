"""
Core utilities for L-MAC (Listenable Maps for Audio Classifiers).

Implements:
- LMACSpeechOceanDataset (reuses evaluate_speechocean pipeline)
- LMACDecoder (light 1D U-Net)
- LMACWrapper (backbone + decoder + loss)
- generate_listenable_map and AI/AD metrics
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Audio, load_dataset

# Add project root to sys.path (for direct script execution)
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.normalize_ipa import IPANormalizer

# Reuse SpeechOcean decoding utilities from evaluate_speechocean
from scripts.evaluation.evaluate_speechocean import (
    _decode_audio_to_16k,
    extract_phones_from_words,
)


# =============================================================================
# DATASET (SpeechOcean762) - same pipeline as evaluate_speechocean.py
# =============================================================================

class LMACSpeechOceanDataset(torch.utils.data.Dataset):
    """SpeechOcean762 dataset wrapper with consistent audio decoding.

    Uses the same pipeline as scripts/evaluation/evaluate_speechocean.py.
    """

    def __init__(
        self,
        split: str = "train",
        target_phoneme: Optional[str] = None,
        full: bool = True,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.split = split
        self.target_phoneme = target_phoneme
        self.full = full
        self.max_samples = max_samples
        self.rng = random.Random(seed)
        self.normalizer = IPANormalizer(mode="strict")

        if not full:
            raise ValueError("LMACSpeechOceanDataset supporta solo full=True (richiesta utente)")

        hf_ds = load_dataset("mispeech/speechocean762", split=split)
        hf_ds = hf_ds.cast_column("audio", Audio(sampling_rate=16000))

        def prepare_example(ex: Dict[str, Any]) -> Dict[str, Any]:
            ref_ipa = extract_phones_from_words(ex.get("words", []))
            ex["reference_ipa"] = ref_ipa
            return ex

        hf_ds = hf_ds.map(prepare_example)
        hf_ds = hf_ds.filter(lambda x: len(x.get("reference_ipa", "")) > 0)

        if max_samples is not None:
            max_samples = int(max_samples)
            hf_ds = hf_ds.select(range(min(max_samples, len(hf_ds))))

        self.ds = hf_ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.ds[int(idx)]
        audio_data = ex.get("audio")
        audio_arr, sr = _decode_audio_to_16k(audio_data, target_sr=16000)

        reference_ipa = self.normalizer.normalize(ex.get("reference_ipa", ""))
        if not reference_ipa:
            reference_ipa = ex.get("reference_ipa", "")

        # Target phoneme selection (single)
        if self.target_phoneme is not None:
            target_ph = self.target_phoneme
        else:
            target_ph = self._random_phoneme(reference_ipa)

        return {
            "audio": audio_arr,
            "sampling_rate": sr,
            "reference_ipa": reference_ipa,
            "target_phoneme": target_ph,
            "text": ex.get("text", ""),
            "age": ex.get("age", None),
            "words": ex.get("words", []),
        }

    def _random_phoneme(self, reference_ipa: str) -> str:
        if not reference_ipa:
            return ""
        # Sample a single IPA symbol (best-effort):
        # split by whitespace if present, else sample character-wise
        parts = reference_ipa.split()
        if len(parts) > 1:
            return self.rng.choice(parts)
        return self.rng.choice(list(reference_ipa))


# =============================================================================
# COLLATE
# =============================================================================


def collate_audio_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    audio_list = [b["audio"] for b in batch]
    target_ph = [b["target_phoneme"] for b in batch]

    lengths = [len(a) for a in audio_list]
    max_len = max(lengths)

    padded = []
    attn = []
    for a, l in zip(audio_list, lengths):
        pad_len = max_len - l
        if pad_len > 0:
            a = np.pad(a, (0, pad_len), mode="constant")
        padded.append(a)
        mask = np.zeros(max_len, dtype=np.int64)
        mask[:l] = 1
        attn.append(mask)

    return {
        "input_values": torch.tensor(np.stack(padded), dtype=torch.float32),
        "attention_mask": torch.tensor(np.stack(attn), dtype=torch.long),
        "target_phoneme": target_ph,
        "reference_ipa": [b.get("reference_ipa", "") for b in batch],
        "text": [b.get("text", "") for b in batch],
    }


# =============================================================================
# DECODER (Light U-Net 1D)
# =============================================================================

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        padding = k // 2
        groups = 8 if out_ch >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=padding),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=padding),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LMACDecoder(nn.Module):
    """Lightweight 1D U-Net decoder for mask prediction."""

    def __init__(
        self,
        in_ch: int,
        base_ch: int = 128,
        use_conditioning: bool = False,
        vocab_size: int = 0,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.use_conditioning = use_conditioning
        
        if use_conditioning:
            if vocab_size <= 0:
                raise ValueError("vocab_size must be > 0 when use_conditioning=True")
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            in_ch += embedding_dim
            
        self.enc1 = _ConvBlock(in_ch, base_ch)
        self.down1 = nn.Conv1d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = _ConvBlock(base_ch * 2, base_ch * 2)
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1)
        self.enc3 = _ConvBlock(base_ch * 4, base_ch * 4)

        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = _ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = _ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv1d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, target_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, C, T]
        if self.use_conditioning:
            if target_ids is None:
                raise ValueError("target_ids required for conditional decoder")
            
            # Embed target phoneme: [B] -> [B, emb_dim]
            emb = self.embedding(target_ids)
            # Expand to time dimension: [B, emb_dim, T]
            emb = emb.unsqueeze(-1).expand(-1, -1, x.size(-1))
            # Concatenate along channel dimension
            x = torch.cat([x, emb], dim=1)
            
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        e3 = self.enc3(d2)

        u2 = self.up2(e3)
        if u2.size(-1) != e2.size(-1):
            u2 = F.interpolate(u2, size=e2.size(-1), mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.size(-1) != e1.size(-1):
            u1 = F.interpolate(u1, size=e1.size(-1), mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        m = self.out_conv(d1)
        return torch.sigmoid(m)


# =============================================================================
# BACKBONE HELPERS
# =============================================================================

@dataclass
class LMACBackboneConfig:
    backbone_type: str
    model_path: str
    hubert_name: str = "facebook/hubert-large-ls960-ft"
    wavlm_name: str = "microsoft/wavlm-large"
    use_weighted_wavlm: bool = True
    layer_ids: Tuple[int, ...] = (6, 12, 18, 24)
    # Late Fusion params
    fusion_weights: Optional[List[float]] = None
    ensemble_models: Optional[List[str]] = None  # List of paths/names
    use_conditioning: bool = False
    vocab_size: int = 0  # Required if use_conditioning=True
    embedding_dim: int = 64


class _LMACEarlyFusionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hubert_name: str,
        wavlm_name: str,
        use_weighted_wavlm: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel

        self.hubert = AutoModel.from_pretrained(
            hubert_name,
            output_hidden_states=True,
        )
        self.wavlm = AutoModel.from_pretrained(
            wavlm_name,
            output_hidden_states=True,
        )
        if hasattr(self.hubert, "freeze_feature_encoder"):
            self.hubert.freeze_feature_encoder()
        if hasattr(self.wavlm, "freeze_feature_encoder"):
            self.wavlm.freeze_feature_encoder()
        self.use_weighted = use_weighted_wavlm
        if use_weighted_wavlm:
            num_layers = int(self.wavlm.config.num_hidden_layers) + 1
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))

        hidden_h = self.hubert.config.hidden_size
        hidden_w = self.wavlm.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.ctc_head = nn.Linear(hidden_h + hidden_w, vocab_size)

    def _get_wavlm_weighted_output(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        weights = F.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(hidden_states, dim=0)
        weights_view = weights.view(-1, 1, 1, 1)
        return (stacked * weights_view).sum(dim=0)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
    ) -> Dict[str, Any]:
        h_out = self.hubert(input_values, attention_mask=attention_mask, output_hidden_states=True)
        w_out = self.wavlm(input_values, attention_mask=attention_mask, output_hidden_states=True)

        h_h = h_out.last_hidden_state
        if self.use_weighted:
            h_w = self._get_wavlm_weighted_output(w_out.hidden_states)
        else:
            h_w = w_out.last_hidden_state

        min_len = min(h_h.size(1), h_w.size(1))
        h_h = h_h[:, :min_len, :]
        h_w = h_w[:, :min_len, :]
        combined = torch.cat([h_h, h_w], dim=-1)
        logits = self.ctc_head(self.dropout(combined))

        return {
            "logits": logits,
            "hubert_hidden_states": h_out.hidden_states,
            "wavlm_hidden_states": w_out.hidden_states,
            "last_hidden_state": combined,
        }


class _LMACLateFusionModel(nn.Module):
    """Ensemble of models for Late Fusion.
    
    Uses features from the 'dominant' model (highest weight) for the L-MAC decoder,
    but computes loss against the weighted ensemble logits.
    """
    def __init__(
        self,
        model_paths: List[str],
        weights: List[float],
        device: str = "cpu"
    ):
        super().__init__()
        self.models = nn.ModuleList()
        self.weights = weights
        self.model_paths = model_paths
        
        # Determine dominant model for feature extraction
        import numpy as np
        self.dominant_idx = int(np.argmax(weights))
        print(f"🧩 Late Fusion: Dominant model for features is #{self.dominant_idx} ({model_paths[self.dominant_idx]})")
        
        from transformers import AutoModel, HubertForCTC, Wav2Vec2ForCTC
        
        for path in model_paths:
            # Check if local path: exists OR looks like an absolute path
            # This prevents HF from validating it as a Repo ID if the file is missing but meant to be local
            is_local = os.path.exists(path) or path.startswith("/") or path.startswith("\\")
            kwargs = {"local_files_only": True} if is_local else {}
            
            # Heuristic loading: try CTC, else generic
            try:
                # Assuming most are Hubert/WavLM and have the standard config
                # We can try loading config first to see architecture
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(path, **kwargs)
                archs = getattr(cfg, "architectures", [])
                
                if archs and "ForCTC" in archs[0]:
                   if "Hubert" in archs[0]:
                       m = HubertForCTC.from_pretrained(path, **kwargs)
                   elif "WavLM" in archs[0]:
                       from transformers import WavLMForCTC
                       m = WavLMForCTC.from_pretrained(path, **kwargs)
                   else:
                       m = AutoModel.from_pretrained(path, **kwargs) # Fallback
                elif "hubert" in str(path).lower():
                    m = HubertForCTC.from_pretrained(path, **kwargs)
                elif "wavlm" in str(path).lower():
                    from transformers import WavLMForCTC
                    m = WavLMForCTC.from_pretrained(path, **kwargs)
                else:
                    m = AutoModel.from_pretrained(path, **kwargs) # Fallback
            except Exception as e:
                print(f"⚠️ Error loading {path}, trying AutoModel directly (local={is_local}): {e}")
                try:
                     m = AutoModel.from_pretrained(path, **kwargs)
                except Exception as e2:
                     # Final fallback: only retry if not local (to avoid HF validation error on paths)
                     if not is_local:
                         print(f"⚠️ AutoModel fallback failed: {e2}. Retrying without restrictions...")
                         m = AutoModel.from_pretrained(path)
                     else:
                         print(f"❌ Failed to load local model at {path}. Check if path exists and contains config.json")
                         raise e2
            
            if hasattr(m, "freeze_feature_encoder"):
                m.freeze_feature_encoder()
            m.eval()
            self.models.append(m)
            
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        return_dict: bool =True
    ) -> Any:
        # 1. Forward all models to get logits
        all_logits = []
        dominant_hidden = None
        
        for i, model in enumerate(self.models):
            # Move individual models to device if needed (or assume wrapper handles it)
            # model.to(input_values.device) 
            
            out = model(input_values, attention_mask=attention_mask, output_hidden_states=True)
            logits = out.logits if hasattr(out, "logits") else out["logits"]
            all_logits.append(logits)
            
            if i == self.dominant_idx:
                dominant_hidden = out.hidden_states if hasattr(out, "hidden_states") else out["hidden_states"]
                
        # 2. Weighted average of logits
        # Stack: [N_models, B, T, Vocab]
        # Check shapes consistency (time dim might vary slightly due to convs?)
        # For now assume consistent architecture (Hubert/WavLM Large)
        
        # Weighted sum
        avg_logits = torch.zeros_like(all_logits[0])
        total_w = sum(self.weights)
        
        for logits, w in zip(all_logits, self.weights):
             avg_logits += w * logits
        
        if total_w != 1.0 and total_w > 0:
            avg_logits /= total_w
            
        # Return object similar to HF output
        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(
            loss=None,
            logits=avg_logits,
            hidden_states=dominant_hidden, # Only dominant features
            attentions=None,
        )


# =============================================================================
# LMAC WRAPPER
# =============================================================================

class LMACWrapper(nn.Module):
    """Wrapper for L-MAC: frozen backbone + trainable decoder + loss."""

    def __init__(
        self,
        config: LMACBackboneConfig,
        target_phoneme: Optional[str] = None,  # None = multi-phoneme mode
        device: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.target_phoneme = target_phoneme
        self.layer_ids = config.layer_ids

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone_type = config.backbone_type
        self._init_backbone()
        self._init_tokenizer()

        in_ch = self._compute_decoder_in_channels()
        self.decoder = LMACDecoder(
            in_ch=in_ch,
            use_conditioning=config.use_conditioning,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
        ).to(self.device)

        for p in self.backbone.parameters():
            p.requires_grad = False

    def _init_tokenizer(self) -> None:
        from transformers import Wav2Vec2CTCTokenizer

        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.config.model_path)
        self.vocab = self.tokenizer.get_vocab()
        
        # Multi-phoneme mode: target_id will be set dynamically per batch
        if self.target_phoneme is None:
            self.target_id = None  # Will be set in compute_loss
        else:
            if self.target_phoneme not in self.vocab:
                raise ValueError(f"Target phoneme '{self.target_phoneme}' non presente nel vocab.")
            self.target_id = self.vocab[self.target_phoneme]

    def _init_backbone(self) -> None:
        if self.backbone_type == "hubert":
            from transformers import HubertForCTC

            self.backbone = HubertForCTC.from_pretrained(
                self.config.model_path,
                output_hidden_states=True,
            ).to(self.device)
            if hasattr(self.backbone, "freeze_feature_encoder"):
                self.backbone.freeze_feature_encoder()
            self.backbone.eval()
        elif self.backbone_type == "early_fusion":
            cfg_path = Path(self.config.model_path) / "config.json"
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                hubert_name = cfg.get("hubert_name", self.config.hubert_name)
                wavlm_name = cfg.get("wavlm_name", self.config.wavlm_name)
                use_weighted = cfg.get("use_weighted_wavlm", True)
                vocab_size = int(cfg.get("vocab_size", 43))
            else:
                hubert_name = self.config.hubert_name
                wavlm_name = self.config.wavlm_name
                use_weighted = True
                vocab_size = 43

            self.backbone = _LMACEarlyFusionModel(
                vocab_size=vocab_size,
                hubert_name=hubert_name,
                wavlm_name=wavlm_name,
                use_weighted_wavlm=use_weighted,
            ).to(self.device)
            self._load_early_fusion_weights(self.config.model_path)
            self.backbone.eval()
        elif self.backbone_type == "late_fusion":
            # Late Fusion support
            if not self.config.fusion_weights or not self.config.ensemble_models:
                raise ValueError("Late Fusion requires fusion_weights and ensemble_models in config")
                
            self.backbone = _LMACLateFusionModel(
                model_paths=self.config.ensemble_models,
                weights=self.config.fusion_weights,
                device=self.device
            ).to(self.device)
            self.backbone.eval()
        else:
            raise ValueError(f"backbone_type non supportato: {self.backbone_type}")

    def _load_early_fusion_weights(self, model_path: str) -> None:
        model_dir = Path(model_path)
        possible_files = [
            model_dir / "pytorch_model.bin",
            model_dir / "model.safetensors",
            model_dir / "model.bin",
        ]
        model_file = None
        for pf in possible_files:
            if pf.exists():
                model_file = pf
                break
        if model_file is None:
            return

        state_dict = None
        if model_file.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(model_file))
            except Exception:
                state_dict = None
        if state_dict is None:
            state_dict = torch.load(model_file, map_location="cpu", weights_only=False)

        model_state = self.backbone.state_dict()
        for key in list(state_dict.keys()):
            if key in model_state:
                model_state[key] = state_dict[key]
        self.backbone.load_state_dict(model_state, strict=False)

    def _compute_decoder_in_channels(self) -> int:
        if self.backbone_type == "hubert":
            hidden = int(self.backbone.config.hidden_size)
            return hidden * len(self.layer_ids)
        if self.backbone_type == "late_fusion":
            # Uses dominant model config
            dom_model = self.backbone.models[self.backbone.dominant_idx]
            hidden = int(dom_model.config.hidden_size)
            return hidden * len(self.layer_ids)
        if self.backbone_type == "early_fusion":
            hidden_h = int(self.backbone.hubert.config.hidden_size)
            hidden_w = int(self.backbone.wavlm.config.hidden_size)
            return (hidden_h + hidden_w) * len(self.layer_ids)
        raise ValueError("Unsupported backbone type")

    def _select_layers(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        layers = []
        total = len(hidden_states)
        for idx in self.layer_ids:
            if idx < 0:
                idx = total + idx
            if idx >= total:
                raise ValueError(f"Layer {idx} fuori range (totale {total})")
            layers.append(hidden_states[idx])
        return torch.cat(layers, dim=-1)

    def _feat_mask_from_attention(self, attn_mask: torch.Tensor, feat_len: int) -> torch.Tensor:
        # attn_mask: [B, T_wave]
        mask = attn_mask.float().unsqueeze(1)  # [B,1,T]
        mask = F.interpolate(mask, size=feat_len, mode="nearest")
        return mask.squeeze(1)  # [B, T_feat]

    def forward(
        self, 
        input_values: torch.Tensor, 
        attention_mask: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        input_values = input_values.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self.backbone_type == "hubert":
            outputs = self.backbone(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            feats = self._select_layers(hidden_states)
            logits_clean = outputs.logits
        else:
            outputs = self.backbone(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            feats_h = self._select_layers(outputs["hubert_hidden_states"])
            feats_w = self._select_layers(outputs["wavlm_hidden_states"])
            # Align temporal length
            min_len = min(feats_h.size(1), feats_w.size(1))
            feats = torch.cat([feats_h[:, :min_len, :], feats_w[:, :min_len, :]], dim=-1)
            logits_clean = outputs["logits"]

        # Decoder expects [B, C, T_feat]
        feats = feats.transpose(1, 2)
        
        # Prepare target_ids for conditioning if needed
        decoder_target_ids = None
        if self.config.use_conditioning:
            if target_ids is not None:
                decoder_target_ids = target_ids
            elif self.target_id is not None:
                # Use fixed target_id for all samples if available (single-phoneme mode or inference)
                decoder_target_ids = torch.full(
                    (feats.size(0),), 
                    self.target_id, 
                    device=self.device, 
                    dtype=torch.long
                )
            else:
                 raise ValueError("Decoder requires conditioning but no target_ids provided and self.target_id is None")

        m_feat = self.decoder(feats, target_ids=decoder_target_ids)  # [B,1,T_feat]

        # Upsample mask to waveform length
        mask = F.interpolate(m_feat, size=input_values.size(1), mode="linear", align_corners=False)
        mask = mask.squeeze(1)
        mask = mask * attention_mask.float()

        return {
            "mask": mask,
            "logits_clean": logits_clean,
        }

    def compute_loss(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        target_phonemes: Optional[List[str]] = None,  # Per multi-phoneme mode
        lambda_out: float = 1.0,
        lambda_reg: float = 1e-4,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        # Resolve target IDs for conditioning (if enabled)
        target_ids = None
        if self.config.use_conditioning:
            if target_phonemes is not None:
                # Convert list of phonemes to tensor of IDs
                tids = []
                for ph in target_phonemes:
                    tids.append(self.vocab.get(ph, 0)) # 0 as fallback
                target_ids = torch.tensor(tids, device=self.device, dtype=torch.long)
            elif self.target_id is not None:
                # Single-phoneme mode
                target_ids = torch.full(
                    (input_values.size(0),), 
                    self.target_id, 
                    device=self.device, 
                    dtype=torch.long
                )
        
        out = self.forward(input_values, attention_mask, target_ids=target_ids)
        mask = out["mask"]

        masked_audio = input_values.to(self.device) * mask
        removed_audio = input_values.to(self.device) * (1.0 - mask)

        # Forward for fidelity losses (no grad on params, but grad wrt mask/input)
        attention_mask_device = attention_mask.to(self.device)
        if self.backbone_type == "hubert":
            logits_in = self.backbone(masked_audio, attention_mask=attention_mask_device, return_dict=True).logits
            logits_out = self.backbone(removed_audio, attention_mask=attention_mask_device, return_dict=True).logits
        else:
            logits_in = self.backbone(masked_audio, attention_mask=attention_mask_device)["logits"]
            logits_out = self.backbone(removed_audio, attention_mask=attention_mask_device)["logits"]

        # Average over time with mask
        feat_mask = self._feat_mask_from_attention(attention_mask, logits_in.size(1))
        feat_mask = feat_mask.to(logits_in.device)

        probs_in = F.softmax(logits_in, dim=-1)
        probs_out = F.softmax(logits_out, dim=-1)

        # Multi-phoneme mode: compute loss per-sample with different targets
        if target_phonemes is not None:
            # Get target IDs for each sample in batch
            batch_size = probs_in.size(0)
            p_in_list = []
            p_out_list = []
            for i, ph in enumerate(target_phonemes):
                tid = self.vocab.get(ph, 0)  # Default to 0 if not found
                p_in_list.append(probs_in[i, :, tid])
                p_out_list.append(probs_out[i, :, tid])
            p_in = torch.stack(p_in_list, dim=0)  # [B, T]
            p_out = torch.stack(p_out_list, dim=0)
        else:
            # Single phoneme mode
            p_in = probs_in[..., self.target_id]
            p_out = probs_out[..., self.target_id]

        p_in_mean = (p_in * feat_mask).sum(dim=1) / (feat_mask.sum(dim=1) + eps)
        p_out_mean = (p_out * feat_mask).sum(dim=1) / (feat_mask.sum(dim=1) + eps)

        # L_fid_in: -log prob target on masked audio
        loss_fid_in = (-torch.log(p_in_mean + eps)).mean()
        # L_fid_out: minimize target prob on removed audio
        loss_fid_out = p_out_mean.mean()

        # L1 sparsity on mask
        valid = attention_mask.to(self.device).float()
        loss_reg = (mask.abs() * valid).sum() / (valid.sum() + eps)

        total = loss_fid_in + lambda_out * loss_fid_out + lambda_reg * loss_reg

        return {
            "loss": total,
            "loss_fid_in": loss_fid_in.detach(),
            "loss_fid_out": loss_fid_out.detach(),
            "loss_reg": loss_reg.detach(),
        }


# =============================================================================
# LISTENABLE MAP GENERATION
# =============================================================================


def generate_listenable_map(
    wrapper: LMACWrapper,
    audio_path: str,
    out_dir: str,
    prefix: str = "lmac",
    target_phoneme: Optional[str] = None,
) -> Dict[str, str]:
    import soundfile as sf
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_arr, sr = _decode_audio_to_16k({"path": audio_path}, target_sr=16000)
    input_values = torch.tensor(audio_arr[None, :], dtype=torch.float32)
    attn_mask = torch.ones_like(input_values, dtype=torch.long)

    # Resolve target_ids if conditional
    target_ids = None
    if wrapper.config.use_conditioning:
        curr_target = target_phoneme if target_phoneme else wrapper.target_phoneme
        
        # If still None, try to infer from metadata if available (requires decoding wrapper or similar context, 
        # but here we only have audio_path. We can't easily get reference IPA without the dataset item.
        # However, for consistency with notebook, we can try to rely on caller passing it.
        # But to prevent crash if simple usage:
        if curr_target:
             tid = wrapper.vocab.get(curr_target, 0)
             target_ids = torch.tensor([tid], dtype=torch.long, device=wrapper.device)
        else:
             # Try to find reference IPA from dataset if available or warn
             # Since we don't have access to dataset here easily without loading it, 
             # we will default to ID 0 with a warning, or fail if we want to be strict.
             # Given user feedback, let's try to be helpful:
             # If reference_ipa arg was passed (it isn't currently), we could use it.
             # For now, default to 0 to allow execution, but print warning.
             print("⚠️ Warning: No target_phoneme provided for conditional L-MAC. Using generic ID 0.")
             target_ids = torch.tensor([0], dtype=torch.long, device=wrapper.device)


    with torch.no_grad():
        out = wrapper.forward(input_values, attn_mask, target_ids=target_ids)
    mask = out["mask"].cpu().numpy()[0]

    masked_audio = audio_arr * mask

    audio_out = out_dir / f"{prefix}_masked.wav"
    sf.write(audio_out, masked_audio, 16000)

    # Plot waveform, mask, spectrogram
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(audio_arr, color="#2c3e50")
    ax1.set_title("Waveform originale")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.imshow(mask[None, :], aspect="auto", cmap="magma")
    
    label_target = target_phoneme if target_phoneme else (wrapper.target_phoneme or "Unknown")
    ax2.set_title(f"Maschera L-MAC (Target: /{label_target}/)")

    ax3 = fig.add_subplot(3, 1, 3)
    try:
        import librosa
        import librosa.display

        S = np.abs(librosa.stft(audio_arr, n_fft=512, hop_length=160))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=16000, hop_length=160, x_axis="time", y_axis="linear", ax=ax3)
    except Exception:
        # Fallback simple spectrogram
        from scipy.signal import spectrogram

        f, t, Sxx = spectrogram(audio_arr, fs=16000)
        ax3.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-9), shading="gouraud")

    ax3.set_title("Spettrogramma")
    plt.tight_layout()

    plot_path = out_dir / f"{prefix}_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "masked_audio": str(audio_out),
        "plot": str(plot_path),
    }


# =============================================================================
# AI / AD METRICS
# =============================================================================


def compute_ai_ad(
    wrapper: LMACWrapper,
    dataloader: torch.utils.data.DataLoader,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute Average Increase (AI) and Average Drop (AD) on test set."""
    wrapper.eval()
    
    # Global stats
    ai_hits = 0
    n = 0
    ad_sum = 0.0
    
    # Per-phoneme stats
    phoneme_stats = {}  # ph -> {'ai_hits': 0, 'ad_sum': 0.0, 'n': 0}
    
    is_multi_phoneme = wrapper.target_id is None

    for b_idx, batch in enumerate(dataloader):
        if max_batches is not None and b_idx >= max_batches:
            break

        input_values = batch["input_values"].to(wrapper.device)
        attention_mask = batch["attention_mask"].to(wrapper.device)

        # 1. Prepare target_ids for conditional forward
        target_ids = None
        batch_target_phonemes = []
        
        if is_multi_phoneme:
            batch_target_phonemes = batch.get("target_phoneme", [])
            if wrapper.config.use_conditioning:
                tids = [wrapper.vocab.get(ph, 0) for ph in batch_target_phonemes]
                target_ids = torch.tensor(tids, device=wrapper.device, dtype=torch.long)
        else:
            # Single-phoneme mode
            batch_target_phonemes = [wrapper.target_phoneme] * input_values.size(0)
            if wrapper.config.use_conditioning:
                target_ids = torch.full((input_values.size(0),), wrapper.target_id, device=wrapper.device, dtype=torch.long)

        # 2. Forward pass to get mask and logits
        with torch.no_grad():
            out = wrapper.forward(input_values, attention_mask, target_ids=target_ids)
            mask = out["mask"]

            # clean logits
            if wrapper.backbone_type == "hubert":
                logits_clean = wrapper.backbone(input_values, attention_mask=attention_mask, return_dict=True).logits
            else:
                logits_clean = wrapper.backbone(input_values, attention_mask=attention_mask)["logits"]

            masked_audio = input_values * mask
            if wrapper.backbone_type == "hubert":
                logits_masked = wrapper.backbone(masked_audio, attention_mask=attention_mask, return_dict=True).logits
            else:
                logits_masked = wrapper.backbone(masked_audio, attention_mask=attention_mask)["logits"]

        # 3. Compute probabilities for the target phonemes
        feat_mask = wrapper._feat_mask_from_attention(attention_mask, logits_clean.size(1)).to(logits_clean.device)
        
        if is_multi_phoneme:
            probs_clean_list = []
            probs_masked_list = []
            for i, ph in enumerate(batch_target_phonemes):
                tid = wrapper.vocab.get(ph, 0)
                probs_clean_list.append(F.softmax(logits_clean[i], dim=-1)[:, tid])
                probs_masked_list.append(F.softmax(logits_masked[i], dim=-1)[:, tid])
            probs_clean = torch.stack(probs_clean_list, dim=0)
            probs_masked = torch.stack(probs_masked_list, dim=0)
        else:
            probs_clean = F.softmax(logits_clean, dim=-1)[..., wrapper.target_id]
            probs_masked = F.softmax(logits_masked, dim=-1)[..., wrapper.target_id]

        p_clean = (probs_clean * feat_mask).sum(dim=1) / (feat_mask.sum(dim=1) + 1e-8)
        p_masked = (probs_masked * feat_mask).sum(dim=1) / (feat_mask.sum(dim=1) + 1e-8)

        # Update global stats
        hits = (p_masked > p_clean).float()
        ad = torch.clamp(p_clean - p_masked, min=0.0) / (p_clean + 1e-8)
        
        ai_hits += hits.sum().item()
        ad_sum += ad.sum().item()
        n += p_clean.numel()
        
        # Update per-phoneme stats
        for i, ph in enumerate(batch_target_phonemes):
            if ph not in phoneme_stats:
                phoneme_stats[ph] = {'ai_hits': 0, 'ad_sum': 0.0, 'n': 0}
            
            phoneme_stats[ph]['ai_hits'] += hits[i].item()
            phoneme_stats[ph]['ad_sum'] += ad[i].item()
            phoneme_stats[ph]['n'] += 1

    if n == 0:
        return {"AI": 0.0, "AD": 0.0}
        
    results = {
        "AI": 100.0 * ai_hits / n,
        "AD": ad_sum / n,
        "per_phoneme": {}
    }
    
    # Calculate aggregated per-phoneme metrics
    for ph, stats in phoneme_stats.items():
        if stats['n'] > 0:
            results["per_phoneme"][ph] = {
                "AI": 100.0 * stats['ai_hits'] / stats['n'],
                "AD": stats['ad_sum'] / stats['n'],
                "count": stats['n']
            }
            
    return results

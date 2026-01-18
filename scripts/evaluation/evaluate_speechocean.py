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
import traceback
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, Audio

# Aggiungi project root al path (2 livelli su da scripts/evaluation/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Importa il modulo di normalizzazione centralizzato
from src.data.normalize_ipa import (
    IPANormalizer,
    normalize_for_evaluation,
    arpa_to_ipa,
    get_corrected_arpa_to_ipa_mapping,
)


def _to_mono_float32(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2:
        # Heuristic: handle both (channels, time) and (time, channels)
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            out = arr.mean(axis=0)
        elif arr.shape[1] <= 8:
            out = arr.mean(axis=1)
        else:
            out = arr.reshape(-1)
    else:
        out = arr.reshape(-1)

    if out.dtype != np.float32:
        out = out.astype(np.float32)
    return out


def _resample_1d(arr: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return arr
    # Prefer librosa if present (consistent with other scripts), else scipy.
    try:
        import librosa

        return librosa.resample(arr, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except Exception:
        pass

    try:
        from scipy.signal import resample_poly
        import math

        g = math.gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // g
        down = int(orig_sr) // g
        return resample_poly(arr, up=up, down=down).astype(np.float32)
    except Exception:
        # Very last resort: linear interpolation
        ratio = float(target_sr) / float(orig_sr)
        n = int(max(1, round(len(arr) * ratio)))
        x_old = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        return np.interp(x_new, x_old, arr).astype(np.float32)


def _decode_audio_to_16k(audio_data: Any, *, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Decode HF datasets audio payload into a mono float32 numpy array.

    Supports:
    - dict: {"array", "sampling_rate"} or {"path"}
    - datasets Audio objects with .array
    - torchcodec AudioDecoder (datasets.features._torchcodec.AudioDecoder) via get_all_samples()
    - generic array-like types
    """

    arr = None
    sr = target_sr

    if isinstance(audio_data, dict):
        if audio_data.get("array") is not None:
            arr = audio_data.get("array")
            sr = int(audio_data.get("sampling_rate", target_sr) or target_sr)
        elif "path" in audio_data:
            import soundfile as sf

            arr, sr = sf.read(audio_data["path"], dtype="float32", always_2d=False)
    elif hasattr(audio_data, "get_all_samples"):
        # TorchCodec AudioDecoder path
        samples = audio_data.get_all_samples()
        sr = int(getattr(samples, "sample_rate", getattr(samples, "sampling_rate", target_sr)) or target_sr)
        data = getattr(samples, "data", samples)
        arr = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
    elif hasattr(audio_data, "array"):
        # datasets.Audio (decoded)
        arr = np.asarray(audio_data.array)
        sr = int(getattr(audio_data, "sampling_rate", target_sr) or target_sr)
    elif callable(audio_data):
        # Some decoders are callable
        decoded = audio_data()
        data = getattr(decoded, "data", decoded)
        arr = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
        sr = int(getattr(decoded, "sample_rate", getattr(decoded, "sampling_rate", target_sr)) or target_sr)
    elif isinstance(audio_data, np.ndarray):
        arr = audio_data
    elif isinstance(audio_data, (list, tuple)):
        arr = np.asarray(audio_data)
    else:
        # Last resort: treat as path-like
        import soundfile as sf

        arr, sr = sf.read(str(audio_data), dtype="float32", always_2d=False)

    if arr is None:
        raise ValueError(f"Audio payload non decodificabile: {type(audio_data)}")

    arr = _to_mono_float32(np.asarray(arr))
    if int(sr) != int(target_sr):
        arr = _resample_1d(arr, int(sr), int(target_sr))
        sr = target_sr

    return arr, int(sr)


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


def ctc_greedy_decode(token_ids, blank_id: int = 0):
    """
    CTC greedy decoding: rimuove blank tokens e collassa token ripetuti.
    
    Args:
        token_ids: Lista o tensor di token IDs
        blank_id: ID del blank token (default 0 per [PAD])
    
    Returns:
        Lista di token IDs decodificati
    """
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    
    result = []
    prev = None
    for tok in token_ids:
        # Skip blank tokens and repeated consecutive tokens
        if tok != blank_id and tok != prev:
            result.append(tok)
        prev = tok
    return result


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _compute_cer(predictions: list[str], references: list[str]) -> float:
    """CER locale (come negli script di training) per evitare evaluate.load("cer") su Kaggle."""
    try:
        import jiwer

        return float(jiwer.cer(references, predictions))
    except Exception:
        pass

    if not references:
        return 0.0
    if len(predictions) != len(references):
        raise ValueError("predictions e references devono avere stessa lunghezza")

    edits = 0
    chars = 0
    for p, r in zip(predictions, references):
        edits += _levenshtein_distance(p, r)
        chars += len(r)
    return float(edits) / float(max(1, chars))


def _get_transformers_version() -> str:
    try:
        import transformers

        return str(getattr(transformers, "__version__", "unknown"))
    except Exception:
        return "unknown"


def evaluate_speechocean(model_path: str, verbose: bool = True, full_dataset: bool = False):
    """Valuta modello su SpeechOcean762 con benchmark scientifico completo."""
    from transformers import Wav2Vec2Processor, WavLMForCTC, WavLMModel
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
    import json
    import torch
    import torch.nn as nn
    
    # Inizializza normalizzatore
    normalizer = IPANormalizer(mode='strict')

    # Sanity check: path locale
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"\n❌ Percorso modello non trovato: {model_path}")
        # Suggerisci automaticamente alcune directory comuni (senza scandire tutta la repo).
        candidates = []
        for root in (Path("outputs/backup"), Path("outputs")):
            if root.exists():
                candidates.extend(sorted(root.glob("*/final_model")))
        if candidates:
            print("\nℹ️ Esempi di model_path validi trovati:")
            for p in candidates[:12]:
                print(f"   - {p.as_posix()}")
        print("\nSuggerimento: passa la cartella che contiene i file del modello (es. config.json, model.safetensors, preprocessor_config.json).")
        return None

    if not model_dir.is_dir():
        print(f"\n❌ model_path non è una directory: {model_path}")
        return None
    
    print("=" * 70)
    print("🔬 BENCHMARK SCIENTIFICO - SPEECHOCEAN762")
    print("=" * 70)
    print("\n📋 Configurazione:")
    print(f"   Modello: {model_path}")
    print(f"   Normalizzazione IPA: {normalizer.mode}")
    print(f"   Dataset completo: {'Sì' if full_dataset else 'No (50 esempi)'}")

    def _is_hf_quantized_model(m: nn.Module) -> bool:
        return bool(
            getattr(m, "is_loaded_in_4bit", False)
            or getattr(m, "is_loaded_in_8bit", False)
            or getattr(m, "quantization_method", None) is not None
            or getattr(m, "quantization_config", None) is not None
        )
    
    # Controlla tipo modello PRIMA di caricare processor
    config_path = Path(model_path) / "config.json"
    is_weighted_model = False
    is_baseline_mlp = False
    is_xlsr_model = False
    is_mms_model = False
    is_data2vec2_model = False
    is_hubert_model = False
    is_distilhubert_model = False
    is_mctct_model = False
    is_parakeet_model = False
    is_speechtokenizer = False
    is_whisper_encoder = False
    is_qwen_audio = False
    is_w2v_bert = False
    is_early_fusion = False  # NEW
    model = None  # Ensure model is initialized
    config = {}
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        is_weighted_model = config.get("model_type") == "wavlm_weighted_layers"
        is_baseline_mlp = config.get("model_type") == "baseline_mlp_ctc"
        is_speechtokenizer = config.get("model_type") == "speechtokenizer_discrete_ctc"
        is_whisper_encoder = config.get("model_type") == "whisper_encoder_ctc"
        is_qwen_audio = config.get("model_type") == "qwen2_audio_ctc"
        is_early_fusion = config.get("model_type") == "early_fusion"  # NEW
        is_w2v_bert = config.get("model_type") == "wav2vec2-bert" or \
                      "wav2vec2bert" in config.get("architectures", [""])[0].lower() or \
                      "w2v-bert" in str(config.get("_name_or_path", "")).lower()

        # Nuovi modelli (coerenti con scripts/training/train_mctct.py e train_parakeet.py)
        arch0 = (config.get("architectures", [""])[0] or "").lower()
        name0 = str(config.get("_name_or_path", "")).lower()
        model_type0 = str(config.get("model_type", "")).lower()
        is_mctct_model = (
            "mctct" in model_type0
            or "m-ctc-t" in model_type0
            or "mctct" in arch0
            or "m-ctc-t" in name0
        )
        is_parakeet_model = (
            "parakeet" in model_type0
            or "parakeet" in arch0
            or "parakeet" in name0
        )
        is_mms_model = "mms-1b" in name0

        # Data2Vec2 (coerente con scripts/training/train_data2vec2.py che usa AutoModelForCTC)
        is_data2vec2_model = "data2vec" in arch0 or "data2vec" in name0 or "data2vec" in model_type0
        is_xlsr_model = (not is_w2v_bert) and (not is_mms_model) and (
            "wav2vec2" in config.get("architectures", [""])[0].lower() or
            "xlsr" in str(config.get("_name_or_path", "")).lower()
        )
        is_hubert_model = config.get("model_type") == "hubert" or \
                          "hubert" in config.get("architectures", [""])[0].lower() or \
                          "hubert" in str(config.get("_name_or_path", "")).lower()

        # DistilHuBERT: stesso loader di HuBERT (HubertForCTC), ma utile per logging/detection.
        is_distilhubert_model = "distilhubert" in name0 or "distilhubert" in arch0
    else:
        print(f"   ⚠️ WARNING: Config file not found at {config_path}")
        # Fallback detection from path string
        path_str = str(model_path).lower()
        if "early_fusion" in path_str:
            print(f"   ℹ️ Detected 'early_fusion' in path. Forcing Early Fusion mode.")
            is_early_fusion = True
            is_w2v_bert = False
            is_xlsr_model = False
        elif "w2v-bert" in path_str or "w2v_bert" in path_str:
            print(f"   ℹ️ Detected 'w2v-bert' in path.")
            is_w2v_bert = True
        elif "distilhubert" in path_str:
            print(f"   ℹ️ Detected 'distilhubert' in path.")
            is_distilhubert_model = True
            is_hubert_model = True
        elif "parakeet" in path_str:
            print(f"   ℹ️ Detected 'parakeet' in path.")
            is_parakeet_model = True
        elif "m-ctc-t" in path_str or "mctct" in path_str:
            print(f"   ℹ️ Detected 'm-ctc-t/mctct' in path.")
            is_mctct_model = True
        elif "mms" in path_str:
            print(f"   ℹ️ Detected 'mms' in path.")
            is_mms_model = True
        elif "data2vec" in path_str:
            print(f"   ℹ️ Detected 'data2vec' in path.")
            is_data2vec2_model = True
        elif "wav2vec2" in path_str or "xlsr" in path_str:
            print(f"   ℹ️ Detected 'wav2vec2/xlsr' in path.")
            is_xlsr_model = True
        elif "whisper" in path_str:
            print(f"   ℹ️ Detected 'whisper' in path.")
            is_whisper_encoder = True
        elif "qwen" in path_str:
            print(f"   ℹ️ Detected 'qwen' in path.")
            is_qwen_audio = True

    
    # Carica modello
    print("\n📦 Caricamento modello...")
    
    # Per modelli custom, usa solo tokenizer (non Wav2Vec2Processor completo)
    if is_mctct_model:
        try:
            from transformers import MCTCTProcessor

            processor = MCTCTProcessor.from_pretrained(model_path)
        except Exception:
            # Fallback: carica processor dal checkpoint base e inietta il tokenizer salvato (come train_mctct.py)
            from transformers import MCTCTProcessor, Wav2Vec2CTCTokenizer

            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
            base_ckpt = config.get("_name_or_path") or "speechbrain/m-ctc-t-large"
            processor = MCTCTProcessor.from_pretrained(base_ckpt)
            processor.tokenizer = tokenizer

    elif is_parakeet_model:
        try:
            from transformers import ParakeetProcessor, Wav2Vec2CTCTokenizer
        except Exception as e:
            tf_ver = _get_transformers_version()
            raise RuntimeError(
                "ParakeetProcessor non disponibile nella tua installazione di transformers. "
                f"Transformers version: {tf_ver}.\n"
                "In Colab/Kaggle: pip install -U transformers accelerate bitsandbytes\n"
                "(se usi un checkpoint quantizzato 4-bit serve anche CUDA + bitsandbytes)."
            ) from e

        try:
            processor = ParakeetProcessor.from_pretrained(model_path)
        except Exception:
            # Fallback: come train_parakeet.py (processor base + tokenizer custom)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
            base_ckpt = config.get("_name_or_path") or "nvidia/parakeet-ctc-1.1b"
            processor = ParakeetProcessor.from_pretrained(base_ckpt)
            processor.tokenizer = tokenizer

    elif is_speechtokenizer or is_whisper_encoder or is_qwen_audio or is_early_fusion:
        from transformers import Wav2Vec2CTCTokenizer
        processor = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
    elif is_w2v_bert:
        from transformers import Wav2Vec2BertProcessor
        processor = Wav2Vec2BertProcessor.from_pretrained(model_path)
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_path)
        except Exception:
            # Fallback robusto: tokenizer dal folder + feature_extractor dal checkpoint base.
            # Utile per modelli HuBERT/DistilHuBERT se manca parte della config del processor.
            if is_hubert_model or is_distilhubert_model:
                from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

                tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
                base_ckpt = config.get("_name_or_path") or (
                    "ntu-spml/distilhubert" if is_distilhubert_model else "facebook/hubert-large-ls960-ft"
                )
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_ckpt)
                processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            else:
                raise
    
    if is_baseline_mlp:
        print("   Tipo: BaselineMLPForCTC (Linear Probe)")
        vocab_size = config["vocab_size"]
        hidden_dim = config.get("hidden_dim", 256)
        backbone = config.get("backbone", "microsoft/wavlm-base")
        
        class BaselineMLPForCTC(nn.Module):
            def __init__(self, vocab_size, hidden_dim=256, backbone_name="microsoft/wavlm-base"):
                super().__init__()
                from transformers import WavLMModel
                self.wavlm = WavLMModel.from_pretrained(backbone_name)
                for param in self.wavlm.parameters():
                    param.requires_grad = False
                
                hidden_size = self.wavlm.config.hidden_size
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, vocab_size),
                )
            
            def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
                outputs = self.wavlm(input_values, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
                logits = self.mlp(hidden_states)
                return {"logits": logits}
        
        model = BaselineMLPForCTC(vocab_size, hidden_dim, backbone)
        model_file = Path(model_path) / "pytorch_model.bin"
        state_dict = torch.load(model_file, map_location="cpu")
        mlp_state_dict = {k: v for k, v in state_dict.items() if k.startswith("mlp.")}
        model.mlp.load_state_dict({k.replace("mlp.", ""): v for k, v in mlp_state_dict.items()})
        print(f"   ✓ Pesi MLP caricati da: {model_file}")
        
    elif is_weighted_model:
        print("   Tipo: WavLMWithWeightedLayers (custom)")
        vocab_size = config["vocab_size"]
        base_model = config.get("base_model", "microsoft/wavlm-large")
        
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
                weights = weights.view(-1, 1, 1, 1)
                weighted_output = (stacked * weights).sum(dim=0)
                weighted_output = self.dropout(weighted_output)
                logits = self.lm_head(weighted_output)
                return {"logits": logits}
        
        model = WavLMWithWeightedLayers(vocab_size, base_model)
        model_file = Path(model_path) / "pytorch_model.bin"
        state_dict = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"   ✓ Pesi caricati da: {model_file}")
        
    elif is_w2v_bert:
        print("   Tipo: Wav2Vec2BertForCTC (W2V-BERT 2.0)")
        from transformers import Wav2Vec2BertForCTC
        model = Wav2Vec2BertForCTC.from_pretrained(model_path)

    elif is_mctct_model:
        print("   Tipo: MCTCTForCTC (M-CTC-T)")
        try:
            from transformers import MCTCTForCTC
        except Exception as e:
            raise RuntimeError(
                "MCTCTForCTC non disponibile: serve una versione recente di transformers. "
                "Vedi scripts/training/train_mctct.py"
            ) from e
        model = MCTCTForCTC.from_pretrained(model_path)

    elif is_parakeet_model:
        print("   Tipo: ParakeetForCTC (Parakeet-CTC 1.1B)")
        try:
            from transformers import ParakeetForCTC
        except Exception as e:
            tf_ver = _get_transformers_version()
            raise RuntimeError(
                "ParakeetForCTC non disponibile: serve una versione di transformers che includa Parakeet.\n"
                f"Transformers version: {tf_ver}.\n"
                "In Colab/Kaggle: pip install -U transformers accelerate bitsandbytes\n"
                "Vedi scripts/training/train_parakeet.py"
            ) from e
        parakeet_kwargs = {}
        if torch.cuda.is_available():
            # Il checkpoint Parakeet nel repo viene tipicamente salvato/allenato in 4-bit (bitsandbytes).
            # Su GPU usiamo device_map='auto' e proviamo a forzare il loader 4-bit.
            parakeet_kwargs["device_map"] = "auto"
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig

                parakeet_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                # Se bitsandbytes manca, il load potrebbe comunque funzionare se il modello non è quantizzato.
                pass

        model = ParakeetForCTC.from_pretrained(model_path, **parakeet_kwargs)

        # Su CPU, un checkpoint 4/8-bit (bitsandbytes) non è valutabile: meglio fallire subito.
        if _is_hf_quantized_model(model) and not torch.cuda.is_available():
            print(
                "\n❌ Il checkpoint Parakeet risulta quantizzato (4/8-bit) e richiede CUDA + bitsandbytes per l'inferenza.\n"
                "   Soluzione: esegui la valutazione su GPU (Kaggle/Colab) oppure salva un checkpoint FP32 non-quantizzato."
            )
            return None

    elif is_data2vec2_model:
        print("   Tipo: AutoModelForCTC (Data2Vec2)")
        from transformers import AutoModelForCTC

        model = AutoModelForCTC.from_pretrained(model_path)
        
    elif is_xlsr_model:
        print("   Tipo: Wav2Vec2ForCTC (XLS-R 1B)")
        from transformers import Wav2Vec2ForCTC
        model_dir = Path(model_path)
        is_adapter_only = (model_dir / "adapter_config.json").exists()
        if is_adapter_only:
            try:
                from peft import PeftConfig, PeftModel
            except Exception as e:
                raise RuntimeError(
                    "Checkpoint PEFT rilevato (adapter_config.json), ma peft non è installato. "
                    "Installa: pip install -q peft"
                ) from e

            peft_cfg = PeftConfig.from_pretrained(model_path)
            base_ckpt = peft_cfg.base_model_name_or_path or "facebook/wav2vec2-xls-r-1b"
            try:
                vocab_size = len(getattr(processor, "tokenizer", processor))
            except Exception:
                vocab_size = None

            base_model = Wav2Vec2ForCTC.from_pretrained(
                base_ckpt,
                vocab_size=vocab_size,
                ignore_mismatched_sizes=True,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            print(f"   ✓ PEFT adapter caricato su base: {base_ckpt}")
        else:
            model = Wav2Vec2ForCTC.from_pretrained(model_path)

    elif is_mms_model:
        print("   Tipo: Wav2Vec2ForCTC (MMS-1B)")
        from transformers import Wav2Vec2ForCTC
        model_dir = Path(model_path)
        is_adapter_only = (model_dir / "adapter_config.json").exists()
        if is_adapter_only:
            try:
                from peft import PeftConfig, PeftModel
            except Exception as e:
                raise RuntimeError(
                    "Checkpoint PEFT rilevato (adapter_config.json), ma peft non è installato. "
                    "Installa: pip install -q peft"
                ) from e

            peft_cfg = PeftConfig.from_pretrained(model_path)
            base_ckpt = peft_cfg.base_model_name_or_path or "facebook/mms-1b-all"
            try:
                vocab_size = len(getattr(processor, "tokenizer", processor))
            except Exception:
                vocab_size = None

            base_model = Wav2Vec2ForCTC.from_pretrained(
                base_ckpt,
                vocab_size=vocab_size,
                ignore_mismatched_sizes=True,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            print(f"   ✓ PEFT adapter caricato su base: {base_ckpt}")
        else:
            model = Wav2Vec2ForCTC.from_pretrained(model_path)
        
    elif is_hubert_model:
        if is_distilhubert_model:
            print("   Tipo: DistilHuBERT (HubertForCTC)")
        else:
            print("   Tipo: HubertForCTC")
        from transformers import HubertForCTC
        model = HubertForCTC.from_pretrained(model_path)
        
    elif is_early_fusion:
        print("   Tipo: EarlyFusionModel (HuBERT + WavLM)")
        vocab_size = config.get("vocab_size", 45)
        
        # Get encoder paths from config
        hubert_name = config.get("hubert_name", "facebook/hubert-large-ls960-ft")
        wavlm_name = config.get("wavlm_name", "microsoft/wavlm-base")
        
        # Check if paths are local - try local backup paths before HuggingFace
        model_base = Path(model_path).parent  # outputs/backup/early_fusion
        
        # Determine root backup dir (outputs/backup)
        # If model_path is a checkpoint inside early_fusion folder:
        # outputs/backup/early_fusion/checkpoint-X -> parent.parent = outputs/backup
        if "checkpoint-" in str(Path(model_path).name):
             backup_root = model_base.parent
        else:
             backup_root = model_base

        # HuBERT fallback logic
        local_hubert = backup_root / "hubert_large" / "final_model_hubert"
        # If explicitly local path but missing, OR if default HF name but local backup exists -> try local
        if (hubert_name.startswith("/") and not Path(hubert_name).exists()) or \
           (hubert_name == "facebook/hubert-large-ls960-ft" and local_hubert.exists()):
            
            if local_hubert.exists():
                hubert_name = str(local_hubert)
                print(f"   ✓ HuBERT: using local backup at {hubert_name}")
            else:
                hubert_name = "facebook/hubert-large-ls960-ft"
                print(f"   ⚠️ HuBERT path not found at {local_hubert}, downloading from HuggingFace...")
        
        # WavLM fallback logic
        local_wavlm = backup_root / "wavLM" / "final_model_aug_comb"
        if (wavlm_name.startswith("/") and not Path(wavlm_name).exists()) or \
           (wavlm_name == "microsoft/wavlm-base" and local_wavlm.exists()):
            
            if local_wavlm.exists():
                wavlm_name = str(local_wavlm)
                print(f"   ✓ WavLM: using local backup at {wavlm_name}")
            else:
                wavlm_name = "microsoft/wavlm-base"
                print(f"   ⚠️ WavLM path not found at {local_wavlm}, downloading from HuggingFace...")
            
        use_weighted = config.get("use_weighted_wavlm", True)
        
        from typing import Tuple, Optional, Dict
        from transformers import HubertModel, HubertForCTC
        from transformers.models.wavlm import WavLMModel, WavLMForCTC
        
        class EarlyFusionModel(nn.Module):
            def __init__(self, vocab_size, hubert_name, wavlm_name, use_weighted=True):
                super().__init__()
                # Load HuBERT encoder
                try:
                    hubert_full = HubertForCTC.from_pretrained(hubert_name)
                    self.hubert = hubert_full.hubert
                except:
                    self.hubert = HubertModel.from_pretrained(hubert_name)
                
                # Load WavLM encoder
                try:
                    wavlm_full = WavLMForCTC.from_pretrained(wavlm_name)
                    self.wavlm = wavlm_full.wavlm
                except:
                    self.wavlm = WavLMModel.from_pretrained(wavlm_name)
                
                # Weighted layer sum for WavLM
                self.use_weighted = use_weighted
                if use_weighted:
                    num_layers = self.wavlm.config.num_hidden_layers + 1
                    self.layer_weights = nn.Parameter(torch.zeros(num_layers))
                    self.wavlm.config.output_hidden_states = True
                
                # CTC Head
                hidden_h = self.hubert.config.hidden_size
                hidden_w = self.wavlm.config.hidden_size
                self.dropout = nn.Dropout(0.1)
                self.ctc_head = nn.Linear(hidden_h + hidden_w, vocab_size)
            
            def _get_wavlm_weighted_output(self, hidden_states):
                weights = F.softmax(self.layer_weights, dim=0)
                stacked = torch.stack(hidden_states, dim=0)
                weights_view = weights.view(-1, 1, 1, 1)
                return (stacked * weights_view).sum(dim=0)
            
            def forward(self, input_values, attention_mask=None, **kwargs):
                # Get target dtype
                target_dtype = next(self.hubert.parameters()).dtype
                if input_values.dtype != target_dtype:
                    input_values = input_values.to(target_dtype)
                
                with torch.no_grad():
                    h_h = self.hubert(input_values, attention_mask=attention_mask).last_hidden_state.clone()
                    outputs_w = self.wavlm(input_values, attention_mask=attention_mask)
                    if self.use_weighted and hasattr(outputs_w, 'hidden_states') and outputs_w.hidden_states:
                        h_w = self._get_wavlm_weighted_output(outputs_w.hidden_states).clone()
                    else:
                        h_w = outputs_w.last_hidden_state.clone()
                
                # Align temporal dimension
                min_len = min(h_h.size(1), h_w.size(1))
                combined = torch.cat([h_h[:,:min_len], h_w[:,:min_len]], dim=-1)
                logits = self.ctc_head(self.dropout(combined))
                return {"logits": logits}
        
        model = EarlyFusionModel(vocab_size, hubert_name, wavlm_name, use_weighted)
        
        # Load trained weights - try multiple file formats
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
            raise FileNotFoundError(f"No model file found in {model_path}")
        
        state_dict = None
        # Try loading - safetensors first if that's the file, fallback to torch.load
        if model_file.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
                state_dict = load_file(str(model_file))
            except Exception as e:
                print(f"   ⚠️ safetensors load failed: {e}")
                print(f"   ⚠️ File may be corrupted. Trying pytorch_model.bin if exists...")
                # Try pytorch_model.bin instead
                alt_file = model_dir / "pytorch_model.bin"
                if alt_file.exists():
                    state_dict = torch.load(alt_file, map_location="cpu", weights_only=False)
                else:
                    raise ValueError(f"Cannot load model: safetensors corrupted and no pytorch_model.bin found")
        else:
            state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
        
        # Load only the trained parts (ctc_head, layer_weights, dropout)
        model_state = model.state_dict()
        loaded_keys = []
        for key in state_dict:
            if key in model_state:
                model_state[key] = state_dict[key]
                loaded_keys.append(key)
        model.load_state_dict(model_state)
        print(f"   ✓ EarlyFusionModel caricato da {model_file.name}")
        print(f"   ✓ Loaded keys: {len(loaded_keys)} ({', '.join(loaded_keys[:5])}...)")
        
    elif is_speechtokenizer:
        print("   Tipo: SpeechTokenizer (Discrete)")
        vocab_size = config.get("vocab_size", 100)
        codebook_size = config.get("codebook_size", 1024)
        embed_dim = config.get("embed_dim", 256)
        num_heads = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 2)
        
        class DiscreteTokenClassifier(nn.Module):
            def __init__(self, vocab_size, codebook_size=1024, embed_dim=256, num_heads=4, num_layers=2):
                super().__init__()
                self.embedding = nn.Embedding(codebook_size, embed_dim)
                self.pos_encoding = nn.Parameter(torch.zeros(1, 2048, embed_dim))
                encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                           dim_feedforward=embed_dim*4, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.dropout = nn.Dropout(0.1)
                self.lm_head = nn.Linear(embed_dim, vocab_size)
            
            def forward(self, input_ids, **kwargs):
                x = self.embedding(input_ids) + self.pos_encoding[:, :input_ids.size(1), :]
                x = self.transformer(x)
                logits = self.lm_head(self.dropout(x))
                return {"logits": logits}
        
        model = DiscreteTokenClassifier(vocab_size, codebook_size, embed_dim, num_heads, num_layers)
        model_file = Path(model_path) / "pytorch_model.bin"
        state_dict = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"   ✓ Classificatore caricato")
        print("   ⚠️ NOTA: SpeechTokenizer richiede preprocessing audio diverso!")
        
    elif is_whisper_encoder:
        print("   Tipo: Whisper Encoder + CTC")
        
        # Store config for later model loading
        whisper_vocab_size = config.get("vocab_size", 43)
        whisper_model_name = config.get("whisper_model_name", "openai/whisper-small")
        whisper_model_path = model_path
        
        import gc
        
        if full_dataset:
            # FULL DATASET MODE
            print("\n📥 Caricamento FULL SpeechOcean762...")
            ds_full = load_dataset("mispeech/speechocean762", split="test")
            ds_full = ds_full.cast_column("audio", Audio(sampling_rate=16000))
            print(f"✓ Caricati {len(ds_full)} esempi")
            
            print("\n🔄 Conversione fonemi ARPABET → IPA...")
            collected_examples = []
            for i in range(len(ds_full)):
                ex = ds_full[i]
                ref_ipa = extract_phones_from_words(ex["words"])
                if len(ref_ipa) > 0:
                    collected_examples.append({
                        "audio": ex["audio"],
                        "reference_ipa": ref_ipa,
                        "text": ex["text"],
                        "accuracy": ex["accuracy"],
                        "age": ex.get("age", 0),
                    })
                if (i + 1) % 500 == 0:
                    print(f"   Processati {i + 1}/{len(ds_full)} esempi")
            
            del ds_full
            gc.collect()
            ds = collected_examples
            print(f"✓ Dataset pronto: {len(ds)} esempi validi")
        else:
            # MINIMAL MODE for Kaggle (50 examples)
            print("   ⚠️ Modalità MINIMAL (50 esempi). Usa --full per dataset completo.")
            print("\n📥 Caricamento MINIMAL SpeechOcean762...")
            
            ds_iter = iter(load_dataset("mispeech/speechocean762", split="test", streaming=True))
            collected_examples = []
            target_count = 50
            
            for i in range(500):
                try:
                    ex = next(ds_iter)
                except StopIteration:
                    break
                
                ref_ipa = extract_phones_from_words(ex["words"])
                if len(ref_ipa) > 0:
                    audio_data = ex["audio"]
                    arr = audio_data["array"]
                    sr = audio_data.get("sampling_rate", 16000)
                    if sr != 16000:
                        import librosa
                        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                    
                    collected_examples.append({
                        "audio": {"array": arr, "sampling_rate": 16000},
                        "reference_ipa": ref_ipa,
                        "text": ex["text"],
                        "accuracy": ex["accuracy"],
                        "age": ex.get("age", 0),
                    })
                    
                    if len(collected_examples) >= target_count:
                        break
                
                if i % 10 == 0:
                    gc.collect()
            
            del ds_iter
            gc.collect()
            ds = collected_examples
            print(f"✓ Dataset pronto: {len(ds)} esempi")
        
        # NOW load Whisper model (after dataset is ready)
        print("\n📦 Caricamento Whisper Encoder (post-dataset)...")
        
        class WhisperEncoderForCTC(nn.Module):
            def __init__(self, vocab_size: int, whisper_model_name: str = "openai/whisper-small"):
                super().__init__()
                from transformers import WhisperModel, WhisperFeatureExtractor
                
                self.whisper = WhisperModel.from_pretrained(whisper_model_name)
                self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
                
                for param in self.whisper.encoder.parameters():
                    param.requires_grad = False
                
                hidden_size = self.whisper.config.d_model
                self.lm_head = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, input_features, **kwargs):
                encoder_outputs = self.whisper.encoder(input_features)
                hidden_states = encoder_outputs.last_hidden_state
                logits = self.lm_head(hidden_states)
                return {"logits": logits}
        
        model = WhisperEncoderForCTC(whisper_vocab_size, whisper_model_name)
        
        pytorch_model_path = Path(whisper_model_path) / "pytorch_model.bin"
        if pytorch_model_path.exists():
            state_dict = torch.load(pytorch_model_path, map_location="cpu")
            lm_head_state = {k.replace("lm_head.", ""): v for k, v in state_dict.items() if k.startswith("lm_head.")}
            if lm_head_state:
                model.lm_head.load_state_dict(lm_head_state)
                print(f"   ✓ lm_head caricata: weight {lm_head_state['weight'].shape}")
        
        whisper_feature_extractor = model.feature_extractor
        model.eval()
        print("✓ Whisper Encoder caricato!")
        
    # For Qwen2-Audio
    if is_qwen_audio:
        import gc
        import librosa
        
        if full_dataset:
            print("\n📥 Caricamento FULL SpeechOcean762...")
            ds_full = load_dataset("mispeech/speechocean762", split="test")
            # Note: NOT casting audio column to avoid TorchCodec issues
            print(f"✓ Caricati {len(ds_full)} esempi")
            
            print("\n🔄 Conversione fonemi ARPABET → IPA...")
            collected_examples = []
            for i in range(len(ds_full)):
                ex = ds_full[i]
                ref_ipa = extract_phones_from_words(ex["words"])
                if len(ref_ipa) > 0:
                    # Load audio manually with librosa from path
                    audio_info = ex["audio"]
                    audio_arr = None
                    
                    # Debug first example
                    if i == 0:
                        print(f"   [DEBUG] Audio info type: {type(audio_info)}")
                        if isinstance(audio_info, dict):
                            print(f"   [DEBUG] Audio keys: {audio_info.keys()}")
                        else:
                            # Show what attributes/methods AudioDecoder has
                            attrs = [a for a in dir(audio_info) if not a.startswith('_')]
                            print(f"   [DEBUG] AudioDecoder attrs: {attrs[:20]}")
                            print(f"   [DEBUG] Is callable: {callable(audio_info)}")
                            print(f"   [DEBUG] Has __getitem__: {hasattr(audio_info, '__getitem__')}")
                    
                    # Handle TorchCodec AudioDecoder - use get_all_samples() method
                    audio_decoded = False
                    
                    # Method 1: TorchCodec AudioDecoder.get_all_samples() - the correct API
                    if hasattr(audio_info, 'get_all_samples'):
                        try:
                            samples = audio_info.get_all_samples()
                            if i == 0:
                                print(f"   [DEBUG] get_all_samples() returned type: {type(samples)}")
                                if hasattr(samples, 'data'):
                                    print(f"   [DEBUG] samples.data shape: {samples.data.shape}")
                            
                            if hasattr(samples, 'data'):
                                data = samples.data
                                if hasattr(data, 'numpy'):
                                    audio_arr = data.numpy()
                                else:
                                    audio_arr = np.asarray(data)
                            else:
                                audio_arr = np.asarray(samples)
                            audio_decoded = True
                        except Exception as e:
                            if i == 0:
                                import traceback
                                print(f"   [DEBUG] get_all_samples() failed: {e}")
                                traceback.print_exc()
                    
                    # Method 2: Try calling it (some older versions are callable)
                    if not audio_decoded and callable(audio_info):
                        try:
                            decoded = audio_info()
                            if hasattr(decoded, 'data'):
                                audio_arr = np.asarray(decoded.data)
                            else:
                                audio_arr = np.asarray(decoded)
                            audio_decoded = True
                        except Exception as e:
                            if i == 0:
                                print(f"   [DEBUG] audio_info() failed: {e}")
                    
                    # Method 4: Access .array or .data directly
                    if not audio_decoded:
                        if hasattr(audio_info, 'array'):
                            audio_arr = np.asarray(audio_info.array)
                            audio_decoded = True
                            if i == 0:
                                print(f"   [DEBUG] Got audio from .array")
                        elif hasattr(audio_info, 'data'):
                            audio_arr = np.asarray(audio_info.data)
                            audio_decoded = True
                            if i == 0:
                                print(f"   [DEBUG] Got audio from .data")
                    
                    # Post-process if we got audio
                    if audio_decoded and audio_arr is not None:
                        # Squeeze to 1D if needed
                        if audio_arr.ndim > 1:
                            audio_arr = audio_arr.squeeze()
                        if audio_arr.ndim > 1:
                            audio_arr = audio_arr.mean(axis=0)
                        audio_arr = audio_arr.astype(np.float32)
                        # Normalize if int16
                        if len(audio_arr) > 0 and np.abs(audio_arr).max() > 1.0:
                            audio_arr = audio_arr / 32768.0
                        if i == 0:
                            print(f"   [DEBUG] Final audio shape: {audio_arr.shape}, dtype: {audio_arr.dtype}")
                    elif i == 0:
                        print(f"   [DEBUG] All audio decode methods failed for first example")
                    
                    # Fallback for dict-based audio (older datasets format)
                    if not audio_decoded and isinstance(audio_info, dict):
                        if "path" in audio_info:
                            try:
                                audio_arr, _ = librosa.load(audio_info["path"], sr=16000)
                                audio_decoded = True
                            except Exception as e:
                                if i == 0:
                                    print(f"   [DEBUG] librosa.load failed: {e}")
                                # Try bytes if path fails
                                if "bytes" in audio_info and audio_info["bytes"]:
                                    import io
                                    import soundfile as sf
                                    try:
                                        audio_arr, sr = sf.read(io.BytesIO(audio_info["bytes"]))
                                        if sr != 16000:
                                            audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
                                        audio_decoded = True
                                    except Exception as e2:
                                        if i == 0:
                                            print(f"   [DEBUG] soundfile from bytes failed: {e2}")
                        elif "bytes" in audio_info and audio_info["bytes"]:
                            import io
                            import soundfile as sf
                            try:
                                audio_arr, sr = sf.read(io.BytesIO(audio_info["bytes"]))
                                if sr != 16000:
                                    audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
                                audio_decoded = True
                            except Exception:
                                pass
                        elif "array" in audio_info:
                            audio_arr = np.asarray(audio_info["array"]).astype(np.float32)
                            sr = audio_info.get("sampling_rate", 16000)
                            if sr != 16000:
                                audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
                            audio_decoded = True
                    
                    if audio_arr is not None:
                        collected_examples.append({
                            "audio": {"array": audio_arr, "sampling_rate": 16000},
                            "reference_ipa": ref_ipa,
                            "text": ex["text"],
                            "accuracy": ex["accuracy"],
                            "age": ex.get("age", 0),
                        })
                if (i + 1) % 500 == 0:
                    print(f"   Processati {i + 1}/{len(ds_full)} esempi")
            
            del ds_full
            gc.collect()
            ds = collected_examples
            print(f"✓ Dataset pronto: {len(ds)} esempi validi")
        else:
            print("\n📥 Caricamento MINIMAL SpeechOcean762 (50 esempi)...")
            
            ds_iter = iter(load_dataset("mispeech/speechocean762", split="test", streaming=True))
            collected_examples = []
            target_count = 50
            
            for i in range(500):
                try:
                    ex = next(ds_iter)
                except StopIteration:
                    break
                
                ref_ipa = extract_phones_from_words(ex["words"])
                if len(ref_ipa) > 0:
                    audio_data = ex["audio"]
                    arr = audio_data["array"]
                    sr = audio_data.get("sampling_rate", 16000)
                    if sr != 16000:
                        import librosa
                        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                    
                    collected_examples.append({
                        "audio": {"array": arr, "sampling_rate": 16000},
                        "reference_ipa": ref_ipa,
                        "text": ex["text"],
                        "accuracy": ex["accuracy"],
                        "age": ex.get("age", 0),
                    })
                    
                    if len(collected_examples) >= target_count:
                        break
                
                if i % 10 == 0:
                    gc.collect()
            
            del ds_iter
            gc.collect()
            ds = collected_examples
            print(f"✓ Dataset pronto: {len(ds)} esempi")
        
        # Now load Qwen2 model
        print("\n📦 Caricamento Qwen2-Audio...")
        vocab_size = config.get("vocab_size", 43)
        
        class Qwen2AudioEncoderForCTC(nn.Module):
            def __init__(self, vocab_size: int, device: str = "cuda"):
                super().__init__()
                from transformers import BitsAndBytesConfig, AutoProcessor
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                
                from transformers import Qwen2AudioForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2-Audio-7B-Instruct",
                    trust_remote_code=True,
                )
                qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-Audio-7B-Instruct",
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self.audio_encoder = qwen_model.audio_tower
                for param in self.audio_encoder.parameters():
                    param.requires_grad = False
                
                hidden_size = 1280
                self.ctc_head = nn.Sequential(
                    nn.Linear(hidden_size, 512),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, vocab_size),
                ).to(device)
                self._device = device
            
            def forward(self, input_features, **kwargs):
                with torch.no_grad():
                    audio_outputs = self.audio_encoder(input_features)
                    hidden_states = audio_outputs.last_hidden_state
                # Convert to float32 for CTC head (encoder outputs fp16 due to quantization)
                hidden_states = hidden_states.float().to(self._device)
                logits = self.ctc_head(hidden_states)
                return {"logits": logits}
        
        try:
            import bitsandbytes
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = Qwen2AudioEncoderForCTC(vocab_size, device=device)
            
            ctc_head_path = Path(model_path) / "ctc_head.bin"
            if ctc_head_path.exists():
                state_dict = torch.load(ctc_head_path, map_location="cpu")
                model.ctc_head.load_state_dict(state_dict)
                print(f"   ✓ CTC head caricata da: {ctc_head_path}")
            
            qwen_feature_extractor = model.processor.feature_extractor
            model.eval()
            print("✓ Modello caricato!")
        except ImportError:
            print("   ⚠️ bitsandbytes non disponibile!")
            model = None
    
    # Check if model was loaded successfully
    if model is None:
        print("\n❌ Impossibile caricare il modello. Uscita.")
        return None
    
    if not is_qwen_audio and not is_whisper_encoder:
        model.eval()
        print("✓ Modello caricato!")

        # Load dataset for other models (wav2vec, wavlm, hubert, mctct, parakeet, etc.)
        try:
            if full_dataset:
                print("\n📥 Scaricamento FULL SpeechOcean762...")
                hf_ds = load_dataset("mispeech/speechocean762", split="test")
                hf_ds = hf_ds.cast_column("audio", Audio(sampling_rate=16000))
                print(f"✓ Caricati {len(hf_ds)} esempi")

                def prepare_example(example):
                    example["reference_ipa"] = extract_phones_from_words(example["words"])
                    return example

                print("\n🔄 Conversione fonemi ARPABET → IPA...")
                hf_ds = hf_ds.map(prepare_example)
                hf_ds = hf_ds.filter(lambda x: len(x["reference_ipa"]) > 0)
                print(f"✓ Esempi validi: {len(hf_ds)}")

                collected_examples = []
                for i in range(len(hf_ds)):
                    ex = hf_ds[i]
                    collected_examples.append({
                        "audio": ex["audio"],
                        "reference_ipa": ex["reference_ipa"],
                        "text": ex["text"],
                        "accuracy": ex["accuracy"],
                        "age": ex.get("age", 0),
                        "words": ex["words"],
                    })
                ds = collected_examples
            else:
                print("   ⚠️ Modalità MINIMAL (50 esempi). Usa --full per dataset completo.")
                print("\n📥 Caricamento MINIMAL SpeechOcean762 (streaming)...")
                ds_iter = iter(load_dataset("mispeech/speechocean762", split="test", streaming=True))
                collected_examples = []
                target_count = 50

                for _ in range(2000):
                    try:
                        ex = next(ds_iter)
                    except StopIteration:
                        break

                    ref_ipa = extract_phones_from_words(ex.get("words", []))
                    if not ref_ipa:
                        continue

                    try:
                        arr, _ = _decode_audio_to_16k(ex["audio"], target_sr=16000)
                    except Exception as e:
                        print(f"   ⚠️ Cannot decode audio (streaming): {e}")
                        continue

                    collected_examples.append({
                        "audio": {"array": arr, "sampling_rate": 16000},
                        "reference_ipa": ref_ipa,
                        "text": ex.get("text", ""),
                        "accuracy": ex.get("accuracy", 0),
                        "age": ex.get("age", 0),
                        "words": ex.get("words", []),
                    })
                    if len(collected_examples) >= target_count:
                        break

                ds = collected_examples
                print(f"✓ Dataset pronto: {len(ds)} esempi")
        except Exception as e:
            print("\n❌ Errore nel caricamento di SpeechOcean762 (mispeech/speechocean762).")
            print("   Cause comuni: connessione, cache HF corrotta, versione datasets/pyarrow, oppure dataset non accessibile.")
            print("   Suggerimenti rapidi:")
            print("   - pip install -U datasets pyarrow soundfile librosa")
            print("   - (se serve) huggingface-cli login")
            raise
    
    # ==========================================================================
    # PREDIZIONE CON CONFIDENCE SCORE (BATCH PROCESSING FOR LIST)
    # ==========================================================================
    print("\n🔄 Esecuzione inferenza con confidence scoring...")
    
    batch_size = 4
    results = []
    
    for batch_start in range(0, len(ds), batch_size):
        batch_end = min(batch_start + batch_size, len(ds))
        batch_examples = ds[batch_start:batch_end]
        
        # Extract audio arrays from batch
        audio_arrays = []
        for ex in batch_examples:

            audio_data = ex["audio"]
            try:
                arr, _ = _decode_audio_to_16k(audio_data, target_sr=16000)
            except Exception as e:
                print(f"   ⚠️ Cannot decode audio: {type(audio_data)} ({e})")
                arr = np.zeros(16000, dtype=np.float32)  # 1 second silence
            
            audio_arrays.append(arr)
        
        # Run inference
        try:
            if is_qwen_audio and model is not None:
                mel_features = []
                for audio in audio_arrays:
                    mel = qwen_feature_extractor(
                        audio, sampling_rate=16000, return_tensors="pt"
                    ).input_features[0]
                    mel_features.append(mel)
                max_len = max(m.shape[-1] for m in mel_features)
                padded_mels = []
                for m in mel_features:
                    if m.shape[-1] < max_len:
                        pad_width = max_len - m.shape[-1]
                        m = torch.nn.functional.pad(m, (0, pad_width))
                    padded_mels.append(m)
                input_features = torch.stack(padded_mels)
                
                # Convert to half precision if model uses half
                if next(model.parameters()).dtype == torch.float16:
                    input_features = input_features.half()
                
                with torch.no_grad():
                    outputs = model(input_features)
                    logits = outputs["logits"]
            elif is_whisper_encoder:
                mel_features = []
                for audio in audio_arrays:
                    mel = whisper_feature_extractor(
                        audio, sampling_rate=16000, return_tensors="pt"
                    ).input_features[0]
                    mel_features.append(mel)
                max_len = max(m.shape[-1] for m in mel_features)
                padded_mels = []
                for m in mel_features:
                    if m.shape[-1] < max_len:
                        pad_width = max_len - m.shape[-1]
                        m = torch.nn.functional.pad(m, (0, pad_width))
                    padded_mels.append(m)
                input_features = torch.stack(padded_mels)
                
                with torch.no_grad():
                    outputs = model(input_features)
                    logits = outputs["logits"]
            elif is_early_fusion:
                # Early Fusion uses raw audio directly, no processor needed for input
                # Just convert audio arrays to tensors
                max_len = max(len(a) for a in audio_arrays)
                padded_audio = []
                for a in audio_arrays:
                    if len(a) < max_len:
                        pad_width = max_len - len(a)
                        a = np.pad(a, (0, pad_width), mode='constant')
                    padded_audio.append(a)
                input_values = torch.tensor(np.stack(padded_audio), dtype=torch.float32)

                device = next(model.parameters()).device
                param_dtype = next(model.parameters()).dtype

                # Su CPU, evita half/bf16: alcune ops non sono supportate o causano mismatch.
                if device.type == "cpu" and param_dtype in (torch.float16, torch.bfloat16):
                    if not _is_hf_quantized_model(model):
                        model = model.float()
                    param_dtype = torch.float32

                input_values = input_values.to(device=device, dtype=param_dtype)
                use_amp = (device.type == "cuda")
                amp_dtype = param_dtype if param_dtype in (torch.float16, torch.bfloat16) else torch.float16

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        outputs = model(input_values)
                    if isinstance(outputs, dict):
                        logits = outputs["logits"]
                    else:
                        logits = outputs.logits
            else:
                inputs = processor(
                    audio_arrays,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )

                device = next(model.parameters()).device
                param_dtype = next(model.parameters()).dtype

                # Su CPU, upcast a fp32 per robustezza.
                if device.type == "cpu" and param_dtype in (torch.float16, torch.bfloat16):
                    if not _is_hf_quantized_model(model):
                        model = model.float()
                    param_dtype = torch.float32

                use_amp = (device.type == "cuda")
                amp_dtype = param_dtype if param_dtype in (torch.float16, torch.bfloat16) else torch.float16

                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        if hasattr(inputs, 'input_features') and inputs.input_features is not None:
                            outputs = model(inputs.input_features.to(device=device, dtype=param_dtype))
                        else:
                            outputs = model(inputs.input_values.to(device=device, dtype=param_dtype))
                    if isinstance(outputs, dict):
                        logits = outputs["logits"]
                    else:
                        logits = outputs.logits
            
            # Predictions - use proper CTC decoding for Whisper/Qwen
            predicted_ids = torch.argmax(logits, dim=-1)

            # Determine padding/blank id for CTC-style decoding & confidence masking
            pad_token_id = (
                getattr(processor, 'pad_token_id', None)
                or getattr(getattr(processor, 'tokenizer', None), 'pad_token_id', 0)
            )
            
            # Apply CTC greedy decoding for models that need it
            if is_whisper_encoder or is_qwen_audio or is_early_fusion:
                # Decode each sequence with CTC blank/repeat collapsing
                predicted_texts = []
                for seq in predicted_ids:
                    decoded_ids = ctc_greedy_decode(seq, blank_id=pad_token_id)
                    if decoded_ids:
                        text = processor.decode(decoded_ids, skip_special_tokens=True)
                    else:
                        text = ""
                    predicted_texts.append(text)
            else:
                # Standard decoding for wav2vec/wavlm/hubert/mctct/parakeet
                if hasattr(processor, "batch_decode"):
                    predicted_texts = processor.batch_decode(predicted_ids)
                elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "batch_decode"):
                    predicted_texts = processor.tokenizer.batch_decode(predicted_ids)
                else:
                    raise AttributeError(f"Processor senza batch_decode: {type(processor)}")
            
            # Confidence scores
            probs = F.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1).values
            
            for i, ex in enumerate(batch_examples):
                # Calculate confidence
                non_pad_mask = predicted_ids[i] != pad_token_id
                if non_pad_mask.sum() > 0:
                    conf = max_probs[i][non_pad_mask].mean().item()
                else:
                    conf = 0.0
                
                # Store result
                results.append({
                    "predicted_ipa": predicted_texts[i],
                    "reference_ipa": ex["reference_ipa"],
                    "confidence_score": conf,
                    "accuracy": ex["accuracy"],
                    "text": ex["text"],
                    "age": ex.get("age", 0),
                })
        except Exception as e:
            print(f"   ⚠️ Errore batch {batch_start}-{batch_end}: {e}")
            for ex in batch_examples:
                results.append({
                    "predicted_ipa": "",
                    "reference_ipa": ex["reference_ipa"],
                    "confidence_score": 0.0,
                    "accuracy": ex["accuracy"],
                    "text": ex["text"],
                    "age": ex.get("age", 0),
                })
        
        if (batch_end) % 200 == 0 or batch_end == len(ds):
            print(f"   Processati {batch_end}/{len(ds)} esempi")
    
    # ==========================================================================
    # PREPARA DATI PER ANALISI
    # ==========================================================================
    print("\n📊 Preparazione dati per analisi...")
    
    all_data = []
    for i in range(len(results)):
        pred = normalizer.normalize(results[i]["predicted_ipa"])
        ref = normalizer.normalize(results[i]["reference_ipa"])
        
        if not pred or not ref:
            continue
        
        per = _compute_cer(predictions=[pred], references=[ref])
        
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
    
    if len(all_data) < 2:
        print("\n❌ Non ci sono abbastanza esempi validi per l'analisi!")
        return None
    
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
    
    high_quality_mask = human_scores >= 8
    high_quality_preds = [d["pred"] for d, m in zip(all_data, high_quality_mask) if m]
    high_quality_refs = [d["ref"] for d, m in zip(all_data, high_quality_mask) if m]
    
    if len(high_quality_preds) > 0:
        per_high = _compute_cer(predictions=high_quality_preds, references=high_quality_refs)
        
        print(f"\n   📊 Risultati su High Quality (score >= 8):")
        print(f"   ─────────────────────────────────────────")
        print(f"   Campioni: {len(high_quality_preds)}")
        print(f"   PER:      {per_high * 100:.2f}%")
        print(f"   Accuracy: {(1 - per_high) * 100:.2f}%")
        
        for threshold in [8, 9, 10]:
            mask = human_scores == threshold
            preds = [d["pred"] for d, m in zip(all_data, mask) if m]
            refs = [d["ref"] for d, m in zip(all_data, mask) if m]
            if len(preds) > 0:
                per_t = _compute_cer(predictions=preds, references=refs)
                print(f"      Score {threshold}: PER = {per_t * 100:.2f}% (n={len(preds)})")
    else:
        print("   ⚠️ Nessun esempio high quality trovato!")
        per_high = 1.0
    
    # ==========================================================================
    # TASK B: SCORING CORRELATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📋 TASK B: SCORING CORRELATION (PER vs Human Score)")
    print("=" * 70)
    print("   Obiettivo: Verificare correlazione tra PER del modello")
    print("              e giudizio umano sulla qualità della pronuncia.")
    print("   Metrica principale: (1 - PER) ↔ Human Score")
    print("-" * 70)
    
    pearson_per, pearson_per_p = pearsonr(1 - pers, human_scores)
    spearman_per, spearman_per_p = spearmanr(1 - pers, human_scores)
    
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
    # TASK C: MISPRONUNCIATION DETECTION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("📋 TASK C: MISPRONUNCIATION DETECTION (PER-based Classification)")
    print("=" * 70)
    print("   Obiettivo: Classificare pronuncia come Corretta/Errata")
    print("              usando il PER (distanza Levenshtein) come predittore.")
    print("   Labels: Errata (score <= 6), Corretta (score > 6)")
    print("   Logica: Alto PER → Alta probabilità di errore di pronuncia")
    print("-" * 70)
    
    y_true = (human_scores <= 6).astype(int)
    y_prob = pers
    
    from sklearn.metrics import precision_recall_curve
    
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
    
    y_pred = (pers >= best_threshold).astype(int)
    
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_roc = 0.5
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    accuracy = ((y_pred == y_true).sum()) / len(y_true)
    
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
    
    if auc_roc >= 0.8:
        auc_interp = "✅ OTTIMO - classificatore affidabile"
    elif auc_roc >= 0.7:
        auc_interp = "✅ BUONO - classificatore discreto"
    elif auc_roc >= 0.6:
        auc_interp = "⚠️ MODERATO - margine di miglioramento"
    else:
        auc_interp = "❌ SCARSO - classificatore poco affidabile"
    print(f"\n   Interpretazione AUC: {auc_interp}")
    
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
        "pearson": pearson_per,
        "spearman": spearman_per,
        "auc_roc": auc_roc,
        "f1": f1,
        "best_threshold": best_threshold,
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
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full dataset (default: 50 examples for Kaggle)"
    )
    
    args = parser.parse_args()
    evaluate_speechocean(args.model_path, verbose=not args.quiet, full_dataset=args.full)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("\n❌ Errore non gestito in evaluate_speechocean.py. Stacktrace completo:")
        traceback.print_exc()
        sys.exit(1)

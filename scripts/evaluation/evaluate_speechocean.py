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

# Aggiungi project root al path (2 livelli su da scripts/evaluation/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    from transformers import Wav2Vec2Processor, WavLMForCTC, WavLMModel
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
    import json
    import torch
    import torch.nn as nn
    
    # Inizializza normalizzatore
    normalizer = IPANormalizer(mode='strict')
    
    print("=" * 70)
    print("🔬 BENCHMARK SCIENTIFICO - SPEECHOCEAN762")
    print("=" * 70)
    print("\n📋 Configurazione:")
    print(f"   Modello: {model_path}")
    print(f"   Normalizzazione IPA: {normalizer.mode}")
    
    # Controlla tipo modello PRIMA di caricare processor
    config_path = Path(model_path) / "config.json"
    is_weighted_model = False
    is_baseline_mlp = False
    is_xlsr_model = False
    is_hubert_model = False
    is_speechtokenizer = False
    is_whisper_encoder = False
    is_qwen_audio = False
    is_w2v_bert = False  # Wav2Vec2-BERT 2.0
    config = {}
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        is_weighted_model = config.get("model_type") == "wavlm_weighted_layers"
        is_baseline_mlp = config.get("model_type") == "baseline_mlp_ctc"
        is_speechtokenizer = config.get("model_type") == "speechtokenizer_discrete_ctc"
        is_whisper_encoder = config.get("model_type") == "whisper_encoder_ctc"
        is_qwen_audio = config.get("model_type") == "qwen2_audio_ctc"
        # Wav2Vec2-BERT 2.0 detection (must come before xlsr check)
        is_w2v_bert = config.get("model_type") == "wav2vec2-bert" or \
                      "wav2vec2bert" in config.get("architectures", [""])[0].lower() or \
                      "w2v-bert" in str(config.get("_name_or_path", "")).lower()
        # XLS-R usa architettura wav2vec2 (but NOT wav2vec2-bert)
        is_xlsr_model = (not is_w2v_bert) and (
            "wav2vec2" in config.get("architectures", [""])[0].lower() or
            "xlsr" in str(config.get("_name_or_path", "")).lower()
        )
        # HuBERT detection
        is_hubert_model = config.get("model_type") == "hubert" or \
                          "hubert" in config.get("architectures", [""])[0].lower() or \
                          "hubert" in str(config.get("_name_or_path", "")).lower()
    
    # Carica modello
    print("\n📦 Caricamento modello...")
    
    # Per modelli custom, usa solo tokenizer (non Wav2Vec2Processor completo)
    if is_speechtokenizer or is_whisper_encoder or is_qwen_audio:
        from transformers import Wav2Vec2CTCTokenizer
        processor = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
    elif is_w2v_bert:
        # Wav2Vec2-BERT needs Wav2Vec2BertProcessor with SeamlessM4TFeatureExtractor
        from transformers import Wav2Vec2BertProcessor
        processor = Wav2Vec2BertProcessor.from_pretrained(model_path)
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_path)
    
    if is_baseline_mlp:
        print("   Tipo: BaselineMLPForCTC (Linear Probe)")
        # Carica modello Baseline MLP
        vocab_size = config["vocab_size"]
        hidden_dim = config.get("hidden_dim", 256)
        backbone = config.get("backbone", "microsoft/wavlm-base")
        
        # Definisci classe inline (stesso codice di train_baseline_mlp.py)
        # NOTA: usa 'wavlm' come nome attributo per matchare il checkpoint salvato
        class BaselineMLPForCTC(nn.Module):
            def __init__(self, vocab_size, hidden_dim=256, backbone_name="microsoft/wavlm-base"):
                super().__init__()
                from transformers import WavLMModel
                self.wavlm = WavLMModel.from_pretrained(backbone_name)
                # Freeze backbone
                for param in self.wavlm.parameters():
                    param.requires_grad = False
                
                hidden_size = self.wavlm.config.hidden_size  # 768 per wavlm-base
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, vocab_size),
                )
            
            def forward(self, input_values, attention_mask=None, labels=None, **kwargs):
                outputs = self.wavlm(input_values, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state  # [batch, frames, 768]
                logits = self.mlp(hidden_states)  # [batch, frames, vocab_size]
                return {"logits": logits}
        
        model = BaselineMLPForCTC(vocab_size, hidden_dim, backbone)
        
        # Carica pesi (solo MLP, backbone viene scaricato da HF)
        model_file = Path(model_path) / "pytorch_model.bin"
        state_dict = torch.load(model_file, map_location="cpu")
        
        # Carica solo i pesi MLP (il backbone è già inizializzato da from_pretrained)
        # Filtra per caricare solo mlp.* keys
        mlp_state_dict = {k: v for k, v in state_dict.items() if k.startswith("mlp.")}
        model.mlp.load_state_dict({k.replace("mlp.", ""): v for k, v in mlp_state_dict.items()})
        
        print(f"   ✓ Pesi MLP caricati da: {model_file}")
        print(f"   ✓ Backbone: {backbone} (FROZEN, caricato da HuggingFace)")
        print(f"   ✓ MLP hidden: {hidden_dim}")
        
    elif is_weighted_model:
        print("   Tipo: WavLMWithWeightedLayers (custom)")
        # Carica modello custom
        vocab_size = config["vocab_size"]
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
                weights = weights.view(-1, 1, 1, 1)
                weighted_output = (stacked * weights).sum(dim=0)
                weighted_output = self.dropout(weighted_output)
                logits = self.lm_head(weighted_output)
                return {"logits": logits}
        
        model = WavLMWithWeightedLayers(vocab_size, base_model)
        
        # Carica pesi
        model_file = Path(model_path) / "pytorch_model.bin"
        state_dict = torch.load(model_file, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"   ✓ Pesi caricati da: {model_file}")
    elif is_w2v_bert:
        print("   Tipo: Wav2Vec2BertForCTC (W2V-BERT 2.0)")
        from transformers import Wav2Vec2BertForCTC
        model = Wav2Vec2BertForCTC.from_pretrained(model_path)
    elif is_xlsr_model:
        print("   Tipo: Wav2Vec2ForCTC (XLS-R)")
        from transformers import Wav2Vec2ForCTC
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    elif is_hubert_model:
        print("   Tipo: HubertForCTC")
        from transformers import HubertForCTC
        model = HubertForCTC.from_pretrained(model_path)
    elif is_speechtokenizer:
        print("   Tipo: SpeechTokenizer (Discrete)")
        # SpeechTokenizer usa un classificatore custom
        vocab_size = config.get("vocab_size", 100)
        codebook_size = config.get("codebook_size", 1024)
        embed_dim = config.get("embed_dim", 256)
        num_heads = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 2)
        
        import torch.nn as nn
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
        print("   ⚠️ La valutazione potrebbe non essere accurata senza SpeechTokenizer encoder.")
    elif is_whisper_encoder:
        print("   Tipo: Whisper Encoder + CTC")
        # Load Whisper Encoder with CTC head
        vocab_size = config.get("vocab_size", 43)
        whisper_model_name = config.get("whisper_model_name", "openai/whisper-small")
        
        class WhisperEncoderForCTC(nn.Module):
            def __init__(self, vocab_size: int, whisper_model_name: str = "openai/whisper-small"):
                super().__init__()
                from transformers import WhisperModel, WhisperFeatureExtractor
                
                self.whisper = WhisperModel.from_pretrained(whisper_model_name)
                self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
                
                # Freeze encoder
                for param in self.whisper.encoder.parameters():
                    param.requires_grad = False
                
                hidden_size = self.whisper.config.d_model
                self.ctc_head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size // 2, vocab_size),
                )
            
            def forward(self, input_features, **kwargs):
                # Use only encoder
                encoder_outputs = self.whisper.encoder(input_features)
                hidden_states = encoder_outputs.last_hidden_state
                logits = self.ctc_head(hidden_states)
                return {"logits": logits}
        
        model = WhisperEncoderForCTC(vocab_size, whisper_model_name)
        
        # Load CTC head weights
        ctc_head_path = Path(model_path) / "ctc_head.bin"
        pytorch_model_path = Path(model_path) / "pytorch_model.bin"
        
        if ctc_head_path.exists():
            state_dict = torch.load(ctc_head_path, map_location="cpu")
            model.ctc_head.load_state_dict(state_dict)
            print(f"   ✓ CTC head caricata da: {ctc_head_path}")
        elif pytorch_model_path.exists():
            # Try loading full model state dict and extract ctc_head
            state_dict = torch.load(pytorch_model_path, map_location="cpu")
            ctc_head_state = {k.replace("ctc_head.", ""): v for k, v in state_dict.items() if k.startswith("ctc_head.")}
            if ctc_head_state:
                model.ctc_head.load_state_dict(ctc_head_state)
                print(f"   ✓ CTC head estratta da: {pytorch_model_path}")
            else:
                print(f"   ⚠️ Nessun CTC head trovato in {pytorch_model_path}")
        else:
            print(f"   ⚠️ Nessun file di pesi trovato!")
        
        # Store feature extractor for later use
        whisper_feature_extractor = model.feature_extractor
        is_whisper_encoder = True  # Flag for inference
        
    # For Qwen2-Audio, load dataset FIRST before model to avoid OOM
    # (Qwen2 7B 4-bit uses ~4GB RAM, need room for dataset preprocessing)
    # STREAMING MODE to avoid Arrow OOM on Kaggle
    if is_qwen_audio:
        print("\n📥 Caricamento SpeechOcean762 (STREAMING MODE per evitare OOM)...")
        # Use streaming=True to avoid loading full dataset into RAM
        ds_stream = load_dataset("mispeech/speechocean762", split="test", streaming=True)
        
        # Manually collect valid examples with IPA conversion
        print("\n🔄 Conversione fonemi ARPABET → IPA (streaming)...")
        collected_examples = []
        for i, example in enumerate(ds_stream):
            ref_ipa = extract_phones_from_words(example["words"])
            if len(ref_ipa) > 0:
                collected_examples.append({
                    "audio": example["audio"],  # raw audio dict
                    "reference_ipa": ref_ipa,
                    "text": example["text"],
                    "accuracy": example["accuracy"],
                    "age": example.get("age", 0),
                    "words": example["words"],
                })
            if (i + 1) % 500 == 0:
                print(f"   Processati {i + 1} esempi, validi: {len(collected_examples)}")
        
        print(f"✓ Esempi validi raccolti: {len(collected_examples)}")
        
        # Convert to a simple list-based dataset (no Arrow)
        ds = collected_examples  # will be processed as list, not HF Dataset
        
        # Now load Qwen2 model
        print("\n📦 Caricamento Qwen2-Audio (post-dataset preprocessing)...")
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
                logits = self.ctc_head(hidden_states.to(self._device))
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
    
    if not is_qwen_audio:  # Qwen already called model.eval() above
        model.eval()
        print("✓ Modello caricato!")
        
        # Load dataset for non-Qwen models - USE STREAMING to avoid OOM
        print("\n📥 Caricamento SpeechOcean762 (STREAMING MODE per evitare OOM)...")
        ds_stream = load_dataset("mispeech/speechocean762", split="test", streaming=True)
        
        # Manually collect valid examples with IPA conversion
        print("\n🔄 Conversione fonemi ARPABET → IPA (streaming)...")
        collected_examples = []
        for i, example in enumerate(ds_stream):
            ref_ipa = extract_phones_from_words(example["words"])
            if len(ref_ipa) > 0:
                collected_examples.append({
                    "audio": example["audio"],  # raw audio dict with 'array' and 'sampling_rate'
                    "reference_ipa": ref_ipa,
                    "text": example["text"],
                    "accuracy": example["accuracy"],
                    "age": example.get("age", 0),
                    "words": example["words"],
                })
            if (i + 1) % 500 == 0:
                print(f"   Processati {i + 1} esempi, validi: {len(collected_examples)}")
        
        print(f"✓ Esempi validi raccolti: {len(collected_examples)}")
        ds = collected_examples  # Will be processed as list, not HF Dataset
    
    # ==========================================================================
    # PREDIZIONE CON CONFIDENCE SCORE (BATCH PROCESSING FOR LIST)
    # ==========================================================================
    print("\n🔄 Esecuzione inferenza con confidence scoring...")
    
    # ds is now a Python list, process in batches manually
    batch_size = 4
    results = []  # Will store processed examples
    
    for batch_start in range(0, len(ds), batch_size):
        batch_end = min(batch_start + batch_size, len(ds))
        batch_examples = ds[batch_start:batch_end]
        
        # Extract audio arrays from batch
        audio_arrays = []
        for ex in batch_examples:
            audio_data = ex["audio"]
            # Handle both dict format (from streaming) and array format
            if isinstance(audio_data, dict):
                arr = audio_data["array"]
                sr = audio_data.get("sampling_rate", 16000)
                # Resample if needed
                if sr != 16000:
                    import librosa
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                audio_arrays.append(arr)
            else:
                audio_arrays.append(audio_data)
        
        # Run inference
        try:
            if is_qwen_audio and model is not None:
                mel_features = []
                for audio in audio_arrays:
                    mel = qwen_feature_extractor(
                        audio, sampling_rate=16000, return_tensors="pt"
                    ).input_features[0]
                    mel_features.append(mel)
                # Stack and pad
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
            elif is_whisper_encoder:
                mel_features = []
                for audio in audio_arrays:
                    mel = whisper_feature_extractor(
                        audio, sampling_rate=16000, return_tensors="pt"
                    ).input_features[0]
                    mel_features.append(mel)
                # Stack and pad
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
            else:
                inputs = processor(
                    audio_arrays,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                
                with torch.no_grad():
                    if hasattr(inputs, 'input_features') and inputs.input_features is not None:
                        outputs = model(inputs.input_features)
                    else:
                        outputs = model(inputs.input_values)
                    if isinstance(outputs, dict):
                        logits = outputs["logits"]
                    else:
                        logits = outputs.logits
            
            # Predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_texts = processor.batch_decode(predicted_ids)
            
            # Confidence scores
            probs = F.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1).values
            
            for i, ex in enumerate(batch_examples):
                # Calculate confidence
                non_pad_mask = predicted_ids[i] != processor.tokenizer.pad_token_id
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
            # Add placeholder results for failed batch
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
    
    args = parser.parse_args()
    evaluate_speechocean(args.model_path, verbose=not args.quiet)


if __name__ == "__main__":
    main()

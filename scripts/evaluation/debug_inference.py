
import torch
import librosa
import numpy as np
import sys
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch.nn.functional as F

def debug_inference(model_path, audio_path):
    print(f"📦 Loading model from {model_path}...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.eval()
    
    print(f"🎤 Loading audio: {audio_path}")
    if audio_path == "dummy":
        # Create dummy audio (silence + some noise)
        audio = np.random.uniform(-0.1, 0.1, 16000*2).astype(np.float32)
    else:
        audio, sr = librosa.load(audio_path, sr=16000)
    
    print(f"   Shape: {audio.shape}")
    print(f"   Range: [{audio.min():.3f}, {audio.max():.3f}]")

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    print(f"\n📊 Logits Shape: {logits.shape}")
    # logits: [1, frames, vocab_size]
    
    # Check max logits
    probs = F.softmax(logits, dim=-1)
    max_probs, max_ids = torch.max(probs, dim=-1)
    
    print("\n🔍 First 20 frames prediction:")
    for i in range(min(20, logits.shape[1])):
        token_id = max_ids[0, i].item()
        token = processor.tokenizer.convert_ids_to_tokens(token_id)
        prob = max_probs[0, i].item()
        print(f"   Frame {i:3d}: ID {token_id:3d} ({token}) | Prob: {prob:.4f}")
        
    # Check if collapsed
    unique_ids = torch.unique(max_ids)
    print(f"\n⚠️ Unique predicted IDs: {unique_ids.tolist()}")
    
    decoded = processor.batch_decode(max_ids)[0]
    print(f"\n📝 Decoded: /{decoded}/")
    
    # Vocab check
    r_id = processor.tokenizer.convert_tokens_to_ids("r")
    print(f"\nℹ️ ID for 'r': {r_id}")
    if 19 in unique_ids:
        print("   -> Model is indeed predicting 'r'")

if __name__ == "__main__":
    model_dir = "outputs/backup/final_model_wav2vec2"
    audio_file = "dummy" 
    # Try to find a real audio if possible
    import glob
    audios = glob.glob("data/processed/audio/*.wav") + glob.glob("data/**/*.wav", recursive=True)
    if audios:
        audio_file = audios[0]
    
    debug_inference(model_dir, audio_file)

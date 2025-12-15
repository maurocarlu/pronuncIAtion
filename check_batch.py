"""Check what training batch looks like"""
import torch
from pathlib import Path
from transformers import Wav2Vec2Processor, WavLMForCTC
from datasets import load_dataset, Audio
import sys
sys.path.insert(0, '.')
from src.training.dataset import DataCollatorCTCWithPadding, prepare_dataset_function

# Setup
model_path = 'outputs/wavlm-phoneme-recognizer/final_model'
csv_path = 'data/processed/phonemeref_processed.csv'

print("=== Checking Training Batch ===\n")

# Load
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = WavLMForCTC.from_pretrained(model_path)

# Load sample data
ds = load_dataset('csv', data_files=csv_path)['train'].select(range(3))
ds = ds.cast_column('audio_path', Audio(sampling_rate=16000))
ds = ds.rename_column('audio_path', 'audio')

# Prepare
prepare_fn = prepare_dataset_function(processor)
encoded = ds.map(prepare_fn, remove_columns=ds.column_names)

print("Encoded dataset columns:", encoded.column_names)
print()

# Sample
for i in range(min(3, len(encoded))):
    print(f"Sample {i}:")
    print(f"  input_values shape: {len(encoded[i]['input_values'])}")
    print(f"  labels: {encoded[i]['labels']}")
    print(f"  labels len: {len(encoded[i]['labels'])}")
    print()

# Test collate
collator = DataCollatorCTCWithPadding(processor=processor)
batch = collator([encoded[i] for i in range(min(3, len(encoded)))])

print("Batch after collation:")
print(f"  input_values shape: {batch['input_values'].shape}")
print(f"  labels shape: {batch['labels'].shape}")
print(f"  labels sample: {batch['labels'][0][:20]}")
print()

# Forward pass
print("Forward pass test:")
with torch.no_grad():
    outputs = model(
        input_values=batch['input_values'],
        labels=batch['labels']
    )
    
print(f"  Loss: {outputs.loss.item():.4f}")
print(f"  Logits shape: {outputs.logits.shape}")

# Decode prediction
pred_ids = torch.argmax(outputs.logits, dim=-1)
print(f"  Predicted IDs unique: {pred_ids[0].unique().tolist()}")
print(f"  Decoded: {repr(processor.decode(pred_ids[0]))}")

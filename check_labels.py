"""Quick diagnostic to check label tokenization."""
import pandas as pd
from transformers import Wav2Vec2Processor

df = pd.read_csv('data/processed/phonemeref_processed.csv').head(5)
processor = Wav2Vec2Processor.from_pretrained('outputs/wavlm-phoneme-recognizer/final_model')

print('=== Label Tokenization Test ===\n')

for idx, row in df.iterrows():
    ipa = row['ipa_clean']
    word = row.get('word', '?')
    
    # Tokenize using tokenizer directly
    tokens = processor.tokenizer(ipa)
    ids = tokens['input_ids']
    
    # Count UNK tokens 
    unk_count = sum(1 for i in ids if i == 1)
    
    print(f'Word: {word}')
    print(f'  IPA: {repr(ipa)}')
    print(f'  Length: {len(ipa)} chars -> {len(ids)} tokens')
    print(f'  Token IDs: {ids[:15]}...' if len(ids) > 15 else f'  Token IDs: {ids}')
    
    if unk_count > 0:
        print(f'  WARNING: {unk_count} UNK tokens!')
    print()

#!/usr/bin/env python3
"""Train L-MAC decoder on SpeechOcean762 (full dataset)."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Add project root to path (robust when executed as a script)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.lmac_core import (
    LMACBackboneConfig,
    LMACSpeechOceanDataset,
    LMACWrapper,
    collate_audio_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train L-MAC decoder")
    parser.add_argument("--model-path", type=str, required=True, help="Path to backbone checkpoint")
    parser.add_argument("--backbone", type=str, default="hubert", choices=["hubert", "early_fusion"])
    parser.add_argument("--target-phoneme", type=str, required=True, help="IPA phoneme to explain")
    parser.add_argument("--layer-ids", type=str, default="6,12,18,24", help="Comma-separated layer ids")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-out", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs/lmac")
    parser.add_argument("--use-conditioning", action="store_true", help="Enable conditional decoder (required for multi-phoneme)")
    return parser.parse_args()


def train_lmac(args) -> Path:
    layer_ids = tuple(int(x.strip()) for x in args.layer_ids.split(",") if x.strip())
    
    # Pre-load tokenizer to get vocab size if conditioning is enabled
    vocab_size = 0
    if args.use_conditioning:
        from transformers import Wav2Vec2CTCTokenizer
        try:
             # Try loading tokenizer from model path
             tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model_path)
             vocab_size = len(tokenizer)
        except Exception as e:
            # Fallback for EarlyFusion where model_path might differ or specific structure
            print(f"⚠️ Could not load tokenizer directly from {args.model_path}: {e}")
            # Try loading from standard Hubert path as fallback (common vocab)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/hubert-large-ls960-ft")
            vocab_size = len(tokenizer)
        print(f"🧠 Conditioning enabled. Vocab size: {vocab_size}")

    config = LMACBackboneConfig(
        backbone_type=args.backbone,
        model_path=args.model_path,
        layer_ids=layer_ids,
        use_conditioning=args.use_conditioning,
        vocab_size=vocab_size,
    )

    # Support None for multi-phoneme mode
    target_phoneme = args.target_phoneme if args.target_phoneme else None
    is_multi_phoneme = target_phoneme is None

    dataset = LMACSpeechOceanDataset(
        split="train",
        target_phoneme=target_phoneme,  # None = random sampling
        full=True,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_audio_batch,
    )

    wrapper = LMACWrapper(config=config, target_phoneme=target_phoneme)
    optimizer = torch.optim.Adam(wrapper.decoder.parameters(), lr=args.lr)

    # Output dir: use "multi" folder for multi-phoneme mode
    phoneme_folder = target_phoneme if target_phoneme else "multi"
    output_dir = Path(args.output_dir) / args.backbone / phoneme_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        wrapper.train()
        total = 0.0
        n = 0
        step_pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False, unit="batch")
        for step, batch in enumerate(step_pbar, start=1):
            optimizer.zero_grad(set_to_none=True)
            
            # Multi-phoneme mode: pass per-batch target phonemes
            if is_multi_phoneme:
                losses = wrapper.compute_loss(
                    batch["input_values"],
                    batch["attention_mask"],
                    target_phonemes=batch["target_phoneme"],  # List of phonemes per sample
                    lambda_out=args.lambda_out,
                    lambda_reg=args.lambda_reg,
                )
            else:
                losses = wrapper.compute_loss(
                    batch["input_values"],
                    batch["attention_mask"],
                    lambda_out=args.lambda_out,
                    lambda_reg=args.lambda_reg,
                )
            
            loss = losses["loss"]
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            n += 1
            
            # Update progress bar with current loss
            step_pbar.set_postfix(loss=f"{total/n:.4f}")

        avg = total / max(1, n)
        epoch_pbar.set_postfix(loss=f"{avg:.4f}")
        tqdm.write(f"Epoch {epoch}/{args.epochs} | loss={avg:.4f}")

        # Save decoder checkpoint each epoch
        torch.save(
            {
                "decoder_state": wrapper.decoder.state_dict(),
                "target_phoneme": target_phoneme,  # None for multi
                "is_multi_phoneme": is_multi_phoneme,
                "layer_ids": layer_ids,
                "backbone": args.backbone,
                "backbone": args.backbone,
                "model_path": args.model_path,
                "use_conditioning": args.use_conditioning,
                "vocab_size": vocab_size,
            },
            output_dir / f"decoder_epoch_{epoch}.pt",
        )

    # Save final decoder
    torch.save(
        {
            "decoder_state": wrapper.decoder.state_dict(),
            "target_phoneme": target_phoneme,
            "is_multi_phoneme": is_multi_phoneme,
            "layer_ids": layer_ids,
            "backbone": args.backbone,
            "backbone": args.backbone,
            "model_path": args.model_path,
            "use_conditioning": args.use_conditioning,
            "vocab_size": vocab_size,
        },
        output_dir / "decoder_final.pt",
    )

    tqdm.write(f"✓ Decoder salvato in: {output_dir}")
    return output_dir


def main():
    args = parse_args()
    train_lmac(args)


if __name__ == "__main__":
    main()

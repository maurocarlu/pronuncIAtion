# Phoneme and Pronunciation Recognition Benchmark

A comprehensive benchmark project for evaluating phoneme recognition and pronunciation assessment systems using deep learning approaches.

## Overview

This project aims to create and test a benchmark for phoneme recognition and pronunciation evaluation systems. The benchmark includes a curated dataset of English words with their corresponding phonetic transcriptions (IPA notation) and audio pronunciations, designed to assess the performance of automatic speech recognition (ASR) and pronunciation assessment models.

## Objectives

- **Dataset Creation**: Build a high-quality dataset of English words with phonetic transcriptions and audio samples
- **Benchmark Development**: Establish standardized evaluation metrics for phoneme recognition accuracy
- **Model Evaluation**: Test and compare different deep learning architectures on phoneme recognition tasks
- **Pronunciation Assessment**: Develop methods to evaluate pronunciation quality and accuracy

## Key Features

- ✅ Large-scale English vocabulary dataset (~10,000 words)
- ✅ IPA (International Phonetic Alphabet) phonetic transcriptions
- ✅ High-quality audio samples (USA English pronunciation)
- ✅ Automated data collection and validation pipeline
- ✅ Standardized train/validation/test splits
- ✅ CSV metadata format for easy integration



## Dataset Format

The main dataset file (`phonemeref_metadata.csv`) contains:

| Column | Description |
|--------|-------------|
| `id` | Unique identifier for each entry |
| `word` | English word |
| `ipa` | IPA phonetic transcription |
| `audio_path` | Relative path to audio file |
| `split` | Dataset split (train/val/test) |


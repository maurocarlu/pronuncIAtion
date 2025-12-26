"""
Script per avviare il training.
"""

# ============================================================================
# SILENZIA LOG PRIMA DI IMPORTARE QUALSIASI COSA
# Questo deve essere PRIMA di qualsiasi import per funzionare
# ============================================================================
import os
import sys
import warnings

# Redirect stderr temporaneamente per catturare warning all'import
class SuppressStderr:
    """Context manager per sopprimere stderr temporaneamente."""
    def __init__(self):
        self.null = open(os.devnull, 'w')
        self.old_stderr = None
    def __enter__(self):
        self.old_stderr = sys.stderr
        sys.stderr = self.null
        return self
    def __exit__(self, *args):
        sys.stderr = self.old_stderr
        self.null.close()

# Silenzia TensorFlow/XLA/CUDA prima dell'import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="  # Disabilita XLA

warnings.filterwarnings("ignore")

# Silenzia logging
import logging
for logger_name in ["tensorflow", "transformers", "datasets", "absl", "urllib3", "filelock"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# ============================================================================
# IMPORT LIBRERIE (con stderr sopresso per catturare warning residui)
# ============================================================================
import argparse
from pathlib import Path

# Sopprimi stderr durante l'import delle librerie pesanti
with SuppressStderr():
    # Aggiungi src al path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.training.trainer import PhonemeTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train phoneme recognition model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=None,
        help="Override CSV path from config"
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (1 epoch, small batch)"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("TRAINING PHONEME RECOGNIZER")
    print("=" * 50)
    
    # Carica config
    if Path(args.config).exists():
        print(f"[INFO] Caricamento config: {args.config}")
        config = TrainingConfig.from_yaml(args.config)
    else:
        print(f"[WARN] Config non trovato: {args.config}, uso defaults")
        config = TrainingConfig()
    
    # Override CSV path se specificato
    if args.data_csv:
        config.csv_path = args.data_csv
        print(f"[INFO] Override CSV: {args.data_csv}")
    
    # Debug mode
    if args.debug:
        config.num_train_epochs = 1
        config.batch_size = 2
        config.eval_steps = 50
        config.save_steps = 50
        config.logging_steps = 10
        print("[INFO] Modalita debug attiva")
    
    # Mostra config
    print(f"\nConfigurazione:")
    print(f"   Modello: {config.model_name}")
    print(f"   Dataset: {config.csv_path}")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Gradient accum: {config.gradient_accumulation_steps}")
    print(f"   Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   FP16: {config.fp16}")
    print(f"   DataLoader workers: {config.dataloader_num_workers}")
    print(f"   Group by length: {config.group_by_length}")
    print(f"   Output: {config.output_dir}")
    
    # Train
    trainer = PhonemeTrainer(config)
    trainer.train(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()

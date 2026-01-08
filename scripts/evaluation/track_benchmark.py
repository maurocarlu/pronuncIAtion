#!/usr/bin/env python3
"""
Benchmark Tracking Script - Gestione risultati esperimenti.

Crea e aggiorna un file Excel per tracciare i risultati dei benchmark
su SpeechOcean762 per diverse configurazioni di modelli.

Uso:
    # Aggiungi nuovo risultato
    python scripts/track_benchmark.py \
        --model_name "WavLM Large Weighted" \
        --architecture "Weighted" \
        --per 30.41 \
        --pearson 0.58 \
        --auc 0.85 \
        --recall 0.72
    
    # Visualizza risultati correnti
    python scripts/track_benchmark.py --show
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

# Path al file Excel (project root)
EXCEL_PATH = Path(__file__).parent.parent.parent / "benchmark_results.xlsx"

# Definizione colonne con valori default
COLUMNS = {
    # Info Modello
    "Model_Name": None,
    "Architecture": None,  # Standard, Weighted, Ensemble
    "Training_Data": None,
    
    # Task A - ASR Robustness
    "TaskA_PER_HighQuality": None,  # %
    "TaskA_Accuracy": None,  # %
    
    # Task B - Scoring Correlation
    "TaskB_Pearson_r": None,  # (1-PER) vs Human
    "TaskB_Spearman_rho": None,
    
    # Task C - Detection Classification
    "TaskC_AUC_ROC": None,
    "TaskC_F1_Score": None,
    "TaskC_Recall_Errors": None,  # Cruciale!
    "TaskC_Precision": None,
    "TaskC_Threshold": None,
    
    # Note
    "Notes": None,
}


def initialize_excel() -> pd.DataFrame:
    """Crea DataFrame vuoto con le colonne corrette."""
    return pd.DataFrame(columns=list(COLUMNS.keys()))


def load_or_create_excel() -> pd.DataFrame:
    """Carica il file Excel esistente o ne crea uno nuovo."""
    if EXCEL_PATH.exists():
        try:
            df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
            # Verifica che tutte le colonne esistano
            for col in COLUMNS.keys():
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception as e:
            print(f"⚠️ Errore lettura Excel: {e}")
            print("   Creazione nuovo file...")
            return initialize_excel()
    else:
        return initialize_excel()


def save_excel(df: pd.DataFrame) -> None:
    """Salva DataFrame su Excel con formattazione."""
    # Riordina colonne
    df = df[list(COLUMNS.keys())]
    
    # Salva con openpyxl
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Benchmark Results")
        
        # Formattazione
        worksheet = writer.sheets["Benchmark Results"]
        
        # Larghezza colonne
        column_widths = {
            "A": 12,  # Date
            "B": 25,  # Model_Name
            "C": 12,  # Architecture
            "D": 15,  # Training_Data
            "E": 20,  # TaskA_PER
            "F": 15,  # TaskA_Accuracy
            "G": 18,  # TaskB_Pearson
            "H": 18,  # TaskB_Spearman
            "I": 15,  # TaskC_AUC
            "J": 15,  # TaskC_F1
            "K": 20,  # TaskC_Recall
            "L": 15,  # TaskC_Precision
            "M": 15,  # TaskC_Threshold
            "N": 30,  # Notes
        }
        for col, width in column_widths.items():
            worksheet.column_dimensions[col].width = width
    
    print(f"✓ Salvato: {EXCEL_PATH}")


def add_entry(
    model_name: str,
    architecture: str = "Standard",
    training_data: str = "Aug_Comb",
    per: Optional[float] = None,
    accuracy: Optional[float] = None,
    pearson: Optional[float] = None,
    spearman: Optional[float] = None,
    auc: Optional[float] = None,
    f1: Optional[float] = None,
    recall: Optional[float] = None,
    precision: Optional[float] = None,
    threshold: Optional[float] = None,
    notes: Optional[str] = None,
) -> None:
    """
    Aggiunge una nuova riga al file Excel dei benchmark.
    
    Args:
        model_name: Nome del modello (es. "WavLM Large")
        architecture: Tipo architettura (Standard, Weighted, Ensemble)
        training_data: Dataset usato per training
        per: Phoneme Error Rate su sample alta qualità (%)
        accuracy: Accuracy ASR (%)
        pearson: Correlazione Pearson (1-PER) vs Human Score
        spearman: Correlazione Spearman
        auc: AUC-ROC per detection errori
        f1: F1-Score per detection
        recall: Recall errori (sensibilità - cruciale!)
        precision: Precision errori
        threshold: Soglia ottimale per classificazione
        notes: Note libere
    """
    df = load_or_create_excel()
    
    # Crea nuova riga
    new_row = {
        "Model_Name": model_name,
        "Architecture": architecture,
        "Training_Data": training_data,
        "TaskA_PER_HighQuality": per,
        "TaskA_Accuracy": accuracy,
        "TaskB_Pearson_r": pearson,
        "TaskB_Spearman_rho": spearman,
        "TaskC_AUC_ROC": auc,
        "TaskC_F1_Score": f1,
        "TaskC_Recall_Errors": recall,
        "TaskC_Precision": precision,
        "TaskC_Threshold": threshold,
        "Notes": notes,
    }
    
    # Append
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Salva
    save_excel(df)
    
    # Mostra riga aggiunta
    print("\n📊 Nuovo risultato aggiunto:")
    print("-" * 50)
    for key, value in new_row.items():
        if value is not None:
            # Formatta percentuali
            if "PER" in key or "Accuracy" in key:
                print(f"   {key}: {value:.2f}%")
            elif isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")


def show_results() -> None:
    """Mostra tutti i risultati in formato tabella."""
    if not EXCEL_PATH.exists():
        print("❌ Nessun file benchmark trovato")
        return
    
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    
    print("\n" + "=" * 80)
    print("📊 BENCHMARK RESULTS")
    print("=" * 80)
    
    if len(df) == 0:
        print("   Nessun risultato ancora registrato")
        return
    
    # Mostra tabella semplificata
    display_cols = [
        "Model_Name", "Architecture",
        "TaskA_PER_HighQuality", "TaskB_Pearson_r", 
        "TaskC_AUC_ROC", "TaskC_Recall_Errors"
    ]
    
    # Filtra colonne esistenti
    display_cols = [c for c in display_cols if c in df.columns]
    
    print(df[display_cols].to_string(index=False))
    print("\n" + "-" * 80)
    print(f"   Totale esperimenti: {len(df)}")
    print(f"   File: {EXCEL_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Gestione benchmark results per esperimenti speech"
    )
    
    # Azioni
    parser.add_argument(
        "--show", action="store_true",
        help="Mostra tutti i risultati correnti"
    )
    parser.add_argument(
        "--init", action="store_true",
        help="Inizializza file Excel (se non esiste)"
    )
    
    # Info modello
    parser.add_argument("--model_name", type=str, help="Nome modello")
    parser.add_argument(
        "--architecture", type=str, default="Standard",
        choices=["Standard", "Weighted", "Ensemble", "Linear Probe", "HuBERT", "XLS-R",
                 "Wav2Vec2", "Whisper", "SpeechTokenizer", "Qwen2-Audio", "Fusion"],
        help="Tipo architettura"
    )
    parser.add_argument(
        "--training_data", type=str, default="Aug_Comb",
        help="Dataset di training"
    )
    
    # Task A - ASR
    parser.add_argument("--per", type=float, help="PER HighQuality (%%)")
    parser.add_argument("--accuracy", type=float, help="Accuracy (%%)")
    
    # Task B - Correlation
    parser.add_argument("--pearson", type=float, help="Pearson r")
    parser.add_argument("--spearman", type=float, help="Spearman rho")
    
    # Task C - Detection
    parser.add_argument("--auc", type=float, help="AUC-ROC")
    parser.add_argument("--f1", type=float, help="F1-Score")
    parser.add_argument("--recall", type=float, help="Recall errori")
    parser.add_argument("--precision", type=float, help="Precision")
    parser.add_argument("--threshold", type=float, help="Threshold ottimale")
    
    # Note
    parser.add_argument("--notes", type=str, help="Note libere")
    
    args = parser.parse_args()
    
    # Azioni
    if args.show:
        show_results()
        return
    
    if args.init:
        if EXCEL_PATH.exists():
            print(f"⚠️ File già esistente: {EXCEL_PATH}")
        else:
            df = initialize_excel()
            save_excel(df)
            print(f"✓ File creato: {EXCEL_PATH}")
        return
    
    # Aggiungi entry
    if args.model_name:
        add_entry(
            model_name=args.model_name,
            architecture=args.architecture,
            training_data=args.training_data,
            per=args.per,
            accuracy=args.accuracy,
            pearson=args.pearson,
            spearman=args.spearman,
            auc=args.auc,
            f1=args.f1,
            recall=args.recall,
            precision=args.precision,
            threshold=args.threshold,
            notes=args.notes,
        )
    else:
        parser.print_help()
        print("\n💡 Esempi:")
        print('   python scripts/track_benchmark.py --show')
        print('   python scripts/track_benchmark.py --init')
        print('   python scripts/track_benchmark.py \\')
        print('       --model_name "WavLM Large Weighted" \\')
        print('       --architecture Weighted \\')
        print('       --per 30.41 --pearson 0.58 --auc 0.85 --recall 0.72')


if __name__ == "__main__":
    main()

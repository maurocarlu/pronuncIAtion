#!/usr/bin/env python3
"""
L-MAC Benchmark Tracking Script - Gestione risultati AI/AD.

Crea e aggiorna un file Excel per tracciare i risultati L-MAC
su SpeechOcean762 per diverse architetture e fonemi target.

Uso:
    # Aggiungi nuovo risultato
    python scripts/evaluation/track_lmac_benchmark.py \
        --architecture "hubert" \
        --target_phoneme "i" \
        --run_id 1 \
        --ai 42.18 \
        --ad 0.2743 \
        --notes "Run singolo - fonema i"

    # Multi-phoneme (None/multi)
    python scripts/evaluation/track_lmac_benchmark.py \
        --architecture "early_fusion" \
        --target_phoneme "multi" \
        --run_id 2 \
        --ai 38.05 \
        --ad 0.3012

    # Visualizza risultati correnti
    python scripts/evaluation/track_lmac_benchmark.py --show
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# Path al file Excel (project root)
EXCEL_PATH = Path(__file__).parent.parent.parent / "lmac_benchmark_results.xlsx"


# Definizione colonne con valori default
COLUMNS = {
    "Date": None,
    "Architecture": None,  # hubert, early_fusion, late_fusion
    "Target_Phoneme": None,  # i, t, multi
    "Run_ID": None,
    "Split": "test",
    "AI_Percent": None,
    "AD": None,
    "Model_Path": None,
    "Decoder_CKPT": None,
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
            for col in COLUMNS.keys():
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception as exc:
            print(f"⚠️ Errore lettura Excel: {exc}")
            print("   Creazione nuovo file...")
            return initialize_excel()
    return initialize_excel()


def save_excel(df: pd.DataFrame) -> None:
    """Salva DataFrame su Excel con formattazione base."""
    df = df[list(COLUMNS.keys())]

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="L-MAC Results")
        worksheet = writer.sheets["L-MAC Results"]

        column_widths = {
            "A": 12,  # Date
            "B": 14,  # Architecture
            "C": 14,  # Target_Phoneme
            "D": 8,   # Run_ID
            "E": 10,  # Split
            "F": 12,  # AI
            "G": 10,  # AD
            "H": 35,  # Model_Path
            "I": 35,  # Decoder_CKPT
            "J": 30,  # Notes
        }
        for col, width in column_widths.items():
            worksheet.column_dimensions[col].width = width

    print(f"✓ Salvato: {EXCEL_PATH}")


def _normalize_target_phoneme(value: Optional[str]) -> str:
    if value is None:
        return "multi"
    v = str(value).strip().lower()
    if v in ("", "none", "null", "multi", "multiple"):
        return "multi"
    return value


def add_entry(
    architecture: str,
    target_phoneme: Optional[str],
    run_id: int,
    ai: Optional[float] = None,
    ad: Optional[float] = None,
    split: str = "test",
    model_path: Optional[str] = None,
    decoder_ckpt: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    """Aggiunge una nuova riga al file Excel L-MAC."""
    df = load_or_create_excel()

    new_row = {
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Architecture": architecture,
        "Target_Phoneme": _normalize_target_phoneme(target_phoneme),
        "Run_ID": int(run_id),
        "Split": split,
        "AI_Percent": ai,
        "AD": ad,
        "Model_Path": model_path,
        "Decoder_CKPT": decoder_ckpt,
        "Notes": notes,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_excel(df)

    print("\n📊 Nuovo risultato L-MAC aggiunto:")
    print("-" * 50)
    for key, value in new_row.items():
        if value is None:
            continue
        if key == "AI_Percent" and isinstance(value, (float, int)):
            print(f"   {key}: {value:.2f}%")
        elif isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")


def show_results() -> None:
    """Mostra tutti i risultati in formato tabella."""
    if not EXCEL_PATH.exists():
        print("❌ Nessun file L-MAC benchmark trovato")
        return

    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    print("\n" + "=" * 80)
    print("📊 L-MAC BENCHMARK RESULTS")
    print("=" * 80)

    if len(df) == 0:
        print("   Nessun risultato ancora registrato")
        return

    display_cols = [
        "Date",
        "Architecture",
        "Target_Phoneme",
        "Run_ID",
        "AI_Percent",
        "AD",
        "Split",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))
    print("\n" + "-" * 80)
    print(f"   Totale esperimenti: {len(df)}")
    print(f"   File: {EXCEL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gestione risultati L-MAC (AI/AD)"
    )

    parser.add_argument(
        "--show", action="store_true",
        help="Mostra tutti i risultati correnti"
    )
    parser.add_argument(
        "--init", action="store_true",
        help="Inizializza file Excel (se non esiste)"
    )

    parser.add_argument("--architecture", type=str, help="Architettura (hubert, early_fusion, late_fusion)")
    parser.add_argument("--target_phoneme", type=str, help="Fonema target (i, t, o 'multi')")
    parser.add_argument("--run_id", type=int, help="ID run (1..N)")
    parser.add_argument("--split", type=str, default="test", help="Split dataset (default: test)")

    parser.add_argument("--ai", type=float, help="AI (percentuale)")
    parser.add_argument("--ad", type=float, help="AD (float)")

    parser.add_argument("--model_path", type=str, help="Path al modello")
    parser.add_argument("--decoder_ckpt", type=str, help="Path al decoder checkpoint")
    parser.add_argument("--notes", type=str, help="Note libere")

    args = parser.parse_args()

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

    if args.architecture and args.run_id is not None:
        add_entry(
            architecture=args.architecture,
            target_phoneme=args.target_phoneme,
            run_id=args.run_id,
            ai=args.ai,
            ad=args.ad,
            split=args.split,
            model_path=args.model_path,
            decoder_ckpt=args.decoder_ckpt,
            notes=args.notes,
        )
    else:
        parser.print_help()
        print("\n💡 Esempi:")
        print("   python scripts/evaluation/track_lmac_benchmark.py --show")
        print("   python scripts/evaluation/track_lmac_benchmark.py --init")
        print("   python scripts/evaluation/track_lmac_benchmark.py \\")
        print("       --architecture hubert \\")
        print("       --target_phoneme i \\")
        print("       --run_id 1 \\")
        print("       --ai 42.18 --ad 0.2743")


if __name__ == "__main__":
    main()
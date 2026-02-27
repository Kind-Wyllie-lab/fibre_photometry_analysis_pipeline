# Imports Block (Add to preprocessing.py - top)
import pandas as pd
import numpy as np
from pathlib import Path
from params import *  # SR, BASELINE_SAMPLES, etc.

FLUO_COLS = ['CH1-410', 'CH1-470', 'CH1-560']


# Method Block 1: Loader (Add this function)
def load_raw_fluorescence(file_path: str = None):
    """Load raw CSV with malformed headers/rows."""
    if file_path is None:
        file_path = DATA_FILES[0]
    if not Path(file_path).exists():
        raise ValueError(f"File not found: {file_path}")

    # FIX: Robust CSV parsing for malformed fibre photometry files
    df = pd.read_csv(
        file_path,
        skiprows=1,  # Skip malformed header row
        on_bad_lines='skip',  # Skip bad rows
        engine='python'  # Handles complex parsing
    )

    print(f"Raw loaded: {file_path}, shape={df.shape}")
    print(f"  Cols preview: {list(df.columns)}")
    print(f"  Events preview: {df.get('Events', pd.Series()).dropna().unique()[:5]}")

    if df.empty or 'TimeStamp' not in df.columns:
        # Fallback: read all lines, infer structure
        lines = Path(file_path).read_text().splitlines()
        data_start = next((i for i, line in enumerate(lines) if 'TimeStamp' in line), 1)
        df = pd.read_csv(file_path, skiprows=data_start - 1, on_bad_lines='skip')
        print(f"  Fallback parse: shape={df.shape}")

    return df


# Method Block 2: Cleaner (Replace clean_and_map_events)
def clean_and_map_events(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize fluorescence data."""
    print("Transforming raw → cleaned...")
    df = df_raw.copy()
    df['TimeStamp'] = pd.to_numeric(df['TimeStamp'], errors='coerce')

    # Flexible fluorescence columns
    fluo_cols = [col for col in df.columns if col.startswith('CH1-')]
    if not fluo_cols:
        raise ValueError("No CH1-* columns")
    print(f"  Fluorescence cols: {fluo_cols}")

    df_clean = df[['TimeStamp'] + fluo_cols].copy()
    df_clean.columns = ['TimeStamp'] + FLUO_COLS[:len(fluo_cols)]
    print(f"  After select: shape={df_clean.shape}")

    # Time + events
    df_clean['time_s'] = (df_clean['TimeStamp'] / 1000.0).round(ROUND_DECIMALS)
    if 'Events' in df_raw.columns:
        df_clean['Events_numeric'] = pd.to_numeric(df_raw['Events'], errors='coerce').fillna(0).astype(int)

    # Round + validate fluorescence
    for col in FLUO_COLS:
        if col in df_clean:
            vals = df_clean[col].round(ROUND_DECIMALS)
            df_clean[col] = vals
            v = vals.dropna()
            print(f"  {col}: [{v.min():.3f}, {v.median():.3f}, {v.max():.3f}]")

    return df_clean


# Method Block 3: Saver + Processor
def save_cleaned(df_clean: pd.DataFrame, stem: str):
    path = OUTPUT_DIR / f"cleaned_{stem}.csv"
    df_clean.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    assert reloaded.shape == df_clean.shape, "Save error"
    print(f"Saved: cleaned_{stem}.csv")


def process_single_file(file_path: str = None):
    """Full preprocessing pipeline for single file."""
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)
    df_raw = load_raw_fluorescence(file_path)
    stem = Path(file_path or DATA_FILES[0]).stem
    df_clean = clean_and_map_events(df_raw)
    save_cleaned(df_clean, stem)
    print(f"SUCCESS: {stem}!")
    return df_clean

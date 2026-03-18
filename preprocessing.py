# Imports Block (Add to preprocessing.py - top)
import pandas as pd
import numpy as np
from pathlib import Path
from params import *  # SR, BASELINE_SAMPLES, etc.

fluo_cols = ['CH1-410', 'CH1-470', 'CH1-560']


# Method Block 1: Loader (Add this function)
def load_raw_fluorescence(file_path: str = None):
    """Load raw CSV with malformed headers/rows."""
    if file_path is None:
        file_path = data_files[0]
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

    df_clean = df[['TimeStamp'] + fluo_cols + ['Events']].copy()
    df_clean.columns = ['TimeStamp'] + fluo_cols[:len(fluo_cols)] + ['Events']
    print(f"  After select: shape={df_clean.shape}")

    # Time + events
    df_clean['time_s'] = (df_clean['TimeStamp'] / 1000.0).round(round_decimals)
    if 'Events' in df_raw.columns:
        df_clean['Events_numeric'] = df_raw['Events'].map(lambda x: 0 if pd.isna(x) else int(str(x)[-1]) + 1)


    return df_clean


# Method Block 3: Saver + Processor
def save_cleaned(df_clean: pd.DataFrame, stem: str):
    path = csv_dir / f"cleaned_{stem}.csv"
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
    stem = Path(file_path or data_files[0]).stem
    df_clean = clean_and_map_events(df_raw)
    save_cleaned(df_clean, stem)
    print(f"SUCCESS: {stem}!")
    return df_clean

# if __name__ == "__main__":
#
#     print("TESTING PREPROCESSING STANDALONE")
#     print(f"DATA_FILES: {[f.name for f in data_files]}")
#     print(f"CSV_DIR: {csv_dir}")
#
#     df_clean = process_single_file()
#     print(f"\nSUCCESS!")
#     print(f"  Shape: {df_clean.shape}")
#     print(f"  Columns: {list(df_clean.columns)}")
#     print(f"  Events: {df_clean['Events_numeric'].value_counts().to_dict()}")
#     print(f"  Saved: {csv_dir / 'cleaned_*.csv'}")

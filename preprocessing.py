import pandas as pd
from pathlib import Path
from params import *

FLU_COLS = ['CH1-410', 'CH1-470', 'CH1-560']  # Exact names

def load_raw_fluorescence(file_path: str = None):
    if file_path is None:
        file_path = DATA_FILES[0]
    if not Path(file_path).exists():
        raise ValueError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, skiprows=SKIPROWS_JSON)
    print(f"Raw loaded: {file_path}, shape={df.shape}")
    print(f"  Cols preview: {list(df.columns)}")
    print(f"  Events unique: {df['Events'].unique()[:5]}")
    if df.empty or 'TimeStamp' not in df.columns or 'Events' not in df.columns:
        raise ValueError(f"Invalid data: shape={df.shape}, cols={list(df.columns)}")
    # Check flu cols
    missing = [c for c in SELECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing cols: {missing}")
    return df

def clean_and_map_events(df_raw: pd.DataFrame):
    print("Transforming raw → cleaned...")
    df = df_raw[SELECTED_COLS].copy()
    print(f"  After select: shape={df.shape}")
    df['Events_numeric'] = df['Events'].map(EVENT_MAPPING).fillna(0).astype(int)
    print(f"  Events mapped: {df['Events_numeric'].value_counts().head()}")
    # FIXED: Round exact col names
    for col in FLU_COLS:
        if col not in df.columns:
            raise ValueError(f"Flu col missing after select: {col}")
        orig_sample = df[col].iloc[:3].tolist()
        df[col] = df[col].round(ROUND_DECIMALS)
        print(f"  {col}: {orig_sample} → {df[col].iloc[:3].tolist()}")
    df['time_s'] = (df['TimeStamp'] / 1000.0).round(ROUND_DECIMALS)
    print(f"Cleaned: {df.shape}, cols={list(df.columns)}")
    return df

# ADD to preprocessing.py save_cleaned() - load-back assert from original
def save_cleaned(df_clean: pd.DataFrame, stem: str):
    path = OUTPUT_DIR / f"cleaned_{stem}.csv"
    df_clean.to_csv(path, index=False)
    # Load-back sanity check (from your original)
    df_reloaded = pd.read_csv(path)
    if df_reloaded.shape != df_clean.shape:
        raise ValueError(f"Save/load mismatch: saved {df_clean.shape}, reloaded {df_reloaded.shape}")
    print(f"Saved + verified: {path} ({len(df_clean)} rows, {len(df_clean.columns)} cols)")


def process_single_file(file_path: str = None):
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)
    df_raw = load_raw_fluorescence(file_path)
    stem = Path(file_path or DATA_FILES[0]).stem
    df_clean = clean_and_map_events(df_raw)
    save_cleaned(df_clean, stem)
    print(f"SUCCESS: {stem}!")
    return df_clean

if __name__ == "__main__":
    process_single_file()
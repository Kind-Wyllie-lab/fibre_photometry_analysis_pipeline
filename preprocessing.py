"""
    Analysis pipeline for Fiber Photometry recordings
    Copyright (C) 2026 Dr Paul Rignanese, Kind Lab, University of Edinburgh

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# Imports Block (Add to preprocessing.py - top)
import os.path

import pandas as pd
import numpy as np
from pathlib import Path
from params import *  # SR, BASELINE_SAMPLES, etc.

fluo_cols = ['CH1-410', 'CH1-470', 'CH1-560']


# Method Block 1: Loader (Add this function)
def load_raw_fluorescence(file_path: str = None):
    """Load raw CSV with malformed headers/rows."""
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
def save_cleaned(df_clean, output_dir):
    path = output_dir / f"cleaned_data.csv"
    df_clean.to_csv(path, index=False)


def extract_session_raw_data(raw_data_path, output_dir):
    """Full preprocessing pipeline for single file."""
    raw_fluorescence_csv_path = os.path.join(raw_data_path, 'Fluorescence.csv')
    df_raw = load_raw_fluorescence(raw_fluorescence_csv_path)
    df_clean = clean_and_map_events(df_raw)
    save_cleaned(df_clean, output_dir)
    return df_clean

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
import re
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from pathlib import Path
from params import *  # SR, BASELINE_SAMPLES, etc.

fluo_cols = ['CH1-410', 'CH1-470', 'CH1-560']

_EVENT_LAST_DIGIT_REGEX = re.compile(r"(\d)(?!.*\d)")
_LED_REGEX = re.compile(r"(?:^|;)\s*(?:CH1_)?Input1\*\d+\*(\d)\s*;?")
_FREEZE_ONSET_REGEX = re.compile(r"(?:^|;)\s*CH1_onset\*\d+\*(\d)\s*;?")
_FREEZE_OFFSET_REGEX = re.compile(r"(?:^|;)\s*CH1_offset\*\d+\*(\d)\s*;?")


def _extract_last_digit_as_int(text: str) -> Optional[int]:
    """
    Extract the last digit of a string.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    int or None
        Last digit found as integer, or None if no digit is found.
    """
    match = _EVENT_LAST_DIGIT_REGEX.search(text)
    if match is None:
        return None
    return int(match.group(1))


def _parse_led_and_freezing_from_event_string(event_string: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Parse LED TTL and freezing onset/offset TTLs from a semicolon-separated event string.

    Examples of supported event strings
    ----------------------------------
    - ' CH1_Input1*2*1;'
    - 'CH1_Input1*2*1;CH1_offset*4*0;CH1_offset*4*1;'
    - 'CH1_onset*4*0;CH1_onset*4*1;'
    - 'CH1_offset*4*0;CH1_offset*4*1;'

    Parameters
    ----------
    event_string : str
        Raw event string for one row.

    Returns
    -------
    events_led : int or None
        Last digit for the Input1 token, if present.
    freezing_onset : int or None
        Last digit for the onset token, if present.
    freezing_offset : int or None
        Last digit for the offset token, if present.
    """
    if event_string is None:
        return None, None, None

    s = str(event_string).strip()
    if s == "" or s.lower() == "nan":
        return None, None, None

    led_match = _LED_REGEX.search(s)
    onset_match = _FREEZE_ONSET_REGEX.search(s)
    offset_match = _FREEZE_OFFSET_REGEX.search(s)

    events_led = int(led_match.group(1)) if led_match is not None else None
    freezing_onset = int(onset_match.group(1)) if onset_match is not None else None
    freezing_offset = int(offset_match.group(1)) if offset_match is not None else None

    return events_led, freezing_onset, freezing_offset


def add_event_columns_from_raw(
    df_clean: pd.DataFrame,
    df_raw: pd.DataFrame,
    source_column_simple: str = "Events",
    source_column_multiplex: str = "Events",
    output_led_column: str = "Events_LED",
    output_freezing_column: str = "freezing_event",
) -> pd.DataFrame:
    """
    Add event columns to `df_clean` handling two different event encodings.

    This function supports:
    1) Simple encoding strings like 'Input1*2*1' / 'Input1*2*0'
       -> extracts the last digit to create `Events_LED`.
    2) Multiplex encoding strings like:
       - 'CH1_Input1*2*1;'
       - 'CH1_Input1*2*1;CH1_offset*4*0;CH1_offset*4*1;'
       - 'CH1_onset*4*0;CH1_onset*4*1;'
       - 'CH1_offset*4*0;CH1_offset*4*1;'
       -> extracts LED digit, freezing onset digit, freezing offset digit.

    The freezing output is encoded as:
    - 1 for onset (freezing starts)
    - 0 for offset (freezing ends)
    - NaN if no freezing token is present in that row

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Clean dataframe to be augmented.
    df_raw : pandas.DataFrame
        Raw dataframe containing event columns.
    source_column_simple : str, default="Events"
        Name of the raw column containing the simple event encoding.
    source_column_multiplex : str, default="Events"
        Name of the raw column containing the multiplex ';'-separated encoding.
        If it is the same as `source_column_simple`, the parser will try both strategies.
    output_led_column : str, default="Events_LED"
        Output LED event column name.
    output_freezing_column : str, default="freezing_event"
        Output freezing event column name.
    """
    result = df_clean.copy()

    # Initialize output columns
    result[output_led_column] = 0
    result[output_freezing_column] = np.nan

    if source_column_multiplex in df_raw.columns:
        multiplex_series = df_raw[source_column_multiplex]

        parsed = multiplex_series.map(
            lambda x: (None, None, None) if pd.isna(x) else _parse_led_and_freezing_from_event_string(str(x))
        )

        parsed_df = pd.DataFrame(parsed.tolist(), columns=["_led", "_freeze_onset", "_freeze_offset"])

        # LED: when multiplex provides a LED value, override the simple LED result for that row
        led_override_mask = parsed_df["_led"].notna()
        if led_override_mask.any():
            result.loc[led_override_mask, output_led_column] = parsed_df.loc[led_override_mask, "_led"].astype(int)

        # Freezing: onset/offset can exist independently (sometimes only onset tokens exist, etc.)
        # If both appear in same row (rare), prefer onset (start) by default.
        onset_mask = parsed_df["_freeze_onset"].notna()
        offset_mask = parsed_df["_freeze_offset"].notna()

        # Encode events: 1 = onset, 0 = offset
        result.loc[offset_mask, output_freezing_column] = 0
        result.loc[onset_mask, output_freezing_column] = 1

    return result


def load_raw_fluorescence(file_path: str = None):
    """Load raw CSV with malformed headers/rows."""
    if not Path(file_path).exists():
        raise ValueError(f"File not found: {file_path}")

    if file_path.endswith('Event.csv'):
        df = pd.read_csv(file_path, skiprows=1)
        df = df.reset_index().iloc[:, :-1].set_axis(df.columns, axis=1)  # drop last col, move first col to index

    else:
        # FIX: Robust CSV parsing for malformed fibre photometry files
        df = pd.read_csv(file_path, skiprows=1,  # Skip malformed header row
            on_bad_lines='skip',  # Skip bad rows
            engine='python')  # Handles complex parsing)

    if df.empty or 'TimeStamp' not in df.columns:
        # Fallback: read all lines, infer structure
        lines = Path(file_path).read_text().splitlines()
        data_start = next((i for i, line in enumerate(lines) if 'TimeStamp' in line), 1)
        df = pd.read_csv(file_path, skiprows=data_start - 1, on_bad_lines='skip')
        print(f"  Fallback parse: shape={df.shape}")

    return df


def clean_and_map_events(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize fluorescence data."""
    print("Transforming raw → cleaned...")
    df = df_raw.copy()
    df['TimeStamp'] = pd.to_numeric(df['TimeStamp'], errors='coerce')

    # Flexible fluorescence columns
    fluo_cols = [col for col in df.columns if col.startswith('CH')]
    if not fluo_cols:
        raise ValueError("No CH1-* columns")
    print(f"  Fluorescence cols: {fluo_cols}")

    df_clean = df[['TimeStamp'] + fluo_cols + ['Events']].copy()
    df_clean.columns = ['TimeStamp'] + fluo_cols[:len(fluo_cols)] + ['Events']
    print(f"  After select: shape={df_clean.shape}")

    # Time + events
    df_clean['time_s'] = (df_clean['TimeStamp'] / 1000.0).round(round_decimals)
    if 'Events' in df_raw.columns:
        df_clean = add_event_columns_from_raw(
            df_clean=df_clean,
            df_raw=df_raw,
            source_column_simple="Events",
            source_column_multiplex="Events",
            output_led_column="Events_LED",
            output_freezing_column="freezing_event",
        )

    return df_clean

def save_cleaned(df_clean, output_dir):
    path = output_dir / f"cleaned_data.csv"
    df_clean.to_csv(path, index=False)


def extract_session_raw_data(raw_data_path, output_dir):
    """Full preprocessing pipeline for single file."""
    file_name = 'Fluorescence_Event.csv' if os.path.exists(os.path.join(raw_data_path, 'Fluorescence_Event.csv')) else 'Fluorescence.csv'
    raw_fluorescence_csv_path = os.path.join(raw_data_path, file_name)
    df_raw = load_raw_fluorescence(raw_fluorescence_csv_path)
    df_clean = clean_and_map_events(df_raw)
    save_cleaned(df_clean, output_dir)
    return df_clean

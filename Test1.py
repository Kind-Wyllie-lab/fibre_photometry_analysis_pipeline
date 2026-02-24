import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, stats
import seaborn as sns
from pathlib import Path
# params.py
import os
FLUORESCENCE_PATH = "Fluorescence.csv"  # Update full path if needed
OUTPUT_DIR = "results"
BASELINE_SAMPLES = 1000  # First 1000 for F0
SAMPLERATE_HZ = 60.0
ZSCORE_WINDOW = 300  # Samples for rolling z-score

df_raw = pd.read_csv('/Users/Lou/Desktop/Fluorescence.csv', skiprows=1)
print(df_raw.head(), df_raw.info(), )
print("Raw shape:", df_raw.shape)
print("Raw columns:", df_raw.columns.tolist())

"""cleaning and parsing"""
df_clean=(
    df_raw
    .dropna(axis='columns', how='all')
    .fillna(0))
print(df_clean.head(), df_clean.info(), df_clean.shape,df_clean.columns.tolist() )
print("Cleaned shape:", df_clean.shape)
print(df_clean['Events'].value_counts(dropna=False))

"""Make Events numeric codes"""
mapping = {
    '0': 0,
    'Input1*2*0': 1,  # e.g. cue
    'Input1*2*1': 2,  # e.g. cue2
}

df_clean['Events'] = df_clean['Events'].astype(str).map(mapping).fillna(0).astype(int)
print(df_clean['Events'].value_counts(dropna=False))

"resampling for even spacing"
"df.set_index('TimeStamp').resample('50ms').mean().reset_index()"

"need to save in a new file"
#output_path = (r"/Users/Lou/Desktop")
#Path("results").mkdir(exist_ok=True)  # Folder if missing

df_save = df_clean[['TimeStamp', 'Events', 'CH1-410', 'CH1-470', 'CH1-560']].round(4)  # 4 decimals
df_save.to_csv(r"/Users/Lou/Desktop/cleaned_fluorescence.csv", index=False, float_format='%.4f')
print("file saved")

# LOAD BACK TEST
df_loaded = pd.read_csv(r"/Users/Lou/Desktop/cleaned_fluorescence.csv")
assert df_loaded.shape == df_clean.shape, "Load mismatch!"
print("Reload success:", df_loaded.head())

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, stats
import seaborn as sns
import os
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
DATA_FILES = ["/Users/Lou/Desktop/cleaned_fluorescence.csv"]  # Bulk: add more paths
OUTPUT_DIR = Path("/Users/Lou/Desktop/Fibre_phto_python_dir")
BASELINE_SAMPLES = 1000
SAMPLERATE_HZ = 60.0
ZSCORE_WINDOW = 300

df_plot = pd.read_csv("/Users/Lou/Desktop/cleaned_fluorescence.csv")
print("Events dtype:", df_plot["Events"].dtype)
print("Events sample (first 10 non-zero):")
print(df_plot[df_plot["Events"] != '0']['Events'].head(10).tolist())
print("Events == 'Input1*2*0' count:", (df_plot["Events"] == 'Input1*2*0').sum())
print("Exact match sample:")
matching = df_plot[df_plot["Events"] == 'Input1*2*0']
if len(matching) > 0:
    print(matching[['TimeStamp', 'Events']].head())
else:
    print("NO MATCHES - check spaces/stripping")

print(df_plot.head())
print(df_plot.columns)
print(df_plot["TimeStamp"].head())
print(df_plot["CH1-470"].head())

df_plot = pd.read_csv("/Users/Lou/Desktop/cleaned_fluorescence.csv")
ch410= df_plot["CH1-410"].values
ch470= df_plot["CH1-470"].values

time_s = df_plot["TimeStamp"].values / 1000.0
events = df_plot["Events"].values
f0 = (ch470 - ch410)[:1000]
dff = (ch470 - ch410) / np.median(f0)
zscore = ((ch470 - ch410) - np.mean(f0)) / np.std(f0)

mask_0 = events == 1   # Input1*2*0
mask_1 = events == 2   # Input1*2*1
event_times_0 = time_s[mask_0]
event_times_1 = time_s[mask_1]


print("n Input1*2*0:", len(event_times_0))
print("n Input1*2*1:", len(event_times_1))

'''plt.subplot(311)
plt.plot(time_s, ch470)
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence 470nm")
plt.title("470nm over time")

plt.subplot(312)
plt.plot(time_s, ch410, color="green")
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence 410nm")
plt.title("410nm over time")


plt.subplot(313)
corrected_signal = ch470 - ch410
#plt.figure(figsize=(8, 4))
plt.plot(time_s, corrected_signal, color="purple")
for t in event_times_0:
    plt.axvline(x=t, color="red", linestyle="--", alpha=0.7, linewidth=0.5)
for t in event_times_1:
    plt.axvline(x=t, color="blue", linestyle="--", alpha=0.7, linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("470nm–410nm")
plt.title("Corrected photometry trace")
plt.grid(True, alpha=0.1)
plt.tight_layout()
plt.show()'''



plt.subplot(211)
plt.plot(time_s, dff, color="purple")
for t in event_times_0:
    plt.axvline(x=t, color="red", linestyle="--", alpha=0.7, linewidth=0.5)
for t in event_times_1:
    plt.axvline(x=t, color="blue", linestyle="--", alpha=0.7, linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("df/f")
plt.title("Corrected photometry trace")
plt.grid(True, alpha=0.1)
plt.tight_layout()


plt.subplot(212)
plt.plot(time_s, zscore, color="green")
for t in event_times_0:
    plt.axvline(x=t, color="red", linestyle="--", alpha=0.7, linewidth=0.5)
for t in event_times_1:
    plt.axvline(x=t, color="blue", linestyle="--", alpha=0.7, linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel("zscore")
plt.title("Corrected photometry trace")
plt.grid(True, alpha=0.1)
plt.tight_layout()
plt.show()
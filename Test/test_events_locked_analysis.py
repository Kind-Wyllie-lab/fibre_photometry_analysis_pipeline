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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# Load data
df = pd.read_csv("/Users/Lou/Desktop/cleaned_fluorescence.csv")
time_s = df["TimeStamp"].values / 1000.0
ch470 = df["CH1-470"].values
ch410 = df["CH1-410"].values
corrected = ch470 - ch410

# baseline segment for both dF/F and z-score
f0 = corrected[:1000]

# ΔF/F
dff = corrected / np.median(f0)

# z-score
baseline_mean = np.mean(f0)
baseline_std  = np.std(f0)
zsig = (corrected - baseline_mean) / baseline_std

# All first-events (e.g. one per cluster)
events_df = pd.read_csv("/Users/Lou/Desktop/first_cluster_events.csv", skiprows=[1]).reset_index(drop=True)
event_times_s = events_df["TimeStamp"].values / 1000.0

print(events_df)

# Peri-event window
t_pre = 2.0
t_post = 48.0
#time_s = [0.01, 0.02, 0.03, ...] → diff = [0.01, 0.01, ...] → median = 0.01
#Function: Auto-detects sampling rate (samples/second)
dt = np.median(np.diff(time_s))
# convert pre-event window from seconds to integer sample count
# (e.g. 2 s → 200 samples at 100 Hz) so we can index the signal array
n_pre = int(np.round(t_pre / dt))
n_post = int(np.round(t_post / dt))
#Creates x-axis with event at t=0
peri_t = np.arange(-n_pre, n_post) * dt

def extract_epoch(signal, time, t_event, n_pre, n_post):
    idx_event = np.searchsorted(time, t_event)
    start = idx_event - n_pre
    end = idx_event + n_post
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]


# Extract epochs_dff
epochs_dff = []
for t_ev in event_times_s:
    ep = extract_epoch(dff, time_s, t_ev, n_pre, n_post)
    if ep is not None:
        epochs_dff.append(ep)

epochs_dff = np.array(epochs_dff)  # (n_events, n_timepoints)

# Extract epochs_z
epochs_z = []
for t_ev in event_times_s:
    ep = extract_epoch(zsig, time_s, t_ev, n_pre, n_post)
    if ep is not None:
        epochs_z.append(ep)

epochs_z = np.array(epochs_z)  # (n_events, n_timepoints)

# Averages + SEM
mean_dff = epochs_dff.mean(axis=0)
sem_dff  = epochs_dff.std(axis=0) / np.sqrt(len(epochs_dff))

mean_z   = epochs_z.mean(axis=0)
sem_z    = epochs_z.std(axis=0) / np.sqrt(len(epochs_z))



plt.subplot(211)
plt.plot(peri_t, mean_dff, color= 'darkcyan' , lw=1, label='Mean ΔF/F')
plt.fill_between(peri_t, mean_dff - sem_dff, mean_dff + sem_dff, alpha=0.4)
plt.axvline(0, color='r', ls='--', lw=0.5, label='Event')
plt.xlabel('Time from event (s)')
plt.ylabel('ΔF/F')
plt.title(f'Peri-event average (n={len(epochs_dff)} trials)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


plt.subplot(212)
plt.plot(peri_t, mean_z, color= 'darkcyan' , lw=1, label='Mean Zscore')
plt.fill_between(peri_t, mean_z - sem_z, mean_z + sem_z, alpha=0.4)
plt.axvline(0, color='r', ls='--', lw=0.5, label='Event')
plt.xlabel('Time from event (s)')
plt.ylabel('Zscore')
plt.title(f'Peri-event average (n={len(epochs_z)} trials)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

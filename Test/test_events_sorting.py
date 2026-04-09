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

'''import pandas as pd

df_events = pd.read_csv("/Users/Lou/Desktop/cleaned_fluorescence.csv")
non_zero_events = df_events[df_events["Events"] != 0][["TimeStamp", "Events"]]
print(non_zero_events)

# Fixed: Add quotes around path string
non_zero_events.to_csv("/Users/Lou/Desktop/events_sorting.csv", index=False)
print(f"Found {len(non_zero_events)} non-zero events")'''

import pandas as pd
from pathlib import Path

# Load the events-only file
events_path = Path("/Users/Lou/Desktop/events_sorting.csv")
df_events = pd.read_csv(events_path)

# 1) Detect clusters by time gap
#    Define a new cluster whenever the gap between consecutive events > 5000 ms (5 s)
df_events["gap_ms"] = df_events["TimeStamp"].diff()
df_events["cluster_id"] = (df_events["gap_ms"] > 5000).cumsum()

# 2) Inspect number of clusters
n_clusters = df_events["cluster_id"].nunique()
print(f"Number of clusters: {n_clusters}")

# 3) Extract the first event timestamp per cluster
first_events = df_events.loc[
    df_events.groupby("cluster_id")["TimeStamp"].idxmin()
][["cluster_id", "TimeStamp", "Events"]].reset_index(drop=True)

print("First event per cluster:")
print(first_events.head())

''''# If you only want the timestamps as a Series/array:
first_timestamps = first_events["TimeStamp"].values
print("First timestamps (ms):", first_timestamps)'''

# Optional: save for later use
out_path = Path("/Users/Lou/Desktop/first_cluster_events.csv")
first_events.to_csv(out_path, index=False)
print(f"\nSaved first events per cluster to: {out_path}")

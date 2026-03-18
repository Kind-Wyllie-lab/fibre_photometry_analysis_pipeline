# event_processing.py
import pandas as pd
from pathlib import Path
from params import output_dir, event_gap_ms, data_files

# ADD to event_processing.py - after imports

def validate_event_data(df: pd.DataFrame, stage: str = "raw"):
    """Sanity checks: shape, timestamps, event counts."""
    print(f"\n--- Sanity Check [{stage}] ---")

    # Shape
    print(f"  Shape: {df.shape}")
    if df.empty:
        raise ValueError(f"[{stage}] DataFrame is empty")

    # Timestamps monotonic
    if "TimeStamp" in df.columns:
        if not df["TimeStamp"].is_monotonic_increasing:
            raise ValueError(f"[{stage}] TimeStamps not monotonically increasing")
        duration_s = (df["TimeStamp"].max() - df["TimeStamp"].min()) / 1000
        print(f"  Duration: {duration_s:.1f}s ({df['TimeStamp'].min():.0f}–{df['TimeStamp'].max():.0f}ms)")

    # Event type counts
    event_col = "Events_numeric" if "Events_numeric" in df.columns else "Events"
    if event_col in df.columns:
        counts = df[event_col].value_counts().sort_index()
        print(f"  Event counts:\n{counts.to_string()}")
        n_events = (df[event_col] != 0).sum()
        if n_events == 0:
            raise ValueError(f"[{stage}] No non-zero events found")
        if n_events < 2:
            print(f"  WARNING: Only {n_events} non-zero event(s) — check recording")

    # Cluster-specific checks
    if "cluster_id" in df.columns:
        cluster_sizes = df.groupby("cluster_id").size()
        print(f"  Clusters: {df['cluster_id'].nunique()} total")
        print(f"  Events/cluster: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")
        if df['cluster_id'].nunique() < 2:
            print(f"  WARNING: Only 1 cluster — EVENT_GAP_MS={event_gap_ms}ms may be too large")

    # Duplicate timestamps
    dupes = df["TimeStamp"].duplicated().sum() if "TimeStamp" in df.columns else 0
    if dupes > 0:
        print(f"  WARNING: {dupes} duplicate timestamps")

    print(f"--- Check passed [{stage}] ---\n")


# UPDATED process_events() with validate calls
def process_events(df_clean: pd.DataFrame, stem: str):
    print("=" * 50)
    print("EVENT PROCESSING PIPELINE")
    print("=" * 50)
    validate_event_data(df_clean, stage="cleaned_input")         # Check before
    df_events = extract_non_zero_events(df_clean)
    df_events_clustered, first_events = detect_clusters_and_first_events(df_events)
    validate_event_data(df_events_clustered, stage="clustered")  # Check after
    save_event_files(df_events_clustered, first_events, stem)
    print(f"SUCCESS: {stem} events processed!")
    return df_events_clustered, first_events


def extract_non_zero_events(df_clean: pd.DataFrame) -> pd.DataFrame:
    print("DEBUG Events columns:", df_clean.columns.tolist())

    events_df = df_clean.loc[df_clean['Events_numeric'] != 0]

    if events_df.empty:
        print("Raw Events sample:")
        print(df_clean['Events'].value_counts())
        raise ValueError("No events matched pattern")

    print(f"Extracted {len(events_df)} events")
    return events_df

def detect_clusters_and_first_events(df_events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster events by gap > EVENT_GAP_MS, return first event per cluster."""
    df = df_events.copy()

    # Gap between consecutive events (ms)
    df["gap_ms"] = df["TimeStamp"].diff()

    # New cluster when gap > threshold
    df["cluster_id"] = (df["gap_ms"] > event_gap_ms).cumsum()

    n_clusters = df["cluster_id"].nunique()
    print(f"Number of clusters: {n_clusters}")

    # First event per cluster (earliest TimeStamp)
    first_events = df.loc[
        df.groupby("cluster_id")["TimeStamp"].idxmin()
    ][["cluster_id", "TimeStamp", "Events_numeric"]].reset_index(drop=True)

    print(f"First events per cluster:\n{first_events}")
    return df, first_events  # clustered_df, first_events


def save_event_files(df_events: pd.DataFrame, first_events: pd.DataFrame, stem: str):
    events_path = output_dir / f"events_sorting_{stem}.csv"
    first_path = output_dir / f"first_cluster_events_{stem}.csv"
    df_events.to_csv(events_path, index=False)
    first_events.to_csv(first_path, index=False)
    print(f"Saved events: {events_path}")
    print(f"Saved first events: {first_path}")

def process_events(df_clean: pd.DataFrame, stem: str):
    print("=" * 50)
    print("EVENT PROCESSING PIPELINE")
    print("=" * 50)
    df_events = extract_non_zero_events(df_clean)
    df_events_clustered, first_events = detect_clusters_and_first_events(df_events)
    save_event_files(df_events_clustered, first_events, stem)
    print(f"SUCCESS: {stem} events processed!")
    return df_events_clustered, first_events

if __name__ == "__main__":
    """STANDALONE TEST - Events processing pipeline"""
    print("TESTING EVENT PROCESSING STANDALONE")
    print(f"DATA_FILES: {[f.name for f in data_files]}")

    from preprocessing import process_single_file
    df_clean = process_single_file()

    print(f"df_clean shape: {df_clean.shape}")
    print(f"Events_numeric: {df_clean['Events_numeric'].value_counts().to_dict()}")

    stem = Path(data_files[0]).stem
    clustered_events, first_events = process_events(df_clean, stem)

    print(f"\nSUCCESS!")
    print(f"  Clustered events: {len(clustered_events)}")
    print(f"  First events/clusters: {len(first_events)}")
    print(f"  Clusters: {clustered_events['cluster_id'].nunique()}")
    print(f"  Saved: {output_dir}/events_sorting_*.csv")
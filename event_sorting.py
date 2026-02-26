# event_processing.py
import pandas as pd
from pathlib import Path
from params import OUTPUT_DIR, EVENT_GAP_MS, DATA_FILES

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
            print(f"  WARNING: Only 1 cluster — EVENT_GAP_MS={EVENT_GAP_MS}ms may be too large")

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
    non_zero = df_clean[df_clean["Events_numeric"] != 0][["TimeStamp", "Events_numeric"]].copy()
    non_zero = non_zero.rename(columns={"Events_numeric": "Events"})
    if non_zero.empty:
        raise ValueError("No non-zero events found in cleaned data")
    print(f"Non-zero events: {len(non_zero)}")
    return non_zero

def detect_clusters_and_first_events(df_events: pd.DataFrame) -> pd.DataFrame:
    df = df_events.copy()
    df["gap_ms"] = df["TimeStamp"].diff()
    df["cluster_id"] = (df["gap_ms"] > EVENT_GAP_MS).cumsum()
    n_clusters = df["cluster_id"].nunique()
    print(f"Number of clusters: {n_clusters}")
    first_events = df.loc[
        df.groupby("cluster_id")["TimeStamp"].idxmin()
    ][["cluster_id", "TimeStamp", "Events"]].reset_index(drop=True)
    print(f"First events per cluster:\n{first_events.head()}")
    return df, first_events

def save_event_files(df_events: pd.DataFrame, first_events: pd.DataFrame, stem: str):
    events_path = OUTPUT_DIR / f"events_sorting_{stem}.csv"
    first_path = OUTPUT_DIR / f"first_cluster_events_{stem}.csv"
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
    from preprocessing import process_single_file
    from pathlib import Path
    df_clean = process_single_file()
    stem = Path(DATA_FILES[0]).stem
    process_events(df_clean, stem)

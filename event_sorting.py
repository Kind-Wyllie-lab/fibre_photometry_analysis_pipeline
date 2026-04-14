from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from params import event_gap_ms

EventEdge = Literal["onset", "offset"]


def validate_event_data(df: pd.DataFrame, stage: str = "raw") -> None:
    """
    Perform sanity checks for event dataframes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    stage : str, default="raw"
        Label for printing context.

    Raises
    ------
    ValueError
        If dataframe is empty or timestamps are not monotonic (when present).
    """
    print(f"\n--- Sanity Check [{stage}] ---")
    print(f"  Shape: {df.shape}")
    if df.empty:
        raise ValueError(f"[{stage}] DataFrame is empty")

    if "TimeStamp" in df.columns:
        if not df["TimeStamp"].is_monotonic_increasing:
            raise ValueError(f"[{stage}] TimeStamp is not monotonically increasing")
        duration_s = (df["TimeStamp"].max() - df["TimeStamp"].min()) / 1000.0
        print(
            f"  Duration: {duration_s:.1f}s "
            f"({df['TimeStamp'].min():.0f}–{df['TimeStamp'].max():.0f}ms)"
        )

    dupes = int(df["TimeStamp"].duplicated().sum()) if "TimeStamp" in df.columns else 0
    if dupes > 0:
        print(f"  WARNING: {dupes} duplicate timestamps")

    print(f"--- Check passed [{stage}] ---\n")


def extract_edge_events_from_binary_column(
    df_clean: pd.DataFrame,
    event_column: str,
    edge: EventEdge,
    timestamp_column: str = "TimeStamp",
) -> pd.DataFrame:
    """
    Extract edge events (onset or offset) from a binary event column.

    This detects transitions in `event_column`:
    - onset  : 0 -> 1
    - offset : 1 -> 0

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Cleaned dataframe with timestamp and event columns.
    event_column : str
        Name of the column containing a binary event signal (0/1).
    edge : {"onset", "offset"}
        Which edge to extract.
    timestamp_column : str, default="TimeStamp"
        Timestamp column name.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing extracted edges with at least:
        ``TimeStamp``, ``edge``, and the original event column.
    """
    if timestamp_column not in df_clean.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")
    if event_column not in df_clean.columns:
        raise ValueError(f"Missing event column: {event_column}")

    df = df_clean[[timestamp_column, event_column]].copy()
    signal = pd.to_numeric(df[event_column], errors="coerce").fillna(0).astype(int).clip(0, 1)

    prev_signal = signal.shift(1, fill_value=0)

    if edge == "onset":
        mask = (prev_signal == 0) & (signal == 1)
    elif edge == "offset":
        mask = (prev_signal == 1) & (signal == 0)
    else:
        raise ValueError(f"Unsupported edge: {edge!r}")

    edges = df.loc[mask, [timestamp_column]].copy()
    edges[event_column] = signal.loc[mask].to_numpy()
    edges["edge"] = edge
    return edges.reset_index(drop=True)


def cluster_events_by_gap(
    df_events: pd.DataFrame,
    gap_threshold_ms: float,
    timestamp_column: str = "TimeStamp",
    cluster_column: str = "cs_n",
) -> pd.DataFrame:
    """
    Cluster events by temporal gaps.

    Parameters
    ----------
    df_events : pandas.DataFrame
        Dataframe with one row per event and a timestamp column.
    gap_threshold_ms : float
        Gap threshold (ms). A new cluster starts when consecutive events are separated
        by more than this value.
    timestamp_column : str, default="TimeStamp"
        Timestamp column name.
    cluster_column : str, default="cs_n"
        Output cluster id column name.

    Returns
    -------
    pandas.DataFrame
        Copy of df_events with added columns: ``gap_ms`` and ``cs_n``.
    """
    if df_events.empty:
        return df_events.copy()

    df = df_events.sort_values(timestamp_column).reset_index(drop=True).copy()
    df["gap_ms"] = df[timestamp_column].diff()
    df[cluster_column] = (df["gap_ms"] > gap_threshold_ms).cumsum().astype(int)
    return df


def summarize_clusters_first_last(
    df_clustered: pd.DataFrame,
    timestamp_column: str = "TimeStamp",
    cluster_column: str = "cs_n",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get first and last event per cluster.

    Parameters
    ----------
    df_clustered : pandas.DataFrame
        Clustered dataframe produced by `cluster_events_by_gap`.
    timestamp_column : str, default="TimeStamp"
        Timestamp column name.
    cluster_column : str, default="cs_n"
        Cluster id column name.

    Returns
    -------
    first_events : pandas.DataFrame
        One row per cluster with the earliest timestamp.
    last_events : pandas.DataFrame
        One row per cluster with the latest timestamp.
    """
    if df_clustered.empty:
        return df_clustered.copy(), df_clustered.copy()

    first_events = df_clustered.loc[
        df_clustered.groupby(cluster_column)[timestamp_column].idxmin()
    ].reset_index(drop=True)

    last_events = df_clustered.loc[
        df_clustered.groupby(cluster_column)[timestamp_column].idxmax()
    ].reset_index(drop=True)

    return first_events, last_events


def process_event_stream(
    df_clean: pd.DataFrame,
    event_column: str,
    gap_threshold_ms: float,
    stream_name: str,
    timestamp_column: str = "TimeStamp",
) -> dict[str, pd.DataFrame]:
    """
    Process one event stream (binary column) into onset/offset events and clusters.

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Cleaned dataframe including `TimeStamp` and event columns.
    event_column : str
        Binary column name (0/1).
    gap_threshold_ms : float
        Cluster separation gap (ms).
    stream_name : str
        Label for outputs and saving.
    timestamp_column : str, default="TimeStamp"
        Timestamp column name.

    Returns
    -------
    dict
        Dictionary containing:
        - ``onsets``: onset events dataframe
        - ``offsets``: offset events dataframe
        - ``onsets_clustered``: clustered onsets dataframe
        - ``offsets_clustered``: clustered offsets dataframe
        - ``cluster_first_onsets``: 1st onset per cluster
        - ``cluster_last_onsets``: last onset per cluster
        - ``cluster_first_offsets``: 1st offset per cluster
        - ``cluster_last_offsets``: last offset per cluster
    """
    onsets = extract_edge_events_from_binary_column(df_clean, event_column=event_column, edge="onset", timestamp_column=timestamp_column)
    offsets = extract_edge_events_from_binary_column(df_clean, event_column=event_column, edge="offset", timestamp_column=timestamp_column)

    onsets_clustered = cluster_events_by_gap(onsets, gap_threshold_ms=gap_threshold_ms, timestamp_column=timestamp_column, cluster_column="cs_n")
    offsets_clustered = cluster_events_by_gap(offsets, gap_threshold_ms=gap_threshold_ms, timestamp_column=timestamp_column, cluster_column="cs_n")

    cluster_first_onsets, cluster_last_onsets = summarize_clusters_first_last(onsets_clustered, timestamp_column=timestamp_column, cluster_column="cs_n")
    cluster_first_offsets, cluster_last_offsets = summarize_clusters_first_last(offsets_clustered, timestamp_column=timestamp_column, cluster_column="cs_n")

    return {
        f"{stream_name}_onsets": onsets,
        f"{stream_name}_offsets": offsets,
        f"{stream_name}_onsets_clustered": onsets_clustered,
        f"{stream_name}_offsets_clustered": offsets_clustered,
        f"{stream_name}_cluster_first_onsets": cluster_first_onsets,
        f"{stream_name}_cluster_last_onsets": cluster_last_onsets,
        f"{stream_name}_cluster_first_offsets": cluster_first_offsets,
        f"{stream_name}_cluster_last_offsets": cluster_last_offsets,
    }


def save_event_tables(output_dir: str | Path, tables: dict[str, pd.DataFrame]) -> None:
    """
    Save multiple event tables to disk.

    Parameters
    ----------
    output_dir : str or Path
        Directory to write CSVs to.
    tables : dict[str, pandas.DataFrame]
        Dictionary mapping filename stems to dataframes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        out_path = output_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


def process_ttl_events(
    df_clean: pd.DataFrame,
    output_dir: str | Path,
    led_column: str = "Events_LED",
    freezing_column: str = "freezing_event",
    gap_threshold_ms: float = event_gap_ms,
) -> dict[str, pd.DataFrame]:
    """
    Event processing pipeline handling both LED (CS) and freezing event streams.

    LED stream:
    - CS onsets: LED 0->1 edges
    - CS offsets: LED 1->0 edges
    - Clustering for onsets and offsets is performed independently using `gap_threshold_ms`.
    - "First CS onsets" (your previous `first_events`) are provided as the first onset per cluster.
    - "CS offsets" are similarly summarized via clustered offset events.

    Freezing stream:
    - `freezing_event` is expected to be a binary state (0/1) if you have it as a state,
      OR an edge-coded column where you stored 1=onset and 0=offset.
    - This function treats it as a *state* by default if it contains continuous 0/1 states.
      If it is *edge-coded* (sparse 0/1 only at event rows), you should convert it to a state
      beforehand or use the onset/offset extraction directly on a state column.

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Cleaned dataframe containing at least TimeStamp plus event columns.
    output_dir : str or Path
        Output directory for generated CSVs.
    led_column : str, default="Events_LED"
        LED binary column name.
    freezing_column : str, default="freezing_event"
        Freezing event column name.
    gap_threshold_ms : float, default=event_gap_ms
        Gap threshold for clustering.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary of all generated event tables.
    """
    validate_event_data(df_clean, stage="cleaned_input")

    tables: dict[str, pd.DataFrame] = {}

    # --- LED CS stream: onsets + offsets ---
    led_tables = process_event_stream(
        df_clean=df_clean,
        event_column=led_column,
        gap_threshold_ms=gap_threshold_ms,
        stream_name="cs_led",
        timestamp_column="TimeStamp",
    )
    tables.update(led_tables)

    # --- Freezing stream: interpret as state when possible ---
    # If freezing_event is sparse-coded (NaN most rows), convert NaN->previous value to build a state.
    if freezing_column in df_clean.columns:
        freezing_series = pd.to_numeric(df_clean[freezing_column], errors="coerce")

        # If column is sparse with NaNs, forward-fill to create a state-like series (starts at 0).
        if freezing_series.isna().any():
            freezing_state = freezing_series.ffill().fillna(0).astype(int).clip(0, 1)
        else:
            freezing_state = freezing_series.fillna(0).astype(int).clip(0, 1)

        df_freeze_state = df_clean[["TimeStamp"]].copy()
        df_freeze_state["freezing_state"] = freezing_state

        freeze_tables = process_event_stream(
            df_clean=df_freeze_state,
            event_column="freezing_state",
            gap_threshold_ms=gap_threshold_ms,
            stream_name="freezing",
            timestamp_column="TimeStamp",
        )
        tables.update(freeze_tables)

    save_event_tables(output_dir, tables)

    print("SUCCESS: events processed (LED CS + freezing)")
    return tables

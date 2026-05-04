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


def _cluster_and_get_first_last_event_times_s(
    df_edges: pd.DataFrame,
    gap_threshold_ms: float,
    timestamp_column: str = "TimeStamp",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster edge events and return first/last timestamp per cluster in seconds.

    Parameters
    ----------
    df_edges : pandas.DataFrame
        Edge dataframe with a TimeStamp column (ms).
    gap_threshold_ms : float
        Gap threshold (ms) for clustering.
    timestamp_column : str, default="TimeStamp"
        Name of timestamp column.

    Returns
    -------
    first_times_s : numpy.ndarray
        First event time per cluster in seconds.
    last_times_s : numpy.ndarray
        Last event time per cluster in seconds.
    """
    if df_edges is None or df_edges.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    clustered = cluster_events_by_gap(
        df_edges,
        gap_threshold_ms=gap_threshold_ms,
        timestamp_column=timestamp_column,
        cluster_column="cs_n",
    )
    first_df, last_df = summarize_clusters_first_last(
        clustered,
        timestamp_column=timestamp_column,
        cluster_column="cs_n",
    )

    first_times_s = first_df[timestamp_column].to_numpy(dtype=float) / 1000.0 if not first_df.empty else np.array([], dtype=float)
    last_times_s = last_df[timestamp_column].to_numpy(dtype=float) / 1000.0 if not last_df.empty else np.array([], dtype=float)

    return first_times_s, last_times_s


def process_ttl_events(
    df_clean: pd.DataFrame,
    output_dir: str | Path,
    gap_threshold_ms: float = event_gap_ms,
) -> dict[str, dict[str, Optional[np.ndarray]]]:
    """
    Extract CS (LED) and freezing onsets/offsets as simple event time arrays.

    This function returns a minimal event representation suitable for epoching:
    - CS onsets  : first onset per "CS burst"/cluster (gap-based clustering of LED rising edges)
    - CS offsets : last offset per "CS burst"/cluster (gap-based clustering of LED falling edges)
    - Freezing onsets/offsets: edges derived from a freezing *state* (0/1), which is
      constructed by forward-filling a sparse `freezing_event` column if needed.

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Clean dataframe containing at least ``TimeStamp`` and the TTL-derived columns.
    output_dir : str or Path
        Output directory for optional debug CSV saving.
    led_column : str, default="Events_LED"
        Binary LED column (0/1).
    freezing_column : str, default="freezing_event"
        Column containing freezing transitions (sparse) or state (dense).
    gap_threshold_ms : float, default=event_gap_ms
        Gap threshold (ms) used to cluster repeated TTL pulses into a single CS burst.

    Returns
    -------
    dict
        Dictionary with two keys:

        - ``"LED_events"``: dict with:
            - ``"cs_onsets"`` : numpy.ndarray (seconds)
            - ``"cs_offsets"``: numpy.ndarray (seconds)

        - ``"freezing_events"``: dict with:
            - ``"freezing_onsets"`` : numpy.ndarray (seconds) or None
            - ``"freezing_offsets"``: numpy.ndarray (seconds) or None

        If no freezing column exists or no freezing transitions are found,
        freezing arrays are returned as None.
    """
    validate_event_data(df_clean, stage="cleaned_input")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # LED (CS) onsets/offsets
    # -----------------------
    led_onset_edges = extract_edge_events_from_binary_column(
        df_clean,
        event_column='Events_LED',
        edge="onset",
        timestamp_column="TimeStamp",
    )
    led_offset_edges = extract_edge_events_from_binary_column(
        df_clean,
        event_column='Events_LED',
        edge="offset",
        timestamp_column="TimeStamp",
    )

    # CS onsets: FIRST onset per cluster (burst)
    cs_onsets_s, _ = _cluster_and_get_first_last_event_times_s(
        df_edges=led_onset_edges,
        gap_threshold_ms=gap_threshold_ms,
        timestamp_column="TimeStamp",
    )

    # CS offsets: LAST offset per cluster (burst)
    _, cs_offsets_s = _cluster_and_get_first_last_event_times_s(
        df_edges=led_offset_edges,
        gap_threshold_ms=gap_threshold_ms,
        timestamp_column="TimeStamp",
    )

    raw_led_mask = pd.to_numeric(df_clean['Events_LED'], errors="coerce").fillna(0).astype(int).clip(0, 1) == 1
    raw_led_events_s = df_clean.loc[raw_led_mask, "TimeStamp"].to_numpy(dtype=float) / 1000.0

    # -----------------------
    # Freezing onsets/offsets
    # -----------------------
    freezing_onsets_s: Optional[np.ndarray] = None
    freezing_offsets_s: Optional[np.ndarray] = None

    if 'freezing_event' in df_clean.columns:
        freezing_series = pd.to_numeric(df_clean['freezing_event'], errors="coerce")

        # Build a binary state even if the column is sparse (NaN except at transitions)
        freezing_state = freezing_series.ffill().fillna(0).astype(int).clip(0, 1)

        df_freeze_state = df_clean[["TimeStamp"]].copy()
        df_freeze_state["freezing_state"] = freezing_state

        freeze_onset_edges = extract_edge_events_from_binary_column(
            df_freeze_state,
            event_column="freezing_state",
            edge="onset",
            timestamp_column="TimeStamp",
        )
        freeze_offset_edges = extract_edge_events_from_binary_column(
            df_freeze_state,
            event_column="freezing_state",
            edge="offset",
            timestamp_column="TimeStamp",
        )

        freezing_onsets_s = freeze_onset_edges["TimeStamp"].to_numpy(dtype=float) / 1000.0 if not freeze_onset_edges.empty else None
        freezing_offsets_s = freeze_offset_edges["TimeStamp"].to_numpy(dtype=float) / 1000.0 if not freeze_offset_edges.empty else None

    #cleanup wrong triggers at the beginning that occur sometimes
    cs_onsets_s = np.array([i for i in cs_onsets_s if i > 90])
    cs_offsets_s = np.array([i for i in cs_offsets_s if i > 120])
    shock = cs_onsets_s + 9 # during conditioning shock comes 9 seconds after cs onset, for recall its "expected shock"

    return {"raw_LED_events": raw_led_events_s,
            "cs_onsets": cs_onsets_s,
            "cs_offsets": cs_offsets_s,
            "shock": shock,
            "freezing_onsets": freezing_onsets_s,
            "freezing_offsets": freezing_offsets_s,
            }

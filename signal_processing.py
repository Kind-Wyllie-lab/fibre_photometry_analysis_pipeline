import numpy as np
import pandas as pd
from params import (CALCIUM_CHANNEL, REF_CHANNEL, BASELINE_SAMPLES,
                    T_PRE_EVENT_S, T_POST_EVENT_S, SKIP_FIRST_EVENT,
                    SYNC_SIGNAL_TO_FIRST_EVENT,SELECTED_CLUSTERS)

def select_events_from_params(first_events: pd.DataFrame) -> pd.DataFrame:
    """Apply selection rules from params.py to first_events."""
    df = first_events.copy()

    # cluster-based selection
    if SELECTED_CLUSTERS is not None:
        df = df[df["cluster_id"].isin(SELECTED_CLUSTERS)]
    if df.empty:
        raise ValueError("No events after applying SELECTED_* filters in params.py")

    return df

def sync_signal_to_first_event(time_s: np.ndarray, dff: np.ndarray,
                                zscore: np.ndarray,
                                first_event_time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim signal to start at first valid event timestamp.
    Synchronizes calcium trace to TTL pulse onset — discards pre-event recording."""
    if not SYNC_SIGNAL_TO_FIRST_EVENT:
        return time_s, dff, zscore
    idx_start = np.searchsorted(time_s, first_event_time_s)
    if idx_start >= len(time_s):
        raise ValueError(f"First event time {first_event_time_s:.2f}s outside signal duration")
    print(f"Synced signal to first event at {first_event_time_s:.2f}s "
          f"(sample {idx_start}/{len(time_s)}, "
          f"{len(time_s) - idx_start} samples remaining)")
    return time_s[idx_start:], dff[idx_start:], zscore[idx_start:]

def compute_corrected_signal(df_clean: pd.DataFrame) -> np.ndarray:
    calcium = df_clean[CALCIUM_CHANNEL].values    # 470nm: calcium + motion
    reference = df_clean[REF_CHANNEL].values      # 410nm: isosbestic control (motion only)

    # Simplified motion correction: MotionCorrected = F470 - F410
    # RWD manual (§8.1.4): full correction uses robust linear fit of 410 onto 470
    # then subtracts: MotionCorrected = F470 - fitted410
    # Here we use direct subtraction as an approximation (no baseline correction applied)
    corrected = calcium - reference

    if corrected.shape[0] < BASELINE_SAMPLES:
        raise ValueError(f"Signal too short: {corrected.shape[0]} < {BASELINE_SAMPLES} baseline samples")
    print(f"Corrected signal: shape={corrected.shape}, range=[{corrected.min():.3f}, {corrected.max():.3f}]")
    return corrected


def compute_dff_and_zscore(corrected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = corrected[:BASELINE_SAMPLES]

    # ΔF/F = MotionCorrected / median(F_baseline)
    # RWD manual (§8.1.4): ΔF/F = MotionCorrected / fitted410
    # Here baseline median approximates F0 (no fitted410 available)
    # median is robust to transient artifacts in the baseline window
    dff = corrected / np.median(baseline)

    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    if baseline_std == 0:
        raise ValueError("Baseline std=0: flat signal, check recording")

    # Z-score = (x - mean) / std
    # RWD manual (§8.2 peri-event): mean and std computed over the baseline interval
    # Here baseline interval = first BASELINE_SAMPLES samples (set in params.py)
    zscore = (corrected - baseline_mean) / baseline_std

    print(f"ΔF/F range: [{dff.min():.3f}, {dff.max():.3f}]")
    print(f"Z-score range: [{zscore.min():.3f}, {zscore.max():.3f}]")
    return dff, zscore


def extract_epoch(signal: np.ndarray, time: np.ndarray,
                  t_event: float, n_pre: int, n_post: int):
    # Find sample index closest to event timestamp using binary search
    idx_event = np.searchsorted(time, t_event)
    start = idx_event - n_pre
    end = idx_event + n_post
    if start < 0 or end > len(signal):
        return None  # Skip epochs that fall outside signal bounds
    return signal[start:end]


def extract_all_epochs(signal: np.ndarray, time_s: np.ndarray,
                       event_times_s: np.ndarray,
                       n_pre: int, n_post: int) -> np.ndarray:
    epochs = []
    for t_ev in event_times_s:
        epoch = extract_epoch(signal, time_s, t_ev, n_pre, n_post)
        if epoch is not None:
            epochs.append(epoch)
    if not epochs:
        raise ValueError("No valid epochs extracted — check event times vs signal duration")

    result = np.array(epochs)  # shape: (n_events, n_timepoints)
    print(f"Epochs extracted: {result.shape[0]} trials, {result.shape[1]} samples each")
    return result

def select_events(first_events: pd.DataFrame,
                  cluster_ids: list[int] | None = None) -> pd.DataFrame:
    """Filter events by cluster_id list and/or absolute time window (s)."""
    df = first_events.copy()

    if cluster_ids is not None:
        df = df[df["cluster_id"].isin(cluster_ids)]


    if df.empty:
        raise ValueError("No events after selection — relax filters")

    return df


def filter_first_event(first_events: pd.DataFrame) -> pd.DataFrame:
    """Drop first cluster event if SKIP_FIRST_EVENT=True in params (spurious trigger)."""
    if SKIP_FIRST_EVENT:
        deleted_row = first_events.iloc[[0]]  # keep as DataFrame for display
        print(f"[SKIP_FIRST_EVENT] Deleted row:\n{deleted_row[['cluster_id', 'TimeStamp', 'Events']].to_string(index=False)}")
        filtered = first_events.iloc[1:].reset_index(drop=True)
        print(f"Skipped first event: {len(first_events)} → {len(filtered)} events")
        return filtered
    return first_events


def process_signals(df_clean: pd.DataFrame,
                    first_events: pd.DataFrame,
                    stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray]:
    """Returns: epochs_dff, epochs_z, peri_t, dff_full, zscore_full"""
    print("=" * 50)
    print("SIGNAL PROCESSING PIPELINE")
    print("=" * 50)

    time_s = df_clean["TimeStamp"].values / 1000.0
    dt = np.median(np.diff(time_s))
    n_pre  = int(np.round(T_PRE_EVENT_S / dt))
    n_post = int(np.round(T_POST_EVENT_S / dt))
    peri_t = np.arange(-n_pre, n_post) * dt
    print(f"dt={dt * 1000:.2f}ms, n_pre={n_pre}, n_post={n_post}")

    corrected = compute_corrected_signal(df_clean)
    dff, zscore = compute_dff_and_zscore(corrected)  # full-length, not trimmed

    events_selected = select_events_from_params(first_events)
    events_to_use = filter_first_event(events_selected)
    if events_to_use.empty:
        raise ValueError("No events remaining after selection/filtering — check params")

    if events_to_use.empty:
        raise ValueError("No events remaining after filtering — check SKIP_FIRST_EVENT")

    first_event_time_s = events_to_use["TimeStamp"].iloc[0] / 1000.0
    # trim only for epoch extraction — dff/zscore full arrays preserved above
    time_sync, dff_sync, zscore_sync = sync_signal_to_first_event(
        time_s, dff, zscore, first_event_time_s
    )

    print(events_to_use[["cluster_id", "TimeStamp", "Events"]].to_string())
    event_times_s = events_to_use["TimeStamp"].values / 1000.0
    epochs_dff = extract_all_epochs(dff_sync, time_sync, event_times_s, n_pre, n_post)
    epochs_z   = extract_all_epochs(zscore_sync, time_sync, event_times_s, n_pre, n_post)

    print(f"SUCCESS: {stem} signals processed!")
    return epochs_dff, epochs_z, peri_t, dff, zscore  # dff/zscore = full, unsynced


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/Lou/PycharmProjects/Fibre_photometry_M2")  # explicit project root

    from preprocessing import process_single_file
    from event_sorting import process_events

    input_df = process_single_file()
    input_stem = "Fluorescence"
    _, input_first_events = process_events(input_df, input_stem)
    process_signals(input_df, input_first_events, input_stem)


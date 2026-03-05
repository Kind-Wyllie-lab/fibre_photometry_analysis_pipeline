# signal_processing.py
import numpy as np
from typing import Tuple, Optional
from scipy.stats import linregress
import pandas as pd
from params import (CALCIUM_CHANNEL, REF_CHANNEL, BASELINE_SAMPLES,
                    T_PRE_EVENT_S, T_POST_EVENT_S, SKIP_FIRST_EVENT,
                    SYNC_SIGNAL_TO_FIRST_EVENT, SELECTED_CLUSTERS)


def select_events_from_params(first_events: pd.DataFrame) -> pd.DataFrame:
    """Apply selection rules from params.py to first_events."""
    df = first_events.copy()
    if SELECTED_CLUSTERS is not None:
        df = df[df["cluster_id"].isin(SELECTED_CLUSTERS)]
    if df.empty:
        raise ValueError("No events after applying SELECTED_* filters in params.py")
    return df


def filter_first_event(first_events: pd.DataFrame) -> pd.DataFrame:
    """Drop first cluster event if SKIP_FIRST_EVENT=True in params (spurious trigger)."""
    if SKIP_FIRST_EVENT:
        deleted_row = first_events.iloc[[0]]
        print(f"[SKIP_FIRST_EVENT] Deleted row:\n{deleted_row[['cluster_id', 'TimeStamp', 'Events']].to_string(index=False)}")
        filtered = first_events.iloc[1:].reset_index(drop=True)
        print(f"Skipped first event: {len(first_events)} → {len(filtered)} events")
        return filtered
    return first_events


def robust_fit_410_to_470(df_clean: pd.DataFrame, max_iter: int = 10, c: float = 4.685) -> np.ndarray:
    """Robust IRLS fit of REF (x) to CALCIUM (y); returns fitted410."""
    calcium = df_clean[CALCIUM_CHANNEL].values
    reference = df_clean[REF_CHANNEL].values
    beta = linregress(reference, calcium).slope
    for _ in range(max_iter):
        residuals = calcium - beta * reference
        weights = np.where(np.abs(residuals) < c, (1 - (residuals / c)**2)**2, 0)
        denom = np.sum(weights * reference**2)
        if denom > 0:
            beta = np.sum(weights * reference * calcium) / denom
    return np.clip(beta * reference, 0, None)


def compute_dff_and_zscore(df_clean: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute motion-corrected ΔF/F and z-score.

    Returns:
        dff_fitted: ΔF/F using fitted410 as F0 (photobleaching correction)
        dff_baseline: ΔF/F using baseline median as F0 (peri-event analysis)
        zscore: z-score computed from dff_baseline
    """
    fitted410 = robust_fit_410_to_470(df_clean)
    corrected = df_clean[CALCIUM_CHANNEL].values - fitted410
    baseline = corrected[:BASELINE_SAMPLES]
    f0_baseline = np.median(baseline)
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    if baseline_std == 0:
        raise ValueError("Baseline std=0")

    dff_fitted = corrected / fitted410
    dff_baseline = corrected / f0_baseline
    zscore = (dff_baseline - baseline_mean) / baseline_std

    print(f"ΔF/F (fitted): [{dff_fitted.min():.3f}, {dff_fitted.max():.3f}]")
    print(f"ΔF/F (baseline): [{dff_baseline.min():.3f}, {dff_baseline.max():.3f}]")
    print(f"Z-score: [{zscore.min():.3f}, {zscore.max():.3f}]")
    return dff_fitted, dff_baseline, zscore


def extract_epoch(signal: np.ndarray, time: np.ndarray,
                  t_event: float, n_pre: int, n_post: int) -> Optional[np.ndarray]:
    """Extract single epoch around event timestamp."""
    idx_event = np.searchsorted(time, t_event)
    start = idx_event - n_pre
    end = idx_event + n_post
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]


def extract_all_epochs(signal: np.ndarray, time_s: np.ndarray,
                       event_times_s: np.ndarray,
                       n_pre: int, n_post: int) -> np.ndarray:
    """Extract all epochs around event timestamps."""
    epochs = []
    for t_ev in event_times_s:
        epoch = extract_epoch(signal, time_s, t_ev, n_pre, n_post)
        if epoch is not None:
            epochs.append(epoch)
    if not epochs:
        raise ValueError("No valid epochs extracted — check event times vs signal duration")

    result = np.array(epochs)
    print(f"Epochs extracted: {result.shape[0]} trials, {result.shape[1]} samples each")
    return result


def process_signals(df_clean: pd.DataFrame,
                    first_events: pd.DataFrame,
                    stem: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Process fiber photometry signals: motion correction, normalization, epoch extraction.

    Returns:
        epochs_dff: Peri-event ΔF/F epochs (n_trials x n_timepoints)
        epochs_z: Peri-event z-score epochs (n_trials x n_timepoints)
        peri_t: Time vector for epochs (s, relative to event)
        dff_baseline: Full-trace ΔF/F (baseline median normalization)
        dff_fitted: Full-trace ΔF/F (fitted410 normalization)
        zscore: Full-trace z-score
        filtered_events: DataFrame of events used for epoch extraction
    """
    print("=" * 50)
    print("SIGNAL PROCESSING PIPELINE")
    print("=" * 50)

    time_s = df_clean["TimeStamp"].values / 1000.0
    dt = np.median(np.diff(time_s))
    n_pre = int(np.round(T_PRE_EVENT_S / dt))
    n_post = int(np.round(T_POST_EVENT_S / dt))
    peri_t = np.arange(-n_pre, n_post) * dt
    print(f"dt={dt * 1000:.2f}ms, n_pre={n_pre}, n_post={n_post}")

    dff_fitted, dff_baseline, zscore = compute_dff_and_zscore(df_clean)

    events_selected = select_events_from_params(first_events)
    events_to_use = filter_first_event(events_selected)
    if events_to_use.empty:
        raise ValueError("No events remaining after selection/filtering — check params")

    print(events_to_use[["cluster_id", "TimeStamp", "Events"]].to_string())
    event_times_s = events_to_use["TimeStamp"].values / 1000.0

    epochs_dff = extract_all_epochs(dff_baseline, time_s, event_times_s, n_pre, n_post)
    epochs_z = extract_all_epochs(zscore, time_s, event_times_s, n_pre, n_post)

    print(f"SUCCESS: {stem} signals processed!")
    return epochs_dff, epochs_z, peri_t, dff_baseline, dff_fitted, zscore, events_to_use


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/Lou/PycharmProjects/Fibre_photometry_M2")

    from preprocessing import process_single_file
    from event_sorting import process_events

    input_df = process_single_file()
    input_stem = "Fluorescence"
    _, input_first_events = process_events(input_df, input_stem)
    process_signals(input_df, input_first_events, input_stem)

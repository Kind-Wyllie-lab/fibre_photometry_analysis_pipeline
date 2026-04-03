# signal_processing.py
import numpy as np
from typing import Tuple, Optional
from scipy.stats import linregress
import pandas as pd
from params import (calcium_channel, ref_channel, baseline_samples,
                    time_pre_event_s, time_post_event_s, selected_clusters, sample_rate_hz)

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def fit_reference_robust_bisquare(
    fluorescence: np.ndarray,
    reference: np.ndarray,
    tuning_constant: float = 4.685,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """
    Robustly fit a reference channel onto a fluorescence channel using
    iteratively reweighted least squares with Tukey bisquare weights.

    The model is constrained to a zero intercept:

    $$ fluorescence \\\\approx \\\\beta \\\\cdot reference $$

    Parameters
    ----------
    fluorescence : np.ndarray
        One-dimensional fluorescence signal of interest.
    reference : np.ndarray
        One-dimensional control/reference signal aligned to `fluorescence`.
    tuning_constant : float, default=4.685
        Tukey bisquare tuning constant controlling outlier rejection.
    max_iterations : int, default=100
        Maximum number of robust reweighting iterations.
    tolerance : float, default=1e-10
        Convergence threshold on the fitted slope.

    Returns
    -------
    fitted_reference : np.ndarray
        Robustly fitted reference signal, equal to `beta * reference`.
    beta : float
        Final fitted slope coefficient.

    Raises
    ------
    ValueError
        If the input arrays are not one-dimensional, do not share the same
        length, or if the fit is ill-posed.

    Notes
    -----
    This implements the motion-fitting step described in the specification:
    "Motion fitting algorithm: robust linear fit".

    The returned fitted signal can be used to compute the motion-corrected
    residual:

    $$ motion\\\\_corrected = fluorescence - fitted\\\\_reference $$
    """
    fluorescence = np.asarray(fluorescence, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if fluorescence.ndim != 1 or reference.ndim != 1:
        raise ValueError("`fluorescence` and `reference` must be one-dimensional.")
    if fluorescence.shape[0] != reference.shape[0]:
        raise ValueError("`fluorescence` and `reference` must have the same length.")

    initial_fit = linregress(reference, fluorescence)
    beta = float(initial_fit.slope)

    for _ in range(max_iterations):
        residuals = fluorescence - beta * reference
        robust_scale = np.median(np.abs(residuals - np.median(residuals))) / 0.6745

        if robust_scale <= np.finfo(float).eps:
            break

        standardized_residuals = residuals / (tuning_constant * robust_scale)
        weights = np.where(
            np.abs(standardized_residuals) < 1.0,
            (1.0 - standardized_residuals**2) ** 2,
            0.0,
        )

        denominator = np.sum(weights * reference**2)
        if denominator <= np.finfo(float).eps:
            raise ValueError("Robust reference fitting failed: zero weighted denominator.")

        updated_beta = np.sum(weights * reference * fluorescence) / denominator

        if np.abs(updated_beta - beta) < tolerance:
            beta = float(updated_beta)
            break

        beta = float(updated_beta)

    fitted_reference = beta * reference
    return fitted_reference, beta


def estimate_baseline_asls(
    signal: np.ndarray,
    smoothness_penalty: float = 1e6,
    asymmetry_penalty: float = 0.01,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Estimate a slowly varying baseline using asymmetric least squares.

    This method is a practical iterative weighted least squares baseline
    estimator consistent with the specification:
    "Baseline fitting algorithm: iterative weighted least squares method".

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional input signal.
    smoothness_penalty : float, default=1e6
        Smoothness regularization parameter. Larger values produce a smoother
        baseline.
    asymmetry_penalty : float, default=0.01
        Asymmetry parameter in the interval (0, 1). Smaller values force the
        baseline to remain below the signal more strongly.
    max_iterations : int, default=50
        Maximum number of reweighting iterations.
    tolerance : float, default=1e-6
        Relative convergence threshold on the weight vector.

    Returns
    -------
    baseline : np.ndarray
        Estimated baseline with the same shape as `signal`.

    Raises
    ------
    ValueError
        If `signal` is not one-dimensional or if the algorithm parameters are
        invalid.

    References
    ----------
    Eilers, P. H. C., & Boelens, H. F. M. (2005).
    Baseline correction with asymmetric least squares smoothing.
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError("`signal` must be one-dimensional.")
    if smoothness_penalty <= 0:
        raise ValueError("`smoothness_penalty` must be positive.")
    if not (0.0 < asymmetry_penalty < 1.0):
        raise ValueError("`asymmetry_penalty` must be in the open interval (0, 1).")

    n_samples = signal.size
    if n_samples < 3:
        return signal.copy()

    second_difference = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n_samples - 2, n_samples))
    penalty_matrix = smoothness_penalty * (second_difference.T @ second_difference)

    weights = np.ones(n_samples, dtype=float)

    for _ in range(max_iterations):
        weighted_diagonal = diags(weights, 0)
        baseline = spsolve(weighted_diagonal + penalty_matrix, weights * signal)
        updated_weights = np.where(
            signal > baseline,
            asymmetry_penalty,
            1.0 - asymmetry_penalty,
        )

        weight_change = np.linalg.norm(updated_weights - weights) / (np.linalg.norm(weights) + np.finfo(float).eps)
        weights = updated_weights
        if weight_change < tolerance:
            break

    return np.asarray(baseline, dtype=float)


def optionally_smooth_signal(
    signal: np.ndarray,
    enable_smoothing: bool = False,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Optionally smooth a one-dimensional signal using a Savitzky-Golay filter.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional input signal.
    enable_smoothing : bool, default=False
        If True, smooth the signal; otherwise return a copy unchanged.
    window_length : int, default=11
        Odd filter window length for Savitzky-Golay smoothing.
    polyorder : int, default=3
        Polynomial order for Savitzky-Golay smoothing.

    Returns
    -------
    smoothed_signal : np.ndarray
        Output signal after optional smoothing.
    """
    signal = np.asarray(signal, dtype=float)
    if not enable_smoothing:
        return signal.copy()

    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, signal.size if signal.size % 2 == 1 else signal.size - 1)
    if window_length <= polyorder:
        return signal.copy()

    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)


def compute_photometry_dff_and_zscore(
    df_clean: pd.DataFrame,
    calcium_channel: str,
    reference_channel: str,
    baseline_interval_samples: int | None = None,
    control_source: str = "410",
    apply_baseline_correction: bool = False,
    enable_smoothing: bool = False,
    smoothing_window_length: int = 11,
    smoothing_polyorder: int = 3,
    background_calcium: float | None = None,
    background_reference: float | None = None,
    baseline_smoothness_penalty: float = 1e6,
    baseline_asymmetry_penalty: float = 0.01,
) -> dict[str, np.ndarray | float]:
    """
    Compute fiber photometry preprocessing outputs according to the specified
    ΔF/F and Z-score definitions.

    Supported ΔF/F modes
    --------------------
    1. Control is 410, with baseline correction:

       $$ \\\\Delta F/F_0 = \\\\frac{
       \\\\mathrm{BaselineCorrected}(F_v) - \\\\mathrm{fitted410}(\\\\mathrm{BaselineCorrected}(410))
       }{
       \\\\mathrm{median}(F_{full})
       } $$

    2. Control is 410, with no baseline correction:

       $$ \\\\Delta F/F_0 = \\\\frac{F_v - \\\\mathrm{fitted410}}{\\\\mathrm{fitted410}} $$

    3. Control is Baseline, with baseline correction:

       $$ \\\\Delta F/F_0 = \\\\frac{
       \\\\mathrm{BaselineCorrected}(F_v) - \\\\mathrm{BaselineFitted}
       }{
       \\\\mathrm{median}(\\\\mathrm{Baseline})
       } $$

    4. Control is Baseline, with no baseline correction:

       $$ \\\\Delta F/F_0 = \\\\frac{
       F_v - \\\\mathrm{median}(\\\\mathrm{Baseline})
       }{
       \\\\mathrm{median}(\\\\mathrm{Baseline})
       } $$

    Z-score
    -------
    The Z-score is computed from the full ΔF/F trace as:

    $$ Z = \\\\frac{x - \\\\mathrm{mean}(x)}{\\\\mathrm{std}(x)} $$

    where `x` is the selected ΔF/F trace over the entire recording duration.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Input table containing the fluorescence and reference channels.
    calcium_channel : str
        Column name of the fluorescence channel of interest.
    reference_channel : str
        Column name of the control/reference channel, typically 410 nm or 560 nm.
    baseline_interval_samples : int or None, default=None
        Number of initial samples defining the baseline interval used when the
        denominator depends on `median(Baseline)`. If None, the full trace is used.
    control_source : {'410', 'baseline'}, default='410'
        Choice of denominator and correction strategy according to the
        specification.
    apply_baseline_correction : bool, default=False
        Whether to baseline-correct the relevant signals prior to ΔF/F
        computation.
    enable_smoothing : bool, default=False
        Whether to smooth both channels before further processing.
    smoothing_window_length : int, default=11
        Savitzky-Golay smoothing window length.
    smoothing_polyorder : int, default=3
        Savitzky-Golay smoothing polynomial order.
    background_calcium : float or None, default=None
        Optional scalar background to subtract from the calcium channel.
    background_reference : float or None, default=None
        Optional scalar background to subtract from the reference channel.
    baseline_smoothness_penalty : float, default=1e6
        Smoothness penalty for asymmetric least squares baseline estimation.
    baseline_asymmetry_penalty : float, default=0.01
        Asymmetry penalty for asymmetric least squares baseline estimation.

    Returns
    -------
    results : dict
        Dictionary containing:
        - ``'fluorescence_processed'`` : processed fluorescence signal
        - ``'reference_processed'`` : processed reference signal
        - ``'fitted_reference'`` : robustly fitted reference signal
        - ``'motion_corrected'`` : fluorescence minus fitted reference
        - ``'fluorescence_baseline'`` : estimated fluorescence baseline
        - ``'reference_baseline'`` : estimated reference baseline
        - ``'baseline_fitted'`` : fitted fluorescence baseline in baseline mode
        - ``'dff'`` : selected ΔF/F trace
        - ``'zscore'`` : full-trace Z-score of ΔF/F
        - ``'beta'`` : robust fit slope

    Raises
    ------
    ValueError
        If inputs are invalid or if a required denominator is numerically zero.
    """
    fluorescence = df_clean[calcium_channel].to_numpy(dtype=float)
    reference = df_clean[reference_channel].to_numpy(dtype=float)

    if background_calcium is not None:
        fluorescence = fluorescence - float(background_calcium)
    if background_reference is not None:
        reference = reference - float(background_reference)

    fluorescence = optionally_smooth_signal(
        fluorescence,
        enable_smoothing=enable_smoothing,
        window_length=smoothing_window_length,
        polyorder=smoothing_polyorder,
    )
    reference = optionally_smooth_signal(
        reference,
        enable_smoothing=enable_smoothing,
        window_length=smoothing_window_length,
        polyorder=smoothing_polyorder,
    )

    n_samples = fluorescence.size
    if baseline_interval_samples is None:
        baseline_interval_samples = n_samples
    baseline_interval_samples = int(baseline_interval_samples)
    if baseline_interval_samples <= 0 or baseline_interval_samples > n_samples:
        raise ValueError("`baseline_interval_samples` must be in the range [1, n_samples].")

    baseline_slice = slice(0, baseline_interval_samples)
    baseline_median_raw = np.median(fluorescence[baseline_slice])
    full_trace_median_raw = np.median(fluorescence)

    fluorescence_baseline = estimate_baseline_asls(
        fluorescence,
        smoothness_penalty=baseline_smoothness_penalty,
        asymmetry_penalty=baseline_asymmetry_penalty,
    )
    reference_baseline = estimate_baseline_asls(
        reference,
        smoothness_penalty=baseline_smoothness_penalty,
        asymmetry_penalty=baseline_asymmetry_penalty,
    )

    if apply_baseline_correction:
        fluorescence_for_motion = fluorescence - fluorescence_baseline
        reference_for_motion = reference - reference_baseline
    else:
        fluorescence_for_motion = fluorescence
        reference_for_motion = reference

    fitted_reference, beta = fit_reference_robust_bisquare(
        fluorescence=fluorescence_for_motion,
        reference=reference_for_motion,
    )
    motion_corrected = fluorescence_for_motion - fitted_reference

    if control_source.lower() == "410" and apply_baseline_correction:
        denominator = full_trace_median_raw
        if np.abs(denominator) <= np.finfo(float).eps:
            raise ValueError("median(Ffull) is zero; cannot compute ΔF/F.")
        dff = motion_corrected / denominator
        baseline_fitted = fluorescence_baseline

    elif control_source.lower() == "410" and not apply_baseline_correction:
        denominator = fitted_reference
        if np.any(np.abs(denominator) <= np.finfo(float).eps):
            raise ValueError("fitted410 contains zeros; cannot compute ΔF/F safely.")
        dff = (fluorescence - fitted_reference) / denominator
        baseline_fitted = fluorescence_baseline

    elif control_source.lower() == "baseline" and apply_baseline_correction:
        denominator = baseline_median_raw
        if np.abs(denominator) <= np.finfo(float).eps:
            raise ValueError("median(Baseline) is zero; cannot compute ΔF/F.")
        baseline_fitted = fluorescence_baseline
        dff = ((fluorescence - fluorescence_baseline) - baseline_fitted) / denominator

    elif control_source.lower() == "baseline" and not apply_baseline_correction:
        denominator = baseline_median_raw
        if np.abs(denominator) <= np.finfo(float).eps:
            raise ValueError("median(Baseline) is zero; cannot compute ΔF/F.")
        baseline_fitted = fluorescence_baseline
        dff = (fluorescence - denominator) / denominator

    else:
        raise ValueError("`control_source` must be either '410' or 'baseline'.")

    dff_mean = np.mean(dff)
    dff_std = np.std(dff)
    if dff_std <= np.finfo(float).eps:
        raise ValueError("ΔF/F standard deviation is zero; cannot compute Z-score.")
    zscore = (dff - dff_mean) / dff_std

    return {
        "fluorescence_processed": fluorescence,
        "reference_processed": reference,
        "fitted_reference": fitted_reference,
        "motion_corrected": motion_corrected,
        "fluorescence_baseline": fluorescence_baseline,
        "reference_baseline": reference_baseline,
        "baseline_fitted": baseline_fitted,
        "dff": dff,
        "zscore": zscore,
        "beta": beta,
    }


def select_events_from_params(first_events: pd.DataFrame) -> pd.DataFrame:
    """Apply selection rules from params.py to first_events."""
    df = first_events.copy()
    if selected_clusters is not None:
        df = df[df["cs_n"].isin(selected_clusters)]
    if df.empty:
        raise ValueError("No events after applying SELECTED_* filters in params.py")
    return df


def filter_first_event(first_events: pd.DataFrame, skip_first_event) -> pd.DataFrame:
    """Drop first cluster event if SKIP_FIRST_EVENT=True in params (spurious trigger)."""
    if skip_first_event:
        deleted_row = first_events.iloc[[0]]
        print(f"[SKIP_FIRST_EVENT] Deleted row:\n{deleted_row[['cs_n', 'TimeStamp', 'Events_numeric']].to_string(index=False)}")
        filtered = first_events.iloc[1:].reset_index(drop=True)
        print(f"Skipped first event: {len(first_events)} → {len(filtered)} events")
        return filtered
    return first_events


def robust_fit_410_to_470(df_clean: pd.DataFrame, max_iter: int = 10, c: float = 4.685) -> np.ndarray:
    """Robust IRLS fit of REF (x) to CALCIUM (y); returns fitted410."""
    calcium = df_clean[calcium_channel].values
    reference = df_clean[ref_channel].values
    beta = linregress(reference, calcium).slope
    for _ in range(max_iter):
        residuals = calcium - beta * reference
        weights = np.where(np.abs(residuals) < c, (1 - (residuals / c)**2)**2, 0)
        denom = np.sum(weights * reference**2)
        if denom > 0:
            beta = np.sum(weights * reference * calcium) / denom
    return np.clip(beta * reference, 0, None)


def extract_epoch(signal: np.ndarray, time: np.ndarray,
                  t_event: float, n_pre: int, n_post: int) -> Optional[np.ndarray]:
    """Extract single epoch around event timestamp."""
    idx_event = np.searchsorted(time, t_event)
    start = idx_event - n_pre
    end = idx_event + n_post
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]


def extract_epoched_data(signal: np.ndarray, time_s: np.ndarray,
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


def process_signals(df_clean,
                    stem, skip_first_event) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
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

    # time_s = df_clean["TimeStamp"].values / 1000.0

    n_pre = int(time_pre_event_s * sample_rate_hz)
    n_post = int(time_post_event_s * sample_rate_hz)
    peri_t = np.arange(-n_pre, n_post)

    photometry_results = compute_photometry_dff_and_zscore(
        df_clean=df_clean,
        calcium_channel=calcium_channel,
        reference_channel=ref_channel,
        baseline_interval_samples=baseline_samples,
        control_source="410",  # "410" or "baseline"
        apply_baseline_correction=False,  # True or False
        enable_smoothing=False,
        smoothing_window_length=11,
        smoothing_polyorder=3,
        background_calcium=None,
        background_reference=None,
        baseline_smoothness_penalty=1e6,
        baseline_asymmetry_penalty=0.01,
    )

    # events_selected = select_events_from_params(first_events)
    # events_to_use = filter_first_event(events_selected, skip_first_event)
    # if events_to_use.empty:
    #     raise ValueError("No events remaining after selection/filtering — check params")
    #
    # event_times_s = events_to_use["TimeStamp"].values / 1000.0
    #
    # epochs_dff = extract_epoched_data(dff_baseline, time_s, event_times_s, n_pre, n_post)
    # epochs_z = extract_epoched_data(zscore, time_s, event_times_s, n_pre, n_post)

    return epochs_dff, epochs_z, peri_t, dff_baseline, dff_fitted, zscore, events_to_use


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/Lou/PycharmProjects/Fibre_photometry_M2")

    from preprocessing import extract_session_raw_data
    from event_sorting import process_events

    input_df = extract_session_raw_data()
    input_stem = "Fluorescence"
    _, input_first_events = process_events(input_df, input_stem)
    process_signals(input_df, input_first_events, input_stem)

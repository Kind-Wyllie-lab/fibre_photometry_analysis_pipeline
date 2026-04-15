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

# signal_processing.py
import numpy as np
from typing import Tuple, Optional

from scipy.signal import savgol_filter
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
    allow_intercept: bool = True,
) -> tuple[np.ndarray, float, float]:
    """
    Robustly fit a reference channel onto a fluorescence channel using
    iteratively reweighted least squares with Tukey bisquare weights.

    The fitted model is either

    $$ fluorescence \\\\approx intercept + \\\\beta \\\\cdot reference $$

    or, if `allow_intercept=False`,

    $$ fluorescence \\\\approx \\\\beta \\\\cdot reference $$

    Parameters
    ----------
    fluorescence : np.ndarray
        One-dimensional fluorescence signal of interest.
    reference : np.ndarray
        One-dimensional reference signal aligned to `fluorescence`.
    tuning_constant : float, default=4.685
        Tukey bisquare tuning constant.
    max_iterations : int, default=100
        Maximum number of robust reweighting iterations.
    tolerance : float, default=1e-10
        Convergence threshold on parameter updates.
    allow_intercept : bool, default=True
        If True, fit both slope and intercept. If False, fit slope only.

    Returns
    -------
    fitted_reference : np.ndarray
        Fitted reference contribution in fluorescence units.
    beta : float
        Fitted slope coefficient.
    intercept : float
        Fitted intercept. Equals 0.0 if `allow_intercept=False`.

    Raises
    ------
    ValueError
        If the input arrays are invalid or if too few finite samples are
        available.

    Notes
    -----
    If a robust iteration produces all-zero weights, the algorithm falls back
    to the previous parameter estimate instead of raising an exception.
    """
    fluorescence = np.asarray(fluorescence, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if fluorescence.ndim != 1 or reference.ndim != 1:
        raise ValueError("`fluorescence` and `reference` must be one-dimensional.")
    if fluorescence.shape[0] != reference.shape[0]:
        raise ValueError("`fluorescence` and `reference` must have the same length.")

    finite_mask = np.isfinite(fluorescence) & np.isfinite(reference)
    if finite_mask.sum() < 3:
        raise ValueError("At least 3 finite paired samples are required for robust fitting.")

    y = fluorescence[finite_mask]
    x = reference[finite_mask]

    if np.allclose(x, 0):
        raise ValueError("`reference` is all zeros or numerically constant at zero.")
    if np.std(x) <= np.finfo(float).eps:
        raise ValueError("`reference` has near-zero variance; fitting is ill-posed.")

    if allow_intercept:
        initial_fit = linregress(x, y)
        beta = float(initial_fit.slope)
        intercept = float(initial_fit.intercept)
    else:
        denominator = np.sum(x**2)
        if denominator <= np.finfo(float).eps:
            raise ValueError("Unweighted denominator is zero; cannot initialize slope.")
        beta = float(np.sum(x * y) / denominator)
        intercept = 0.0

    for _ in range(max_iterations):
        fitted = intercept + beta * x
        residuals = y - fitted

        robust_scale = np.median(np.abs(residuals - np.median(residuals))) / 0.6745
        if robust_scale <= np.finfo(float).eps:
            break

        standardized_residuals = residuals / (tuning_constant * robust_scale)
        weights = np.where(
            np.abs(standardized_residuals) < 1.0,
            (1.0 - standardized_residuals**2) ** 2,
            0.0,
        )

        if np.sum(weights) <= np.finfo(float).eps:
            break

        if allow_intercept:
            design_matrix = np.column_stack([np.ones_like(x), x])
            weighted_design = design_matrix * np.sqrt(weights)[:, None]
            weighted_response = y * np.sqrt(weights)
            parameters, *_ = np.linalg.lstsq(weighted_design, weighted_response, rcond=None)
            updated_intercept = float(parameters[0])
            updated_beta = float(parameters[1])

            parameter_shift = max(
                np.abs(updated_intercept - intercept),
                np.abs(updated_beta - beta),
            )
            intercept = updated_intercept
            beta = updated_beta
        else:
            denominator = np.sum(weights * x**2)
            if denominator <= np.finfo(float).eps:
                break

            updated_beta = float(np.sum(weights * x * y) / denominator)
            parameter_shift = np.abs(updated_beta - beta)
            beta = updated_beta
            intercept = 0.0

        if parameter_shift < tolerance:
            break

    fitted_reference_full = intercept + beta * reference
    return fitted_reference_full, beta, intercept



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


def preprocess_photometry_dff_and_zscore(
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
        Input table containing the calcium_fluorescence and reference_fluorescence channels.
    calcium_channel : str
        Column name of the calcium_fluorescence channel of interest.
    reference_channel : str
        Column name of the control/reference_fluorescence channel, typically 410 nm or 560 nm.
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
        Optional scalar background to subtract from the reference_fluorescence channel.
    baseline_smoothness_penalty : float, default=1e6
        Smoothness penalty for asymmetric least squares baseline estimation.
    baseline_asymmetry_penalty : float, default=0.01
        Asymmetry penalty for asymmetric least squares baseline estimation.

    Returns
    -------
    results : dict
        Dictionary containing:
        - ``'fluorescence_processed'`` : processed calcium_fluorescence signal
        - ``'reference_processed'`` : processed reference_fluorescence signal
        - ``'fitted_reference'`` : robustly fitted reference_fluorescence signal
        - ``'motion_corrected'`` : calcium_fluorescence minus fitted reference_fluorescence
        - ``'fluorescence_baseline'`` : estimated calcium_fluorescence baseline
        - ``'reference_baseline'`` : estimated reference_fluorescence baseline
        - ``'baseline_fitted'`` : fitted calcium_fluorescence baseline in baseline mode
        - ``'dff'`` : selected ΔF/F trace
        - ``'zscore'`` : full-trace Z-score of ΔF/F
        - ``'beta'`` : robust fit slope

    Raises
    ------
    ValueError
        If inputs are invalid or if a required denominator is numerically zero.
    """
    calcium_fluorescence = df_clean[calcium_channel].to_numpy(dtype=float)
    reference_fluorescence = df_clean[reference_channel].to_numpy(dtype=float)

    if background_calcium is not None:
        calcium_fluorescence = calcium_fluorescence - float(background_calcium)
    if background_reference is not None:
        reference_fluorescence = reference_fluorescence - float(background_reference)

    calcium_fluorescence = optionally_smooth_signal(
        calcium_fluorescence,
        enable_smoothing=enable_smoothing,
        window_length=smoothing_window_length,
        polyorder=smoothing_polyorder,
    )
    reference_fluorescence = optionally_smooth_signal(
        reference_fluorescence,
        enable_smoothing=enable_smoothing,
        window_length=smoothing_window_length,
        polyorder=smoothing_polyorder,
    )

    n_samples = calcium_fluorescence.size
    if baseline_interval_samples is None:
        baseline_interval_samples = n_samples
    baseline_interval_samples = int(baseline_interval_samples)
    if baseline_interval_samples <= 0 or baseline_interval_samples > n_samples:
        raise ValueError("`baseline_interval_samples` must be in the range [1, n_samples].")

    baseline_slice = slice(0, baseline_interval_samples)
    baseline_median_raw = np.median(calcium_fluorescence[baseline_slice])
    full_trace_median_raw = np.median(calcium_fluorescence)

    fluorescence_baseline = estimate_baseline_asls(
        calcium_fluorescence,
        smoothness_penalty=baseline_smoothness_penalty,
        asymmetry_penalty=baseline_asymmetry_penalty,
    )
    reference_baseline = estimate_baseline_asls(
        reference_fluorescence,
        smoothness_penalty=baseline_smoothness_penalty,
        asymmetry_penalty=baseline_asymmetry_penalty,
    )

    if apply_baseline_correction:
        fluorescence_for_motion = calcium_fluorescence - fluorescence_baseline
        reference_for_motion = reference_fluorescence - reference_baseline
    else:
        fluorescence_for_motion = calcium_fluorescence
        reference_for_motion = reference_fluorescence

    fitted_reference, beta, intercept = fit_reference_robust_bisquare(
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
        dff = (calcium_fluorescence - fitted_reference) / denominator
        baseline_fitted = fluorescence_baseline

    elif control_source.lower() == "baseline" and apply_baseline_correction:
        denominator = baseline_median_raw
        if np.abs(denominator) <= np.finfo(float).eps:
            raise ValueError("median(Baseline) is zero; cannot compute ΔF/F.")
        baseline_fitted = fluorescence_baseline
        dff = ((calcium_fluorescence - fluorescence_baseline) - baseline_fitted) / denominator

    elif control_source.lower() == "baseline" and not apply_baseline_correction:
        denominator = baseline_median_raw
        if np.abs(denominator) <= np.finfo(float).eps:
            raise ValueError("median(Baseline) is zero; cannot compute ΔF/F.")
        baseline_fitted = fluorescence_baseline
        dff = (calcium_fluorescence - denominator) / denominator

    else:
        raise ValueError("`control_source` must be either '410' or 'baseline'.")

    dff_mean = np.mean(dff)
    dff_std = np.std(dff)
    if dff_std <= np.finfo(float).eps:
        raise ValueError("ΔF/F standard deviation is zero; cannot compute Z-score.")
    zscore = (dff - dff_mean) / dff_std

    return {
        "fluorescence_processed": calcium_fluorescence,
        "reference_processed": reference_fluorescence,
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

def extract_epoched_data(
    signal: np.ndarray,
    time_s: np.ndarray,
    event_times_s: np.ndarray,
    n_pre: int,
    n_post: int,
    compute_trial_baseline_dff: bool = False,
    trial_baseline_window_s: float = 5.0,
    trial_baseline_statistic: str = "mean",
    minimum_baseline_value: Optional[float] = None,
) -> np.ndarray:
    """
    Extract peri-event epochs and optionally normalize each epoch to its own
    pre-trigger baseline to obtain trial-local ΔF/F.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional signal array from which epochs are extracted.
    time_s : numpy.ndarray
        One-dimensional time vector in seconds aligned to `signal`.
    event_times_s : numpy.ndarray
        Event timestamps in seconds used as epoch centers.
    n_pre : int
        Number of samples to include before each event.
    n_post : int
        Number of samples to include after each event.
    compute_trial_baseline_dff : bool, default=False
        If ``True``, compute trial-local ΔF/F for each extracted epoch using
        its own pre-trigger baseline window.
    trial_baseline_window_s : float, default=2.0
        Duration in seconds of the pre-trigger baseline window used for
        trial-local ΔF/F normalization.
    trial_baseline_statistic : {"mean", "median"}, default="median"
        Summary statistic used to estimate the baseline fluorescence per trial.
    minimum_baseline_value : float or None, default=None
        Optional lower bound for acceptable baseline values. If provided,
        epochs with baseline values less than or equal to this threshold are
        rejected.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_trials, n_timepoints)`` containing extracted epochs.
        If `compute_trial_baseline_dff` is ``True``, the returned values are
        trial-local ΔF/F.

    Raises
    ------
    ValueError
        If no valid epochs are extracted, if the baseline window is invalid,
        or if the baseline statistic is unsupported.

    Notes
    -----
    When `compute_trial_baseline_dff` is enabled, the baseline is computed from
    the interval immediately preceding the trigger, corresponding to the last
    `trial_baseline_window_s` seconds of the pre-event segment.
    """
    epochs = []
    for t_ev in event_times_s:
        epoch = extract_epoch(signal, time_s, t_ev, n_pre, n_post)
        if epoch is None:
            continue

        if compute_trial_baseline_dff:
            if n_pre <= 0:
                raise ValueError("n_pre must be > 0 to compute trial-local baseline ΔF/F")

            if len(time_s) < 2:
                raise ValueError("time_s must contain at least 2 samples to infer sampling interval")

            sampling_interval_s = float(np.median(np.diff(time_s)))
            if sampling_interval_s <= 0:
                raise ValueError("Invalid non-positive sampling interval inferred from time_s")

            baseline_samples = int(round(trial_baseline_window_s / sampling_interval_s))
            if baseline_samples <= 0:
                raise ValueError(
                    f"trial_baseline_window_s={trial_baseline_window_s} yields zero baseline samples"
                )
            if baseline_samples > n_pre:
                raise ValueError(
                    f"trial_baseline_window_s={trial_baseline_window_s}s requires {baseline_samples} samples, "
                    f"but only {n_pre} pre-event samples are available"
                )

            baseline_segment = epoch[n_pre - baseline_samples:n_pre]

            if trial_baseline_statistic == "mean":
                baseline_value = float(np.mean(baseline_segment))
            elif trial_baseline_statistic == "median":
                baseline_value = float(np.median(baseline_segment))
            else:
                raise ValueError(
                    "trial_baseline_statistic must be either 'mean' or 'median'"
                )

            if not np.isfinite(baseline_value):
                continue
            if baseline_value == 0:
                continue
            if minimum_baseline_value is not None and baseline_value <= minimum_baseline_value:
                continue

            epoch = (epoch - baseline_value) / baseline_value

        epochs.append(epoch)

    if not epochs:
        raise ValueError("No valid epochs extracted — check event times vs signal duration")

    result = np.asarray(epochs, dtype=float)
    print(f"Epochs extracted: {result.shape[0]} trials, {result.shape[1]} samples each")

    if compute_trial_baseline_dff:
        print(
            "Applied trial-local ΔF/F normalization "
            f"using {trial_baseline_window_s:.3f}s pre-trigger baseline "
            f"({trial_baseline_statistic})"
        )

    return result

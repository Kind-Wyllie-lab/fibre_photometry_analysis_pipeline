from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

class EventEpochExtractor:
    """
    Extract peri-event epochs from preprocessed signals based on event tables.

    This object is intentionally separate from `PhotometrySession` so that:
    - Session remains a processor/orchestrator, not an analysis/plotting decision-maker.
    - Different analyses can request different epoching specs without modifying Session.

    Notes
    -----
    This class expects that event tables are produced by `process_events(...)` and
    stored in a dict (as in the updated event_sorting module).

    Epoch extraction is performed on any 1D signal using the provided `extract_epoched_data`
    function from your signal processing utilities.

    """

    @staticmethod
    def extract_epochs_for_signals(
        time_s: np.ndarray,
        event_times_s: np.ndarray,
        preprocessed_signals: dict[str, np.ndarray],
        n_pre: int,
        n_post: int,
        extract_epoched_data_callable,
    ) -> dict[str, np.ndarray]:
        """
        Extract epochs for multiple named signals.

        Parameters
        ----------
        time_s : numpy.ndarray
            Continuous time base in seconds (same length as signals).
        event_times_s : numpy.ndarray
            Event times (seconds).
        preprocessed_signals : dict[str, numpy.ndarray]
            Dictionary of continuous signals (e.g. ``{"dff": ..., "zscore": ...}``).
        n_pre : int
            Number of samples before event.
        n_post : int
            Number of samples after event.
        extract_epoched_data_callable : callable
            Function with signature:
            ``extract_epoched_data(signal, time_s, event_times_s, n_pre, n_post)``.

        Returns
        -------
        dict[str, numpy.ndarray]
            Dictionary mapping each input signal name to its epochs array.
        """
        epochs_by_signal: dict[str, np.ndarray] = {}
        for signal_name, signal in preprocessed_signals.items():
            epochs_by_signal[signal_name] = extract_epoched_data_callable(
                signal=signal,
                time_s=time_s,
                event_times_s=event_times_s,
                n_pre=n_pre,
                n_post=n_post,
            )
        return epochs_by_signal

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats


SignalName = Literal["dff", "zscore"]


@dataclass
class SessionBootstrapAUCAnalyzer:
    """
    Session-level bootstrap test of peri-event AUC responses using a *single* null distribution.

    Workflow
    --------
    1) Build a global null distribution by sampling mock event times that:
       - are within valid time bounds for epoch extraction
       - do not overlap with each other
       - do not overlap with real events
    2) For each real event, compute:
       - peri-event epoch
       - baseline reference using the pre-trigger baseline window (default 2 s)
       - AUC from 0 to 2 s post-trigger on baselined epoch
    3) Compare each real-event AUC to the null distribution quantiles (2.5% and 97.5%)
    4) Plot one histogram+Gaussian of null, and draw one axvline per real event.

    Parameters
    ----------
    session : object
        Session-like object with:
        - df_clean["TimeStamp"] in ms
        - preprocessed_signals dict with keys "dff" and "zscore"
    event_times_s : numpy.ndarray
        Real event times in seconds.
    event_name : str
        Event label for titles/filenames.
    baseline_window_s : float, default=2.0
        Baseline duration immediately preceding t=0 for baseline reference.
    auc_window_s : tuple[float, float], default=(0.0, 2.0)
        AUC integration window relative to t=0.
    n_mocks : int, default=5000
        Number of mock events used to build the null distribution.
    min_separation_s : float or None, default=None
        Minimum separation between mock events and real events (and between mocks).
        If None, uses epoch length (baseline_window + max(auc_window)).
    random_seed : int or None, default=0
        RNG seed.
    """

    session: object
    event_times_s: np.ndarray
    event_name: str

    baseline_window_s: float = 2.0
    auc_window_s: tuple[float, float] = (0.0, 2.0)
    n_mocks: int = 5000
    min_separation_s: Optional[float] = None
    random_seed: Optional[int] = 0

    def __post_init__(self) -> None:
        self.event_times_s = np.asarray(self.event_times_s, dtype=float)
        if self.event_times_s.ndim != 1:
            raise ValueError("event_times_s must be 1D")

        if "TimeStamp" not in self.session.df_clean.columns:
            raise ValueError("session.df_clean must contain TimeStamp column (ms)")

        self.time_s = self.session.df_clean["TimeStamp"].to_numpy(dtype=float) / 1000.0
        if self.time_s.size < 2:
            raise ValueError("Need >=2 samples to infer sampling interval")

        self.dt_s = float(np.median(np.diff(self.time_s)))
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("Invalid dt inferred from TimeStamp")

        if not hasattr(self.session, "preprocessed_signals"):
            raise ValueError("session must have preprocessed_signals dict")
        for key in ("dff", "zscore"):
            if key not in self.session.preprocessed_signals:
                raise ValueError(f"session.preprocessed_signals missing {key!r}")

        self.epoch_pre_s = float(self.baseline_window_s)
        self.epoch_post_s = float(max(self.auc_window_s[1], 0.0))
        self.epoch_len_s = self.epoch_pre_s + self.epoch_post_s

        if self.min_separation_s is None:
            self.min_separation_s = self.epoch_len_s

        self.rng = np.random.default_rng(self.random_seed)

    # -------------------------
    # epoch utilities
    # -------------------------
    def _extract_epoch(self, signal: np.ndarray, t_event_s: float) -> Optional[np.ndarray]:
        n_pre = int(round(self.epoch_pre_s / self.dt_s))
        n_post = int(round(self.epoch_post_s / self.dt_s))
        idx = int(np.searchsorted(self.time_s, t_event_s))
        start = idx - n_pre
        end = idx + n_post
        if start < 0 or end > signal.size:
            return None
        return signal[start:end]

    def _baseline_reference_epoch(self, epoch: np.ndarray) -> Optional[np.ndarray]:
        n_pre = int(round(self.epoch_pre_s / self.dt_s))
        if n_pre < 2:
            return None
        baseline_segment = epoch[:n_pre]
        baseline_value = float(np.median(baseline_segment))
        if not np.isfinite(baseline_value):
            return None
        return epoch - baseline_value

    def _auc_from_epoch(self, baselined_epoch: np.ndarray) -> Optional[float]:
        auc_start_s, auc_end_s = self.auc_window_s
        if auc_end_s <= auc_start_s:
            raise ValueError("auc_window_s must satisfy auc_end > auc_start")

        n_pre = int(round(self.epoch_pre_s / self.dt_s))
        peri_t = (np.arange(baselined_epoch.size, dtype=float) - n_pre) * self.dt_s
        mask = (peri_t >= auc_start_s) & (peri_t <= auc_end_s)
        if np.sum(mask) < 2:
            return None
        return float(np.trapz(baselined_epoch[mask], x=peri_t[mask]))

    # -------------------------
    # mock sampling
    # -------------------------
    def _sample_mock_times(self) -> np.ndarray:
        """
        Sample mock event times uniformly from the valid range, rejecting only:
        - times too close to any real event (within min_separation_s)

        Notes
        -----
        Unlike the previous implementation, this does *not* enforce any separation
        between mock events, so mock windows may overlap heavily and can be adjacent.

        Returns
        -------
        numpy.ndarray
            Mock event times in seconds.
        """
        t_start_s = float(self.time_s[0] + self.epoch_pre_s)
        t_end_s = float(self.time_s[-1] - self.epoch_post_s)
        if t_end_s <= t_start_s:
            raise ValueError("Recording too short for requested epoch windows")

        real = self.event_times_s[np.isfinite(self.event_times_s)]
        real = np.sort(real)

        mocks: list[float] = []
        max_tries = max(50_000, 50 * self.n_mocks)
        tries = 0

        # If you want "one sample away" to be meaningful, sample from the discrete time grid:
        # this guarantees minimum step = dt_s.
        valid_time_grid = self.time_s[(self.time_s >= t_start_s) & (self.time_s <= t_end_s)]
        if valid_time_grid.size < 2:
            raise ValueError("Not enough valid samples to draw mock events")

        while len(mocks) < self.n_mocks and tries < max_tries:
            tries += 1

            # draw a candidate mock from the discrete sampling grid (adjacent allowed)
            t = float(self.rng.choice(valid_time_grid))

            # reject if too close to any real event
            if real.size:
                # O(log n) nearest distance using searchsorted
                j = int(np.searchsorted(real, t))
                d_left = abs(t - real[j - 1]) if j > 0 else np.inf
                d_right = abs(real[j] - t) if j < real.size else np.inf
                if min(d_left, d_right) < float(self.min_separation_s):
                    continue

            mocks.append(t)

        if len(mocks) < self.n_mocks:
            print(
                f"[MOCK SAMPLING WARNING] Requested n_mocks={self.n_mocks}, obtained={len(mocks)} "
                f"(increase max_tries, decrease min_separation_s, or check event density)"
            )

        return np.asarray(mocks, dtype=float)

    # -------------------------
    # public compute
    # -------------------------
    def compute_auc_real_and_null(self, signal_name: SignalName) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute AUC for real events and for one shared null distribution.

        Returns
        -------
        auc_real : numpy.ndarray
            Real-event AUC values (length = n_valid_events).
        auc_null : numpy.ndarray
            Null AUC values computed from mock times (length ~ n_mocks, may be smaller if some epochs invalid).
        """
        signal = np.asarray(self.session.preprocessed_signals[signal_name], dtype=float)
        if signal.size != self.time_s.size:
            raise ValueError(f"{signal_name} trace length != time vector length")

        auc_real = []
        for t_event in self.event_times_s:
            epoch = self._extract_epoch(signal, float(t_event))
            if epoch is None:
                continue
            epoch_b = self._baseline_reference_epoch(epoch)
            if epoch_b is None:
                continue
            auc = self._auc_from_epoch(epoch_b)
            if auc is None:
                continue
            auc_real.append(auc)

        auc_real = np.asarray(auc_real, dtype=float)

        mock_times = self._sample_mock_times()
        mock_times = np.asarray(mock_times, dtype=float)
        mock_times = np.unique(mock_times)
        auc_null = []
        for t_mock in tqdm(mock_times):
            epoch = self._extract_epoch(signal, float(t_mock))
            if epoch is None:
                continue
            epoch_b = self._baseline_reference_epoch(epoch)
            if epoch_b is None:
                continue
            auc = self._auc_from_epoch(epoch_b)
            if auc is None:
                continue
            auc_null.append(auc)

        auc_null = np.asarray(auc_null, dtype=float)
        return auc_real, auc_null

    def plot_null_with_all_event_axvlines(
            self,
            signal_name: SignalName,
            bins: int = 60,
            tail_mode: Literal["two_sided", "right"] = "two_sided",
            alpha_level: float = 0.05,
            kde: bool = False,
    ) -> plt.Figure:
        """
        Plot a single null distribution and overlay one axvline per real event AUC.

        The axvline is green if significant, else black.
        Title reports responsive/total.

        Parameters
        ----------
        signal_name : {"dff","zscore"}
            Signal to analyze.
        bins : int, default=60
            Histogram bin count.
        tail_mode : {"two_sided","right"}, default="two_sided"
            - "two_sided": significance if AUC < q(alpha/2) OR AUC > q(1-alpha/2)
            - "right": significance if AUC > q(1-alpha)
        alpha_level : float, default=0.05
            Significance level.
        kde : bool, default=False
            If True, overlay seaborn KDE for the null.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object.
        """
        if not (0.0 < alpha_level < 1.0):
            raise ValueError("alpha_level must be in (0, 1)")

        auc_real, auc_null = self.compute_auc_real_and_null(signal_name)

        fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.6))

        if auc_null.size < 20:
            raise ValueError("Too few mock AUC samples to plot a null distribution")

        # --- quantiles + decision rule ---
        if tail_mode == "two_sided":
            q_lo = float(np.quantile(auc_null, alpha_level / 2.0))
            q_hi = float(np.quantile(auc_null, 1.0 - (alpha_level / 2.0)))
            responsive_mask = (auc_real < q_lo) | (auc_real > q_hi)
            quantile_label = f"{alpha_level / 2.0:.3f} / {1.0 - alpha_level / 2.0:.3f} quantiles"
        elif tail_mode == "right":
            q_lo = np.nan
            q_hi = float(np.quantile(auc_null, 1.0 - alpha_level))
            responsive_mask = (auc_real > q_hi)
            quantile_label = f"{1.0 - alpha_level:.3f} quantile"
        else:
            raise ValueError("tail_mode must be 'two_sided' or 'right'")

        n_resp = int(np.sum(responsive_mask))
        n_total = int(auc_real.size)

        # --- null histogram (density) ---
        sns.histplot(auc_null, bins=bins, stat="density", color="0.25", alpha=0.35, ax=ax)
        if kde:
            sns.kdeplot(auc_null, color="0.15", lw=1.2, ax=ax)

        mu, sigma = float(np.mean(auc_null)), float(np.std(auc_null))
        xgrid = np.linspace(np.min(auc_null), np.max(auc_null), 500)
        ax.plot(
            xgrid,
            stats.norm.pdf(xgrid, loc=mu, scale=max(sigma, 1e-12)),
            color="0.15",
            lw=1.4,
            alpha=0.9,
        )

        # --- quantile lines ---
        if np.isfinite(q_lo):
            ax.axvline(q_lo, color="red", lw=1.0, ls="--", alpha=0.85, label=quantile_label)
            ax.axvline(q_hi, color="red", lw=1.0, ls="--", alpha=0.85)
        else:
            ax.axvline(q_hi, color="red", lw=1.0, ls="--", alpha=0.85, label=quantile_label)

        # --- one line per event ---
        for auc_value, is_resp in zip(auc_real, responsive_mask):
            ax.axvline(
                float(auc_value),
                color=("green" if is_resp else "black"),
                lw=1.2,
                alpha=0.7,
            )

        ax.set_xlabel(f"AUC of baselined {signal_name} ({self.auc_window_s[0]:.1f}–{self.auc_window_s[1]:.1f}s)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"{self.event_name} | {signal_name} | responsive {n_resp}/{n_total} | "
            f"{tail_mode}, alpha={alpha_level:.2f}"
        )
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig

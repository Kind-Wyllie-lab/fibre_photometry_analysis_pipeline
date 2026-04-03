from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from params import (
    save_figures,
    preview_figures,
    figure_format,
    figure_dpi,
    figure_size_trace,
    figure_size_peri,
    heatmap_figsize,
    color_470_nm,
    color_410_nm,
    color_dff,
    color_zscore,
    color_peri_mean,
    color_event_onset,
    lw_trace,
    lw_peri_mean,
    lw_event_marker,
    alpha_event_lines,
    alpha_sem_fill,
    xtick_fontsize,
    xtick_direction,
    xtick_width_major,
    xtick_length_major,
    xtick_width_minor,
    xtick_length_minor,
    xtick_nbins,
    ytick_fontsize,
    ytick_direction,
    ytick_width_major,
    ytick_length_major,
    ytick_width_minor,
    ytick_length_minor,
    ytick_nbins,
    heatmap_cmap_dff,
    heatmap_cmap_z,
    heatmap_vmin_dff,
    heatmap_vmax_dff,
    heatmap_vmin_z,
    heatmap_vmax_z,
    heatmap_xtick_major,
    heatmap_xtick_minor,
    heatmap_show_all_ylabels,
    heatmap_ytick_major_step,
    heatmap_ytick_minor_step,
)



@dataclass
class PhotometryPlotter:
    """
    Object-oriented plotting interface for a single fiber photometry session.

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Cleaned fluorescence dataframe containing at least ``TimeStamp``,
        calcium channel, and reference channel columns.
    dff_fitted : numpy.ndarray
        Full-session motion-corrected ΔF/F trace normalized to fitted reference.
    zscore : numpy.ndarray
        Full-session z-score trace.
    epochs_dff : numpy.ndarray
        Peri-event ΔF/F array with shape ``(n_trials, n_timepoints)``.
    epochs_z : numpy.ndarray
        Peri-event z-score array with shape ``(n_trials, n_timepoints)``.
    peri_t : numpy.ndarray
        Peri-event time vector in seconds.
    stem : str
        Session identifier used for titles and output filenames.
    event_times_s : numpy.ndarray, optional
        Event timestamps in seconds for vertical markers on full-trace plots.
    t_zero_s : float, optional
        Time origin in seconds used to realign the full trace.
    filtered_events : pandas.DataFrame, optional
        Event dataframe used for epoch extraction. If present, trial labels
        can be inferred for heatmaps.
    figure_output_directory : str or Path, optional
        Directory where figures are saved. If ``None``, uses the directory
        defined in ``params.py``.

    Attributes
    ----------
    save_figures_enabled : bool
        Whether figures should be written to disk.
    preview_figures_enabled : bool
        Whether figures should be displayed interactively.
    output_directory : pathlib.Path
        Destination directory for figure export.
    """

    df_clean: pd.DataFrame
    dff_fitted: np.ndarray
    zscore: np.ndarray
    epochs_dff: np.ndarray
    epochs_z: np.ndarray
    peri_t: np.ndarray
    stem: str
    event_times_s: Optional[np.ndarray] = None
    t_zero_s: Optional[float] = None
    filtered_events: Optional[pd.DataFrame] = None
    figure_output_directory: str = './'

    def __post_init__(self) -> None:
        """
        Validate inputs and initialize plotting output settings.

        Raises
        ------
        ValueError
            If mandatory arrays are inconsistent in size or dimension.
        """
        self.save_figures_enabled = save_figures
        self.preview_figures_enabled = preview_figures
        self.output_directory = Path(self.figure_output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self._validate_core_inputs()

    def _validate_core_inputs(self) -> None:
        """
        Validate input array shapes and required dataframe columns.

        Raises
        ------
        ValueError
            If required columns or array dimensions are invalid.
        """
        required_columns = {"TimeStamp"}
        missing_columns = required_columns.difference(self.df_clean.columns)
        if missing_columns:
            raise ValueError(f"df_clean missing required columns: {sorted(missing_columns)}")

        if len(self.dff_fitted) != len(self.df_clean):
            raise ValueError(
                f"dff_fitted length {len(self.dff_fitted)} != df_clean length {len(self.df_clean)}"
            )
        if len(self.zscore) != len(self.df_clean):
            raise ValueError(
                f"zscore length {len(self.zscore)} != df_clean length {len(self.df_clean)}"
            )

        if self.epochs_dff.ndim != 2:
            raise ValueError(f"epochs_dff must be 2D, got {self.epochs_dff.ndim}D")
        if self.epochs_z.ndim != 2:
            raise ValueError(f"epochs_z must be 2D, got {self.epochs_z.ndim}D")

        if self.epochs_dff.shape != self.epochs_z.shape:
            raise ValueError(
                f"epochs_dff shape {self.epochs_dff.shape} != epochs_z shape {self.epochs_z.shape}"
            )

        if self.epochs_dff.shape[1] != len(self.peri_t):
            raise ValueError(
                f"peri_t length {len(self.peri_t)} != epoch timepoints {self.epochs_dff.shape[1]}"
            )

    @property
    def time_s(self) -> np.ndarray:
        """
        Session time vector in seconds.

        Returns
        -------
        numpy.ndarray
            Time vector derived from ``TimeStamp``.
        """
        return self.df_clean["TimeStamp"].to_numpy(dtype=float) / 1000.0

    def _autoscale_y_to_signal(self, ax: plt.Axes, y: np.ndarray, pad_percent: float = 5.0) -> None:
        """
        Set y-axis limits based on finite signal range.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        y : numpy.ndarray
            Signal values.
        pad_percent : float, default=5.0
            Padding percentage added above and below the signal range.
        """
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return

        y_min, y_max = y.min(), y.max()
        if y_min == y_max:
            ax.set_ylim(y_min - 1.0, y_max + 1.0)
            return

        span = y_max - y_min
        pad = span * pad_percent / 100.0
        ax.set_ylim(y_min - pad, y_max + pad)

    def _finalize_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Save and/or preview a figure according to project parameters.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to finalize.
        filename : str
            Output filename stem without extension.
        """
        if self.save_figures_enabled:
            output_path = self.output_directory / f"{filename}.{figure_format}"
            fig.savefig(output_path, dpi=figure_dpi, format=figure_format, bbox_inches="tight")
            print(f"Saved: {output_path}")

        if self.preview_figures_enabled:
            plt.show()

        if not self.preview_figures_enabled:
            plt.close(fig)

    def _set_xtick_params(self, ax: plt.Axes) -> None:
        """
        Apply project-wide x-axis tick formatting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        """
        ax.tick_params(axis="x", reset=True)
        ax.tick_params(
            axis="x",
            bottom=True,
            top=False,
            labelbottom=True,
            labeltop=False,
            labelsize=xtick_fontsize,
            direction=xtick_direction,
            width=xtick_width_major,
            length=xtick_length_major,
            which="major",
        )
        ax.tick_params(
            axis="x",
            bottom=True,
            top=False,
            width=xtick_width_minor,
            length=xtick_length_minor,
            which="minor",
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(xtick_nbins))

    def _set_ytick_params(self, ax: plt.Axes) -> None:
        """
        Apply project-wide y-axis tick formatting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        """
        ax.tick_params(axis="y", reset=True)
        ax.tick_params(
            axis="y",
            left=True,
            right=False,
            labelleft=True,
            labelright=False,
            labelsize=ytick_fontsize,
            direction=ytick_direction,
            width=ytick_width_major,
            length=ytick_length_major,
            which="major",
        )
        ax.tick_params(
            axis="y",
            left=True,
            right=False,
            width=ytick_width_minor,
            length=ytick_length_minor,
            which="minor",
        )
        ax.yaxis.set_major_locator(plt.MaxNLocator(ytick_nbins))

    def _build_peri_event_long_dataframe(self) -> pd.DataFrame:
        """
        Build a long-form dataframe for seaborn peri-event plotting.

        Returns
        -------
        pandas.DataFrame
            Long-format dataframe with one row per trial-timepoint-signal value.
            Columns are ``trial``, ``time_s``, ``signal_kind``, and ``signal_value``.
        """
        n_trials, n_timepoints = self.epochs_dff.shape
        if self.epochs_z.shape != (n_trials, n_timepoints):
            raise ValueError(
                f"epochs_z shape {self.epochs_z.shape} does not match epochs_dff shape {self.epochs_dff.shape}"
            )
        if len(self.peri_t) != n_timepoints:
            raise ValueError(
                f"peri_t length {len(self.peri_t)} != number of epoch timepoints {n_timepoints}"
            )

        trial_index = np.arange(1, n_trials + 1, dtype=int)

        epochs_dff_wide = pd.DataFrame(self.epochs_dff, index=trial_index, columns=self.peri_t)
        epochs_dff_wide.index.name = "trial"
        epochs_dff_long = (
            epochs_dff_wide
            .reset_index()
            .melt(id_vars="trial", var_name="time_s", value_name="signal_value")
            .assign(signal_kind="ΔF/F")
        )

        epochs_z_wide = pd.DataFrame(self.epochs_z, index=trial_index, columns=self.peri_t)
        epochs_z_wide.index.name = "trial"
        epochs_z_long = (
            epochs_z_wide
            .reset_index()
            .melt(id_vars="trial", var_name="time_s", value_name="signal_value")
            .assign(signal_kind="Z-score")
        )

        peri_event_long_dataframe = pd.concat(
            [epochs_dff_long, epochs_z_long],
            axis=0,
            ignore_index=True,
        )
        peri_event_long_dataframe["time_s"] = peri_event_long_dataframe["time_s"].astype(float)

        return peri_event_long_dataframe

    def _get_heatmap_trial_labels(self, n_trials: int) -> list[str]:
        """
        Infer heatmap trial labels from the filtered events table when available.

        Parameters
        ----------
        n_trials : int
            Number of extracted peri-event trials.

        Returns
        -------
        list of str
            Trial labels for the heatmap y-axis.
        """
        if self.filtered_events is None:
            return [str(i) for i in range(1, n_trials + 1)]

        if "cluster_id" in self.filtered_events.columns:
            labels = self.filtered_events["cluster_id"].to_numpy()[:n_trials]
            return [str(label) for label in labels]

        if "cs_n" in self.filtered_events.columns:
            labels = self.filtered_events["cs_n"].to_numpy()[:n_trials]
            return [str(label) for label in labels]

        return [str(i) for i in range(1, n_trials + 1)]

    def plot_full_fluorescence(self) -> None:
        """
        Plot full-session raw fluorescence traces for calcium and isosbestic channels.

        Raises
        ------
        ValueError
            If required fluorescence columns are absent.
        """
        if "CH1-470" not in self.df_clean.columns or "CH1-410" not in self.df_clean.columns:
            raise ValueError("df_clean must contain 'CH1-470' and 'CH1-410' columns")

        fig, axes = plt.subplots(2, 1, figsize=figure_size_trace, sharex=True)

        y_470 = self.df_clean["CH1-470"].to_numpy(dtype=float)
        axes[0].plot(self.time_s, y_470, color=color_470_nm, lw=lw_trace, label="470nm (calcium)")
        self._autoscale_y_to_signal(axes[0], y_470)
        axes[0].set_ylabel("470nm (AU)")
        axes[0].set_title(f"Full-session raw fluorescence ({self.stem})")
        axes[0].grid(alpha=0.15)
        axes[0].legend(loc="upper right")
        self._set_ytick_params(axes[0])

        y_410 = self.df_clean["CH1-410"].to_numpy(dtype=float)
        axes[1].plot(self.time_s, y_410, color=color_410_nm, lw=lw_trace, label="410nm (isosbestic)")
        self._autoscale_y_to_signal(axes[1], y_410)
        axes[1].set_ylabel("410nm (AU)")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(alpha=0.15)
        axes[1].legend(loc="upper right")
        self._set_xtick_params(axes[1])
        self._set_ytick_params(axes[1])

        fig.tight_layout()
        self._finalize_figure(fig, f"raw_traces_{self.stem}")

    def plot_full_trace(self) -> None:
        """
        Plot motion-corrected full-session ΔF/F and z-score traces with event markers.
        """
        origin_s = self.t_zero_s if self.t_zero_s is not None else self.time_s[0]

        start_index = np.searchsorted(self.time_s, origin_s)
        time_plot = self.time_s[start_index:] - origin_s
        dff_plot = self.dff_fitted[start_index:]
        zscore_plot = self.zscore[start_index:]

        if self.event_times_s is not None:
            event_times_plot = np.asarray(self.event_times_s, dtype=float) - origin_s
        else:
            if "Events_numeric" in self.df_clean.columns:
                event_mask = self.df_clean["Events_numeric"].to_numpy()[start_index:] == 1
                event_times_plot = time_plot[event_mask]
            else:
                event_times_plot = np.array([], dtype=float)

        fig, axes = plt.subplots(2, 1, figsize=figure_size_trace, sharex=True)

        trace_specs = [
            (dff_plot, "ΔF/F", color_dff),
            (zscore_plot, "Z-score", color_zscore),
        ]

        for ax, (signal_values, y_label, color) in zip(axes, trace_specs):
            ax.plot(time_plot, signal_values, color=color, lw=lw_trace, label=y_label)
            for event_time in event_times_plot:
                ax.axvline(
                    x=event_time,
                    color=color_event_onset,
                    ls="--",
                    alpha=alpha_event_lines,
                    lw=lw_event_marker,
                )
            ax.set_ylabel(y_label)
            ax.grid(alpha=0.15)
            ax.legend(loc="upper right")
            self._set_ytick_params(ax)

        n_events = len(event_times_plot)
        zero_label = (
            f"t=0 at first TTL ({origin_s:.2f}s)"
            if self.t_zero_s is not None
            else "t=0 at recording start"
        )
        axes[0].set_title(
            f"Full session — motion-corrected signal ({self.stem}) | {n_events} events | {zero_label}"
        )
        axes[-1].set_xlabel("Time from first TTL pulse (s)")
        self._set_xtick_params(axes[-1])

        fig.tight_layout()
        self._finalize_figure(fig, f"dff_zscore_traces_{self.stem}")

    def plot_peri_event_average(self) -> None:
        """
        Plot trial-averaged peri-event ΔF/F and z-score using seaborn on long-form data.

        Returns
        -------
        None
            The figure is finalized according to the plotting configuration.
        """
        peri_event_long_dataframe = self._build_peri_event_long_dataframe()

        fig, axes = plt.subplots(2, 1, figsize=figure_size_peri, sharex=True)

        axis_signal_pairs = [
            (axes[0], "ΔF/F", color_dff),
            (axes[1], "Z-score", color_zscore),
        ]

        for axis, signal_kind, signal_color in axis_signal_pairs:
            signal_subset = peri_event_long_dataframe.loc[
                peri_event_long_dataframe["signal_kind"] == signal_kind
                ]

            sns.lineplot(
                data=signal_subset,
                x="time_s",
                y="signal_value",
                estimator="mean",
                errorbar="se",
                color=signal_color,
                linewidth=lw_peri_mean,
                ax=axis,
                label=f"Mean {signal_kind}",
            )

            axis.axvline(
                0,
                color=color_event_onset,
                ls="--",
                lw=0.8,
                label="Event onset",
            )
            axis.set_ylabel(signal_kind)
            axis.set_title(
                f"Peri-event average — {signal_kind} (n={self.epochs_dff.shape[0]} trials)"
            )
            axis.grid(alpha=0.3)
            self._set_ytick_params(axis)
            axis.legend()

        axes[-1].set_xlabel("Time from event (s)")
        self._set_xtick_params(axes[-1])

        fig.tight_layout()
        self._finalize_figure(fig, f"peri_event_average_{self.stem}")

    def plot_peri_event_heatmaps(self) -> None:
        """
        Plot peri-event heatmaps for ΔF/F and z-score across trials.
        """
        n_trials, n_timepoints = self.epochs_dff.shape
        if len(self.peri_t) != n_timepoints:
            raise ValueError(f"peri_t length {len(self.peri_t)} != n_timepoints {n_timepoints}")

        trial_labels = self._get_heatmap_trial_labels(n_trials)

        fig, axes = plt.subplots(2, 1, figsize=heatmap_figsize, sharex=True)

        heatmap_specs = [
            (self.epochs_dff, "ΔF/F", heatmap_cmap_dff, heatmap_vmin_dff, heatmap_vmax_dff),
            (self.epochs_z, "Z-score", heatmap_cmap_z, heatmap_vmin_z, heatmap_vmax_z),
        ]

        for ax, (data, label, cmap, vmin, vmax) in zip(axes, heatmap_specs):
            image = ax.imshow(
                data,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                extent=[self.peri_t[0], self.peri_t[-1], 1, n_trials],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xlim(self.peri_t[0], self.peri_t[-1])
            ax.xaxis.set_major_locator(mticker.MultipleLocator(heatmap_xtick_major))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(heatmap_xtick_minor))

            if heatmap_show_all_ylabels:
                yticks = np.arange(1, n_trials + 1, dtype=float)
                yticklabels = trial_labels
            else:
                yticks = np.arange(1, n_trials + 1, heatmap_ytick_major_step, dtype=float)
                yticklabels = [trial_labels[int(i) - 1] for i in yticks]

            ax.set_yticks(yticks, minor=False)
            ax.set_yticklabels(yticklabels)

            if n_trials > 1 and heatmap_ytick_minor_step > 0:
                minor_ticks = np.arange(1.5, n_trials, heatmap_ytick_minor_step, dtype=float)
                ax.set_yticks(minor_ticks, minor=True)
                ax.grid(
                    which="minor",
                    axis="y",
                    color="w",
                    linestyle="-",
                    linewidth=0.3,
                    alpha=1.0,
                )

            self._set_xtick_params(ax)
            self._set_ytick_params(ax)

            ax.axvline(0, color=color_event_onset, ls="--", lw=0.8)
            ax.set_ylabel("Trial")
            ax.set_title(f"Peri-event heatmap — {label} (n={n_trials} trials)")
            fig.colorbar(image, ax=ax, label=label)

        axes[-1].set_xlabel("Time from event (s)")
        fig.tight_layout()
        self._finalize_figure(fig, f"peri_event_heatmaps_{self.stem}")

    def run_all(self) -> None:
        """
        Generate all standard figures for a session.
        """
        print("=" * 50)
        print("PLOTTING PIPELINE")
        print("=" * 50)

        self.plot_full_fluorescence()
        self.plot_full_trace()
        self.plot_peri_event_average()
        self.plot_peri_event_heatmaps()

        if self.save_figures_enabled:
            print(f"SUCCESS: all figures saved to {self.output_directory}")
        else:
            print("SUCCESS: plotting completed (SAVE_FIGURES=False)")

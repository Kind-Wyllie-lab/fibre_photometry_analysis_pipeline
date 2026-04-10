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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors as mcolors

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
    peri_event_plot_mode: str = "trials"
    trial_alpha: float = 0.9

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

    def _build_progressive_lightness_palette(
            self,
            base_color: str,
            n_colors: int,
            min_lightness_mix: float = 0.0,
            max_lightness_mix: float = 0.75,
    ) -> list[tuple[float, float, float]]:
        """
        Generate a sequential palette from dark to light using a base color.

        Parameters
        ----------
        base_color : str
            Matplotlib-compatible base color specification.
        n_colors : int
            Number of colors to generate.
        min_lightness_mix : float, default=0.0
            Fraction of white mixed into the darkest color.
        max_lightness_mix : float, default=0.75
            Fraction of white mixed into the lightest color.

        Returns
        -------
        list of tuple of float
            RGB colors ordered from darkest to lightest.
        """
        if n_colors <= 0:
            return []

        base_rgb = np.array(mcolors.to_rgb(base_color), dtype=float)
        white_rgb = np.ones(3, dtype=float)

        if n_colors == 1:
            return [tuple(base_rgb)]

        mix_values = np.linspace(min_lightness_mix, max_lightness_mix, n_colors)
        palette = [
            tuple((1.0 - mix_value) * base_rgb + mix_value * white_rgb)
            for mix_value in mix_values
        ]
        return palette

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
        Build a long-form dataframe for peri-event plotting.

        Returns
        -------
        pandas.DataFrame
            Long-format dataframe with columns ``trial``, ``time_s``,
            ``signal_value``, and ``signal_kind``.
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

        epochs_dff_long = (
            pd.DataFrame(self.epochs_dff, index=trial_index, columns=self.peri_t)
            .rename_axis(index="trial")
            .reset_index()
            .melt(id_vars="trial", var_name="time_s", value_name="signal_value")
            .assign(signal_kind="ΔF/F")
        )

        epochs_z_long = (
            pd.DataFrame(self.epochs_z, index=trial_index, columns=self.peri_t)
            .rename_axis(index="trial")
            .reset_index()
            .melt(id_vars="trial", var_name="time_s", value_name="signal_value")
            .assign(signal_kind="Z-score")
        )

        peri_event_long_dataframe = pd.concat(
            [epochs_dff_long, epochs_z_long],
            ignore_index=True,
        )
        peri_event_long_dataframe["time_s"] = peri_event_long_dataframe["time_s"].astype(float)
        peri_event_long_dataframe["trial"] = peri_event_long_dataframe["trial"].astype(int)

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

    def plot_peri_event_trials(self) -> None:
        """
        Plot all peri-event z-score trials individually with a dark-to-light progression.

        Notes
        -----
        Trial order is preserved from first to last extracted event. The first trial
        is plotted with the darkest color and the last trial with the lightest color.
        The x-axis is expressed in peri-event seconds.
        """
        n_trials, n_timepoints = self.epochs_z.shape
        if len(self.peri_t) != n_timepoints:
            raise ValueError(
                f"peri_t length {len(self.peri_t)} != z-score epoch length {n_timepoints}"
            )

        zscore_long_dataframe = (
            pd.DataFrame(self.epochs_z, index=np.arange(1, n_trials + 1), columns=self.peri_t)
            .rename_axis(index="trial")
            .reset_index()
            .melt(id_vars="trial", var_name="time_s", value_name="signal_value")
        )
        zscore_long_dataframe["trial"] = zscore_long_dataframe["trial"].astype(int)
        zscore_long_dataframe["time_s"] = zscore_long_dataframe["time_s"].astype(float)

        figure, axis = plt.subplots(1, 1, figsize=figure_size_peri, sharex=True)

        trial_palette = self._build_progressive_lightness_palette(
            base_color=color_zscore,
            n_colors=n_trials,
            min_lightness_mix=0.0,
            max_lightness_mix=0.8,
        )
        trial_to_color = {
            trial_number: trial_palette[trial_number - 1]
            for trial_number in range(1, n_trials + 1)
        }

        sns.lineplot(
            data=zscore_long_dataframe,
            x="time_s",
            y="signal_value",
            hue="trial",
            units="trial",
            estimator=None,
            palette=trial_to_color,
            linewidth=1.2,
            alpha=self.trial_alpha,
            legend=False,
            ax=axis,
        )

        baseline_window_s = getattr(self, "peri_event_local_baseline_s", 2.0)
        axis.axvspan(
            -baseline_window_s,
            0,
            color="grey",
            alpha=0.15,
            label=f"Baseline window ({baseline_window_s:.1f} s)",
        )
        axis.axvline(0, color=color_event_onset, ls="--", lw=0.8, label="Event onset")

        axis.set_xlim(float(self.peri_t[0]), float(self.peri_t[-1]))
        axis.set_xlabel("Time from event (s)")
        axis.set_ylabel("Z-score")
        axis.set_title(f"Peri-event individual trials — Z-score (n={n_trials} trials)")
        axis.grid(alpha=0.3)
        self._set_xtick_params(axis)
        self._set_ytick_params(axis)
        axis.legend()

        figure.tight_layout()
        self._finalize_figure(figure, f"peri_event_trials_zscore_{self.stem}")

    def plot_peri_event_average(self) -> None:
        """
        Plot trial-averaged peri-event z-score using seaborn on long-form data.

        Notes
        -----
        The x-axis is expressed in peri-event seconds. A shaded region indicates
        the amount of pre-trigger baseline used for epoch-local ΔF/F computation.
        """
        n_trials, n_timepoints = self.epochs_z.shape
        if len(self.peri_t) != n_timepoints:
            raise ValueError(
                f"peri_t length {len(self.peri_t)} != z-score epoch length {n_timepoints}"
            )

        zscore_long_dataframe = (
            pd.DataFrame(self.epochs_z, index=np.arange(1, n_trials + 1), columns=self.peri_t)
            .rename_axis(index="trial")
            .reset_index()
            .melt(id_vars="trial", var_name="time_s", value_name="signal_value")
        )
        zscore_long_dataframe["trial"] = zscore_long_dataframe["trial"].astype(int)
        zscore_long_dataframe["time_s"] = zscore_long_dataframe["time_s"].astype(float)

        figure, axis = plt.subplots(1, 1, figsize=figure_size_peri, sharex=True)

        sns.lineplot(
            data=zscore_long_dataframe,
            x="time_s",
            y="signal_value",
            estimator="mean",
            errorbar="se",
            color=color_zscore,
            linewidth=lw_peri_mean,
            ax=axis,
            label="Mean Z-score",
        )

        baseline_window_s = getattr(self, "peri_event_local_baseline_s", 2.0)
        axis.axvspan(
            -baseline_window_s,
            0,
            color="grey",
            alpha=0.15,
            label=f"Baseline window ({baseline_window_s:.1f} s)",
        )
        axis.axvline(
            0,
            color=color_event_onset,
            ls="--",
            lw=0.8,
            label="Event onset",
        )

        axis.set_xlim(float(self.peri_t[0]), float(self.peri_t[-1]))
        axis.set_xlabel("Time from event (s)")
        axis.set_ylabel("Z-score")
        axis.set_title(f"Peri-event average — Z-score (n={n_trials} trials)")
        axis.grid(alpha=0.3)
        self._set_xtick_params(axis)
        self._set_ytick_params(axis)
        axis.legend()

        figure.tight_layout()
        self._finalize_figure(figure, f"peri_event_average_zscore_{self.stem}")

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

    def plot_peri_event_summary(self) -> None:
        """
        Plot peri-event data using the configured display mode.

        Raises
        ------
        ValueError
            If the configured peri-event plot mode is invalid.
        """
        if self.peri_event_plot_mode == "average":
            self.plot_peri_event_average()
            return

        if self.peri_event_plot_mode == "trials":
            self.plot_peri_event_trials()
            return

        raise ValueError(
            f"Unsupported peri_event_plot_mode: {self.peri_event_plot_mode!r}"
        )

    def run_all(self) -> None:
        """
        Generate all standard figures for a session.
        """
        print("=" * 50)
        print("PLOTTING PIPELINE")
        print("=" * 50)

        self.plot_full_fluorescence()
        self.plot_full_trace()
        self.plot_peri_event_summary()
        self.plot_peri_event_heatmaps()

        if self.save_figures_enabled:
            print(f"SUCCESS: all figures saved to {self.output_directory}")
        else:
            print("SUCCESS: plotting completed (SAVE_FIGURES=False)")

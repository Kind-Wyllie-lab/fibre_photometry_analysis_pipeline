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
from typing import Optional, Sequence, Literal
import matplotlib as mpl
from matplotlib.cm import ScalarMappable

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
from stats import _two_group_independent_test, _p_to_stars, _annotate_two_group_stars


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
        n_trials, n_timepoints = self.epochs_z.shape

        trial_index = np.arange(1, n_trials + 1, dtype=int)

        epochs_z_long = (
            pd.DataFrame(self.epochs_z, index=trial_index, columns=self.peri_t)
            .rename_axis(index="trial")
            .reset_index()
            .melt(id_vars="trial", var_name="time_s", value_name="signal_value")
            .assign(signal_kind="Z-score")
        )

        peri_event_long_dataframe = epochs_z_long
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
        # axes[0].grid(alpha=0.15)
        axes[0].legend(loc="upper right")
        self._set_ytick_params(axes[0])

        y_410 = self.df_clean["CH1-410"].to_numpy(dtype=float)
        axes[1].plot(self.time_s, y_410, color=color_410_nm, lw=lw_trace, label="410nm (baseline)")
        self._autoscale_y_to_signal(axes[1], y_410)
        axes[1].set_ylabel("410nm (AU)")
        axes[1].set_xlabel("Time (s)")
        # axes[1].grid(alpha=0.15)
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

    def plot_peri_event_trials(
            self,
            colormap_name: str = "Spectral",
            line_alpha: float = 0.55,
            linewidth: float = 1.1,
            legend_max_trials: int = 12,
            show_colorbar: bool = True,
    ) -> None:
        """
        Plot all peri-event z-score trials individually using a diverging colormap.

        Parameters
        ----------
        colormap_name : str, default="Spectral"
            Matplotlib colormap name. Diverging colormaps that work well include
            "Spectral", "coolwarm", "RdBu_r", and "PiYG".
        line_alpha : float, default=0.55
            Alpha transparency for trial traces to improve overlap visibility.
        linewidth : float, default=1.1
            Line width for trial traces.
        legend_max_trials : int, default=12
            Maximum number of trial entries to include in the legend. If there are
            more trials than this, the legend is suppressed (use the colorbar).
        show_colorbar : bool, default=True
            Whether to show a colorbar mapping trial index to colormap color.

        Raises
        ------
        ValueError
            If peri-event time base is inconsistent with epoch array shape.
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

        # Diverging palette across trials
        cmap = mpl.cm.get_cmap(colormap_name, n_trials if n_trials > 1 else 2)
        norm = mpl.colors.Normalize(vmin=1, vmax=max(1, n_trials))

        trial_to_color = {
            trial_idx: cmap(norm(trial_idx))
            for trial_idx in range(1, n_trials + 1)
        }

        sns.lineplot(
            data=zscore_long_dataframe,
            x="time_s",
            y="signal_value",
            hue="trial",
            units="trial",
            estimator=None,
            palette=trial_to_color,
            linewidth=linewidth,
            alpha=line_alpha,
            legend=(n_trials <= legend_max_trials),
            ax=axis,
        )

        # baseline_window_s = getattr(self, "peri_event_local_baseline_s", 2.0)
        # axis.axvspan(
        #     -baseline_window_s,
        #     0,
        #     color="grey",
        #     alpha=0.15,
        #     label=f"Baseline window ({baseline_window_s:.1f} s)",
        #     zorder=0,
        # )
        axis.axvline(0, color=color_event_onset, ls="--", lw=0.8, label="Event onset")

        axis.set_xlim(float(self.peri_t[0]), float(self.peri_t[-1]))
        axis.set_xlabel("Time from event (s)")
        axis.set_ylabel("Z-score")
        axis.set_title(f"Peri-event individual trials — Z-score (n={n_trials} trials)")
        axis.grid(alpha=0.3)
        self._set_xtick_params(axis)
        self._set_ytick_params(axis)

        # If legend is shown, make it compact and move it outside
        if n_trials <= legend_max_trials:
            axis.legend(
                title="Trial",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
            )
        else:
            # Keep only baseline/event labels in a small legend
            axis.legend(
                loc="upper right",
                frameon=False,
            )

        # Colorbar gives a clean mapping from trial index -> color
        if show_colorbar and n_trials > 1:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            colorbar = figure.colorbar(sm, ax=axis, pad=0.02)
            colorbar.set_label("Trial index")
            # Optionally reduce ticks density
            if n_trials > 12:
                colorbar.set_ticks(np.linspace(1, n_trials, 6, dtype=int))

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
        # axis.axvspan(
        #     -baseline_window_s,
        #     0,
        #     color="grey",
        #     alpha=0.15,
        #     label=f"Baseline window ({baseline_window_s:.1f} s)",
        # )
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
        n_trials, n_timepoints = self.epochs_z.shape
        if len(self.peri_t) != n_timepoints:
            raise ValueError(f"peri_t length {len(self.peri_t)} != n_timepoints {n_timepoints}")

        trial_labels = self._get_heatmap_trial_labels(n_trials)

        fig, axes = plt.subplots(2, 1, figsize=heatmap_figsize, sharex=True)

        heatmap_specs = [
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
        self.plot_peri_event_trials()
        self.plot_peri_event_heatmaps()

        if self.save_figures_enabled:
            print(f"SUCCESS: all figures saved to {self.output_directory}")
        else:
            print("SUCCESS: plotting completed (SAVE_FIGURES=False)")
def plot_freezing_ratio_profiles(
    freezing_profile_df: pd.DataFrame,
    mode: Literal["group_mean_sem", "individual_animals"] = "group_mean_sem",
    group_column: str = "group",
    animal_column: str = "animal",
    session_column: str = "session_name",
    session_name: Optional[str] = None,
    include_post_cs12: bool = True,
    groups_order: Sequence[str] = ("wt", "het", "gcamp"),
    figure_size: tuple[float, float] = (14.0, 5.0),
    colormap_animals: str = "tab20",
    legend_max_items: int = 30,
) -> plt.Figure:
    """
    Plot freezing ratio behavior profiles across bouts using seaborn.

    Parameters
    ----------
    freezing_profile_df : pandas.DataFrame
        Wide dataframe with one row per animal-session and freezing ratios in bout columns.
    mode : {"group_mean_sem", "individual_animals"}, default="group_mean_sem"
        Plotting mode:
        - "group_mean_sem": group mean ± SEM (seaborn errorbar="se")
        - "individual_animals": one curve per animal (hue=animal)
    group_column : str, default="group"
        Group/genotype column name.
    animal_column : str, default="animal"
        Animal identifier column name.
    session_column : str, default="session_name"
        Session identifier column name (optional).
    session_name : str or None, default=None
        If provided and `session_column` exists, filter to this session.
    include_post_cs12 : bool, default=True
        Whether to include post_cs12 bout.
    groups_order : sequence of str, default=("wt", "het", "gcamp")
        Ordering for group plots.
    figure_size : tuple of float, default=(14.0, 5.0)
        Figure size.
    colormap_animals : str, default="tab20"
        Matplotlib colormap name for individual animal curves.
    legend_max_items : int, default=30
        If too many animals are plotted, legend is truncated to at most this many items.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    required = {animal_column, group_column}
    missing = required.difference(freezing_profile_df.columns)
    if missing:
        raise ValueError(f"freezing_profile_df missing required columns: {sorted(missing)}")

    df = freezing_profile_df.copy()
    if session_name is not None and session_column in df.columns:
        df = df.loc[df[session_column] == session_name].copy()

    # Build ordered bout columns: pre_cs, cs_1, noncs_1, cs_2, noncs_2, ...
    bout_columns = ["pre_cs"]
    for i in range(1, 12 + 1):
        bout_columns.append(f"cs_{i}")
        if i <= 10:
            bout_columns.append(f"noncs_{i}")
    if include_post_cs12 and "post_cs12" in df.columns:
        bout_columns.append("post_cs12")
    bout_columns = [c for c in bout_columns if c in df.columns]
    if not bout_columns:
        raise ValueError("No bout columns found in freezing_profile_df")

    id_vars = [animal_column, group_column]
    if session_column in df.columns:
        id_vars.append(session_column)

    df_long = (
        df[id_vars + bout_columns]
        .melt(id_vars=id_vars, var_name="bout", value_name="freezing_ratio")
        .dropna(subset=["freezing_ratio"])
        .copy()
    )
    df_long["freezing_ratio"] = pd.to_numeric(df_long["freezing_ratio"], errors="coerce")
    df_long = df_long.dropna(subset=["freezing_ratio"])

    # Preserve x-order
    df_long["bout"] = pd.Categorical(df_long["bout"], categories=bout_columns, ordered=True)

    # Numeric x for lineplot ensures correct ordering & consistent ticks
    df_long["bout_index"] = df_long["bout"].cat.codes.astype(int)

    # For individual animals: label legend with "animal (group)"
    if mode == "individual_animals":
        df_long["animal_with_group"] = (
            df_long[animal_column].astype(str) + " (" + df_long[group_column].astype(str) + ")"
        )

    fig, ax = plt.subplots(1, 1, figsize=figure_size)

    if mode == "group_mean_sem":
        # seaborn does mean + SEM in one call
        sns.lineplot(
            data=df_long,
            x="bout_index",
            y="freezing_ratio",
            hue=group_column,
            hue_order=list(groups_order),
            estimator="mean",
            errorbar="se",
            lw=2.2,
            ax=ax,
        )
        title = "Freezing ratio profile (group mean ± SEM across animals)"

    elif mode == "individual_animals":
        animal_labels = sorted(df_long["animal_with_group"].unique().tolist())
        n_animals = len(animal_labels)

        cmap = plt.get_cmap(colormap_animals, max(n_animals, 2))
        palette = {label: cmap(i) for i, label in enumerate(animal_labels)}

        sns.lineplot(
            data=df_long,
            x="bout_index",
            y="freezing_ratio",
            hue="animal_with_group",
            estimator=None,
            units="animal_with_group",
            lw=1.4,
            alpha=0.6,
            palette=palette,
            ax=ax,
        )
        title = "Freezing ratio profile (individual animals)"

        # Truncate legend if too large
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > legend_max_items + 1:  # +1 because seaborn includes title entry
            ax.legend(
                handles=handles[: legend_max_items + 1],
                labels=labels[: legend_max_items + 1],
                title="Animal (group)",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
            )
        else:
            ax.legend(
                title="Animal (group)",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
            )

    else:
        raise ValueError("mode must be 'group_mean_sem' or 'individual_animals'")

    # Axes cosmetics
    ax.set_xlim(-0.5, len(bout_columns) - 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Session bout")
    ax.set_ylabel("Freezing ratio (time freezing / total time)")
    ax.set_title(title + (f" | session={session_name}" if session_name is not None else ""))
    ax.grid(alpha=0.25)

    ax.set_xticks(np.arange(len(bout_columns)))
    ax.set_xticklabels(bout_columns, rotation=45, ha="right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig

def plot_extinction_index_wt_vs_het(
    ext_df,
    genotype_col: str = "genotype",
    value_col: str = "ext_index",
    palette: dict | None = None,
    show_points: bool = True,
    point_alpha: float = 0.8,
    errorbar: str | tuple = "se",
    stats_enabled: bool = True,
    alpha: float = 0.05,
    ax: plt.Axes | None = None,
) -> plt.Axes | None:
    """
    Plot extinction index (EI) for wt vs het with mean±SE overlay and star-annotated stats.

    Parameters
    ----------
    ext_df : pandas.DataFrame
        Dataframe with columns [genotype_col, value_col].
    genotype_col : str, default="genotype"
        Genotype column name (values expected: 'wt' and 'het').
    value_col : str, default="ext_index"
        EI column name.
    palette : dict or None, default=None
        Color mapping for groups, default {'wt':'k','het':'b'}.
    show_points : bool, default=True
        Show individual animal dots.
    point_alpha : float, default=0.8
        Alpha for dots.
    errorbar : str or tuple, default="se"
        Seaborn errorbar specification for pointplot.
    stats_enabled : bool, default=True
        Run `_two_group_independent_test` and annotate stars.
    alpha : float, default=0.05
        Significance level for normality/variance checks.
    ax : matplotlib.axes.Axes or None, default=None
        Axis to draw into.

    Returns
    -------
    matplotlib.axes.Axes or None
        Axis with plot, or None if no data.
    """
    if ext_df is None or ext_df.empty:
        print("No extinction index data to plot.")
        return None

    df_plot = ext_df.dropna(subset=[value_col]).copy()
    if df_plot.empty:
        print("All extinction index values are NaN.")
        return None

    if palette is None:
        palette = {"wt": "k", "het": "b"}

    # Enforce wt/het only and fixed order
    df_plot[genotype_col] = df_plot[genotype_col].astype(str)
    df_plot = df_plot.loc[df_plot[genotype_col].isin(["wt", "het"])].copy()
    if df_plot.empty:
        print("No wt/het rows found.")
        return None

    order = ["wt", "het"]
    present = set(df_plot[genotype_col].unique())
    if present != {"wt", "het"}:
        print(f"Expected both wt and het for stats; got {sorted(present)}. Plotting without stats.")
        stats_enabled = False

    if ax is None:
        plt.figure(figsize=(6.5, 4.2))
        ax = plt.gca()

    if show_points:
        sns.stripplot(
            data=df_plot,
            x=genotype_col,
            y=value_col,
            order=order,
            dodge=False,
            alpha=point_alpha,
            palette=palette,
            ax=ax,
        )

    sns.pointplot(
        data=df_plot,
        x=genotype_col,
        y=value_col,
        order=order,
        dodge=0.2,
        join=False,
        markers="D",
        linestyles="",
        errorbar=errorbar,
        palette=palette,
        ax=ax,
    )

    ax.axhline(0.0, color="k", lw=1, alpha=0.35)
    ax.set_ylabel("Extinction index")
    ax.set_xlabel("Genotype")
    ax.set_title("Extinction index (wt vs het)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if stats_enabled:
        x = df_plot.loc[df_plot[genotype_col] == "wt", value_col].to_numpy(dtype=float)
        y = df_plot.loc[df_plot[genotype_col] == "het", value_col].to_numpy(dtype=float)

        res = _two_group_independent_test(x, y, alpha=alpha)
        stars = _p_to_stars(res["p"])
        _annotate_two_group_stars(ax, x_positions=(0, 1), stars=stars)

        print(
            f"EI stats: {res['test']}; p={res['p']:.3g}; normal={res['normal']}; "
            f"equal_var={res['equal_var']}; n={res['n1']} vs {res['n2']}"
        )

    return ax

def plot_modulation_index_wt_vs_het(
    mi_df: pd.DataFrame,
    genotype_col: str = "genotype",
    value_col: str = "mod_index",
    palette: dict | None = None,
    show_points: bool = True,
    point_alpha: float = 0.8,
    errorbar: str | tuple = "se",
    stats_enabled: bool = True,
    alpha: float = 0.05,
    ax: plt.Axes | None = None,
) -> plt.Axes | None:
    """
    Plot modulation index (MI) for wt vs het with mean±SE overlay and star-annotated stats.
    """
    if mi_df is None or mi_df.empty:
        print("No modulation index data to plot.")
        return None

    df_plot = mi_df.dropna(subset=[value_col]).copy()
    if df_plot.empty:
        print("All modulation index values are NaN.")
        return None

    if palette is None:
        palette = {"wt": "k", "het": "b"}

    df_plot[genotype_col] = df_plot[genotype_col].astype(str)
    df_plot = df_plot.loc[df_plot[genotype_col].isin(["wt", "het"])].copy()
    if df_plot.empty:
        print("No wt/het rows found.")
        return None

    order = ["wt", "het"]
    present = set(df_plot[genotype_col].unique())
    if present != {"wt", "het"}:
        print(f"Expected both wt and het for stats; got {sorted(present)}. Plotting without stats.")
        stats_enabled = False

    if ax is None:
        plt.figure(figsize=(6.5, 4.2))
        ax = plt.gca()

    if show_points:
        sns.stripplot(
            data=df_plot,
            x=genotype_col,
            y=value_col,
            order=order,
            dodge=False,
            alpha=point_alpha,
            palette=palette,
            ax=ax,
        )

    sns.pointplot(
        data=df_plot,
        x=genotype_col,
        y=value_col,
        order=order,
        dodge=0.2,
        join=False,
        markers="D",
        linestyles="",
        errorbar=errorbar,
        palette=palette,
        ax=ax,
    )

    ax.axhline(0.0, color="k", lw=1, alpha=0.35)
    ax.set_ylabel("Modulation index")
    ax.set_xlabel("Genotype")
    ax.set_title("Modulation index (wt vs het)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if stats_enabled:
        x = df_plot.loc[df_plot[genotype_col] == "wt", value_col].to_numpy(dtype=float)
        y = df_plot.loc[df_plot[genotype_col] == "het", value_col].to_numpy(dtype=float)

        res = _two_group_independent_test(x, y, alpha=alpha)
        stars = _p_to_stars(res["p"])
        _annotate_two_group_stars(ax, x_positions=(0, 1), stars=stars)

        print(
            f"MI stats: {res['test']}; p={res['p']:.3g}; normal={res['normal']}; "
            f"equal_var={res['equal_var']}; n={res['n1']} vs {res['n2']}"
        )

    return ax

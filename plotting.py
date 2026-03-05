# plotting.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional

from params import *

def _autoscale_y_to_signal(ax, y, pad_percent: float = 5.0):
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return
    y_min, y_max = y.min(), y.max()
    if y_min == y_max:
        ax.set_ylim(y_min - 1, y_max + 1)
        return
    span = y_max - y_min
    pad = span * pad_percent / 100.0
    ax.set_ylim(y_min - pad, y_max + pad)


def _finalise_figure(fig: plt.Figure, filename: str):
    if SAVE_FIGURES:
        out_path = FIGURES_DIR / f"{filename}.{FIGURE_FORMAT}"
        fig.savefig(out_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        print(f"Saved: {out_path}")
    if PREVIEW_FIGURES:
        plt.show()
    if not PREVIEW_FIGURES:
        plt.close(fig)


def _set_xtick_params(ax):
    ax.tick_params(axis="x", reset=True)
    ax.tick_params(
        axis="x",
        bottom=True, top=False,
        labelbottom=True, labeltop=False,
        labelsize=XTICK_FONTSIZE,
        direction=XTICK_DIRECTION,
        width=XTICK_WIDTH_MAJOR,
        length=XTICK_LENGTH_MAJOR,
        which="major",
    )
    ax.tick_params(
        axis="x",
        bottom=True, top=False,
        width=XTICK_WIDTH_MINOR,
        length=XTICK_LENGTH_MINOR,
        which="minor",
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(XTICK_NBINS))


def _set_ytick_params(ax):
    ax.tick_params(axis="y", reset=True)
    ax.tick_params(
        axis="y",
        left=True, right=False,
        labelleft=True, labelright=False,
        labelsize=YTICK_FONTSIZE,
        direction=YTICK_DIRECTION,
        width=YTICK_WIDTH_MAJOR,
        length=YTICK_LENGTH_MAJOR,
        which="major",
    )
    ax.tick_params(
        axis="y",
        left=True, right=False,
        width=YTICK_WIDTH_MINOR,
        length=YTICK_LENGTH_MINOR,
        which="minor",
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(YTICK_NBINS))


def plot_full_fluorescence(df_clean: pd.DataFrame, stem: str):
    if "CH1-470" not in df_clean.columns or "CH1-410" not in df_clean.columns:
        raise ValueError("df_clean missing CH1-470 or CH1-410 columns")

    time_s = df_clean["TimeStamp"].values / 1000.0
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE_TRACE, sharex=True)

    y470 = df_clean["CH1-470"].values
    axes[0].plot(time_s, y470, color=COLOR_470, lw=LW_TRACE, label="470nm (calcium)")
    _autoscale_y_to_signal(axes[0], y470)
    axes[0].set_ylabel("470nm (AU)")
    axes[0].set_title(f"Full-session raw fluorescence ({stem})")
    axes[0].grid(alpha=0.15)
    axes[0].legend(loc="upper right")
    _set_ytick_params(axes[0])

    y410 = df_clean["CH1-410"].values
    axes[1].plot(time_s, y410, color=COLOR_410, lw=LW_TRACE, label="410nm (isosbestic)")
    _autoscale_y_to_signal(axes[1], y410)
    axes[1].set_ylabel("410nm (AU)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.15)
    axes[1].legend(loc="upper right")
    _set_xtick_params(axes[1])
    _set_ytick_params(axes[1])

    fig.tight_layout()
    _finalise_figure(fig, f"raw_Traces_{stem}")


def plot_full_trace(df_clean: pd.DataFrame, dff_fitted: np.ndarray,
                    zscore: np.ndarray, stem: str,
                    event_times_s: Optional[np.ndarray] = None,
                    t_zero_s: Optional[float] = None):
    if len(dff_fitted) != len(df_clean) or len(zscore) != len(df_clean):
        raise ValueError(f"dff/zscore length {len(dff_fitted)} != df_clean length {len(df_clean)}")

    time_s = df_clean["TimeStamp"].values / 1000.0
    origin = t_zero_s if t_zero_s is not None else time_s[0]

    idx_start = np.searchsorted(time_s, origin)
    time_plot = time_s[idx_start:] - origin
    dff_plot = dff_fitted[idx_start:]
    zscore_plot = zscore[idx_start:]

    if event_times_s is not None:
        event_times_plot = event_times_s - origin
    else:
        event_times_plot = time_plot[df_clean["Events_numeric"].values[idx_start:] == 1]

    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE_TRACE, sharex=True)

    for ax, sig, ylabel, color in zip(
            axes,
            [dff_plot, zscore_plot],
            ["ΔF/F", "Z-score"],
            [COLOR_DFF, COLOR_ZSCORE],
    ):
        ax.plot(time_plot, sig, color=color, lw=LW_TRACE, label=ylabel)
        for t in event_times_plot:
            ax.axvline(x=t, color=COLOR_EVENT_ONSET, ls="--",
                       alpha=ALPHA_EVENT_LINES, lw=LW_EVENT_MARKER)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right")
        _set_ytick_params(ax)

    n_events = len(event_times_plot)
    skipped_label = " (first skipped)" if SKIP_FIRST_EVENT and event_times_s is not None else ""
    zero_label = f"t=0 at first TTL ({origin:.2f}s)" if t_zero_s is not None else "t=0 at recording start"
    axes[0].set_title(
        f"Full session — motion-corrected signal ({stem}) "
        f"| {n_events} events{skipped_label} | {zero_label}"
    )
    axes[-1].set_xlabel("Time from first TTL pulse (s)")
    _set_xtick_params(axes[-1])

    fig.tight_layout()
    _finalise_figure(fig, f"DFF_Zs_traces_{stem}")


def plot_peri_event_average(peri_t: np.ndarray, epochs_dff: np.ndarray,
                            epochs_z: np.ndarray, stem: str):
    if epochs_dff.ndim != 2 or epochs_z.ndim != 2:
        raise ValueError(
            f"epochs must be 2D (n_trials x n_timepoints), "
            f"got dff={epochs_dff.ndim}D, z={epochs_z.ndim}D"
        )

    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE_PERI, sharex=True)

    for ax, epochs, ylabel in zip(
            axes,
            [epochs_dff, epochs_z],
            ["ΔF/F", "Z-score"],
    ):
        mean = epochs.mean(axis=0)
        sem = epochs.std(axis=0) / np.sqrt(len(epochs))

        ax.plot(peri_t, mean, color=COLOR_PERI_MEAN,
                lw=LW_PERI_MEAN, label=f"Mean {ylabel}")
        ax.fill_between(peri_t, mean - sem, mean + sem,
                        alpha=ALPHA_SEM_FILL, color=COLOR_PERI_MEAN)
        ax.axvline(0, color=COLOR_EVENT_ONSET, ls="--", lw=0.8, label="Event onset")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Peri-event average — {ylabel} "
            f"(n={len(epochs)} trials"
            f"{', first skipped' if SKIP_FIRST_EVENT else ''})"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        _set_ytick_params(ax)

    axes[-1].set_xlabel("Time from event (s)")
    _set_xtick_params(axes[-1])

    fig.tight_layout()
    _finalise_figure(fig, f"peri_event_{stem}")


def plot_peri_event_heatmaps(peri_t: np.ndarray,
                             epochs_dff: np.ndarray,
                             epochs_z: np.ndarray,
                             stem: str,
                             filtered_events: Optional[pd.DataFrame] = None):
    if epochs_dff.ndim != 2 or epochs_z.ndim != 2:
        raise ValueError(
            f"epochs must be 2D, got dff={epochs_dff.ndim}D, z={epochs_z.ndim}D"
        )

    n_trials, n_time = epochs_dff.shape
    if len(peri_t) != n_time:
        raise ValueError(f"peri_t length {len(peri_t)} != n_timepoints {n_time}")

    if filtered_events is not None and "cluster_id" in filtered_events.columns:
        cluster_ids = filtered_events["cluster_id"].to_numpy()
        cluster_ids = cluster_ids[:n_trials].astype(str)
    else:
        cluster_ids = [str(i) for i in range(1, n_trials + 1)]

    fig, axes = plt.subplots(2, 1, figsize=HEATMAP_FIGSIZE, sharex=True)

    subplot_specs = [
        ("ΔF/F", HEATMAP_CMAP_DFF, HEATMAP_VMIN_DFF, HEATMAP_VMAX_DFF),
        ("Z-score", HEATMAP_CMAP_Z, HEATMAP_VMIN_Z, HEATMAP_VMAX_Z),
    ]

    for ax, data, (y_label, cmap, v_min, v_max) in zip(
            axes, [epochs_dff, epochs_z], subplot_specs
    ):
        im = ax.imshow(
            data,
            aspect="auto",
            interpolation="nearest",
            alpha=1,
            origin="lower",
            extent=[peri_t[0], peri_t[-1], 1, n_trials],
            cmap=cmap,
            vmin=v_min,
            vmax=v_max,
        )

        ax.set_xlim(peri_t[0], peri_t[-1])
        major_locator = mticker.MultipleLocator(HEATMAP_XTICK_MAJOR)
        minor_locator = mticker.MultipleLocator(HEATMAP_XTICK_MINOR)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)

        if HEATMAP_SHOW_ALL_YLABELS:
            yticks = np.arange(1, n_trials + 1, dtype=float)
            yticklabels = cluster_ids
        else:
            step = HEATMAP_YTICK_MAJOR_STEP
            yticks = np.arange(1, n_trials + 1, step, dtype=float)
            yticklabels = [cluster_ids[int(i) - 1] for i in yticks]

        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels(yticklabels)

        if n_trials > 1 and HEATMAP_YTICK_MINOR_STEP > 0:
            minor_ticks = np.arange(1.5, n_trials, HEATMAP_YTICK_MINOR_STEP, dtype=float)
            ax.set_yticks(minor_ticks, minor=True)
            ax.grid(which="minor", axis="y",
                    color="w", linestyle="-", linewidth=0.3, alpha=1.0)

        _set_xtick_params(ax)
        _set_ytick_params(ax)

        ax.axvline(0, color=COLOR_EVENT_ONSET, ls="--", lw=0.8)
        ax.set_ylabel("Trial")
        ax.set_title(
            f"Peri-event heatmap — {y_label} (n={n_trials} trials"
            f"{', first skipped' if SKIP_FIRST_EVENT else ''})"
        )
        fig.colorbar(im, ax=ax, label=y_label)

    axes[-1].set_xlabel("Time from event (s)")
    fig.tight_layout()
    _finalise_figure(fig, f"peri_event_heatmaps_{stem}")


def run_all_plots(df_clean: pd.DataFrame,
                  dff_fitted: np.ndarray,
                  zscore: np.ndarray,
                  epochs_dff: np.ndarray,
                  epochs_z: np.ndarray,
                  peri_t: np.ndarray,
                  stem: str,
                  event_times_s: Optional[np.ndarray] = None,
                  t_zero_s: Optional[float] = None,
                  filtered_events: Optional[pd.DataFrame] = None):
    """Generate all plots for fiber photometry analysis.

    Args:
        df_clean: Preprocessed DataFrame with TimeStamp, CH1-470, CH1-410
        dff_fitted: Full-trace ΔF/F normalized to fitted410 (photobleaching correction)
        zscore: Full-trace z-score
        epochs_dff: Peri-event ΔF/F epochs (n_trials x n_timepoints)
        epochs_z: Peri-event z-score epochs (n_trials x n_timepoints)
        peri_t: Time vector for epochs (s, relative to event)
        stem: Filename stem for saved figures
        event_times_s: Event timestamps (s) for vertical lines in full trace
        t_zero_s: Time to sync full trace to (first event time)
        filtered_events: DataFrame with cluster_id for heatmap y-labels
    """
    print("=" * 50)
    print("PLOTTING PIPELINE")
    print("=" * 50)

    plot_full_fluorescence(df_clean, stem)
    plot_full_trace(df_clean, dff_fitted, zscore, stem,
                    event_times_s=event_times_s, t_zero_s=t_zero_s)
    plot_peri_event_average(peri_t, epochs_dff, epochs_z, stem)
    plot_peri_event_heatmaps(peri_t, epochs_dff, epochs_z, stem, filtered_events)

    if SAVE_FIGURES:
        print(f"SUCCESS: all figures saved to {FIGURES_DIR}")
    else:
        print("SUCCESS: plotting completed (no files saved, SAVE_FIGURES=False)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/Lou/PycharmProjects/Fibre_photometry_M2")

    from preprocessing import process_single_file
    from event_sorting import process_events
    from signal_processing import process_signals

    input_df = process_single_file()
    input_stem = "Fluorescence"

    _, input_first_events = process_events(input_df, input_stem)

    (
        input_epochs_dff,
        input_epochs_z,
        input_peri_t,
        input_dff_baseline,
        input_dff_fitted,
        input_zscore,
        input_filtered_events
    ) = process_signals(input_df, input_first_events, input_stem)

    input_t_zero_s = input_filtered_events.iloc[0]["TimeStamp"] / 1000.0
    input_event_times_s = input_filtered_events["TimeStamp"].values / 1000.0

    run_all_plots(
        input_df,
        input_dff_fitted,
        input_zscore,
        input_epochs_dff,
        input_epochs_z,
        input_peri_t,
        input_stem,
        event_times_s=input_event_times_s,
        t_zero_s=input_t_zero_s,
        filtered_events=input_filtered_events
    )

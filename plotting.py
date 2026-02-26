# plotting.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from params import (OUTPUT_DIR, SKIP_FIRST_EVENT,
                    FIGURE_DPI, FIGURE_FORMAT,
                    FIGURE_SIZE_TRACE, FIGURE_SIZE_PERI,
                    COLOR_DFF, COLOR_ZSCORE, COLOR_PERI_MEAN,
                    COLOR_EVENT_0, COLOR_EVENT_1, COLOR_EVENT_ONSET,
                    ALPHA_EVENT_LINES, ALPHA_SEM_FILL,
                    LW_TRACE, LW_PERI_MEAN, LW_EVENT_MARKER,
                    SAVE_FIGURES, PREVIEW_FIGURES)

FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


def _finalise_figure(fig: plt.Figure, filename: str):
    """Save and/or preview figure based on SAVE_FIGURES/PREVIEW_FIGURES in params."""
    if SAVE_FIGURES:
        out_path = FIGURES_DIR / f"{filename}.{FIGURE_FORMAT}"
        fig.savefig(out_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        print(f"Saved: {out_path}")
    if PREVIEW_FIGURES:
        plt.show()
    if not PREVIEW_FIGURES:
        plt.close(fig)


def plot_full_trace(df_clean: pd.DataFrame, dff: np.ndarray,
                    zscore: np.ndarray, stem: str):
    """Full-session ΔF/F and z-score traces with colour-coded event onset markers."""
    time_s = df_clean["TimeStamp"].values / 1000.0
    event_times_0 = time_s[df_clean["Events_numeric"].values == 1]  # Input1*2*0
    event_times_1 = time_s[df_clean["Events_numeric"].values == 2]  # Input1*2*1

    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE_TRACE, sharex=True)

    for ax, sig, ylabel, color in zip(
        axes,
        [dff, zscore],
        ["ΔF/F", "Z-score"],
        [COLOR_DFF, COLOR_ZSCORE]
    ):
        ax.plot(time_s, sig, color=color, lw=LW_TRACE, label=ylabel)
        for t in event_times_0:
            ax.axvline(x=t, color=COLOR_EVENT_0, ls="--",
                       alpha=ALPHA_EVENT_LINES, lw=LW_EVENT_MARKER)
        for t in event_times_1:
            ax.axvline(x=t, color=COLOR_EVENT_1, ls="--",
                       alpha=ALPHA_EVENT_LINES, lw=LW_EVENT_MARKER)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"Full session — motion-corrected signal ({stem})")
    fig.tight_layout()
    _finalise_figure(fig, f"full_trace_{stem}")


def plot_peri_event_average(peri_t: np.ndarray, epochs_dff: np.ndarray,
                             epochs_z: np.ndarray, stem: str):
    """Peri-event mean ± SEM for ΔF/F and z-score aligned to event onset."""
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE_PERI)

    for ax, epochs, ylabel in zip(
        axes,
        [epochs_dff, epochs_z],
        ["ΔF/F", "Z-score"]
    ):
        mean = epochs.mean(axis=0)
        # SEM = σ / √n — uncertainty on the mean across trials
        sem = epochs.std(axis=0) / np.sqrt(len(epochs))

        ax.plot(peri_t, mean, color=COLOR_PERI_MEAN, lw=LW_PERI_MEAN, label=f"Mean {ylabel}")
        ax.fill_between(peri_t, mean - sem, mean + sem,
                        alpha=ALPHA_SEM_FILL, color=COLOR_PERI_MEAN)
        ax.axvline(0, color=COLOR_EVENT_ONSET, ls="--", lw=0.8, label="Event onset")
        ax.set_xlabel("Time from event (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Peri-event average — {ylabel} "
                     f"(n={len(epochs)} trials"
                     f"{', first skipped' if SKIP_FIRST_EVENT else ''})")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    _finalise_figure(fig, f"peri_event_{stem}")


def run_all_plots(df_clean: pd.DataFrame, dff: np.ndarray, zscore: np.ndarray,
                  epochs_dff: np.ndarray, epochs_z: np.ndarray,
                  peri_t: np.ndarray, stem: str):
    print("=" * 50)
    print("PLOTTING PIPELINE")
    print("=" * 50)
    plot_full_trace(df_clean, dff, zscore, stem)
    plot_peri_event_average(peri_t, epochs_dff, epochs_z, stem)
    print(f"SUCCESS: all figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/Users/Lou/PycharmProjects/Fibre_photometry_M2")

    from preprocessing import process_single_file
    from event_sorting import process_events
    from signal_processing import (process_signals,
                                   compute_corrected_signal,
                                   compute_dff_and_zscore)

    input_df = process_single_file()
    input_stem = "Fluorescence"
    _, input_first_events = process_events(input_df, input_stem)
    epochs_dff, epochs_z, peri_t = process_signals(input_df, input_first_events, input_stem)

    # Recompute dff/zscore arrays for full trace plot
    corrected = compute_corrected_signal(input_df)
    input_dff, input_zscore = compute_dff_and_zscore(corrected)

    run_all_plots(input_df, input_dff, input_zscore,
                  epochs_dff, epochs_z, peri_t, input_stem)

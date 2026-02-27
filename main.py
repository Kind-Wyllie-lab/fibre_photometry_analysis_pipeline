# main.py
from pathlib import Path
from params import DATA_FILES


# === RUN FLAGS: set True/False to enable/disable each pipeline stage ===
RUN_PREPROCESSING      = True
RUN_EVENT_SORTING      = True
RUN_SIGNAL_PROCESSING  = True
RUN_PLOTTING           = True


def run_pipeline(file_path: str):
    stem         = Path(file_path).stem
    df_clean     = None
    first_events = None
    epochs_dff   = epochs_z = peri_t = None
    dff          = zscore = None
    event_times_s = None

    if RUN_PREPROCESSING:
        from preprocessing import process_single_file
        df_clean = process_single_file(file_path)

    if RUN_EVENT_SORTING:
        if df_clean is None:
            raise ValueError("RUN_PREPROCESSING must be True before RUN_EVENT_SORTING")
        from event_sorting import process_events
        _, first_events = process_events(df_clean, stem)

    if RUN_SIGNAL_PROCESSING:
        from signal_processing import process_signals, filter_first_event

        epochs_dff, epochs_z, peri_t, dff, zscore = process_signals(
            df_clean, first_events, stem
        )

        # t_zero_s = very first raw event (before skip) — x=0 on full trace plot
        t_zero_s = first_events.iloc[0]["TimeStamp"] / 1000.0

        # filtered events = post-skip — used for epoch extraction and plot markers
        filtered_events = filter_first_event(first_events)
        event_times_s = filtered_events["TimeStamp"].values / 1000.0

    if RUN_PLOTTING:
        from plotting import run_all_plots
        run_all_plots(df_clean, dff, zscore,
                      epochs_dff, epochs_z, peri_t, stem,
                      event_times_s=event_times_s,
                      t_zero_s=t_zero_s)


if __name__ == "__main__":
    for file_path in DATA_FILES:
        print(f"\n{'=' * 50}")
        print(f"PROCESSING: {Path(file_path).name}")
        print(f"{'=' * 50}")
        run_pipeline(file_path)

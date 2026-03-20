# main.py
import os
from pathlib import Path

from matplotlib import pyplot as plt
from preprocessing import process_session
from signal_processing import process_signals, filter_first_event
from plotting import run_all_plots
from event_sorting import process_events

from params import base_path, sessions, animal_output_dirs
import matplotlib
matplotlib.use('TkAgg')

# === RUN FLAGS: set True/False to enable/disable each pipeline stage ===
run_preprocessing      = True
run_event_sorting      = True
run_signal_processing  = True
run_plotting           = True


def run_pipeline():
    df_clean     = None
    first_events = None
    epochs_dff   = epochs_z = peri_t = None
    dff          = zscore = None
    event_times_s = None

    for animal in [i for i in os.listdir(base_path) if i.startswith("Rat")]:
        animal_folder_path = os.path.join(base_path, animal)
        for session in sessions:
            session_dir = os.path.join(animal_folder_path, session)
            raw_data_dir = [i for i in os.listdir(session_dir) if i.startswith(animal)][0]
            raw_data_dir = os.path.join(session_dir, raw_data_dir)
            output_dir = animal_output_dirs(animal, session)

            if session == "cond":
                skip_first_event = True
            elif session == "Recall":
                skip_first_event = False

            if run_preprocessing:
                df_clean = process_session(raw_data_dir, output_dir)

            if run_event_sorting:
                if df_clean is None:
                    raise ValueError("RUN_PREPROCESSING must be True before RUN_EVENT_SORTING")
                _, first_events = process_events(df_clean, output_dir)

            if run_signal_processing:

                epochs_dff, epochs_z, peri_t, dff_baseline, dff_fitted, zscore, events_to_use = process_signals(
                    df_clean, first_events, animal_folder_path, skip_first_event
                )

                # t_zero_s = very first raw event (before skip) — x=0 on full trace plot
                t_zero_s = first_events.iloc[0]["TimeStamp"] / 1000.0

                # filtered events = post-skip — used for epoch extraction and plot markers
                filtered_events = filter_first_event(first_events, skip_first_event)
                event_times_s = filtered_events["TimeStamp"].values / 1000.0

            if run_plotting:
                run_all_plots(df_clean, dff_fitted, zscore,
                              epochs_dff, epochs_z, peri_t, animal_folder_path,
                              event_times_s=event_times_s,
                              t_zero_s=t_zero_s)


if __name__ == "__main__":

    run_pipeline()
    plt.show()
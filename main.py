# Replace main.py content with this version, or adapt the run section

import matplotlib
matplotlib.use("TkAgg")

from matplotlib import pyplot as plt

from pipeline import PhotometryPipeline
from params import base_path, sessions

# === RUN FLAGS: set True/False to enable/disable each pipeline stage ===
run_preprocessing = True
run_event_sorting = True
run_signal_processing = True
run_plotting = True


def run_pipeline():
    """
    Run the batch fiber photometry pipeline across all configured animals and sessions.

    Returns
    -------
    list of PhotometrySession
        Completed session-level processing objects.
    """
    pipeline = PhotometryPipeline(
        base_directory=base_path,
        session_names=sessions,
        animal_names=None,  # None => auto-detect folders starting with "Rat"
        run_preprocessing=run_preprocessing,
        run_event_sorting=run_event_sorting,
        run_signal_processing=run_signal_processing,
        run_plotting=run_plotting,
    )
    return pipeline.run()


if __name__ == "__main__":
    completed_sessions = run_pipeline()
    plt.show()

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

from pathlib import Path
from group_analysis import PhotometryGroupAnalyzer

def run_pipeline():
    """
    Run the batch fiber photometry pipeline across all configured animals and sessions.

    Returns
    -------
    PhotometryPipeline
        Completed batch pipeline object.
    """
    pipeline = PhotometryPipeline(
        base_directory=base_path,
        session_names=sessions,
        animal_names=None,
        run_preprocessing=run_preprocessing,
        run_event_sorting=run_event_sorting,
        run_signal_processing=run_signal_processing,
        run_plotting=run_plotting,
    )
    pipeline.run()
    return pipeline


if __name__ == "__main__":
    qcompleted_pipeline = run_pipeline()

    group_peri_event_dataframe = completed_pipeline.build_group_peri_event_dataframe()

    group_analyzer = PhotometryGroupAnalyzer(
        session_level_peri_event_dataframe=group_peri_event_dataframe,
        output_directory=Path(base_path) / "group_outputs",
    )

    group_analyzer.export_group_tables()
    group_analyzer.plot_group_average_all_events()

    # Example: plot first event only across animals
    group_analyzer.plot_group_average_single_event(event_index=1)

    plt.show()

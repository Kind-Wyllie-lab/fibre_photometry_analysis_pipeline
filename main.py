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

# Replace main.py content with this version, or adapt the run section

import matplotlib

import params

matplotlib.use("TkAgg")

from matplotlib import pyplot as plt

from pipeline import PhotometryPipeline
from params import base_path, sessions

# === RUN FLAGS: set True/False to enable/disable each pipeline stage ===
run_preprocessing = True
run_event_sorting = True
run_signal_processing = True
run_plotting = False

from pathlib import Path
from group_analysis import PhotometryGroupAnalyzer, run_group_level_plots_for_event_types


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
        run_plotting=run_plotting
    )
    pipeline.run()
    return pipeline


if __name__ == "__main__":
    completed_pipeline = run_pipeline()

    event_types = {
        "cs_onsets": "cs_led_cluster_first_onsets",
        "cs_offsets": "cs_led_cluster_first_offsets",
        # "freezing_onsets": "freezing_cluster_first_onsets",
        # "freezing_offsets": "freezing_cluster_first_offsets",
    }

    run_group_level_plots_for_event_types(
        completed_pipeline=completed_pipeline,
        event_types=event_types,
        group_output_root=Path(base_path) / "group_outputs",
        auc_window_start_s=0.0,
        auc_window_end_s=5.0,
        max_event_index=12,
        session_name_for_auc="Recall",
    )

    plt.show()
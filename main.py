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
        run_plotting=run_plotting
    )
    pipeline.run()
    return pipeline


if __name__ == "__main__":
    completed_pipeline = run_pipeline()

    group_peri_event_dataframe = completed_pipeline.build_group_peri_event_dataframe()

    group_analyzer = PhotometryGroupAnalyzer(
        session_level_peri_event_dataframe=group_peri_event_dataframe,
        output_directory=Path(base_path) / "group_outputs",
    )

    # group_analyzer.export_group_tables()
    group_analyzer.plot_group_average_all_events()

    # Example: plot first event only across animals
    # group_analyzer.plot_group_average_single_event(event_index=1)
    # group_analyzer.plot_group_average_all_events()
    # group_analyzer.plot_all_single_event_group_averages()
    # group_analyzer.plot_group_event_auc_across_first_events(
    #     auc_window_start_s=0.0,
    #     auc_window_end_s=5.0,
    #     max_event_index=12,
    #     session_name="Recall",
    # )

    for animal in group_analyzer.metadata_dataframe['animal'].values:
        group_analyzer.plot_group_event_auc_across_first_events(
        # animal=animal,
        auc_window_start_s=0.0,
        auc_window_end_s=5.0,
        max_event_index=12,
        session_name="Recall",
        )
        plt.show()
        # group_analyzer.plot_event_trace_stack_by_animal_3d(
        #     event_index=1,
        #     session_name="cond",
        #     group_name="wt",
        # )
        # group_analyzer.plot_event_trace_stack_by_animal_3d(
        #     event_index=1,
        #     session_name="cond",
        #     group_name=None,
        # )
        # plt.show()
        # group_analyzer.plot_group_average_trace_stack_by_event_3d(
        #     group_name="wt",
        #     max_event_index=12,
        #     session_name="cond",
        # )
        # plt.show()
        # group_analyzer.plot_group_average_trace_stack_by_event_3d(
        #     group_name="het",
        #     max_event_index=12,
        #     session_name="cond",
        # )
        # plt.show()
        # group_analyzer.plot_group_average_trace_stack_by_event_3d(
        #     group_name="gcamp",
        #     max_event_index=12,
        #     session_name="cond",
        # )
        # plt.show()
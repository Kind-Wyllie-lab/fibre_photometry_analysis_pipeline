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
import pandas as pd

import params
from behaviour_processing import build_freezing_behavior_profile_table, freezing_profile_wide_to_tidy, \
    compute_extinction_index, compute_modulation_index
from plotting import plot_freezing_ratio_profiles, plot_extinction_index_wt_vs_het, plot_modulation_index_wt_vs_het

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

    event_types = ["cs_onsets","shock", "cs_offsets", "freezing_onsets", "freezing_offsets"]

    run_group_level_plots_for_event_types(
        completed_pipeline=completed_pipeline,
        event_types=event_types,
        group_output_root=Path(base_path) / "group_outputs",
        auc_window_start_s=0.0,
        auc_window_end_s=5.0,
        session_name_for_auc="Cond",
    )

    metadata_df = pd.read_excel(
        Path(base_path) / "animals_metadata.ods",
        engine="odf",
    )[["animal_name", "group"]].rename(columns={"animal_name": "animal"})

    freezing_profile_df = build_freezing_behavior_profile_table(
        completed_sessions=completed_pipeline.results,
        metadata_dataframe=metadata_df,
    )

    recall_df = freezing_profile_df.loc[freezing_profile_df["session_name"] == "Recall"]

    fig = plot_freezing_ratio_profiles(
        freezing_profile_df=freezing_profile_df,
        mode="group_mean_sem",
        session_name="Recall",  # optional if column exists
    )

    fig2 = plot_freezing_ratio_profiles(
        freezing_profile_df=freezing_profile_df,
        mode="individual_animals",
        session_name="Recall",  # optional
        colormap_animals="tab20",
        legend_max_items=40,
    )

    # 1) convert wide -> tidy
    df_freeze_tidy = freezing_profile_wide_to_tidy(
        freezing_profile_df=freezing_profile_df,
        animal_col="animal",
        genotype_col="group",
        session_col="session_name",  # omit or keep if present
        value_col_out="freeze_pct",
        segment_col_out="segment",
    )

    # 2) compute EI per animal (and genotype) using cs_1..cs_12
    ext_df = compute_extinction_index(
        df_freeze=df_freeze_tidy,
        group_by=("animal", "genotype"),
        require_min_cs=10,
        n_first=3,
        n_last=3,
        cs_regex=r"^cs_(\d+)$",
    )

    ext_df = ext_df.replace('gcamp', 'wt')
    # 3) plot EI
    ax = plot_extinction_index_wt_vs_het(
        ext_df=ext_df,  # output of compute_extinction_index(...)
        genotype_col="genotype",  # or "group" depending on your ext_df
        value_col="ext_index",
        palette={"wt": "k", "het": "b"},
        errorbar="se",
        stats_enabled=True,
        alpha=0.05,
    )

    # 2) compute MI (CS vs NONCS)
    mi_df = compute_modulation_index(
        df_freeze=df_freeze_tidy,
        group_by=("animal", "genotype"),
        require_min_cs=1,
        require_min_noncs=1,
        cs_regex=r"^cs_(\d+)$",
        noncs_regex=r"^noncs_(\d+)$",
    )
    mi_df = mi_df.replace('gcamp', 'wt')

    # 3) plot MI wt vs het + stats
    ax = plot_modulation_index_wt_vs_het(
        mi_df=mi_df,
        genotype_col="genotype",
        value_col="mod_index",
        palette={"wt": "k", "het": "b"},
        errorbar="se",
        stats_enabled=True,
        alpha=0.05,
    )
    plt.show()


    print()
    # plt.show()
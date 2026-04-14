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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import params
from epoching import EventEpochExtractor
from params import animal_output_dirs
from session import PhotometrySession
from group_analysis import (
    build_session_peri_event_long_dataframe,
    PhotometryGroupAnalyzer,
)
from signal_processing import extract_epoched_data


@dataclass
class PhotometryPipeline:
    """
    End-to-end pipeline for batch processing fiber photometry data across animals and sessions.

    Parameters
    ----------
    base_directory : str or Path
        Root directory containing animal folders.
    session_names : sequence of str
        Session names to evaluate for each animal.
    animal_names : sequence of str or None, default=None
        Explicit animal identifiers to process. If ``None``, all folders beginning
        with ``"Rat"`` in `base_directory` are used.
    run_preprocessing : bool, default=True
        Whether to execute preprocessing.
    run_event_sorting : bool, default=True
        Whether to execute event sorting.
    run_signal_processing : bool, default=True
        Whether to execute signal processing.
    run_plotting : bool, default=True
        Whether to execute plotting.

    Attributes
    ----------
    results : list of PhotometrySession
        Completed session objects in processing order.
    """

    base_directory: Path | str
    session_names: Sequence[str]
    animal_names: Optional[Sequence[str]] = None
    run_preprocessing: bool = True
    run_event_sorting: bool = True
    run_signal_processing: bool = True
    run_plotting: bool = True

    results: list[PhotometrySession] = field(default_factory=list, init=False)

    def get_animal_names(self) -> list[str]:
        """
        Resolve the list of animals to process.

        Returns
        -------
        list of str
            Sorted animal identifiers.
        """
        if self.animal_names is not None:
            return list(self.animal_names)

        base_directory = Path(self.base_directory)
        return sorted(
            [name for name in os.listdir(base_directory) if name.startswith("Rat")]
        )

    def build_group_peri_event_dataframe(
            self,
            event_table_key: str = None,
            signal_key: str = "zscore",
    ) -> pd.DataFrame:
        """
        Build a concatenated long-form peri-event dataframe from all completed sessions,
        using an event-table-driven epoching specification.

        Parameters
        ----------
        epoching_spec : EpochingSpec
            Specifies which event table to use for epoch extraction.
            Example event keys:
            - "cs_led_cluster_first_onsets"
            - "cs_led_cluster_first_offsets"
            - "freezing_cluster_first_onsets"
            - "freezing_cluster_first_offsets"
        signal_key : str, default="zscore"
            Which preprocessed continuous signal to epoch. Typically "zscore".

        Returns
        -------
        pandas.DataFrame
            Long-format dataframe across all processed animal/session pairs.

        Raises
        ------
        ValueError
            If no sessions contribute data for the requested epoching specification.
        """
        session_level_dataframes: list[pd.DataFrame] = []

        for completed_session in self.results:
            time_s = completed_session.df_clean["TimeStamp"].to_numpy(dtype=float) / 1000.0

            # Select event times from the requested event table
            event_times_s = completed_session.event_tables[event_table_key]

            # Skip sessions with no such event type (e.g. no freezing)
            if event_times_s is None or event_times_s.size == 0:
                continue

            n_pre = int(params.time_pre_event_s * params.sample_rate_hz)
            n_post = int(params.time_post_event_s * params.sample_rate_hz)

            # Epoch just the requested signal (zscore)
            epochs_by_signal = EventEpochExtractor.extract_epochs_for_signals(
                time_s=time_s,
                event_times_s=event_times_s,
                preprocessed_signals={signal_key: completed_session.preprocessed_signals[signal_key]},
                n_pre=n_pre,
                n_post=n_post,
                extract_epoched_data_callable=extract_epoched_data,
            )

            epochs_z = epochs_by_signal[signal_key]

            dt_s = float(np.median(np.diff(time_s)))
            peri_t = (np.arange(-n_pre, n_post, dtype=float) * dt_s)

            session_dataframe = build_session_peri_event_long_dataframe(
                animal=completed_session.animal,
                session_name=completed_session.session_name,
                peri_t=peri_t,
                epochs_z=epochs_z,
            )
            session_dataframe["event_type"] = event_table_key

            session_level_dataframes.append(session_dataframe)

        if not session_level_dataframes:
            raise ValueError(
                "No peri-event session outputs available for group analysis "
                f"for event_table_key={event_table_key!r}"
            )

        return pd.concat(session_level_dataframes, ignore_index=True)

    def build_session(self, animal: str, session_name: str) -> PhotometrySession:
        """
        Create a session object for one animal/session pair.

        Parameters
        ----------
        animal : str
            Animal identifier.
        session_name : str
            Session identifier.

        Returns
        -------
        PhotometrySession
            Configured session object.
        """
        return PhotometrySession(
            animal=animal,
            session_name=session_name,
            base_directory=self.base_directory,
            output_directory=animal_output_dirs(animal, session_name),
            run_preprocessing=self.run_preprocessing,
            run_event_sorting=self.run_event_sorting,
            run_signal_processing=self.run_signal_processing,
            run_plotting=self.run_plotting,
        )

    def run(self) -> list[PhotometrySession]:
        """
        Run the full pipeline across all requested animals and sessions.

        Returns
        -------
        list of PhotometrySession
            Completed session objects.
        """
        self.results = []

        for animal in ['Rat_4879']:#self.get_animal_names():
            print(animal)
            for session_name in self.session_names:
                print(session_name)
                session_processor = self.build_session(animal, session_name)
                session_processor.run()
                self.results.append(session_processor)
        return self.results

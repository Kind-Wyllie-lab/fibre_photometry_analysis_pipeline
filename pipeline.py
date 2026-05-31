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
from matplotlib import pyplot as plt

import params
from bootstrap_analysis import SessionBootstrapAUCAnalyzer
from epoching import EventEpochExtractor
from params import animal_output_dirs
from session import PhotometrySession, session_has_raw_data
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

        xls_animals_matadata_path = self.base_directory / 'animals_metadata.xlsx'
        xls_animals_matadata = pd.read_excel(xls_animals_matadata_path).dropna()

        return xls_animals_matadata['animal_name'].values.tolist()

    def build_group_peri_event_dataframe(
            self,
            event_table_key: str,
            signal_key: str = "zscore",
            channel: str = "CH1",
    ) -> pd.DataFrame:
        """
        Build a concatenated long-form peri-event dataframe across completed sessions for a given
        event type, signal, and photometry channel.

        Parameters
        ----------
        event_table_key : str
            Event key stored in `completed_session.event_tables`, e.g.:
            - "raw_LED_events"
            - "cs_onsets"
            - "cs_offsets"
            - "freezing_onsets"
            - "freezing_offsets"
            (depending on how you store your refactored TTL dict)
        signal_key : {"dff","zscore"}, default="zscore"
            Which signal from the chosen channel to epoch.
        channel : {"CH1","CH2","both"}, default="CH1"
            Which photometry channel to use. If "both", concatenates CH1 and CH2
            when available and adds a `photometry_channel` column.

        Returns
        -------
        pandas.DataFrame
            Long-format peri-event dataframe with at least columns:
            animal, session_name, event_index, time_s, zscore
            plus `event_type` and `photometry_channel`.

        Raises
        ------
        ValueError
            If no sessions contribute data.
        """
        if channel not in {"CH1", "CH2", "both"}:
            raise ValueError("channel must be one of {'CH1','CH2','both'}")
        if signal_key not in {"dff", "zscore"}:
            raise ValueError("signal_key must be 'dff' or 'zscore'")

        requested_channels = ["CH1", "CH2"] if channel == "both" else [channel]

        session_level_dataframes: list[pd.DataFrame] = []
        if 'freezing' in event_table_key:
            completed_sessions = [i for i in self.results.copy() if i.animal not in ['Rat_5091', 'Rat_5181']] # These rat freezes he whole session, will fail if we keep it
        else:
            completed_sessions = self.results.copy()

        for completed_session in completed_sessions:
            if completed_session.df_clean is None:
                continue
            if getattr(completed_session, "event_tables", None) is None:
                continue
            if getattr(completed_session, "preprocessed_signals", None) is None:
                continue

            time_s = completed_session.df_clean["TimeStamp"].to_numpy(dtype=float) / 1000.0
            dt_s = float(np.median(np.diff(time_s)))

            # --- event times (seconds) ---
            # Your process_ttl_events returns nested dicts; support both flat and nested keys.
            event_times_s = None
            if isinstance(completed_session.event_tables, dict):
                # try direct key first
                if event_table_key in completed_session.event_tables:
                    event_times_s = completed_session.event_tables[event_table_key]
                else:
                    # try nested keys: "LED_events"/"freezing_events"
                    for parent_key in ("LED_events", "freezing_events"):
                        if parent_key in completed_session.event_tables and event_table_key in \
                                completed_session.event_tables[parent_key]:
                            event_times_s = completed_session.event_tables[parent_key][event_table_key]
                            break

            if event_times_s is None or (isinstance(event_times_s, np.ndarray) and event_times_s.size == 0):
                continue

            event_times_s = np.asarray(event_times_s, dtype=float)
            if event_times_s.size == 0:
                continue

            n_pre = int(params.time_pre_event_s * params.sample_rate_hz)
            n_post = int(params.time_post_event_s * params.sample_rate_hz)
            peri_t = (np.arange(-n_pre, n_post, dtype=float) * dt_s)

            for ch in requested_channels:
                if ch not in list(completed_session.preprocessed_signals.keys()):
                    continue

                signals_for_channel = completed_session.preprocessed_signals[ch]
                if signal_key not in list(signals_for_channel.keys()):
                    continue

                print('#################################')
                print(completed_session.animal, completed_session.session_name, event_table_key, ch)

                epochs = EventEpochExtractor.extract_epochs_for_signals(
                    time_s=time_s,
                    event_times_s=event_times_s,
                    preprocessed_signals={'dff': signals_for_channel['dff'], 'zscore': signals_for_channel['zscore']},
                    n_pre=n_pre,
                    n_post=n_post,
                    extract_epoched_data_callable=extract_epoched_data,
                )[signal_key]

                # Build long dataframe (group_analysis version expects epochs_z argument name)
                session_dataframe = build_session_peri_event_long_dataframe(
                    animal=completed_session.animal,
                    session_name=completed_session.session_name,
                    peri_t=peri_t,
                    epochs_z=epochs,
                )
                session_dataframe["event_type"] = event_table_key
                session_dataframe["photometry_channel"] = ch
                session_dataframe["signal_key"] = signal_key

                session_level_dataframes.append(session_dataframe)

        if not session_level_dataframes:
            raise ValueError(
                "No peri-event session outputs available for group analysis "
                f"for event_table_key={event_table_key!r} and channel={channel!r}"
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
        animals_to_include = self.get_animal_names()

        animals_to_include = [i for i in animals_to_include if i not in ['Rat_4987', 'Rat_4988', 'Rat_4990', 'Rat_391', 'Rat_5091', 'Rat_5092', 'Rat_5093', 'Rat_5094' ]] #, 'Rat_5162'
        #animals_to_include = [i for i in animals_to_include if i not in ['Rat_391', 'Rat_390','Rat_389' ]] #, 'Rat_5162'

        for animal in animals_to_include:
            print(animal)
            for session_name in self.session_names:
                print(session_name)

                # Check existence BEFORE build_session() (prevents output dir creation crash)
                if not session_has_raw_data(self.base_directory, animal, session_name):
                    print(f"[PIPELINE] Skipping {animal} | {session_name} (no folder/raw data)")
                    continue

                session_processor = self.build_session(animal, session_name)

                print('Running', animal, session_name)
                session_processor.run()
                self.results.append(session_processor)


        return self.results

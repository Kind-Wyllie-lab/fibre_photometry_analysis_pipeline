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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import params
from preprocessing import extract_session_raw_data
from event_sorting import process_events
from signal_processing import filter_first_event, preprocess_photometry_dff_and_zscore, \
    extract_epoched_data, select_events_from_params
from plotting import PhotometryPlotter


@dataclass
class PhotometrySession:
    """
    Container and execution interface for a single fiber photometry recording session.

    Parameters
    ----------
    animal : str
        Animal identifier, typically a folder name such as ``"Rat01"``.
    session_name : str
        Session identifier, e.g. ``"cond"`` or ``"Recall"``.
    base_directory : str or Path
        Root directory containing all animal folders.
    output_directory : str or Path
        Directory where derived outputs for this session are saved.
    run_preprocessing : bool, default=True
        Whether to execute raw fluorescence preprocessing.
    run_event_sorting : bool, default=True
        Whether to execute event extraction/sorting.
    run_signal_processing : bool, default=True
        Whether to compute full-trace and peri-event photometry signals.
    run_plotting : bool, default=True
        Whether to generate all plotting outputs.

    Attributes
    ----------
    animal_path : Path
        Absolute path to the animal folder.
    session_path : Path
        Absolute path to the session folder.
    raw_data_path : Path
        Absolute path to the raw acquisition directory inside the session folder.
    skip_first_event : bool
        Whether the first sorted event should be ignored for epoch extraction.
    df_clean : pandas.DataFrame or None
        Cleaned fluorescence dataframe.
    first_events : pandas.DataFrame or None
        Sorted first-event dataframe returned by the event sorting stage.
    epochs_dff : numpy.ndarray or None
        Peri-event ΔF/F epochs.
    epochs_z : numpy.ndarray or None
        Peri-event z-score epochs.
    peri_t : numpy.ndarray or None
        Peri-event time vector.
    dff_baseline : numpy.ndarray or None
        Full-trace ΔF/F normalized to baseline.
    dff_fitted : numpy.ndarray or None
        Full-trace ΔF/F normalized to fitted reference.
    zscore : numpy.ndarray or None
        Full-trace z-score.
    events_to_use : pandas.DataFrame or None
        Event dataframe ultimately used for signal extraction.
    event_times_s : numpy.ndarray or None
        Event timestamps in seconds after filtering.
    t_zero_s : float or None
        Timestamp in seconds of the very first event before skip filtering.
    """

    animal: str
    session_name: str
    base_directory: Path | str
    output_directory: Path | str
    run_preprocessing: bool = True
    run_event_sorting: bool = True
    run_signal_processing: bool = True
    run_plotting: bool = True

    animal_path: Path = field(init=False)
    session_path: Path = field(init=False)
    raw_data_path: Path = field(init=False)
    skip_first_event: bool = field(init=False)

    df_clean: Optional[pd.DataFrame] = field(default=None, init=False)
    first_events: Optional[pd.DataFrame] = field(default=None, init=False)

    epochs_dff: Optional[object] = field(default=None, init=False)
    epochs_z: Optional[object] = field(default=None, init=False)
    peri_t: Optional[object] = field(default=None, init=False)
    dff_baseline: Optional[object] = field(default=None, init=False)
    dff_fitted: Optional[object] = field(default=None, init=False)
    zscore: Optional[object] = field(default=None, init=False)
    events_to_use: Optional[pd.DataFrame] = field(default=None, init=False)

    event_times_s: Optional[object] = field(default=None, init=False)
    t_zero_s: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """
        Initialize derived directory attributes and session-specific behavior.

        Raises
        ------
        FileNotFoundError
            If the animal or session directory does not exist.
        ValueError
            If the session raw data directory cannot be identified.
        """
        self.base_directory = Path(self.base_directory)
        self.output_directory = Path(self.output_directory)

        self.animal_path = self.base_directory / self.animal
        self.session_path = self.animal_path / self.session_name

        if not self.animal_path.exists():
            raise FileNotFoundError(f"Animal directory not found: {self.animal_path}")
        if not self.session_path.exists():
            raise FileNotFoundError(f"Session directory not found: {self.session_path}")

        candidate_raw_dirs = sorted(
            [p for p in self.session_path.iterdir() if p.is_dir() and p.name.startswith(self.animal)]
        )
        if not candidate_raw_dirs:
            raise ValueError(
                f"No raw data directory starting with '{self.animal}' found in {self.session_path}"
            )
        self.raw_data_path = candidate_raw_dirs[0]

        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.skip_first_event = self._infer_skip_first_event()

    def _infer_skip_first_event(self) -> bool:
        """
        Infer whether the first cluster event should be excluded for this session.

        Returns
        -------
        bool
            ``True`` for conditioning sessions where the first event is spurious,
            ``False`` otherwise.
        """
        session_name_lower = self.session_name.lower()
        if session_name_lower == "cond":
            return True
        if session_name_lower == "recall":
            return False
        return False

    def raw_data_preprocessing(self) -> pd.DataFrame:
        """
        Execute preprocessing for the current session.

        Returns
        -------
        pandas.DataFrame
            Cleaned fluorescence dataframe.
        """
        self.df_clean = extract_session_raw_data(str(self.raw_data_path), self.output_directory)
        return self.df_clean

    def ttl_events_processing(self) -> pd.DataFrame:
        """
        Execute event sorting for the current session.

        Returns
        -------
        pandas.DataFrame
            First-event dataframe.

        Raises
        ------
        ValueError
            If preprocessing data are unavailable.
        """
        if self.df_clean is None:
            raise ValueError("Preprocessing must be run before event sorting.")
        _, self.first_events = process_events(self.df_clean, self.output_directory)
        return self.first_events

    def fluorescence_processing(self) -> None:
        """
        Execute signal processing and peri-event extraction for the current session.

        Raises
        ------
        ValueError
            If cleaned fluorescence or event data are unavailable.
        """
        if self.df_clean is None:
            raise ValueError("Preprocessing must be run before signal processing.")
        if self.first_events is None:
            raise ValueError("Event sorting must be run before signal processing.")

        self.preprocessed_signals = preprocess_photometry_dff_and_zscore(
            df_clean=self.df_clean,
            calcium_channel='CH1-470',
            reference_channel='CH1-410',
            baseline_interval_samples=None,
            control_source="410",  # "410" or "baseline"
            apply_baseline_correction=False,  # True or False
            enable_smoothing=False,
            smoothing_window_length=11,
            smoothing_polyorder=3,
            background_calcium=None,
            background_reference=None,
            baseline_smoothness_penalty=1e6,
            baseline_asymmetry_penalty=0.01,
        )

    def run_plotting_stage(self) -> None:
        """
        Execute all plotting routines for the current session.

        Raises
        ------
        ValueError
            If signal processing outputs are unavailable.
        """
        if self.preprocessed_signals is None:
            raise ValueError("Signal processing must be run before plotting.")

        session_stem = f"{self.animal}_{self.session_name}"

        plotter = PhotometryPlotter(
            df_clean=self.df_clean,
            dff_fitted=self.preprocessed_signals['dff'],
            zscore=self.preprocessed_signals['zscore'],
            epochs_dff=self.epochs_dff,
            epochs_z=self.epochs_z,
            peri_t=self.peri_t,
            stem=session_stem,
            event_times_s=self.event_times_s,
            t_zero_s=self.t_zero_s,
            filtered_events=self.events_to_use,
            figure_output_directory=self.output_directory,
            peri_event_plot_mode="trials",
        )
        plotter.run_all()

    def run(self) -> "PhotometrySession":
        """
        Run the enabled processing stages for the current session.

        Returns
        -------
        PhotometrySession
            The current session instance after execution.
        """
        print("=" * 80)
        print(f"RUNNING SESSION | animal={self.animal} | session={self.session_name}")
        print("=" * 80)

        if self.run_preprocessing:
            self.raw_data_preprocessing()

        if self.run_event_sorting:
            self.ttl_events_processing()

        if self.run_signal_processing:
            self.fluorescence_processing()

        self.t_zero_s = self.first_events.iloc[0]["TimeStamp"] / 1000.0

        time_s = self.df_clean["TimeStamp"].values / 1000.0

        n_pre = int(params.time_pre_event_s * params.sample_rate_hz)
        n_post = int(params.time_post_event_s * params.sample_rate_hz)
        self.peri_t = np.arange(-n_pre, n_post)/params.sample_rate_hz
        events_selected = select_events_from_params(self.first_events)
        events_to_use = filter_first_event(events_selected, self.skip_first_event)

        if events_to_use.empty:
            raise ValueError("No events remaining after selection/filtering — check params")

        event_times_s = events_to_use["TimeStamp"].values / 1000.0

        self.epochs_dff = extract_epoched_data(self.preprocessed_signals['dff'], time_s, event_times_s, n_pre, n_post)
        self.epochs_z = extract_epoched_data(self.preprocessed_signals['zscore'], time_s, event_times_s, n_pre, n_post)

        if self.run_plotting:
            self.run_plotting_stage()

        return self


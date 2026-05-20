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
from bootstrap_analysis import SessionBootstrapAUCAnalyzer
from epoching import EventEpochExtractor
from preprocessing import extract_session_raw_data
from event_sorting import process_ttl_events
from signal_processing import preprocess_photometry_dff_and_zscore, \
    extract_epoched_data, select_events_from_params, preprocess_photometry_multichannel_dff_and_zscore
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

        Notes
        -----
        This method no longer raises if the session is missing. Instead it allows
        the pipeline to skip animals that do not have a given session.
        """
        # Defer strict validation to `run()` so the pipeline can skip missing sessions
        self.base_directory = Path(self.base_directory)
        self.output_directory = Path(self.output_directory)

        self.animal_path = self.base_directory / self.animal
        self.session_path = self.animal_path / self.session_name

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

        time_s = self.df_clean["TimeStamp"].to_numpy(dtype=float) / 1000.0

        if self.run_preprocessing == 2:
            channels = ['CH1', 'CH2']
        else:
            channels = ['CH1']

        for channel in channels:
            self.preprocessed_signals_channel = self.preprocessed_signals[channel]

            if self.animal not in ['Rat_5091', 'Rat_5181']:
                timings = ['cs_onsets', 'cs_offsets', 'shock']
            else:
                timings = ['cs_onsets', 'cs_offsets', 'freezing_onsets', 'freezing_offsets', 'shock']

            for timing in timings:

                if timing in self.event_tables.keys():
                    self.event_times_s = self.event_tables[timing]
                    print('Epoching ', timing, self.event_times_s, self.event_times_s)

                    if self.event_times_s is not None:
                        n_pre = int(params.time_pre_event_s * params.sample_rate_hz)
                        n_post = int(params.time_post_event_s * params.sample_rate_hz)

                        # Epoch just the requested signal (zscore)
                        epochs_by_signal = EventEpochExtractor.extract_epochs_for_signals(
                            time_s=time_s,
                            event_times_s=self.event_times_s,
                            preprocessed_signals={'dff': self.preprocessed_signals_channel['dff'], 'zscore': self.preprocessed_signals_channel['zscore']},
                            n_pre=n_pre,
                            n_post=n_post,
                            extract_epoched_data_callable=extract_epoched_data,
                        )
                        dt_s = float(np.median(np.diff(time_s)))
                        self.peri_t = (np.arange(-n_pre, n_post, dtype=float) * dt_s)

                        self.epochs_z = epochs_by_signal['zscore']
                        self.epochs_dff = epochs_by_signal['dff']

                        plotter = PhotometryPlotter(
                            channel_name=channel,
                            df_clean=self.df_clean,
                            event_name=timing,
                            dff_fitted=self.preprocessed_signals_channel['dff'],
                            zscore=self.preprocessed_signals_channel['zscore'],
                            epochs_dff=self.epochs_dff,
                            epochs_z=self.epochs_z,
                            peri_t=self.peri_t,
                            stem=session_stem,
                            event_times_s=self.event_times_s,
                            t_zero_s=self.t_zero_s,
                            filtered_events=self.events_to_use,
                            figure_output_directory=self.output_directory,
                            peri_event_plot_mode="trials"
                        )


                        analyzer = SessionBootstrapAUCAnalyzer(
                            session=self,
                            channel=channel,
                            event_times_s=self.event_times_s,
                            event_name=timing,
                            baseline_window_s=2.0,
                            auc_window_s=(0.0, 2.0),
                            n_mocks=5000,
                            random_seed=0,
                        )

                        fig_z = analyzer.plot_null_with_all_event_axvlines("zscore", bins=60, tail_mode="right", alpha_level=0.05)
                        fig_d = analyzer.plot_null_with_all_event_axvlines("dff", bins=60, tail_mode="right", alpha_level=0.05)

                        plotter.run_all()

    def session_exists(self) -> bool:
        """
        Check whether this animal has this session folder and a valid raw data directory.

        Returns
        -------
        bool
            True if session folder and raw acquisition directory exist, otherwise False.
        """
        animal_path = Path(self.base_directory) / self.animal
        session_path = animal_path / self.session_name
        if not animal_path.exists() or not session_path.exists():
            return False

        candidate_raw_dirs = [
            p for p in session_path.iterdir()
            if p.is_dir() and p.name.startswith(self.animal)
        ]
        return len(candidate_raw_dirs) > 0

    def _try_resolve_paths_or_skip(self) -> bool:
        """
        Resolve session paths if possible; otherwise mark as skipped.

        Returns
        -------
        bool
            True if paths resolved and processing can proceed, False if session should be skipped.
        """
        self.base_directory = Path(self.base_directory)
        self.output_directory = Path(self.output_directory)

        self.animal_path = self.base_directory / self.animal
        self.session_path = self.animal_path / self.session_name

        if not self.animal_path.exists():
            print(f"[SKIP] Missing animal folder: {self.animal_path}")
            return False

        if not self.session_path.exists():
            print(f"[SKIP] Missing session folder: {self.session_path}")
            return False

        candidate_raw_dirs = sorted(
            [p for p in self.session_path.iterdir() if p.is_dir() and p.name.startswith(self.animal)]
        )
        if not candidate_raw_dirs:
            print(f"[SKIP] No raw acquisition dir found in: {self.session_path}")
            return False

        self.raw_data_path = candidate_raw_dirs[0]
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.skip_first_event = self._infer_skip_first_event()
        return True

    def run(self) -> "PhotometrySession":
        """
        Run the enabled processing stages for the current session.

        Returns
        -------
        PhotometrySession
            The current session instance after execution.
        """
        can_run = self._try_resolve_paths_or_skip()
        if not can_run:
            return self

        self.df_clean = extract_session_raw_data(str(self.raw_data_path), self.output_directory)

        if 'CH2' in self.df_clean.columns:
            self.n_channels = 2
        else:
            self.n_channels = 1

        self.event_tables = process_ttl_events(self.df_clean, self.output_directory)

        self.preprocessed_signals = preprocess_photometry_multichannel_dff_and_zscore(
            df_clean=self.df_clean,
            calcium_wavelength_nm=470,
            reference_wavelength_nm=410,
            baseline_interval_samples=None,
            control_source="410",
            apply_baseline_correction=True,
            enable_smoothing=True,
            smoothing_window_length=15,
            smoothing_polyorder=3,
            background_calcium=None,
            background_reference=None,
            baseline_smoothness_penalty=1e6,
            baseline_asymmetry_penalty=0.01,
        )

        if params.run_single_animal_plots:
            self.run_plotting_stage()
        return self

def session_has_raw_data(base_directory: str | Path, animal: str, session_name: str) -> bool:
    """
    Check whether an animal has a session folder containing a raw acquisition directory.

    Parameters
    ----------
    base_directory : str or Path
        Root directory containing animal folders.
    animal : str
        Animal identifier.
    session_name : str
        Session identifier.

    Returns
    -------
    bool
        True if session exists and contains a raw dir starting with the animal name.
    """
    base_directory = Path(base_directory)
    session_path = base_directory / animal / session_name
    if not session_path.exists():
        return False
    raw_dirs = [p for p in session_path.iterdir() if p.is_dir() and p.name.startswith(animal)]
    return len(raw_dirs) > 0
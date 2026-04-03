from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from params import (
    figure_size_peri,
    figure_dpi,
    figure_format,
    save_figures,
    preview_figures,
    color_dff,
    color_zscore,
    color_event_onset,
    lw_peri_mean,
)

def build_session_peri_event_long_dataframe(
    animal: str,
    session_name: str,
    peri_t: np.ndarray,
    epochs_dff: np.ndarray,
    epochs_z: np.ndarray,
) -> pd.DataFrame:
    """
    Build a long-form peri-event dataframe for one animal/session pair.

    Parameters
    ----------
    animal : str
        Animal identifier.
    session_name : str
        Session identifier.
    peri_t : numpy.ndarray
        Peri-event time vector in seconds.
    epochs_dff : numpy.ndarray
        Peri-event ΔF/F array with shape ``(n_events, n_timepoints)``.
    epochs_z : numpy.ndarray
        Peri-event z-score array with shape ``(n_events, n_timepoints)``.

    Returns
    -------
    pandas.DataFrame
        Long-format dataframe with one row per animal, event, signal type,
        and peri-event timepoint.
    """
    if epochs_dff.ndim != 2 or epochs_z.ndim != 2:
        raise ValueError("epochs_dff and epochs_z must be 2D arrays")
    if epochs_dff.shape != epochs_z.shape:
        raise ValueError(
            f"epochs_dff shape {epochs_dff.shape} != epochs_z shape {epochs_z.shape}"
        )
    if epochs_dff.shape[1] != len(peri_t):
        raise ValueError(
            f"epoch length {epochs_dff.shape[1]} != peri_t length {len(peri_t)}"
        )

    n_events, _ = epochs_dff.shape
    event_numbers = np.arange(1, n_events + 1, dtype=int)

    dff_long = (
        pd.DataFrame(epochs_dff, index=event_numbers, columns=peri_t)
        .rename_axis(index="event_index")
        .reset_index()
        .melt(id_vars="event_index", var_name="time_s", value_name="signal_value")
        .assign(
            animal=animal,
            session_name=session_name,
            signal_kind="ΔF/F",
        )
    )

    z_long = (
        pd.DataFrame(epochs_z, index=event_numbers, columns=peri_t)
        .rename_axis(index="event_index")
        .reset_index()
        .melt(id_vars="event_index", var_name="time_s", value_name="signal_value")
        .assign(
            animal=animal,
            session_name=session_name,
            signal_kind="Z-score",
        )
    )

    peri_event_long_dataframe = pd.concat([dff_long, z_long], ignore_index=True)
    peri_event_long_dataframe["time_s"] = peri_event_long_dataframe["time_s"].astype(float)
    peri_event_long_dataframe["event_index"] = peri_event_long_dataframe["event_index"].astype(int)

    return peri_event_long_dataframe


@dataclass
class PhotometryGroupAnalyzer:
    """
    Group-level peri-event aggregation and plotting across animals.

    Parameters
    ----------
    session_level_peri_event_dataframe : pandas.DataFrame
        Long-format peri-event dataframe across animals and sessions.
    output_directory : str or Path
        Directory where group-level figures and tables are saved.
    """

    session_level_peri_event_dataframe: pd.DataFrame
    output_directory: str | Path

    def __post_init__(self) -> None:
        """
        Validate dataframe schema and create output directory.
        """
        self.output_directory = Path(self.output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        required_columns = {
            "animal",
            "session_name",
            "event_index",
            "time_s",
            "signal_kind",
            "signal_value",
        }
        missing_columns = required_columns.difference(self.session_level_peri_event_dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"session_level_peri_event_dataframe missing columns: {sorted(missing_columns)}"
            )

    def compute_animal_averaged_all_events(self) -> pd.DataFrame:
        """
        Compute all-event peri-event means within animal.

        Returns
        -------
        pandas.DataFrame
            Dataframe with one row per animal, session, signal type, and timepoint.
        """
        animal_averaged_dataframe = (
            self.session_level_peri_event_dataframe
            .groupby(["animal", "session_name", "signal_kind", "time_s"], as_index=False)["signal_value"]
            .mean()
            .rename(columns={"signal_value": "animal_mean_signal"})
        )
        return animal_averaged_dataframe

    def compute_group_summary_all_events(self) -> pd.DataFrame:
        """
        Compute group mean and SEM across animal-level all-event averages.

        Returns
        -------
        pandas.DataFrame
            Group summary dataframe with mean, standard deviation, SEM, and
            number of animals contributing at each timepoint.
        """
        animal_averaged_dataframe = self.compute_animal_averaged_all_events()

        group_summary_dataframe = (
            animal_averaged_dataframe
            .groupby(["session_name", "signal_kind", "time_s"], as_index=False)
            .agg(
                group_mean_signal=("animal_mean_signal", "mean"),
                group_standard_deviation=("animal_mean_signal", "std"),
                n_animals=("animal_mean_signal", "count"),
            )
        )
        group_summary_dataframe["group_sem_signal"] = (
            group_summary_dataframe["group_standard_deviation"] / np.sqrt(group_summary_dataframe["n_animals"])
        )

        return group_summary_dataframe

    def compute_animal_averaged_single_event(self, event_index: int) -> pd.DataFrame:
        """
        Extract a single aligned event across animals.

        Parameters
        ----------
        event_index : int
            Event number to analyze, indexed from 1.

        Returns
        -------
        pandas.DataFrame
            One row per animal, session, signal type, and timepoint for the
            requested event.

        Raises
        ------
        ValueError
            If no animal contains the requested event.
        """
        single_event_dataframe = self.session_level_peri_event_dataframe.loc[
            self.session_level_peri_event_dataframe["event_index"] == event_index
        ].copy()

        if single_event_dataframe.empty:
            raise ValueError(f"No data found for event_index={event_index}")

        animal_event_dataframe = (
            single_event_dataframe
            .groupby(["animal", "session_name", "signal_kind", "time_s"], as_index=False)["signal_value"]
            .mean()
            .rename(columns={"signal_value": "animal_mean_signal"})
        )
        return animal_event_dataframe

    def compute_group_summary_single_event(self, event_index: int) -> pd.DataFrame:
        """
        Compute group mean and SEM across animals for one aligned event index.

        Parameters
        ----------
        event_index : int
            Event number to analyze, indexed from 1.

        Returns
        -------
        pandas.DataFrame
            Group summary dataframe for the requested event.
        """
        animal_event_dataframe = self.compute_animal_averaged_single_event(event_index)

        group_summary_dataframe = (
            animal_event_dataframe
            .groupby(["session_name", "signal_kind", "time_s"], as_index=False)
            .agg(
                group_mean_signal=("animal_mean_signal", "mean"),
                group_standard_deviation=("animal_mean_signal", "std"),
                n_animals=("animal_mean_signal", "count"),
            )
        )
        group_summary_dataframe["group_sem_signal"] = (
            group_summary_dataframe["group_standard_deviation"] / np.sqrt(group_summary_dataframe["n_animals"])
        )
        group_summary_dataframe["event_index"] = event_index

        return group_summary_dataframe

    def plot_group_average_all_events(self, session_name: Optional[str] = None) -> None:
        """
        Plot group mean peri-event traces across all events, with SEM across animals.

        Parameters
        ----------
        session_name : str, optional
            If provided, restrict plotting to one session.
        """
        group_summary_dataframe = self.compute_group_summary_all_events()

        if session_name is not None:
            group_summary_dataframe = group_summary_dataframe.loc[
                group_summary_dataframe["session_name"] == session_name
            ].copy()

        if group_summary_dataframe.empty:
            raise ValueError("No group data available for all-event plotting")

        figure, axes = plt.subplots(2, 1, figsize=figure_size_peri, sharex=True)

        signal_axis_pairs = [
            ("ΔF/F", axes[0], color_dff),
            ("Z-score", axes[1], color_zscore),
        ]

        for signal_kind, axis, signal_color in signal_axis_pairs:
            signal_dataframe = group_summary_dataframe.loc[
                group_summary_dataframe["signal_kind"] == signal_kind
            ].sort_values("time_s")

            sns.lineplot(
                data=signal_dataframe,
                x="time_s",
                y="group_mean_signal",
                errorbar=None,
                color=signal_color,
                linewidth=lw_peri_mean,
                ax=axis,
            )
            axis.fill_between(
                signal_dataframe["time_s"].to_numpy(),
                (signal_dataframe["group_mean_signal"] - signal_dataframe["group_sem_signal"]).to_numpy(),
                (signal_dataframe["group_mean_signal"] + signal_dataframe["group_sem_signal"]).to_numpy(),
                color=signal_color,
                alpha=0.25,
            )
            axis.axvline(0, color=color_event_onset, linestyle="--", linewidth=0.8)
            axis.set_ylabel(signal_kind)
            axis.set_title(
                f"Group peri-event average — {signal_kind} "
                f"(all events, animal-level SEM)"
            )
            axis.grid(alpha=0.3)

        axes[-1].set_xlabel("Time from event (s)")
        figure.tight_layout()

        output_stem = "group_peri_event_all_events"
        if session_name is not None:
            output_stem += f"_{session_name}"

        self._finalize_figure(figure, output_stem)

    def plot_group_average_single_event(
        self,
        event_index: int,
        session_name: Optional[str] = None,
    ) -> None:
        """
        Plot group peri-event traces for one event index across animals.

        Parameters
        ----------
        event_index : int
            Event number to analyze, indexed from 1.
        session_name : str, optional
            If provided, restrict plotting to one session.
        """
        group_summary_dataframe = self.compute_group_summary_single_event(event_index)

        if session_name is not None:
            group_summary_dataframe = group_summary_dataframe.loc[
                group_summary_dataframe["session_name"] == session_name
            ].copy()

        if group_summary_dataframe.empty:
            raise ValueError(
                f"No group data available for event_index={event_index}"
            )

        figure, axes = plt.subplots(2, 1, figsize=figure_size_peri, sharex=True)

        signal_axis_pairs = [
            ("ΔF/F", axes[0], color_dff),
            ("Z-score", axes[1], color_zscore),
        ]

        for signal_kind, axis, signal_color in signal_axis_pairs:
            signal_dataframe = group_summary_dataframe.loc[
                group_summary_dataframe["signal_kind"] == signal_kind
            ].sort_values("time_s")

            sns.lineplot(
                data=signal_dataframe,
                x="time_s",
                y="group_mean_signal",
                errorbar=None,
                color=signal_color,
                linewidth=lw_peri_mean,
                ax=axis,
            )
            axis.fill_between(
                signal_dataframe["time_s"].to_numpy(),
                (signal_dataframe["group_mean_signal"] - signal_dataframe["group_sem_signal"]).to_numpy(),
                (signal_dataframe["group_mean_signal"] + signal_dataframe["group_sem_signal"]).to_numpy(),
                color=signal_color,
                alpha=0.25,
            )
            axis.axvline(0, color=color_event_onset, linestyle="--", linewidth=0.8)
            axis.set_ylabel(signal_kind)
            axis.set_title(
                f"Group peri-event average — {signal_kind} "
                f"(event {event_index}, animal-level SEM)"
            )
            axis.grid(alpha=0.3)

        axes[-1].set_xlabel("Time from event (s)")
        figure.tight_layout()

        output_stem = f"group_peri_event_event_{event_index}"
        if session_name is not None:
            output_stem += f"_{session_name}"

        self._finalize_figure(figure, output_stem)

    def plot_all_single_event_group_averages(self, session_name: Optional[str] = None) -> None:
        """
        Plot group averages for every event index available across animals.

        Parameters
        ----------
        session_name : str, optional
            If provided, restrict to one session.
        """
        dataframe = self.session_level_peri_event_dataframe
        if session_name is not None:
            dataframe = dataframe.loc[dataframe["session_name"] == session_name]

        event_indices = sorted(dataframe["event_index"].dropna().unique().astype(int))
        for event_index in event_indices:
            try:
                self.plot_group_average_single_event(
                    event_index=event_index,
                    session_name=session_name,
                )
            except ValueError as error:
                print(f"Skipping event {event_index}: {error}")

    def export_group_tables(self) -> None:
        """
        Export group summary tables for downstream statistics and figure reuse.
        """
        all_events_summary = self.compute_group_summary_all_events()
        all_events_summary.to_csv(
            self.output_directory / "group_summary_all_events.csv",
            index=False,
        )

    def _finalize_figure(self, figure: plt.Figure, filename_stem: str) -> None:
        """
        Save and/or preview a figure according to project configuration.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure to finalize.
        filename_stem : str
            Output file stem.
        """
        if save_figures:
            output_path = self.output_directory / f"{filename_stem}.{figure_format}"
            figure.savefig(output_path, dpi=figure_dpi, format=figure_format, bbox_inches="tight")
            print(f"Saved: {output_path}")

        if preview_figures:
            plt.show()
        else:
            plt.close(figure)

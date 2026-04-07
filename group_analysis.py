from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import params
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
    epochs_z: np.ndarray,
) -> pd.DataFrame:
    """
    Build a long-form peri-event z-score dataframe for one animal/session pair.

    Parameters
    ----------
    animal : str
        Animal identifier.
    session_name : str
        Session identifier.
    peri_t : numpy.ndarray
        Peri-event time vector in seconds.
    epochs_z : numpy.ndarray
        Peri-event z-score array with shape ``(n_events, n_timepoints)``.

    Returns
    -------
    pandas.DataFrame
        Long-format dataframe with one row per animal, event, and peri-event timepoint.
    """
    if epochs_z.ndim != 2:
        raise ValueError("epochs_z must be a 2D array")
    if epochs_z.shape[1] != len(peri_t):
        raise ValueError(
            f"epoch length {epochs_z.shape[1]} != peri_t length {len(peri_t)}"
        )

    n_events, _ = epochs_z.shape
    event_numbers = np.arange(1, n_events + 1, dtype=int)

    peri_event_long_dataframe = (
        pd.DataFrame(epochs_z, index=event_numbers, columns=peri_t)
        .rename_axis(index="event_index")
        .reset_index()
        .melt(id_vars="event_index", var_name="time_s", value_name="zscore")
        .assign(
            animal=animal,
            session_name=session_name,
        )
    )

    peri_event_long_dataframe["time_s"] = peri_event_long_dataframe["time_s"].astype(float)
    peri_event_long_dataframe["event_index"] = peri_event_long_dataframe["event_index"].astype(int)

    return peri_event_long_dataframe

@dataclass
class PhotometryGroupAnalyzer:
    """
    Group-level peri-event z-score aggregation and plotting across animals.

    Parameters
    ----------
    session_level_peri_event_dataframe : pandas.DataFrame
        Long-format peri-event z-score dataframe across animals and sessions.
    output_directory : str or Path
        Directory where group-level figures and tables are saved.
    metadata_file_path : str or Path or None, default=None
        Path to the animal metadata table. If ``None``, defaults to
        ``base_path / "animals_metadata.ods"``.
    animal_name_column : str, default="animal_name"
        Column name in the metadata table containing animal identifiers.
    group_column : str, default="group"
        Column name in the metadata table containing animal group labels.
    subplot_group_order : tuple of str, default=("wt", "het", "gcamp")
        Group order used for subplot layout.
    """

    session_level_peri_event_dataframe: pd.DataFrame
    output_directory: str | Path
    metadata_file_path: Optional[str | Path] = None
    animal_name_column: str = "animal_name"
    group_column: str = "group"
    subplot_group_order: tuple[str, ...] = ("wt", "het", "gcamp")

    def __post_init__(self) -> None:
        """
        Validate inputs, create the output directory, and merge animal metadata.
        """
        self.output_directory = Path(self.output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        if self.metadata_file_path is None:
            self.metadata_file_path = Path(params.base_path) / "animals_metadata.ods"
        else:
            self.metadata_file_path = Path(self.metadata_file_path)

        required_columns = {
            "animal",
            "session_name",
            "event_index",
            "time_s",
            "zscore",
        }
        missing_columns = required_columns.difference(self.session_level_peri_event_dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"session_level_peri_event_dataframe missing columns: {sorted(missing_columns)}"
            )

        self.metadata_dataframe = self._load_animal_metadata()
        self.session_level_peri_event_dataframe = self._attach_group_metadata(
            self.session_level_peri_event_dataframe
        )

    def _load_animal_metadata(self) -> pd.DataFrame:
        """
        Load animal metadata from the ODS file.

        Returns
        -------
        pandas.DataFrame
            Metadata table containing at least animal name and group columns.
        """
        if not self.metadata_file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file_path}")

        metadata_dataframe = pd.read_excel(
            self.metadata_file_path,
            engine="odf",
        )

        required_columns = {self.animal_name_column, self.group_column}
        missing_columns = required_columns.difference(metadata_dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"Metadata file missing required columns: {sorted(missing_columns)}"
            )

        metadata_dataframe = metadata_dataframe[[self.animal_name_column, self.group_column]].copy()
        metadata_dataframe = metadata_dataframe.rename(
            columns={
                self.animal_name_column: "animal",
                self.group_column: "group",
            }
        )
        metadata_dataframe["animal"] = metadata_dataframe["animal"].astype(str)
        metadata_dataframe["group"] = metadata_dataframe["group"].astype(str)

        return metadata_dataframe

    def _attach_group_metadata(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Merge animal group labels into the session-level peri-event dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Session-level peri-event dataframe.

        Returns
        -------
        pandas.DataFrame
            Input dataframe augmented with a ``group`` column.
        """
        merged_dataframe = dataframe.merge(
            self.metadata_dataframe,
            on="animal",
            how="left",
            validate="many_to_one",
        )

        if merged_dataframe["group"].isna().any():
            missing_animals = sorted(merged_dataframe.loc[merged_dataframe["group"].isna(), "animal"].unique())
            raise ValueError(
                f"Some animals are missing group metadata: {missing_animals}"
            )

        return merged_dataframe

    def compute_animal_averaged_all_events(self) -> pd.DataFrame:
        """
        Compute animal-level average z-score across all events.

        Returns
        -------
        pandas.DataFrame
            One row per animal, group, session, and timepoint.
        """
        animal_averaged_dataframe = (
            self.session_level_peri_event_dataframe
            .groupby(["animal", "group", "session_name", "time_s"], as_index=False)["zscore"]
            .mean()
            .rename(columns={"zscore": "animal_mean_zscore"})
        )
        return animal_averaged_dataframe

    def compute_group_summary_all_events(self) -> pd.DataFrame:
        """
        Compute group mean and SEM across animal-level all-event z-score averages.

        Returns
        -------
        pandas.DataFrame
            Group summary table for all-event peri-event z-score.
        """
        animal_averaged_dataframe = self.compute_animal_averaged_all_events()

        group_summary_dataframe = (
            animal_averaged_dataframe
            .groupby(["group", "session_name", "time_s"], as_index=False)
            .agg(
                group_mean_zscore=("animal_mean_zscore", "mean"),
                group_standard_deviation=("animal_mean_zscore", "std"),
                n_animals=("animal_mean_zscore", "count"),
            )
        )
        group_summary_dataframe["group_sem_zscore"] = (
            group_summary_dataframe["group_standard_deviation"] / np.sqrt(group_summary_dataframe["n_animals"])
        )

        return group_summary_dataframe

    def compute_animal_averaged_single_event(self, event_index: int) -> pd.DataFrame:
        """
        Extract and average one aligned event across animals.

        Parameters
        ----------
        event_index : int
            Event number to analyze, indexed from 1.

        Returns
        -------
        pandas.DataFrame
            One row per animal, group, session, and timepoint for the requested event.
        """
        single_event_dataframe = self.session_level_peri_event_dataframe.loc[
            self.session_level_peri_event_dataframe["event_index"] == event_index
        ].copy()

        if single_event_dataframe.empty:
            raise ValueError(f"No data found for event_index={event_index}")

        animal_event_dataframe = (
            single_event_dataframe
            .groupby(["animal", "group", "session_name", "time_s"], as_index=False)["zscore"]
            .mean()
            .rename(columns={"zscore": "animal_mean_zscore"})
        )
        return animal_event_dataframe

    def compute_group_summary_single_event(self, event_index: int) -> pd.DataFrame:
        """
        Compute group mean and SEM across animals for one event index.

        Parameters
        ----------
        event_index : int
            Event number to analyze, indexed from 1.

        Returns
        -------
        pandas.DataFrame
            Group summary table for the requested event.
        """
        animal_event_dataframe = self.compute_animal_averaged_single_event(event_index)

        group_summary_dataframe = (
            animal_event_dataframe
            .groupby(["group", "session_name", "time_s"], as_index=False)
            .agg(
                group_mean_zscore=("animal_mean_zscore", "mean"),
                group_standard_deviation=("animal_mean_zscore", "std"),
                n_animals=("animal_mean_zscore", "count"),
            )
        )
        group_summary_dataframe["group_sem_zscore"] = (
            group_summary_dataframe["group_standard_deviation"] / np.sqrt(group_summary_dataframe["n_animals"])
        )
        group_summary_dataframe["event_index"] = event_index

        return group_summary_dataframe

    def plot_group_average_all_events(self, session_name: Optional[str] = None) -> None:
        """
        Plot group mean peri-event z-score across all events with SEM across animals.

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

        figure, axes = plt.subplots(
            len(self.subplot_group_order),
            1,
            figsize=(figure_size_peri[0], figure_size_peri[1] * len(self.subplot_group_order)),
            sharex=True,
            sharey=True,
        )

        if len(self.subplot_group_order) == 1:
            axes = [axes]

        for axis, group_name in zip(axes, self.subplot_group_order):
            signal_dataframe = group_summary_dataframe.loc[
                group_summary_dataframe["group"] == group_name
            ].sort_values("time_s")

            if signal_dataframe.empty:
                axis.set_title(f"{group_name} (no data)")
                axis.set_ylabel("Z-score")
                axis.grid(alpha=0.3)
                continue

            sns.lineplot(
                data=signal_dataframe,
                x="time_s",
                y="group_mean_zscore",
                errorbar=None,
                color=color_zscore,
                linewidth=lw_peri_mean,
                ax=axis,
            )
            axis.fill_between(
                signal_dataframe["time_s"].to_numpy(),
                (signal_dataframe["group_mean_zscore"] - signal_dataframe["group_sem_zscore"]).to_numpy(),
                (signal_dataframe["group_mean_zscore"] + signal_dataframe["group_sem_zscore"]).to_numpy(),
                color=color_zscore,
                alpha=0.25,
            )
            axis.axvline(0, color=color_event_onset, linestyle="--", linewidth=0.8)
            axis.set_ylabel("Z-score")
            axis.set_title(
                f"Group peri-event average — {group_name} "
                f"(all events, animal-level SEM, n={signal_dataframe['n_animals'].max()})"
            )
            axis.grid(alpha=0.3)

        axes[-1].set_xlabel("Time from event (s)")
        figure.tight_layout()

        output_stem = "group_peri_event_all_events_zscore"
        if session_name is not None:
            output_stem += f"_{session_name}"

        self._finalize_figure(figure, output_stem)

    def plot_group_average_single_event(
        self,
        event_index: int,
        session_name: Optional[str] = None,
    ) -> None:
        """
        Plot group peri-event z-score for one event index across animals.

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

        figure, axes = plt.subplots(
            len(self.subplot_group_order),
            1,
            figsize=(figure_size_peri[0], figure_size_peri[1] * len(self.subplot_group_order)),
            sharex=True,
            sharey=True,
        )

        if len(self.subplot_group_order) == 1:
            axes = [axes]

        for axis, group_name in zip(axes, self.subplot_group_order):
            signal_dataframe = group_summary_dataframe.loc[
                group_summary_dataframe["group"] == group_name
            ].sort_values("time_s")

            if signal_dataframe.empty:
                axis.set_title(f"{group_name} (no data)")
                axis.set_ylabel("Z-score")
                axis.grid(alpha=0.3)
                continue

            sns.lineplot(
                data=signal_dataframe,
                x="time_s",
                y="group_mean_zscore",
                errorbar=None,
                color=color_zscore,
                linewidth=lw_peri_mean,
                ax=axis,
            )
            axis.fill_between(
                signal_dataframe["time_s"].to_numpy(),
                (signal_dataframe["group_mean_zscore"] - signal_dataframe["group_sem_zscore"]).to_numpy(),
                (signal_dataframe["group_mean_zscore"] + signal_dataframe["group_sem_zscore"]).to_numpy(),
                color=color_zscore,
                alpha=0.25,
            )
            axis.axvline(0, color=color_event_onset, linestyle="--", linewidth=0.8)
            axis.set_ylabel("Z-score")
            axis.set_title(
                f"Group peri-event average — {group_name} "
                f"(event {event_index}, animal-level SEM, n={signal_dataframe['n_animals'].max()})"
            )
            axis.grid(alpha=0.3)

        axes[-1].set_xlabel("Time from event (s)")
        figure.tight_layout()

        output_stem = f"group_peri_event_event_{event_index}_zscore"
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
            self.output_directory / "group_summary_all_events_zscore.csv",
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

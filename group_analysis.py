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
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import params
from params import (
    figure_size_peri,
    figure_dpi,
    figure_format,
    save_figures,
    preview_figures,
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

    def compute_animal_event_auc(
            self,
            auc_window_start_s: float = 0.0,
            auc_window_end_s: float = 5.0,
            max_event_index: int = 12,
            session_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute peri-event z-score area under the curve for each animal and event.

        The AUC is computed over the interval from `auc_window_start_s` to
        `auc_window_end_s` using trapezoidal integration.

        Parameters
        ----------
        auc_window_start_s : float, default=0.0
            Start time of the post-trigger integration window in seconds.
        auc_window_end_s : float, default=5.0
            End time of the post-trigger integration window in seconds.
        max_event_index : int, default=12
            Maximum event index to retain.
        session_name : str, optional
            If provided, restrict computation to one session.

        Returns
        -------
        pandas.DataFrame
            Dataframe with one row per animal and event, containing the computed AUC.

        Raises
        ------
        ValueError
            If no samples are available in the requested integration window.
        """
        auc_input_dataframe = self.session_level_peri_event_dataframe.copy()

        if session_name is not None:
            auc_input_dataframe = auc_input_dataframe.loc[
                auc_input_dataframe["session_name"] == session_name
                ].copy()

        auc_input_dataframe = auc_input_dataframe.loc[
            (auc_input_dataframe["time_s"] >= auc_window_start_s)
            & (auc_input_dataframe["time_s"] <= auc_window_end_s)
            & (auc_input_dataframe["event_index"] >= 1)
            & (auc_input_dataframe["event_index"] <= max_event_index)
            ].copy()

        if auc_input_dataframe.empty:
            raise ValueError(
                "No peri-event samples found in the requested AUC window and event range"
            )

        animal_event_auc_rows = []

        grouping_columns = ["animal", "group", "session_name", "event_index"]
        for grouping_values, event_dataframe in auc_input_dataframe.groupby(grouping_columns):
            event_dataframe = event_dataframe.sort_values("time_s")
            time_values_s = event_dataframe["time_s"].to_numpy(dtype=float)
            zscore_values = event_dataframe["zscore"].to_numpy(dtype=float)

            if len(time_values_s) < 2:
                continue

            auc_value = float(np.trapz(zscore_values, x=time_values_s))

            animal_event_auc_rows.append(
                {
                    "animal": grouping_values[0],
                    "group": grouping_values[1],
                    "session_name": grouping_values[2],
                    "event_index": grouping_values[3],
                    "auc_0_to_5s": auc_value,
                }
            )

        animal_event_auc_dataframe = pd.DataFrame(animal_event_auc_rows)
        if animal_event_auc_dataframe.empty:
            raise ValueError("No valid animal-level AUC values could be computed")

        return animal_event_auc_dataframe

    def compute_group_event_auc_summary(
            self,
            auc_window_start_s: float = 0.0,
            auc_window_end_s: float = 5.0,
            max_event_index: int = 12,
            session_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute group-level mean and SEM of animal peri-event AUC values.

        Parameters
        ----------
        auc_window_start_s : float, default=0.0
            Start time of the post-trigger integration window in seconds.
        auc_window_end_s : float, default=5.0
            End time of the post-trigger integration window in seconds.
        max_event_index : int, default=12
            Maximum event index to retain.
        session_name : str, optional
            If provided, restrict computation to one session.

        Returns
        -------
        pandas.DataFrame
            Group-level summary dataframe with mean AUC, standard deviation,
            SEM, and animal count for each event index.
        """
        animal_event_auc_dataframe = self.compute_animal_event_auc(
            auc_window_start_s=auc_window_start_s,
            auc_window_end_s=auc_window_end_s,
            max_event_index=max_event_index,
            session_name=session_name,
        )

        group_event_auc_summary = (
            animal_event_auc_dataframe
            .groupby(["group", "session_name", "event_index"], as_index=False)
            .agg(
                group_mean_auc=("auc_0_to_5s", "mean"),
                group_standard_deviation=("auc_0_to_5s", "std"),
                n_animals=("auc_0_to_5s", "count"),
            )
        )
        group_event_auc_summary["group_sem_auc"] = (
                group_event_auc_summary["group_standard_deviation"]
                / np.sqrt(group_event_auc_summary["n_animals"])
        )

        return group_event_auc_summary

    def plot_animal_auc_window_benchmark_3d(
            self,
            animal: str,
            auc_window_start_s: float = 0.0,
            auc_window_end_s: float = 5.0,
            max_event_index: int = 12,
            session_name: Optional[str] = None,
            azimuth_degrees: float = -60.0,
            elevation_degrees: float = 25.0,
    ) -> None:
        """
        Plot peri-event z-score traces for one animal in 3D with event index as the third axis.

        The x-axis represents peri-event time in seconds, the y-axis represents
        z-score amplitude, and the z-axis represents event index. The portion of
        each trace used for AUC extraction is shown as a shaded polygon under the
        curve within the specified AUC window.

        Parameters
        ----------
        animal : str
            Animal identifier to plot.
        auc_window_start_s : float, default=0.0
            Start time of the AUC integration window in seconds.
        auc_window_end_s : float, default=5.0
            End time of the AUC integration window in seconds.
        max_event_index : int, default=12
            Maximum event index to display.
        session_name : str, optional
            If provided, restrict plotting to one session.
        azimuth_degrees : float, default=-60.0
            Azimuth angle used for the 3D camera.
        elevation_degrees : float, default=25.0
            Elevation angle used for the 3D camera.

        Raises
        ------
        ValueError
            If no peri-event data are available for the requested animal.
        """
        animal_dataframe = self.session_level_peri_event_dataframe.loc[
            self.session_level_peri_event_dataframe["animal"] == animal
            ].copy()

        if session_name is not None:
            animal_dataframe = animal_dataframe.loc[
                animal_dataframe["session_name"] == session_name
                ].copy()

        animal_dataframe = animal_dataframe.loc[
            (animal_dataframe["event_index"] >= 1)
            & (animal_dataframe["event_index"] <= max_event_index)
            ].copy()

        if animal_dataframe.empty:
            raise ValueError(
                f"No peri-event data available for animal={animal!r}"
                + (f", session_name={session_name!r}" if session_name is not None else "")
            )

        animal_group = str(animal_dataframe["group"].iloc[0])

        animal_mean_trace_dataframe = (
            animal_dataframe
            .groupby(["animal", "group", "session_name", "event_index", "time_s"], as_index=False)["zscore"]
            .mean()
            .rename(columns={"zscore": "animal_mean_zscore"})
        )

        event_indices = list(range(1, max_event_index + 1))
        available_event_dataframe = animal_mean_trace_dataframe.loc[
            animal_mean_trace_dataframe["event_index"].isin(event_indices)
        ].copy()

        if available_event_dataframe.empty:
            raise ValueError(
                f"No event traces available for animal={animal!r} in the requested event range"
            )

        figure = plt.figure(figsize=(figure_size_peri[0] * 1.2, figure_size_peri[1] * 1.2))
        axis = figure.add_subplot(111, projection="3d")

        poly_vertices = []
        poly_event_positions = []
        auc_summary_rows = []

        trace_palette = sns.color_palette("crest", n_colors=len(event_indices))

        for color_index, event_index in enumerate(event_indices):
            event_dataframe = available_event_dataframe.loc[
                available_event_dataframe["event_index"] == event_index
                ].sort_values("time_s")

            if event_dataframe.empty:
                continue

            time_values_s = event_dataframe["time_s"].to_numpy(dtype=float)
            mean_zscore_values = event_dataframe["animal_mean_zscore"].to_numpy(dtype=float)

            axis.plot(
                time_values_s,
                mean_zscore_values,
                zs=event_index,
                zdir="z",
                color=trace_palette[color_index],
                linewidth=lw_peri_mean,
            )

            auc_mask = (
                    (time_values_s >= auc_window_start_s)
                    & (time_values_s <= auc_window_end_s)
            )

            if np.sum(auc_mask) >= 2:
                auc_time_values_s = time_values_s[auc_mask]
                auc_zscore_values = mean_zscore_values[auc_mask]

                auc_value = float(np.trapz(auc_zscore_values, x=auc_time_values_s))

                polygon_vertices = [(auc_time_values_s[0], 0.0)]
                polygon_vertices.extend(list(zip(auc_time_values_s, auc_zscore_values)))
                polygon_vertices.append((auc_time_values_s[-1], 0.0))

                poly_vertices.append(polygon_vertices)
                poly_event_positions.append(event_index)

                auc_summary_rows.append(
                    {
                        "event_index": event_index,
                        "auc_value": auc_value,
                    }
                )
            else:
                auc_summary_rows.append(
                    {
                        "event_index": event_index,
                        "auc_value": np.nan,
                    }
                )

        if poly_vertices:
            polygon_collection = PolyCollection(
                poly_vertices,
                facecolors=[mcolors.to_rgba(color_zscore, alpha=0.25)] * len(poly_vertices),
                edgecolors="none",
            )
            axis.add_collection3d(polygon_collection, zs=poly_event_positions, zdir="z")

        y_min = float(available_event_dataframe["animal_mean_zscore"].min())
        y_max = float(available_event_dataframe["animal_mean_zscore"].max())
        z_min = float(min(event_indices))
        z_max = float(max(event_indices))

        reference_plane_specifications = [
            (0.0, color_event_onset, 0.20),
            (auc_window_start_s, "black", 0.10),
            (auc_window_end_s, "black", 0.10),
        ]

        for reference_x, plane_color, plane_alpha in reference_plane_specifications:
            reference_plane_vertices = [[
                (reference_x, y_min, z_min),
                (reference_x, y_max, z_min),
                (reference_x, y_max, z_max),
                (reference_x, y_min, z_max),
            ]]

            reference_plane = Poly3DCollection(
                reference_plane_vertices,
                facecolors=mcolors.to_rgba(plane_color, alpha=plane_alpha),
                edgecolors=plane_color,
                linewidths=0.8,
            )
            axis.add_collection3d(reference_plane)

        axis.set_xlabel("Time from event (s)")
        axis.set_ylabel("Z-score")
        axis.set_zlabel("Event index")
        axis.set_zticks(event_indices)
        axis.view_init(elev=elevation_degrees, azim=azimuth_degrees)
        axis.set_title(
            f"{animal} ({animal_group}) — 3D peri-event z-score traces\n"
            f"AUC window: [{auc_window_start_s:.1f}, {auc_window_end_s:.1f}] s"
        )

        figure.tight_layout()

        output_stem = f"auc_benchmark_3d_{animal}_first_{max_event_index}_events"
        if session_name is not None:
            output_stem += f"_{session_name}"

        self._finalize_figure(figure, output_stem)

    def plot_event_trace_stack_by_animal_3d(
            self,
            event_index: int,
            session_name: Optional[str] = None,
            group_name: Optional[str] = None,
            azimuth_degrees: float = -60.0,
            elevation_degrees: float = 25.0,
    ) -> None:
        """
        Plot one peri-event z-score trace per animal in 3D for a selected event index.

        The x-axis represents peri-event time in seconds, the y-axis represents
        animal-level mean z-score, and the z-axis indexes animals.

        Parameters
        ----------
        event_index : int
            Event index to display.
        session_name : str, optional
            If provided, restrict plotting to one session.
        group_name : str, optional
            If provided, restrict plotting to one animal group.
        azimuth_degrees : float, default=-60.0
            Azimuth angle used for the 3D camera.
        elevation_degrees : float, default=25.0
            Elevation angle used for the 3D camera.

        Raises
        ------
        ValueError
            If no data are available for the requested selection.
        """
        selection_dataframe = self.session_level_peri_event_dataframe.loc[
            self.session_level_peri_event_dataframe["event_index"] == event_index
            ].copy()

        if session_name is not None:
            selection_dataframe = selection_dataframe.loc[
                selection_dataframe["session_name"] == session_name
                ].copy()

        if group_name is not None:
            selection_dataframe = selection_dataframe.loc[
                selection_dataframe["group"] == group_name
                ].copy()

        if selection_dataframe.empty:
            raise ValueError(
                f"No peri-event data available for event_index={event_index}, "
                f"session_name={session_name!r}, group_name={group_name!r}"
            )

        animal_mean_trace_dataframe = (
            selection_dataframe
            .groupby(["animal", "group", "session_name", "time_s"], as_index=False)["zscore"]
            .mean()
            .rename(columns={"zscore": "animal_mean_zscore"})
        )

        animal_labels = sorted(animal_mean_trace_dataframe["animal"].unique().tolist())
        animal_to_z_position = {
            animal_label: z_position
            for z_position, animal_label in enumerate(animal_labels, start=1)
        }

        figure = plt.figure(figsize=(figure_size_peri[0] * 1.2, figure_size_peri[1] * 1.2))
        axis = figure.add_subplot(111, projection="3d")

        animal_palette = sns.color_palette("viridis", n_colors=len(animal_labels))

        for color_index, animal_label in enumerate(animal_labels):
            animal_dataframe = animal_mean_trace_dataframe.loc[
                animal_mean_trace_dataframe["animal"] == animal_label
                ].sort_values("time_s")

            if animal_dataframe.empty:
                continue

            axis.plot(
                animal_dataframe["time_s"].to_numpy(dtype=float),
                animal_dataframe["animal_mean_zscore"].to_numpy(dtype=float),
                zs=animal_to_z_position[animal_label],
                zdir="z",
                color=animal_palette[color_index],
                linewidth=lw_peri_mean,
            )

        y_min = float(animal_mean_trace_dataframe["animal_mean_zscore"].min())
        y_max = float(animal_mean_trace_dataframe["animal_mean_zscore"].max())
        z_min = 1.0
        z_max = float(len(animal_labels))

        onset_plane_vertices = [[
            (0.0, y_min, z_min),
            (0.0, y_max, z_min),
            (0.0, y_max, z_max),
            (0.0, y_min, z_max),
        ]]
        onset_plane = Poly3DCollection(
            onset_plane_vertices,
            facecolors=mcolors.to_rgba(color_event_onset, alpha=0.18),
            edgecolors=color_event_onset,
            linewidths=0.8,
        )
        axis.add_collection3d(onset_plane)

        axis.set_xlabel("Time from event (s)")
        axis.set_ylabel("Z-score")
        axis.set_zlabel("Animal")
        axis.set_zticks(list(animal_to_z_position.values()))
        axis.set_zticklabels(animal_labels)
        axis.view_init(elev=elevation_degrees, azim=azimuth_degrees)

        title_suffix = f" | group={group_name}" if group_name is not None else ""
        if session_name is not None:
            title_suffix += f" | session={session_name}"
        axis.set_title(
            f"Animal-stacked peri-event z-score traces — event {event_index}{title_suffix}"
        )

        figure.tight_layout()

        output_stem = f"event_{event_index}_animal_stack_3d"
        if group_name is not None:
            output_stem += f"_{group_name}"
        if session_name is not None:
            output_stem += f"_{session_name}"

        self._finalize_figure(figure, output_stem)

    def plot_group_average_trace_stack_by_event_3d(
            self,
            group_name: str,
            max_event_index: int = 12,
            session_name: Optional[str] = None,
            azimuth_degrees: float = -60.0,
            elevation_degrees: float = 25.0,
    ) -> None:
        """
        Plot group-average peri-event z-score traces in 3D with event index as the z-axis.

        The x-axis represents peri-event time in seconds, the y-axis represents
        group-mean z-score, and the z-axis represents event index.

        Parameters
        ----------
        group_name : str
            Animal group to display.
        max_event_index : int, default=12
            Maximum event index to display.
        session_name : str, optional
            If provided, restrict plotting to one session.
        azimuth_degrees : float, default=-60.0
            Azimuth angle used for the 3D camera.
        elevation_degrees : float, default=25.0
            Elevation angle used for the 3D camera.

        Raises
        ------
        ValueError
            If no data are available for the requested group.
        """
        group_mean_trace_dataframe = self.compute_group_mean_event_traces(
            max_event_index=max_event_index,
            session_name=session_name,
        )

        selection_dataframe = group_mean_trace_dataframe.loc[
            group_mean_trace_dataframe["group"] == group_name
            ].copy()

        if selection_dataframe.empty:
            raise ValueError(
                f"No peri-event group mean traces available for group_name={group_name!r}, "
                f"session_name={session_name!r}"
            )

        event_indices = list(range(1, max_event_index + 1))
        figure = plt.figure(figsize=(figure_size_peri[0] * 1.2, figure_size_peri[1] * 1.2))
        axis = figure.add_subplot(111, projection="3d")

        event_palette = sns.color_palette("crest", n_colors=len(event_indices))

        for color_index, event_index in enumerate(event_indices):
            event_dataframe = selection_dataframe.loc[
                selection_dataframe["event_index"] == event_index
                ].sort_values("time_s")

            if event_dataframe.empty:
                continue

            axis.plot(
                event_dataframe["time_s"].to_numpy(dtype=float),
                event_dataframe["group_mean_zscore"].to_numpy(dtype=float),
                zs=event_index,
                zdir="z",
                color=event_palette[color_index],
                linewidth=lw_peri_mean,
            )

        y_min = float(selection_dataframe["group_mean_zscore"].min())
        y_max = float(selection_dataframe["group_mean_zscore"].max())
        z_min = float(min(event_indices))
        z_max = float(max(event_indices))

        onset_plane_vertices = [[
            (0.0, y_min, z_min),
            (0.0, y_max, z_min),
            (0.0, y_max, z_max),
            (0.0, y_min, z_max),
        ]]
        onset_plane = Poly3DCollection(
            onset_plane_vertices,
            facecolors=mcolors.to_rgba(color_event_onset, alpha=0.18),
            edgecolors=color_event_onset,
            linewidths=0.8,
        )
        axis.add_collection3d(onset_plane)

        axis.set_xlabel("Time from event (s)")
        axis.set_ylabel("Group mean z-score")
        axis.set_zlabel("Event index")
        axis.set_zticks(event_indices)
        axis.view_init(elev=elevation_degrees, azim=azimuth_degrees)

        title_suffix = f" | session={session_name}" if session_name is not None else ""
        axis.set_title(
            f"Group-average peri-event z-score traces — {group_name}{title_suffix}"
        )

        figure.tight_layout()

        output_stem = f"group_average_event_stack_3d_{group_name}"
        if session_name is not None:
            output_stem += f"_{session_name}"

        self._finalize_figure(figure, output_stem)

    def plot_group_event_auc_across_first_events(
            self,
            auc_window_start_s: float = 0.0,
            auc_window_end_s: float = 5.0,
            max_event_index: int = 12,
            session_name: Optional[str] = None,
    ) -> None:
        """
        Plot group-level peri-event z-score AUC across the first events.

        One subplot is generated per animal group. For each event index, the mean
        AUC across animals is shown with SEM.

        Parameters
        ----------
        auc_window_start_s : float, default=0.0
            Start time of the post-trigger integration window in seconds.
        auc_window_end_s : float, default=5.0
            End time of the post-trigger integration window in seconds.
        max_event_index : int, default=12
            Number of first events to plot.
        session_name : str, optional
            If provided, restrict plotting to one session.

        Raises
        ------
        ValueError
            If no AUC summary data are available for plotting.
        """
        group_event_auc_summary = self.compute_group_event_auc_summary(
            auc_window_start_s=auc_window_start_s,
            auc_window_end_s=auc_window_end_s,
            max_event_index=max_event_index,
            session_name=session_name,
        )

        if session_name is not None:
            group_event_auc_summary = group_event_auc_summary.loc[
                group_event_auc_summary["session_name"] == session_name
                ].copy()

        if group_event_auc_summary.empty:
            raise ValueError("No group-level event AUC data available for plotting")

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
            group_dataframe = group_event_auc_summary.loc[
                group_event_auc_summary["group"] == group_name
                ].sort_values("event_index")

            if group_dataframe.empty:
                axis.set_title(f"{group_name} (no data)")
                axis.set_ylabel("AUC (z-score·s)")
                axis.grid(alpha=0.3)
                continue

            sns.lineplot(
                data=group_dataframe,
                x="event_index",
                y="group_mean_auc",
                errorbar=None,
                marker="o",
                color=color_zscore,
                linewidth=lw_peri_mean,
                ax=axis,
            )

            axis.errorbar(
                group_dataframe["event_index"].to_numpy(dtype=int),
                group_dataframe["group_mean_auc"].to_numpy(dtype=float),
                yerr=group_dataframe["group_sem_auc"].to_numpy(dtype=float),
                fmt="none",
                ecolor=color_zscore,
                elinewidth=1.2,
                capsize=3,
            )

            axis.set_xticks(np.arange(1, max_event_index + 1, dtype=int))
            axis.set_ylabel("AUC (z-score·s)")
            axis.set_title(
                f"{group_name} — mean z-score AUC from "
                f"{auc_window_start_s:.1f} to {auc_window_end_s:.1f} s "
                f"(n≤{group_dataframe['n_animals'].max()})"
            )
            axis.grid(alpha=0.3)

        axes[-1].set_xlabel("Event index")
        figure.tight_layout()

        output_stem = f"group_event_auc_first_{max_event_index}_events"
        if session_name is not None:
            output_stem += f"_{session_name}"

        self._finalize_figure(figure, output_stem)

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

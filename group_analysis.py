from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from scipy import stats

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
    channel: str
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
            self.metadata_file_path = Path(params.base_path) / "animals_metadata.xlsx"
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
            engine="openpyxl",
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

        return metadata_dataframe.dropna()

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

        self.session_level_peri_event_dataframe["time_s"] = (
            self.session_level_peri_event_dataframe["time_s"].astype(float).round(3)
        )

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
        self.session_level_peri_event_dataframe["time_s"] = (
            self.session_level_peri_event_dataframe["time_s"].astype(float).round(3)
        )

        animal_averaged_dataframe = self.compute_animal_averaged_all_events()

        group_summary_dataframe = (
            animal_averaged_dataframe
            .groupby(["group", "session_name", "time_s"], as_index=False)
            .agg(
                group_mean_zscore=("animal_mean_zscore", "mean"),
                group_standard_deviation=("animal_mean_zscore", "std"),
                n_animals=("animal", "nunique"),
            )
        )

       # group_summary_dataframe["group_sem_zscore"] = (
       #         group_summary_dataframe["group_standard_deviation"]
       #         / np.sqrt(group_summary_dataframe["n_animals"])
        #)

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
        single_event_dataframe["time_s"] = (
            single_event_dataframe["time_s"].astype(float).round(3)
        )
        if single_event_dataframe.empty:
            raise ValueError(f"No data found for event_index={event_index}")

        animal_event_dataframe = (
            single_event_dataframe
            .groupby(["animal","group", "session_name", "time_s"], as_index=False)["zscore"]
            .mean()
            .rename(columns={"zscore": "animal_mean_zscore"})
        )
        print()
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
                n_animals=("animal", "nunique"),
            )
        )

       #group_summary_dataframe["group_sem_zscore"] = (
       #      group_summary_dataframe["group_standard_deviation"]
       #      / np.sqrt(group_summary_dataframe["n_animals"])
        #)

        group_summary_dataframe["event_index"] = event_index

        return group_summary_dataframe

    def compute_animal_event_auc(
            self,
            auc_window_start_s: float = 0.0,
            auc_window_end_s: float = 5.0,
            max_event_index: Optional[int] = 12,
            session_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute peri-event z-score AUC for each animal and event.

        If `max_event_index` is None, all available event indices are included.
        """
        auc_input_dataframe = self.session_level_peri_event_dataframe.copy()

        if session_name is not None:
            auc_input_dataframe = auc_input_dataframe.loc[
                auc_input_dataframe["session_name"] == session_name
                ].copy()

        event_mask = (auc_input_dataframe["event_index"] >= 1)
        if max_event_index is not None:
            event_mask &= (auc_input_dataframe["event_index"] <= int(max_event_index))

        auc_input_dataframe = auc_input_dataframe.loc[
            (auc_input_dataframe["time_s"] >= auc_window_start_s)
            & (auc_input_dataframe["time_s"] <= auc_window_end_s)
            & event_mask
            ].copy()

        if auc_input_dataframe.empty:
            raise ValueError("No peri-event samples found in the requested AUC window and event range")

        animal_event_auc_rows = []
        grouping_columns = ["animal","group", "session_name", "event_index"]

        for grouping_values, event_dataframe in auc_input_dataframe.groupby(grouping_columns):
            event_dataframe = event_dataframe.sort_values("time_s")
            time_values_s = event_dataframe["time_s"].to_numpy(dtype=float)
            zscore_values = event_dataframe["zscore"].to_numpy(dtype=float)

            if time_values_s.size < 2:
                continue

            auc_value = float(np.trapezoid(zscore_values, x=time_values_s))

            animal_event_auc_rows.append(
                {
                    "animal": grouping_values[0],
                    "group": grouping_values[1],
                    "session_name": grouping_values[2],
                    "event_index": int(grouping_values[3]),
                    "auc": auc_value,
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
            max_event_index: Optional[int] = 12,
            session_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute group-level mean and SEM of animal peri-event AUC values.

        If `max_event_index` is None, all available event indices are included.
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
                group_mean_auc=("auc", "mean"),
                group_standard_deviation=("auc", "std"),
                n_animals=("animal", "nunique"),
            )
        )

        group_event_auc_summary["group_sem_auc"] = (
            group_event_auc_summary["group_standard_deviation"]
            / np.sqrt(group_event_auc_summary["n_animals"])
        )

        return group_event_auc_summary

    def compute_group_mean_auc(
                self,
                auc_window_start_s: float = 0.0,
                auc_window_end_s: float = 5.0,
                max_event_index: Optional[int] = 12,
                session_name: Optional[str] = None,
        ) -> pd.DataFrame:
            """
            Compute mean AUC for each group from animal-level mean peri-event AUC values.

            This method first computes one AUC value per animal and event, then averages
            AUC across events within each animal, and finally computes the group-level
            mean and SEM across animals.

            Parameters
            ----------
            auc_window_start_s : float, default=0.0
                Start time of the AUC integration window in seconds.
            auc_window_end_s : float, default=5.0
                End time of the AUC integration window in seconds.
            max_event_index : int or None, default=12
                Maximum event index to include. If None, all available events are used.
            session_name : str or None, default=None
                If provided, restrict computation to one session.

            Returns
            -------
            pandas.DataFrame
                Table with one row per group and session, containing mean AUC,
                standard deviation, SEM, and number of animals.
            """
            animal_event_auc_dataframe = self.compute_animal_event_auc(
                auc_window_start_s=auc_window_start_s,
                auc_window_end_s=auc_window_end_s,
                max_event_index=max_event_index,
                session_name=session_name,
            )

            animal_mean_auc_dataframe = (
                animal_event_auc_dataframe
                .groupby(["animal", "group", "session_name"], as_index=False)
                .agg(
                    animal_mean_auc=("auc", "mean"),
                )
            )

            group_mean_auc_dataframe = (
                animal_mean_auc_dataframe
                .groupby(["group", "session_name"], as_index=False)
                .agg(
                    group_mean_auc=("animal_mean_auc", "mean"),
                    group_standard_deviation=("animal_mean_auc", "std"),
                    n_animals=("animal", "nunique"),
                )
            )

            group_mean_auc_dataframe["group_sem_auc"] = (
                    group_mean_auc_dataframe["group_standard_deviation"]
                    / np.sqrt(group_mean_auc_dataframe["n_animals"])
            )

            return group_mean_auc_dataframe




    def plot_group_event_auc_across_first_events(
            self,
            auc_window_start_s: float = 0.0,
            auc_window_end_s: float = 5.0,
            max_event_index: Optional[int] = 12,
            session_name: Optional[str] = None,
    ) -> None:
        """
        Plot group AUC across event indices.

        If `max_event_index` is None, all available event indices are plotted.
        """
        animal_event_auc_dataframe = self.compute_animal_event_auc(
            auc_window_start_s=auc_window_start_s,
            auc_window_end_s=auc_window_end_s,
            max_event_index=max_event_index,
            session_name=session_name,
        )

        if animal_event_auc_dataframe.empty:
            raise ValueError("No group-level event AUC data available for plotting")

        if max_event_index is None:
            event_indices_to_plot = np.sort(animal_event_auc_dataframe["event_index"].unique().astype(int))
        else:
            event_indices_to_plot = np.arange(1, int(max_event_index) + 1, dtype=int)

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
            group_dataframe = animal_event_auc_dataframe.loc[
                animal_event_auc_dataframe["group"] == group_name
                ].sort_values("event_index")

            if group_dataframe.empty:
                axis.set_title(f"{group_name} (no data)")
                axis.set_ylabel("AUC (z-score·s)")
                axis.grid(alpha=0.3)
                continue

            sns.lineplot(
                data=group_dataframe,
                x="event_index",
                y="auc",
                estimator="mean",
                errorbar="se",
                marker="o",
                color=color_zscore,
                linewidth=lw_peri_mean,
                ax=axis,
            )

            axis.set_xticks(event_indices_to_plot)
            axis.set_ylabel("AUC (z-score·s)")

            max_n = int(group_dataframe["animal"].nunique())
            suffix = "all events" if max_event_index is None else f"first {max_event_index} events"
            axis.set_title(
                f"{group_name} — mean z-score AUC from "
                f"{auc_window_start_s:.1f} to {auc_window_end_s:.1f} s "
                f"({suffix}, n={max_n})"
            )
            axis.grid(alpha=0.3)

        axes[-1].set_xlabel("Event index")
        figure.tight_layout()

        output_stem = "group_event_auc"
        if max_event_index is None:
            output_stem += "_all_events"
        else:
            output_stem += f"_first_{max_event_index}_events"
        if session_name is not None:
            output_stem += f"_{session_name}"

        self._finalize_figure(figure, output_stem)


    def plot_group_event_auc_single_axis_with_hue(
                self,
                auc_window_start_s: float = 0.0,
                auc_window_end_s: float = 5.0,
                max_event_index: Optional[int] = 12,
                session_name: Optional[str] = None,
                hue_order: Optional[list[str]] = None,
        ) -> None:
            """
            Plot group AUC across event indices on a single axis using seaborn hue for groups.

            If `max_event_index` is None, all available event indices are plotted.

            Parameters
            ----------
            auc_window_start_s : float, default=0.0
                Start time of AUC integration window (s).
            auc_window_end_s : float, default=5.0
                End time of AUC integration window (s).
            max_event_index : int or None, default=12
                If None, includes all events.
            session_name : str or None, default=None
                If provided, restrict to one session.
            hue_order : list of str or None, default=None
                Order for group hue. If None, uses `self.subplot_group_order`.
            """
            animal_event_auc_dataframe = self.compute_animal_event_auc(
                auc_window_start_s=auc_window_start_s,
                auc_window_end_s=auc_window_end_s,
                max_event_index=max_event_index,
                session_name=session_name,
            )

            if animal_event_auc_dataframe.empty:
                raise ValueError("No group-level event AUC data available for plotting")

            if max_event_index is None:
                event_indices_to_plot = np.sort(animal_event_auc_dataframe["event_index"].unique().astype(int))
            else:
                event_indices_to_plot = np.arange(1, int(max_event_index) + 1, dtype=int)

            if hue_order is None:
                hue_order = list(self.subplot_group_order)

            palette = sns.color_palette("colorblind", n_colors=len(hue_order))
            group_to_color = {g: palette[i] for i, g in enumerate(hue_order)}

            figure, ax = plt.subplots(1, 1, figsize=figure_size_peri, sharex=True, sharey=True)

            sns.lineplot(
                data=animal_event_auc_dataframe,
                x="event_index",
                y="auc",
                hue="group",
                hue_order=hue_order,
                estimator="mean",
                errorbar="se",
                palette=group_to_color,
                marker="o",
                linewidth=lw_peri_mean,
                ax=ax,
            )

            ax.set_xticks(event_indices_to_plot)
            ax.set_xlabel("Event index")
            ax.set_ylabel("AUC (z-score·s)")

            suffix = "all events" if max_event_index is None else f"first {max_event_index} events"
            session_suffix = f" | {session_name}" if session_name is not None else ""
            ax.set_title(
                f"Group mean z-score AUC [{auc_window_start_s:.1f}, {auc_window_end_s:.1f}] s "
                f"({suffix}){session_suffix}"
            )

            group_counts = (
                animal_event_auc_dataframe.groupby("group")["animal"]
                .nunique()
                .to_dict()
            )

            handles, labels = ax.get_legend_handles_labels()
            if labels and labels[0] == "group":
                handles = handles[1:]
                labels = labels[1:]

            new_labels = [
                f"{label} (n={group_counts.get(label, 0)})"
                for label in labels
            ]

            ax.legend(handles, new_labels, title="Group", frameon=False)
            ax.grid(alpha=0.3)
            figure.tight_layout()

            output_stem = "group_event_auc_single_plot"
            if max_event_index is None:
                output_stem += "_all_events"
            else:
                output_stem += f"_first_{max_event_index}_events"
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
        animal_averaged_dataframe = self.compute_animal_averaged_all_events()

        if session_name is not None:
            animal_averaged_dataframe = animal_averaged_dataframe.loc[
                animal_averaged_dataframe["session_name"] == session_name
                ].copy()

        if animal_averaged_dataframe.empty:
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
            signal_dataframe = animal_averaged_dataframe.loc[
                animal_averaged_dataframe["group"] == group_name
                ].sort_values("time_s")

            if signal_dataframe.empty:
                axis.set_title(f"{group_name}  (no data)")
                axis.set_ylabel("Z-score")
                axis.grid(alpha=0.3)
                continue

            sns.lineplot(
                data=signal_dataframe,
                x="time_s",
                y="animal_mean_zscore",
                estimator="mean",
                errorbar="se",
                color=color_zscore,
                linewidth=lw_peri_mean,
                ax=axis,
            )

            axis.axvline(0, color=color_event_onset, linestyle="--", linewidth=0.8)
            axis.set_ylabel("Z-score")
            n_animals = int(signal_dataframe["animal"].nunique())
            axis.set_title(
                f"Group peri-event average {params.sessions[0]} — {group_name} "
                f"(all events, n={n_animals})"
            )
            axis.grid(alpha=0.3)

        axes[-1].set_xlabel("Time from event (s)")
        figure.tight_layout()

        output_stem = "group_peri_event_all_events_zscore"
        if session_name is not None:
            output_stem += f"_{session_name}"

        self._finalize_figure(figure, output_stem)

    def plot_group_average_all_events_wt_het_superimposed(
            self,
            session_name: Optional[str] = None,
    ) -> None:
        """
        Plot wt and het group peri-event averages across all events on a single axis.

        The trace is computed from animal-level mean peri-event responses, with
        mean and SEM across animals shown for each group.

        Parameters
        ----------
        session_name : str, optional
            If provided, restrict plotting to one session.

        Returns
        -------
        None
            The figure is saved and/or displayed according to project configuration.
        """
        animal_averaged_dataframe = self.compute_animal_averaged_all_events()

        if session_name is not None:
            animal_averaged_dataframe = animal_averaged_dataframe.loc[
                animal_averaged_dataframe["session_name"] == session_name
                ].copy()

        animal_averaged_dataframe = animal_averaged_dataframe.loc[
            animal_averaged_dataframe["group"].isin(["wt", "het"])
        ].copy()

        if animal_averaged_dataframe.empty:
            raise ValueError("No wt/het group data available for all-event plotting")

        group_color_map = {
            "wt": "black",
            "het": "blue",
        }

        figure, axis = plt.subplots(
            1,
            1,
            figsize=figure_size_peri,
            sharex=True,
            sharey=True,
        )

        for group_name in ["wt", "het"]:
            signal_dataframe = animal_averaged_dataframe.loc[
                animal_averaged_dataframe["group"] == group_name
                ].sort_values("time_s")

            if signal_dataframe.empty:
                continue

            sns.lineplot(
                data=signal_dataframe,
                x="time_s",
                y="animal_mean_zscore",
                estimator="mean",
                errorbar="se",
                color=group_color_map[group_name],
                linewidth=lw_peri_mean,
                ax=axis,
                label=f"{group_name} (n={signal_dataframe['animal'].nunique()})",
            )

        axis.axvline(0, color=color_event_onset, linestyle="--", linewidth=0.8)
        axis.set_xlabel("Time from event (s)")
        axis.set_ylabel("Z-score")

        session_label = session_name if session_name is not None else "all sessions"
        axis.set_title(f"Group peri-event average — wt vs het (all events, {session_label})")
        axis.grid(alpha=0.3)
        axis.legend(frameon=False)

        figure.tight_layout()

        output_stem = "group_peri_event_all_events_wt_het_superimposed_zscore"
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
        animal_event_dataframe = self.compute_animal_averaged_single_event(event_index)

        if session_name is not None:
            animal_event_dataframe = animal_event_dataframe.loc[
                animal_event_dataframe["session_name"] == session_name
                ].copy()

        if animal_event_dataframe.empty:
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
            signal_dataframe = animal_event_dataframe.loc[
                animal_event_dataframe["group"] == group_name
                ].sort_values("time_s")

            if signal_dataframe.empty:
                axis.set_title(f"{group_name} (no data)")
                axis.set_ylabel("Z-score")
                axis.grid(alpha=0.3)
                continue

            sns.lineplot(
                data=signal_dataframe,
                x="time_s",
                y="animal_mean_zscore",
                estimator="mean",
                errorbar="se",
                color=color_zscore,
                linewidth=lw_peri_mean,
                ax=axis,
            )

            axis.axvline(0, color=color_event_onset, linestyle="--", linewidth=0.8)
            axis.set_ylabel("Z-score")
            n_animals = int(signal_dataframe["animal"].nunique())
            axis.set_title(
                f"Group peri-event average — {group_name} "
                f"(event {event_index}, n={n_animals})"
            )
            axis.grid(alpha=0.3)

        axes[-1].set_xlabel("Time from event (s)")
        figure.tight_layout()

        output_stem = f"group_peri_event_{event_index}_zscore"
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
            self.plot_group_average_single_event(
                event_index=event_index,
                session_name=session_name,
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
            output_path = self.output_directory / f"{filename_stem}_{self.channel}.{figure_format}"
            figure.savefig(output_path, dpi=figure_dpi, format=figure_format, bbox_inches="tight")
            print(f"Saved: {output_path}")

        if preview_figures:
            plt.show()
        else:
            plt.close(figure)


def run_group_level_plots_for_event_types(
        completed_pipeline,
        event_types: list,
        group_output_root: Path,
        auc_window_start_s: float = 0.0,
        auc_window_end_s: float = 2.0,
        max_event_index: int = 12,
        session_name_for_auc: str | None = "Recall",
) -> None:
    """
    Run the full set of group-level plots for multiple event types.

    Parameters
    ----------
    completed_pipeline : PhotometryPipeline
        Completed pipeline object holding processed sessions.
    event_types : dict[str, str]
        Mapping from a human-readable label to an event table key to epoch on.
        Example:
        - {"cs_onset": "cs_led_cluster_first_onsets",
           "cs_offset": "cs_led_cluster_first_offsets",
           "freeze_onset": "freezing_cluster_first_onsets"}
    group_output_root : pathlib.Path
        Root output directory for group plots. Each event type writes into a subfolder.
    auc_window_start_s : float, default=0.0
        AUC start time in seconds.
    auc_window_end_s : float, default=5.0
        AUC end time in seconds.
    max_event_index : int, default=12
        Number of first events for AUC and single-event plots.
    session_name_for_auc : str or None, default="Recall"
        Session name restriction for AUC plotting. Set to None to compute across all sessions.
    """
    group_output_root.mkdir(parents=True, exist_ok=True)
    auc_group_mean_output_directory = Path(r"E:\Fibre_photmetry\auc group means")
    auc_group_mean_output_directory.mkdir(parents=True, exist_ok=True)

    for event_table_key in event_types:

        output_dir = group_output_root / event_table_key
        output_dir.mkdir(parents=True, exist_ok=True)

        for channel in ["CH1", "CH2"]:
            print(event_table_key, channel)

            group_peri_event_dff = completed_pipeline.build_group_peri_event_dataframe(
                event_table_key,
                channel=channel,
                signal_key="dff",
            )
            group_peri_event_zscore = completed_pipeline.build_group_peri_event_dataframe(
                event_table_key,
                channel=channel,
                signal_key="zscore",
            )

            if group_peri_event_dff.empty:
                print(f"[GROUP] No data for event type {event_table_key!r} -> skipping")
                continue

            group_analyzer_dff = PhotometryGroupAnalyzer(
                session_level_peri_event_dataframe=group_peri_event_dff,
                output_directory=output_dir,
                channel=channel,
            )

            group_analyzer_zscore = PhotometryGroupAnalyzer(
                session_level_peri_event_dataframe=group_peri_event_zscore,
                output_directory=output_dir,
                channel=channel,
            )

            for signal_type in [group_analyzer_dff, group_analyzer_zscore]:
                signal_label = "zscore" if signal_type is group_analyzer_zscore else "dff"

                available_sessions = sorted(
                    signal_type.session_level_peri_event_dataframe["session_name"]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                for current_session_name in available_sessions:
                    if "cs" in event_table_key or "shock" in event_table_key:
                        if current_session_name == "Recall":
                            current_max_event_index = 12
                        elif current_session_name == "Cond":
                            if event_table_key in {"shock", "cs_offsets"}:
                                current_max_event_index = 5
                            else:
                                current_max_event_index = 6
                        else:
                            current_max_event_index = max_event_index
                    else:
                        current_max_event_index = None

                    try:
                        group_mean_auc_dataframe = signal_type.compute_group_event_auc_summary(
                            auc_window_start_s=auc_window_start_s,
                            auc_window_end_s=auc_window_end_s,
                            max_event_index=current_max_event_index,
                            session_name=current_session_name,
                        )
                    except ValueError:
                        print(
                            f"[GROUP AUC] No data for event_type={event_table_key}, "
                            f"signal={signal_label}, channel={channel}, session={current_session_name}"
                        )
                        continue

                    print(group_mean_auc_dataframe)

                    output_csv_path = auc_group_mean_output_directory / (
                        f"group_mean_auc_{event_table_key}_{current_session_name}_"
                        f"{auc_window_start_s:.1f}_to_{auc_window_end_s:.1f}s_"
                        f"{signal_label}_{channel}.csv"
                    )

                    group_mean_auc_dataframe.to_csv(output_csv_path, index=False)
                    print(f"Saved: {output_csv_path}")

                signal_type.plot_group_average_all_events(session_name=session_name_for_auc)
                signal_type.plot_group_average_all_events(session_name=session_name_for_auc)
                signal_type.plot_group_average_all_events_wt_het_superimposed(
                    session_name=session_name_for_auc
                )

                if "cs" in event_table_key or "shock" in event_table_key:
                    if session_name_for_auc == "Recall":
                        plotting_max_event_index = 12
                    elif session_name_for_auc == "Cond":
                        if event_table_key in {"shock", "cs_offsets"}:
                            plotting_max_event_index = 5
                        else:
                            plotting_max_event_index = 6
                    else:
                        plotting_max_event_index = max_event_index

                    for event_index in range(plotting_max_event_index):
                        signal_type.plot_group_average_single_event(
                            event_index=event_index + 1
                        )
                else:
                    plotting_max_event_index = None

                signal_type.plot_group_event_auc_across_first_events(
                    auc_window_start_s=auc_window_start_s,
                    auc_window_end_s=auc_window_end_s,
                    max_event_index=plotting_max_event_index,
                    session_name=session_name_for_auc,
                )

                signal_type.plot_group_event_auc_single_axis_with_hue(
                    auc_window_start_s=auc_window_start_s,
                    auc_window_end_s=auc_window_end_s,
                    max_event_index=plotting_max_event_index,
                    session_name=session_name_for_auc,
                )

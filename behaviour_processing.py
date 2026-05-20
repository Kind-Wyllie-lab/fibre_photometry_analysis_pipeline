from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

def _interval_duration_s(interval: tuple[float, float]) -> float:
    start_s, end_s = interval
    return max(0.0, float(end_s) - float(start_s))


def _compute_overlap_duration_s(
    interval: tuple[float, float],
    freezing_intervals_s: np.ndarray,
) -> float:
    """
    Compute total overlap duration between one interval and a set of freezing intervals.

    Parameters
    ----------
    interval : tuple of float
        (start_s, end_s) for the analysis window.
    freezing_intervals_s : numpy.ndarray
        Array of shape (n_intervals, 2) with freezing (start_s, end_s).

    Returns
    -------
    float
        Total overlapped duration in seconds.
    """
    start_s, end_s = interval
    if end_s <= start_s or freezing_intervals_s.size == 0:
        return 0.0

    fs = freezing_intervals_s[:, 0]
    fe = freezing_intervals_s[:, 1]

    overlap_start = np.maximum(fs, start_s)
    overlap_end = np.minimum(fe, end_s)
    overlap = np.maximum(0.0, overlap_end - overlap_start)
    return float(np.sum(overlap))


def _pair_onsets_offsets_to_intervals(
    onsets_s: np.ndarray,
    offsets_s: np.ndarray,
    recording_end_s: Optional[float] = None,
) -> np.ndarray:
    """
    Pair freezing onsets and offsets into intervals.

    Parameters
    ----------
    onsets_s : numpy.ndarray
        Freezing onset times (seconds).
    offsets_s : numpy.ndarray
        Freezing offset times (seconds).
    recording_end_s : float or None, default=None
        Optional recording end time to close an open interval.

    Returns
    -------
    numpy.ndarray
        Freezing intervals array with shape (n_intervals, 2).

    Notes
    -----
    - If an onset occurs without a later offset, and `recording_end_s` is provided,
      the interval is closed at `recording_end_s`.
    - If offsets are found before the first onset they are ignored.
    """
    onsets_s = np.asarray(onsets_s, dtype=float) if onsets_s is not None else np.array([], dtype=float)
    offsets_s = np.asarray(offsets_s, dtype=float) if offsets_s is not None else np.array([], dtype=float)

    onsets_s = np.sort(onsets_s[np.isfinite(onsets_s)])
    offsets_s = np.sort(offsets_s[np.isfinite(offsets_s)])

    if onsets_s.size == 0:
        return np.zeros((0, 2), dtype=float)

    intervals = []
    j = 0
    for onset in onsets_s:
        while j < offsets_s.size and offsets_s[j] <= onset:
            j += 1
        if j < offsets_s.size:
            intervals.append((float(onset), float(offsets_s[j])))
            j += 1
        else:
            if recording_end_s is not None and recording_end_s > onset:
                intervals.append((float(onset), float(recording_end_s)))

    if not intervals:
        return np.zeros((0, 2), dtype=float)

    return np.asarray(intervals, dtype=float)


def build_behavior_bout_intervals_from_cs(
    cs_onsets_s: np.ndarray,
    cs_offsets_s: np.ndarray,
    pre_cs_duration_s: float = 120.0,
    post_last_cs_duration_s: float = 30.0,
    n_cs: int = 12,
) -> dict[str, tuple[float, float]]:
    """
    Build named session intervals (pre-cs, cs_i, noncs_i, post) from CS onsets/offsets.

    Parameters
    ----------
    cs_onsets_s : numpy.ndarray
        CS onset times (seconds), expected at least `n_cs` entries.
    cs_offsets_s : numpy.ndarray
        CS offset times (seconds), expected at least `n_cs` entries.
    pre_cs_duration_s : float, default=120.0
        Duration before CS1 onset used as baseline/pre-cs interval.
    post_last_cs_duration_s : float, default=30.0
        Duration after CS12 offset used as post interval.
    n_cs : int, default=12
        Number of CS bouts.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from bout name -> (start_s, end_s).

    Raises
    ------
    ValueError
        If not enough CS onsets/offsets are provided or if times are inconsistent.
    """
    cs_onsets_s = np.asarray(cs_onsets_s, dtype=float)
    cs_offsets_s = np.asarray(cs_offsets_s, dtype=float)

    if cs_onsets_s.size < n_cs or cs_offsets_s.size < n_cs:
        raise ValueError(f"Need at least {n_cs} cs onsets/offsets, got onsets={cs_onsets_s.size}, offsets={cs_offsets_s.size}")

    cs_onsets_s = np.sort(cs_onsets_s)[:n_cs]
    cs_offsets_s = np.sort(cs_offsets_s)[:n_cs]

    if np.any(cs_offsets_s <= cs_onsets_s):
        raise ValueError("Some CS offsets occur before/on CS onset; check CS timing extraction")

    intervals: dict[str, tuple[float, float]] = {}

    # pre-cs: 2 minutes before CS1 onset
    cs1_onset = float(cs_onsets_s[0])
    intervals["pre_cs"] = (cs1_onset - float(pre_cs_duration_s), cs1_onset)

    # cs_i intervals
    for i in range(n_cs):
        intervals[f"cs_{i+1}"] = (float(cs_onsets_s[i]), float(cs_offsets_s[i]))

    # noncs_i: between cs_i offset and cs_{i+1} onset, for i=1..(n_cs-1)
    # You requested noncs_1..10 specifically; that corresponds to gaps after cs_1..cs_10 (i=0..9)
    n_noncs = min(10, n_cs - 1)
    for i in range(n_noncs):
        intervals[f"noncs_{i+1}"] = (float(cs_offsets_s[i]), float(cs_onsets_s[i + 1]))

    # post: 30 seconds after CS12 offset
    intervals["post_cs12"] = (float(cs_offsets_s[n_cs - 1]), float(cs_offsets_s[n_cs - 1]) + float(post_last_cs_duration_s))

    return intervals


def compute_freezing_ratio_per_bout(
    bout_intervals: dict[str, tuple[float, float]],
    freezing_onsets_s: Optional[np.ndarray],
    freezing_offsets_s: Optional[np.ndarray],
    recording_end_s: Optional[float] = None,
) -> dict[str, float]:
    """
    Compute freezing ratio per named bout interval.

    Parameters
    ----------
    bout_intervals : dict[str, tuple[float, float]]
        Named bout intervals.
    freezing_onsets_s : numpy.ndarray or None
        Freezing onset times (seconds). If None/empty, ratios will be 0.
    freezing_offsets_s : numpy.ndarray or None
        Freezing offset times (seconds). If None/empty, ratios will be 0.
    recording_end_s : float or None, default=None
        Used if there is an unclosed freezing onset.

    Returns
    -------
    dict[str, float]
        Mapping bout name -> freezing ratio in [0, 1].
    """
    if freezing_onsets_s is None or freezing_offsets_s is None:
        return {name: 0.0 for name in bout_intervals.keys()}

    freezing_intervals_s = _pair_onsets_offsets_to_intervals(
        onsets_s=freezing_onsets_s,
        offsets_s=freezing_offsets_s,
        recording_end_s=recording_end_s,
    )

    ratios: dict[str, float] = {}
    for name, interval in bout_intervals.items():
        total_s = _interval_duration_s(interval)
        if total_s <= 0:
            ratios[name] = np.nan
            continue
        freeze_s = _compute_overlap_duration_s(interval, freezing_intervals_s)
        ratios[name] = float(freeze_s / total_s)

    return ratios

def build_freezing_behavior_profile_table(
    completed_sessions: list,
    metadata_dataframe: pd.DataFrame,
    animal_column: str = "animal",
    group_column: str = "group",
    session_column_name: str = "session_name",
    pre_cs_duration_s: float = 120.0,
    post_last_cs_duration_s: float = 30.0,
) -> pd.DataFrame:
    """
    Build a table of freezing ratios per session bout for all animals with freezing scored.

    Parameters
    ----------
    completed_sessions : list
        Iterable of session objects (from your pipeline). Each session should expose:
        - animal (str)
        - session_name (str)
        - cs_onsets_s (np.ndarray)
        - cs_offsets_s (np.ndarray)
        - freezing_onsets_s (np.ndarray or None)
        - freezing_offsets_s (np.ndarray or None)
    metadata_dataframe : pandas.DataFrame
        Table with `animal` and `group` columns (genotype).
    animal_column : str, default="animal"
        Column name in metadata for animal id.
    group_column : str, default="group"
        Column name in metadata for genotype/group.
    session_column_name : str, default="session_name"
        Output column name for session id.
    pre_cs_duration_s : float, default=120.0
        Pre-CS interval duration.
    post_last_cs_duration_s : float, default=30.0
        Post-CS12 interval duration.

    Returns
    -------
    pandas.DataFrame
        One row per animal-session with freezing ratios for each bout and genotype/group.

    Notes
    -----
    Sessions without freezing scored (missing or empty freezing onsets/offsets)
    are skipped.
    """
    required_meta = {animal_column, group_column}
    if not required_meta.issubset(metadata_dataframe.columns):
        raise ValueError(f"metadata_dataframe must contain columns {required_meta}")

    meta = metadata_dataframe[[animal_column, group_column]].copy()
    meta = meta.rename(columns={animal_column: "animal", group_column: "group"})
    meta["animal"] = meta["animal"].astype(str)
    meta["group"] = meta["group"].astype(str)

    rows = []

    for sess in completed_sessions:
        if sess.session_name == 'Recall':
            n_cs = 12
        elif sess.session_name == 'Cond':
            n_cs = 6
        animal = str(getattr(sess, "animal"))
        sess_name = str(getattr(sess, "session_name"))

        cs_onsets_s = sess.event_tables['cs_onsets']
        cs_offsets_s = sess.event_tables['cs_offsets']

        freezing_onsets_s = sess.event_tables['freezing_onsets']
        freezing_offsets_s = sess.event_tables['freezing_offsets']

        # Skip sessions with no freezing scored
        if freezing_onsets_s is None or freezing_offsets_s is None:
            continue

        if cs_onsets_s is None or cs_offsets_s is None:
            continue

        bout_intervals = build_behavior_bout_intervals_from_cs(
            cs_onsets_s=cs_onsets_s,
            cs_offsets_s=cs_offsets_s,
            pre_cs_duration_s=pre_cs_duration_s,
            post_last_cs_duration_s=post_last_cs_duration_s,
            n_cs=n_cs,
        )

        # Use end of post bout as recording_end for closing open freezing intervals if needed
        recording_end_s = bout_intervals["post_cs12"][1]

        ratios = compute_freezing_ratio_per_bout(
            bout_intervals=bout_intervals,
            freezing_onsets_s=freezing_onsets_s,
            freezing_offsets_s=freezing_offsets_s,
            recording_end_s=recording_end_s,
        )

        row = {
            "animal": animal,
            session_column_name: sess_name,
            'group': meta.loc[meta['animal']== animal]['group'].values[0]
        }
        row.update(ratios)
        rows.append(row)

    behavior_df = pd.DataFrame(rows)
    if behavior_df.empty:
        return behavior_df

    # behavior_df = behavior_df.merge(meta, on="animal", how="left", validate="many_to_one")
    if behavior_df["group"].isna().any():
        missing = sorted(behavior_df.loc[behavior_df["group"].isna(), "animal"].unique())
        raise ValueError(f"Missing genotype/group metadata for animals: {missing}")

    # Put columns in a sensible order
    bout_cols = (
        ["pre_cs"]
        + [f"cs_{i}" for i in range(1, 13)]
        + [f"noncs_{i}" for i in range(1, 11)]
        + ["post_cs12"]
    )
    existing_bout_cols = [c for c in bout_cols if c in behavior_df.columns]
    ordered_cols = ["animal", "group", session_column_name] + existing_bout_cols
    behavior_df = behavior_df[ordered_cols]

    return behavior_df

def freezing_profile_wide_to_tidy(
    freezing_profile_df: pd.DataFrame,
    animal_col: str = "animal",
    genotype_col: str = "group",
    session_col: str = "session_name",
    value_col_out: str = "freeze_pct",
    segment_col_out: str = "segment",
) -> pd.DataFrame:
    """
    Convert a wide freezing profile table into a tidy long table.

    Parameters
    ----------
    freezing_profile_df : pandas.DataFrame
        Wide table with one row per animal/session and columns like
        pre_cs, cs_1..cs_12, noncs_1..noncs_10, post_cs12.
    animal_col : str, default="animal"
        Animal identifier column.
    genotype_col : str, default="group"
        Genotype/group column.
    session_col : str, default="session_name"
        Session identifier column (optional).
    value_col_out : str, default="freeze_pct"
        Output column for freezing ratio.
    segment_col_out : str, default="segment"
        Output column for segment label.

    Returns
    -------
    pandas.DataFrame
        Tidy dataframe with columns:
        - animal
        - genotype
        - session (if present)
        - segment
        - freeze_pct
    """
    required = {animal_col, genotype_col}
    missing = required.difference(freezing_profile_df.columns)
    if missing:
        raise ValueError(f"freezing_profile_df missing required columns: {sorted(missing)}")

    id_vars = [animal_col, genotype_col]
    if session_col in freezing_profile_df.columns:
        id_vars.append(session_col)

    value_vars = [c for c in freezing_profile_df.columns if c not in id_vars]

    df_tidy = (
        freezing_profile_df[id_vars + value_vars]
        .melt(id_vars=id_vars, var_name=segment_col_out, value_name=value_col_out)
        .dropna(subset=[value_col_out])
        .copy()
    )

    df_tidy = df_tidy.rename(
        columns={
            animal_col: "animal",
            genotype_col: "genotype",
            session_col: "session",
        }
    )
    df_tidy[value_col_out] = pd.to_numeric(df_tidy[value_col_out], errors="coerce")
    df_tidy = df_tidy.dropna(subset=[value_col_out])

    return df_tidy



def compute_extinction_index(
    df_freeze: pd.DataFrame,
    group_by: tuple[str, ...] = ("animal", "genotype"),
    require_min_cs: int = 10,
    n_first: int = 3,
    n_last: int = 3,
    cs_regex: str = r"^cs_(\d+)$",
    segment_col: str = "segment",
    freeze_col: str = "freeze_pct",
) -> pd.DataFrame:
    """
    Compute extinction index (EI) from freezing ratios in CS segments.

    EI per group is:
        EI = (sum_first - sum_last) / (sum_first + sum_last)

    where sum_first is the sum of freezing ratios for the first n_first CS,
    and sum_last is the sum for the last n_last CS.

    Parameters
    ----------
    df_freeze : pandas.DataFrame
        Tidy dataframe with columns including group_by, `segment_col`, and `freeze_col`.
    group_by : tuple of str, default=("animal","genotype")
        Grouping columns (animal-level EI by default).
    require_min_cs : int, default=10
        Minimum number of CS segments needed to compute EI (else NaN).
    n_first : int, default=3
        Number of earliest CS used.
    n_last : int, default=3
        Number of latest CS used.
    cs_regex : str, default="^cs_(\\d+)$"
        Regex to match CS segments and extract numeric index.
    segment_col : str, default="segment"
        Segment column name.
    freeze_col : str, default="freeze_pct"
        Freezing ratio column name.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns:
        - group_by...
        - ext_index
        - sum_first
        - sum_last
        - n_cs
    """
    required_cols = set(group_by) | {segment_col, freeze_col}
    missing = required_cols.difference(df_freeze.columns)
    if missing:
        raise ValueError(f"df_freeze missing required columns: {sorted(missing)}")

    min_needed = max(require_min_cs, n_first + n_last)

    cs = df_freeze[df_freeze[segment_col].astype(str).str.match(cs_regex, na=False)].copy()
    if cs.empty:
        return pd.DataFrame(columns=[*group_by, "ext_index", "sum_first", "sum_last", "n_cs"])

    cs["cs_num"] = cs[segment_col].astype(str).str.extract(cs_regex)[0].astype(float)
    cs = cs.dropna(subset=["cs_num", freeze_col])
    cs["cs_num"] = cs["cs_num"].astype(int)

    results: list[dict] = []
    group_cols = list(group_by)

    for gvals, sub in cs.groupby(group_cols, dropna=False, observed=False):
        sub = sub.sort_values("cs_num")
        unique_cs = sub["cs_num"].unique()
        n_cs = len(unique_cs)

        payload = dict(zip(group_cols, gvals if isinstance(gvals, tuple) else (gvals,)))

        if n_cs < min_needed:
            results.append({**payload, "ext_index": np.nan, "sum_first": np.nan, "sum_last": np.nan, "n_cs": n_cs})
            continue

        first_ids = unique_cs[:n_first]
        last_ids = unique_cs[-n_last:]

        sum_first = float(sub.loc[sub["cs_num"].isin(first_ids), freeze_col].sum())
        sum_last = float(sub.loc[sub["cs_num"].isin(last_ids), freeze_col].sum())

        denom = sum_first + sum_last
        ext_idx = (sum_first - sum_last) / denom if denom > 0 else np.nan

        results.append({**payload, "ext_index": ext_idx, "sum_first": sum_first, "sum_last": sum_last, "n_cs": n_cs})

    return pd.DataFrame(results)

def compute_modulation_index(
    df_freeze: pd.DataFrame,
    group_by: tuple[str, ...] = ("animal", "genotype"),
    require_min_cs: int = 1,
    require_min_noncs: int = 1,
    segment_col: str = "segment",
    freeze_col: str = "freeze_pct",
    cs_regex: str = r"^cs_(\d+)$",
    noncs_regex: str = r"^noncs_(\d+)$",
) -> pd.DataFrame:
    """
    Compute modulation index (MI) from per-segment freezing ratios.

    MI per group is:
        MI = (sum_cs - sum_noncs) / (sum_cs + sum_noncs)

    Parameters
    ----------
    df_freeze : pandas.DataFrame
        Tidy dataframe with columns including group_by, `segment_col`, and `freeze_col`.
        `freeze_col` is expected to be a ratio (0..1) or percent (0..100); MI is scale-invariant.
    group_by : tuple of str, default=("animal","genotype")
        Grouping columns.
    require_min_cs : int, default=1
        Minimum number of CS segments required.
    require_min_noncs : int, default=1
        Minimum number of NONCS segments required.
    segment_col : str, default="segment"
        Segment label column.
    freeze_col : str, default="freeze_pct"
        Freezing ratio/percent column.
    cs_regex : str, default="^cs_(\\d+)$"
        Regex to identify CS segments.
    noncs_regex : str, default="^noncs_(\\d+)$"
        Regex to identify NONCS segments.

    Returns
    -------
    pandas.DataFrame
        Dataframe with:
        - group_by columns
        - mod_index
        - sum_cs
        - sum_noncs
        - n_cs
        - n_noncs
    """
    required_cols = set(group_by) | {segment_col, freeze_col}
    missing = required_cols.difference(df_freeze.columns)
    if missing:
        raise ValueError(f"df_freeze missing required columns: {sorted(missing)}")

    df = df_freeze.copy()
    df[segment_col] = df[segment_col].astype(str)
    df[freeze_col] = pd.to_numeric(df[freeze_col], errors="coerce")

    cs_df = df.loc[df[segment_col].str.match(cs_regex, na=False)].dropna(subset=[freeze_col]).copy()
    non_df = df.loc[df[segment_col].str.match(noncs_regex, na=False)].dropna(subset=[freeze_col]).copy()

    if cs_df.empty and non_df.empty:
        return pd.DataFrame(columns=[*group_by, "mod_index", "sum_cs", "sum_noncs", "n_cs", "n_noncs"])

    group_cols = list(group_by)

    def _group_keys(d: pd.DataFrame) -> set[tuple]:
        if d.empty:
            return set()
        return set(d[group_cols].itertuples(index=False, name=None))

    keys = _group_keys(cs_df) | _group_keys(non_df)

    rows: list[dict] = []
    for gvals in keys:
        gvals_tuple = gvals if isinstance(gvals, tuple) else (gvals,)
        payload = dict(zip(group_cols, gvals_tuple))

        mask_cs = np.ones(len(cs_df), dtype=bool)
        mask_non = np.ones(len(non_df), dtype=bool)
        for col, val in zip(group_cols, gvals_tuple):
            if not cs_df.empty:
                mask_cs &= (cs_df[col] == val)
            if not non_df.empty:
                mask_non &= (non_df[col] == val)

        sub_cs = cs_df.loc[mask_cs] if not cs_df.empty else cs_df
        sub_non = non_df.loc[mask_non] if not non_df.empty else non_df

        n_cs = int(sub_cs.shape[0])
        n_non = int(sub_non.shape[0])

        if (n_cs < require_min_cs) or (n_non < require_min_noncs):
            rows.append({**payload, "mod_index": np.nan, "sum_cs": np.nan, "sum_noncs": np.nan, "n_cs": n_cs, "n_noncs": n_non})
            continue

        sum_cs = float(sub_cs[freeze_col].sum())
        sum_noncs = float(sub_non[freeze_col].sum())

        denom = sum_cs + sum_noncs
        mi = (sum_cs - sum_noncs) / denom if denom > 0 else np.nan

        rows.append({**payload, "mod_index": mi, "sum_cs": sum_cs, "sum_noncs": sum_noncs, "n_cs": n_cs, "n_noncs": n_non})

    return pd.DataFrame(rows)

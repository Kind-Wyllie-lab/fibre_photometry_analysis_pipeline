from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stats import _two_group_independent_test, _p_to_stars, _annotate_two_group_stars


input_dir = Path(r"E:\Fibre_photmetry\auc group means")
output_dir = input_dir / "plots"
output_dir.mkdir(exist_ok=True)

group_order = ["wt", "het", "gcamp"]
group_colors = {
    "wt": "black",
    "het": "blue",
    "gcamp": "green",
}

all_files = sorted(input_dir.glob("*.csv"))

global_ymin = np.inf
global_ymax = -np.inf
summary_by_file = []
stats_rows = []

for file in all_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    required_columns = {"group", "event_index", "group_mean_auc"}
    if not required_columns.issubset(df.columns):
        print(f"Skipping {file.name}: missing required columns")
        continue

    df = df.copy()
    df["group"] = df["group"].astype(str)
    df = df.loc[df["group"].isin(group_order)].copy()

    if df.empty:
        continue

    summary = (
        df.groupby("group", as_index=False)
        .agg(
            mean_auc=("group_mean_auc", "mean"),
            std_auc=("group_mean_auc", "std"),
            n_events=("event_index", "nunique"),
        )
    )

    summary["sem_auc"] = summary["std_auc"] / np.sqrt(summary["n_events"])
    summary["group"] = pd.Categorical(summary["group"], categories=group_order, ordered=True)
    summary = summary.sort_values("group").dropna(subset=["mean_auc"]).reset_index(drop=True)

    if summary.empty:
        continue

    summary_by_file.append((file, df, summary))

    ymin = np.nanmin(summary["mean_auc"] - summary["sem_auc"].fillna(0))
    ymax = np.nanmax(summary["mean_auc"] + summary["sem_auc"].fillna(0))

    global_ymin = min(global_ymin, ymin)
    global_ymax = max(global_ymax, ymax)

if not summary_by_file:
    raise ValueError(f"No valid CSV files found in {input_dir}")

y_padding = 0.1 * (global_ymax - global_ymin if global_ymax > global_ymin else 1.0)
shared_ylim = (global_ymin - y_padding, global_ymax + y_padding)

for file, df, summary in summary_by_file:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    x_positions = np.arange(len(summary))

    ax.bar(
        x_positions,
        summary["mean_auc"].to_numpy(dtype=float),
        yerr=summary["sem_auc"].fillna(0).to_numpy(dtype=float),
        capsize=4,
        color=[group_colors[group_name] for group_name in summary["group"].astype(str)],
        edgecolor="black",
        linewidth=1.0,
        width=0.7,
        zorder=2,
    )

    for x_position, group_name in zip(x_positions, summary["group"].astype(str)):
        group_event_values = df.loc[df["group"] == group_name, "group_mean_auc"].to_numpy(dtype=float)

        if group_event_values.size == 0:
            continue

        if group_event_values.size == 1:
            x_jittered = np.array([x_position], dtype=float)
        else:
            x_jittered = x_position + np.linspace(-0.08, 0.08, group_event_values.size)

        ax.scatter(
            x_jittered,
            group_event_values,
            s=45,
            color=group_colors[group_name],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=group_name,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary["group"].astype(str), fontsize=15)
    ax.set_ylabel("Mean AUC across events", fontsize=18)
    ax.set_xlabel("Group", fontsize=16)
#    ax.set_ylim(shared_ylim)
    ax.set_title(file.stem.replace("_", " "), fontsize=12, pad=12)

    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=15, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    legend_title = "Group"

    wt_event_values = df.loc[df["group"] == "wt", "group_mean_auc"].to_numpy(dtype=float)
    het_event_values = df.loc[df["group"] == "het", "group_mean_auc"].to_numpy(dtype=float)

    if wt_event_values.size >= 2 and het_event_values.size >= 2:
        stats_result = _two_group_independent_test(wt_event_values, het_event_values, alpha=0.05)
        stars = _p_to_stars(stats_result["p"])

        wt_mask = summary["group"].astype(str) == "wt"
        het_mask = summary["group"].astype(str) == "het"
        wt_x = x_positions[wt_mask][0]
        het_x = x_positions[het_mask][0]

        _annotate_two_group_stars(ax, x_positions=(wt_x, het_x), stars=stars)

        is_significant = bool(stats_result["p"] < 0.05)
        if is_significant:
            legend_title = f"Group\nwt vs het: p={stats_result['p']:.3g} {stars}"

        stats_rows.append(
            {
                "file_name": file.name,
                "comparison": "wt_vs_het",
                "test": stats_result["test"],
                "p_value": stats_result["p"],
                "stars": stars,
                "significant": is_significant,
                "normal": stats_result["normal"],
                "equal_var": stats_result["equal_var"],
                "n_wt": stats_result["n1"],
                "n_het": stats_result["n2"],
            }
        )

        print(
            f"{file.name} | wt vs het | "
            f"{stats_result['test']} | p={stats_result['p']:.3g} | "
            f"normal={stats_result['normal']} | "
            f"equal_var={stats_result['equal_var']} | "
            f"n={stats_result['n1']} vs {stats_result['n2']}"
        )
    else:
        stats_rows.append(
            {
                "file_name": file.name,
                "comparison": "wt_vs_het",
                "test": "not_run",
                "p_value": np.nan,
                "stars": "",
                "significant": False,
                "normal": np.nan,
                "equal_var": np.nan,
                "n_wt": wt_event_values.size,
                "n_het": het_event_values.size,
            }
        )
        print(f"{file.name} | wt vs het stats skipped: not enough event values")

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    ax.legend(
        unique_handles,
        unique_labels,
        title=legend_title,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"{file.stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

stats_dataframe = pd.DataFrame(stats_rows)
stats_output_path = output_dir / "wt_vs_het_stats_summary.csv"
stats_dataframe.to_csv(stats_output_path, index=False)
print(f"Saved stats summary: {stats_output_path}")

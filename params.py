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

#params.py
from pathlib import Path
from typing import Literal

base_path = Path(r"/media/prignane/data_fast/Fibre_photmetry")

sessions =  ["Cond"]

def animal_output_dirs(animal_num: str, session:str):
    """Return (figures_dir, csv_dir). Create if missing."""
    out_root = base_path / f"{animal_num}/{session}/output"
    out_root.mkdir(exist_ok=True)
    return out_root

sample_rate_hz = 30.0

round_decimals = 4

# === SIGNALS ===
calcium_channel = "CH1-470"
ref_channel = "CH1-410"
baseline_samples = 1000

event_gap_ms = 5000

time_pre_event_s = 10
time_post_event_s = 40

selected_clusters = list(range(0, 20)) #SELECTED_CLUSTERS = list(range(5, 15)), select Clusters 5-14


"""SMOOTHNESS = 15  # Manual default
OFFSET_SUB_470 = 0  # Dark fiber /2
OFFSET_SUB_410 = 0
BASELINE_METHOD = 'PLS'  # or 'Exponential'
USE_MOTION_CORR = True"""

# === FIGURE CORE ===
figure_dpi = 300
figure_format = "png"
figure_size_trace = (14, 6)
figure_size_peri = (10, 6)

save_figures = False
preview_figures = True
#FIGURES_DIR = OUTPUT_DIR / "figures"
#FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# === COLORS & LINES ===
color_470_nm = "#1f77b4"      # Blue: calcium (470nm)
color_410_nm = "#ff7f0e"      # Orange: isosbestic (410nm)

color_dff = "purple"       # Processed ΔF/F
color_zscore = "green"     # Z-normalized signal
color_peri_mean = "darkcyan"  # Peri-event average line
color_event_onset = "red"  # Event time markers

alpha_event_lines = 0.5    # Vertical event line transparency
alpha_sem_fill = 0.4       # SEM shading transparency
lw_trace = 0.1             # Raw trace thickness
lw_peri_mean = 1.0         # Mean line thickness
lw_event_marker = 0.4      # Event marker thickness

# === HEATMAP (SIMPLE) ===
heatmap_figsize = (12, 6)
heatmap_cmap_dff = "viridis"
heatmap_cmap_z = "viridis"

# vmin/vmax: None = auto based on data
heatmap_vmin_dff = None
heatmap_vmax_dff = None
heatmap_vmin_z = None
heatmap_vmax_z = None

# === TICKS (SEPARATE X/Y) ===
# X-axis (time)
xtick_fontsize = 11
xtick_direction = "out"
xtick_width_major = 1.4
xtick_width_minor = 0.9
xtick_length_major = 5
xtick_length_minor = 2.5
xtick_nbins = None   # default max ticks on traces

# Y-axis (signal / trials)
ytick_fontsize = 10
ytick_direction = "out"
ytick_width_major = 1.0
ytick_width_minor = 0.6
ytick_length_major = 3.5
ytick_length_minor = 1.8
ytick_nbins = None

# === HEATMAP TICKS ===
heatmap_xtick_major = 2   # seconds between major ticks
heatmap_xtick_minor = 1.0   # seconds between minor ticks

heatmap_ytick_major_step = 1  # every Nth trial gets major tick label
heatmap_ytick_major_step = 1  # every Nth trial gets major tick label
heatmap_ytick_minor_step = 1  # minor grid lines every N trials
heatmap_show_all_ylabels = True  # True=label every trial, False=every Nth

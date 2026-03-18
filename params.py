#params.py
from pathlib import Path
from typing import Literal

# === PATHS ===
base_path = Path(r"/media/prignane/data_fast/Fibre_photmetry")
animal_id = "Rat_4987"  # Works for "4879" or "Rat_4879"

output_dir = base_path / f"{animal_id}/output"
output_dir.mkdir(exist_ok=True, parents=True)

# === CONFIGURATION PARAMETERS ===
session: Literal["Hab1", "Hab2", "Cond", "Recall"] = "Recall"
events_type: Literal["fluorescence", "fluorescence_event"] = "fluorescence"

output_dir = base_path / f"{animal_id}/{session}/output"
output_dir.mkdir(exist_ok=True, parents=True)

def animal_output_dirs(animal_num: str) -> tuple[Path, Path]:
    """Return (figures_dir, csv_dir). Create if missing."""
    out_root = base_path / f"{animal_num}_output"
    figs_dir = out_root / "figures"
    csv_dir = out_root / "csv"

    created = False
    if not figs_dir.exists():
        figs_dir.mkdir(parents=True)
        created = True
    if not csv_dir.exists():
        csv_dir.mkdir(parents=True)
        created = True

    status = "created" if created else "exists"
    print(f"{animal_num}_output/ [{status}]")

    return figs_dir, csv_dir


def select_data_files(recording_type: str, events_type: str, animal_id: str = animal_id,
                      base_path: Path = base_path) -> list[Path]:
    """Select fluorescence CSV from animalid_cleaned folder.

    - 'fluorescence': picks aligned/regular file (EXCLUDES *-unaligned.csv)
    - 'fluorescence_event': any fluorescence CSV (first match)
    """
    cleaned_dir = base_path / f"{animal_id}"
    root_dir = cleaned_dir / recording_type

    print(f"DEBUG: Looking in '{root_dir}' (events: {events_type})")

    all_files = list(root_dir.rglob("*")) if root_dir.exists() else []

    # Case-insensitive fluorescence search
    data_files = sorted([f for f in all_files if
                         ('fluorescence' in f.name.lower() or 'fluo' in f.name.lower())
                         and f.suffix.lower() == '.csv'])

    if not data_files:
        print(f"Warning: No fluorescence CSV under {root_dir}")
        return []

    print(f"Available: {[f.name for f in data_files]}")

    # Selection by TYPE_OF_EVENTS
    if events_type == "fluorescence":
        # EXCLUDE unaligned - pick first NON-unaligned
        selected = next((f for f in data_files if "unaligned" not in f.name.lower()), None)
        if selected is None:
            print("Warning: No non-unaligned file found, using first")
            selected = data_files[0]
        print(f"Selected (no-unaligned): {selected.name}")
    else:  # fluorescence_event
        selected = data_files[0]
        print(f"Selected (first): {selected.name}")

    return [selected]



# === FILES SELECTION ===
data_files = select_data_files(session, events_type)
figures_dir, csv_dir = animal_output_dirs(animal_id)
# === SAMPLING ===
sample_rate_hz = 60.0
dt_ms = 1000.0 / sample_rate_hz #TODO clarify

# === PREPROCESSING ===
skiprows_json = 1
event_mapping = {
    "0": 0,
    "Input1*2*0": 1,
    "Input1*2*1": 2,
}
selected_cols = ["TimeStamp", "Events", "CH1-410", "CH1-470", "CH1-560"]
round_decimals = 4

# === SIGNALS ===
calcium_channel = "CH1-470"
ref_channel = "CH1-410"
baseline_samples = 1000
zscore_window = 300
sync_signal_to_first_event = True

# === EVENTS / EPOCHS ===
event_gap_ms = 5000
skip_first_event = True
epoch_min_length_samples = 100

time_pre_event_s = 2
time_post_event_s = 10.0

baseline_start_s = -5.0
baseline_end_s = 0.0

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
heatmap_ytick_minor_step = 1  # minor grid lines every N trials
heatmap_show_all_ylabels = True  # True=label every trial, False=every Nth

# Optional: different density on heatmaps only

heatmap_xtick_nbins = None
heatmap_ytick_nbins = None

# === LOG SUMMARY ===
print("Params loaded:")
#print(f"  Output: {OUTPUT_DIR}")
print(f"  SR: {sample_rate_hz}Hz, Baseline: {baseline_samples} samples")
print(f"  Epoch window: {time_pre_event_s}-{time_post_event_s}s")


# if __name__ == "__main__":
#     print("=" * 50)
#     print(f"ANIMAL_ID: {animal_id}")
#     print(f"TYPE_OF_RECORDING: {recording_type}")
#     print(f"TYPE_OF_EVENTS: {events_type}")
#
#     data_files = select_data_files(recording_type, events_type)
#     print(f"DATA_FILES ({len(data_files)}): {[f.name for f in data_files]}")
#
#     figs_dir, csv_dir = animal_output_dirs(animal_id)
#     print(f"OUTPUT FIGS: {figs_dir}")
#     print(f"OUTPUT CSV:  {csv_dir}")
#     print("=" * 50)
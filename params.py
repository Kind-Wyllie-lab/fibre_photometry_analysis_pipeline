from pathlib import Path
from typing import Literal

# === PATHS ===
BASE_PROJECT_PATH = Path(r"\\cmvm.datastore.ed.ac.uk\cmvm\sbms\users\s2830349\Win7\Desktop\Fibre_phot_project_2026")
ANIMAL_ID = "4987"  # Works for "4879" or "Rat_4879"
OUTPUT_DIR = BASE_PROJECT_PATH / "output"

# === CONFIGURATION PARAMETERS ===
TYPE_OF_RECORDING: Literal["Hab1", "Hab2", "Cond", "Recall"] = "Hab1"
TYPE_OF_EVENTS: Literal["fluorescence", "fluorescence_event"] = "fluorescence"

def animal_output_dirs(animal_num: str) -> tuple[Path, Path]:
    """Return (figures_dir, csv_dir). Create if missing."""
    out_root = BASE_PROJECT_PATH / f"{animal_num}_output"
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

def select_data_files(recording_type: str, events_type: str, animal_id: str = ANIMAL_ID, base_path: Path = BASE_PROJECT_PATH) -> list[Path]:
    root_dir = base_path / "data" / animal_id / recording_type
    print(f"DEBUG: root_dir '{root_dir}' exists: {root_dir.exists()}")

    all_files = []
    if root_dir.exists():
        all_files = list(root_dir.rglob("*"))  # All files recursively
        print(f"DEBUG: Found {len(all_files)} total files/subdirs")
        for f in all_files[:5]:  # First 5
            print(f"  {f.relative_to(base_path)} {'DIR' if f.is_dir() else 'FILE'}")

    # Case-insensitive fluorescence search
    data_files = [f for f in all_files if
                  ('fluorescence' in f.name.lower() or 'fluo' in f.name.lower())
                  and f.suffix.lower() == '.csv']

    if not data_files:
        print(f"Warning: No fluorescence CSV files under {root_dir}")
        print("  Try: TYPE_OF_RECORDING='Hab1/Hab1'")
    else:
        print(f"Found {len(data_files)} fluorescence files")

    return sorted(data_files)


if __name__ == "__main__":
    print("=" * 50)
    print(f"ANIMAL_ID: {ANIMAL_ID}")
    print(f"TYPE_OF_RECORDING: {TYPE_OF_RECORDING}")
    print(f"TYPE_OF_EVENTS: {TYPE_OF_EVENTS}")

    DATA_FILES = select_data_files(TYPE_OF_RECORDING, TYPE_OF_EVENTS)
    print(f"DATA_FILES ({len(DATA_FILES)}): {[f.name for f in DATA_FILES]}")

    figs_dir, csv_dir = animal_output_dirs(ANIMAL_ID)
    print(f"OUTPUT FIGS: {figs_dir}")
    print(f"OUTPUT CSV:  {csv_dir}")
    print("=" * 50)


""" 
# === SAMPLING ===
SAMPLERATE_HZ = 60.0
DT_MS = 1000.0 / SAMPLERATE_HZ

# === PREPROCESSING ===
SKIPROWS_JSON = 1
EVENT_MAPPING = {
    "0": 0,
    "Input1*2*0": 1,
    "Input1*2*1": 2,
}
SELECTED_COLS = ["TimeStamp", "Events", "CH1-410", "CH1-470", "CH1-560"]
ROUND_DECIMALS = 4

# === SIGNALS ===
CALCIUM_CHANNEL = "CH1-470"
REF_CHANNEL = "CH1-410"
BASELINE_SAMPLES = 1000
ZSCORE_WINDOW = 300
SYNC_SIGNAL_TO_FIRST_EVENT = True

# === EVENTS / EPOCHS ===
EVENT_GAP_MS = 5000
SKIP_FIRST_EVENT = True
EPOCH_MIN_LENGTH_SAMPLES = 100

T_PRE_EVENT_S = 2
T_POST_EVENT_S = 10.0
SELECTED_CLUSTERS = list(range(0,4)) #SELECTED_CLUSTERS = list(range(5, 15)), select Clusters 5-14


# === FIGURE CORE ===
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURE_SIZE_TRACE = (14, 6)
FIGURE_SIZE_PERI = (10, 6)

SAVE_FIGURES = False
PREVIEW_FIGURES = True
#FIGURES_DIR = OUTPUT_DIR / "figures"
#FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# === COLORS & LINES ===
COLOR_470 = "#1f77b4"      # Blue: calcium (470nm)
COLOR_410 = "#ff7f0e"      # Orange: isosbestic (410nm)

COLOR_DFF = "purple"       # Processed ΔF/F
COLOR_ZSCORE = "green"     # Z-normalized signal
COLOR_PERI_MEAN = "darkcyan"  # Peri-event average line
COLOR_EVENT_ONSET = "red"  # Event time markers

ALPHA_EVENT_LINES = 0.5    # Vertical event line transparency
ALPHA_SEM_FILL = 0.4       # SEM shading transparency
LW_TRACE = 0.6             # Raw trace thickness
LW_PERI_MEAN = 1.0         # Mean line thickness
LW_EVENT_MARKER = 0.4      # Event marker thickness

# === HEATMAP (SIMPLE) ===
HEATMAP_FIGSIZE = (12, 6)
HEATMAP_CMAP_DFF = "viridis"
HEATMAP_CMAP_Z = "viridis"

# vmin/vmax: None = auto based on data
HEATMAP_VMIN_DFF = None
HEATMAP_VMAX_DFF = None
HEATMAP_VMIN_Z = None
HEATMAP_VMAX_Z = None

# === TICKS (SEPARATE X/Y) ===
# X-axis (time)
XTICK_FONTSIZE = 11
XTICK_DIRECTION = "out"
XTICK_WIDTH_MAJOR = 1.4
XTICK_WIDTH_MINOR = 0.9
XTICK_LENGTH_MAJOR = 5
XTICK_LENGTH_MINOR = 2.5
XTICK_NBINS = None   # default max ticks on traces

# Y-axis (signal / trials)
YTICK_FONTSIZE = 10
YTICK_DIRECTION = "out"
YTICK_WIDTH_MAJOR = 1.0
YTICK_WIDTH_MINOR = 0.6
YTICK_LENGTH_MAJOR = 3.5
YTICK_LENGTH_MINOR = 1.8
YTICK_NBINS = None

# === HEATMAP TICKS ===
HEATMAP_XTICK_MAJOR = 2   # seconds between major ticks
HEATMAP_XTICK_MINOR = 1.0   # seconds between minor ticks

HEATMAP_YTICK_MAJOR_STEP = 1  # every Nth trial gets major tick label
HEATMAP_YTICK_MINOR_STEP = 1  # minor grid lines every N trials
HEATMAP_SHOW_ALL_YLABELS = True  # True=label every trial, False=every Nth

# Optional: different density on heatmaps only

HEATMAP_XTICK_NBINS = None
HEATMAP_YTICK_NBINS = None

# === LOG SUMMARY ===
print("Params loaded:")
#print(f"  Output: {OUTPUT_DIR}")
print(f"  SR: {SAMPLERATE_HZ}Hz, Baseline: {BASELINE_SAMPLES} samples")
print(f"  Epoch window: {T_PRE_EVENT_S}-{T_POST_EVENT_S}s")
"""
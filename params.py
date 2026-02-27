from pathlib import Path

# === PATHS ===
'''DATA_ROOT = Path("/Users/Lou/Desktop/Fibre_phto_python_dir")
OUTPUT_DIR = DATA_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

RAW_FILES = sorted(DATA_ROOT.glob("/Users/Lou/Library/CloudStorage/OneDrive-UniversityofEdinburgh/M2/fibre_photometry/20260212_Test_Lou/4879/4879_Fluorescence_Event_freezing_cs.csv")) #RAW_FILES = sorted(DATA_ROOT.glob("**/Fluorescence*.csv"))

DATA_FILES = [str(f) for f in RAW_FILES]
print(f"Data root: {DATA_ROOT}")
print(f"Found Fluorescence files: {len(DATA_FILES)}")
for f in RAW_FILES:
    print(f"  {f.relative_to(DATA_ROOT)}")
if not DATA_FILES:
    raise ValueError(f"No Fluorescence*.csv under {DATA_ROOT}")
print(f"Using first: {Path(DATA_FILES[0]).name}")'''
# File Discovery Block (Replace existing)
DATA_ROOT = Path("/Users/Lou/Library/CloudStorage/OneDrive-UniversityofEdinburgh/M2/fibre_photometry/20260212_Test_Lou/4879")
OUTPUT_DIR = DATA_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Direct path or glob pattern
RAW_FILES = [DATA_ROOT / "4879_Fluorescence_Event_freezing_cs.csv"]  # Single file
# Or glob siblings: RAW_FILES = sorted(DATA_ROOT.glob("4879_Fluorescence*.csv"))

DATA_FILES = [str(f) for f in RAW_FILES]
print(f"Data root: {DATA_ROOT}")
print(f"Found files: {len(DATA_FILES)}")
for f in RAW_FILES:
    print(f"  {f.name}")  # Just filename for cleaner output
if not DATA_FILES:
    raise ValueError(f"No files found at {DATA_ROOT}")
print(f"Using: {Path(DATA_FILES[0]).name}")

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
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

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
print(f"  Output: {OUTPUT_DIR}")
print(f"  SR: {SAMPLERATE_HZ}Hz, Baseline: {BASELINE_SAMPLES} samples")
print(f"  Epoch window: {T_PRE_EVENT_S}-{T_POST_EVENT_S}s")

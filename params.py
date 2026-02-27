from pathlib import Path

DATA_ROOT = Path("/Users/Lou/Desktop/Fibre_phto_python_dir")  # YOUR FULL PATH
OUTPUT_DIR = DATA_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# RECURSIVE SEARCH - finds Fluorescence*.csv anywhere under DATA_ROOT
RAW_FILES = sorted(DATA_ROOT.glob("**/Fluorescence*.csv"))
DATA_FILES = [str(f) for f in RAW_FILES]
print(f"Data root: {DATA_ROOT}")
print(f"Found Fluorescence files: {len(DATA_FILES)}")
for f in RAW_FILES:
    print(f"  {f.relative_to(DATA_ROOT)}")

if not DATA_FILES:
    print("No Fluorescence*.csv under", DATA_ROOT)
    print("Check: ls -R", DATA_ROOT, "| grep Fluorescence")
    raise ValueError("No data files found")

print(f"Using first: {Path(DATA_FILES[0]).name}")


# === SAMPLING ===
SAMPLERATE_HZ = 60.0  # Override auto-detect if known
DT_MS = 1000.0 / SAMPLERATE_HZ  # Derived

# === PREPROCESSING ===
SKIPROWS_JSON = 1
EVENT_MAPPING = {
    '0': 0,
    'Input1*2*0': 1,  # Red events
    'Input1*2*1': 2,  # Blue events
}
SELECTED_COLS = ['TimeStamp', 'Events', 'CH1-410', 'CH1-470', 'CH1-560']
ROUND_DECIMALS = 4

# === SIGNALS ===
CALCIUM_CHANNEL = 'CH1-470'  # Calcium
REF_CHANNEL = 'CH1-410'  # Motion
BASELINE_SAMPLES = 1000
ZSCORE_WINDOW = 300  # Rolling if implemented
SYNC_SIGNAL_TO_FIRST_EVENT = True  # Trim signal to start at first event timestamp

# === EVENTS ===
EVENT_GAP_MS = 5000  # Cluster threshold
T_PRE_EVENT_S = 10.0 #peri_events params
T_POST_EVENT_S = 40.0
SKIP_FIRST_EVENT = True
SELECTED_CLUSTERS = list(range(1, 5))
# Examples (copy-paste and uncomment the one you want):
# SELECTED_CLUSTERS = [0, 1, 2]              # first 3 clusters only
# SELECTED_CLUSTERS = None                   # all clusters (no filtering)
# SELECTED_CLUSTERS = list(range(1, 21))     # all except first (clusters 1–20)
# SELECTED_CLUSTERS = [c for c in range(21) if c != 5]  # all except cluster 5
# SELECTED_CLUSTERS = list(range(3, 11))     # clusters 3–10 only
#SELECTED_CLUSTERS: list[int] | None = None   # default: use all clusters
# === EPOCHS ===
EPOCH_MIN_LENGTH_SAMPLES = 100

# === PLOT PARAMS ===
FIGURE_DPI = 300
FIGURE_SIZE_TRACE = (14, 6)    # Full session trace
FIGURE_SIZE_PERI = (10, 6)     # Peri-event average (12, 5) would be wider

# Raw channel colours
COLOR_470 = "#1f77b4"   # blue  — 470nm calcium signal
COLOR_410 = "#ff7f0e"   # orange — 410nm isosbestic reference

COLOR_DFF = "purple"
COLOR_ZSCORE = "green"
COLOR_PERI_MEAN = "darkcyan"
COLOR_EVENT_0 = "red"          # Input1*2*0 marker
COLOR_EVENT_1 = "blue"         # Input1*2*1 marker
COLOR_EVENT_ONSET = "red"      # Peri-event t=0 line

ALPHA_EVENT_LINES = 0.5        # Transparency of event markers on full trace
ALPHA_SEM_FILL = 0.4           # Transparency of SEM shading
LW_TRACE = 0.6                 # Line width full trace
LW_PERI_MEAN = 1.0             # Line width peri-event mean
LW_EVENT_MARKER = 0.4          # Line width event markers on full trace

# === HEATMAP PARAMS ===
HEATMAP_COLORMAP_DFF = "viridis"     # e.g. "viridis", "YlGnBu_r", "turbo"
HEATMAP_COLORMAP_Z   = "RdBu_r"      # diverging for z-score

# value limits; use None for auto
HEATMAP_VMIN_DFF = None           # e.g. 0.9
HEATMAP_VMAX_DFF = None            # e.g. 1.1
HEATMAP_VMIN_Z   = None
HEATMAP_VMAX_Z   = None

#TODO : check what these params are for

# trial sorting and smoothing
HEATMAP_SORT_TRIALS = False           # sort by response size
HEATMAP_SORT_WINDOW = (0.0, 10.0)    # seconds, window used for sorting
HEATMAP_SMOOTH_WIN  = 5              # moving-average window (samples); 1 = off

# figure size
FIGURE_SIZE_HEATMAP = (12, 5)        # width, height
# === HEATMAP SPECS ===
HEATMAP_FIGSIZE = (12, 6)       # Width for time-series; height for trials
HEATMAP_DPI = 300               # Publication quality

# ΔF/F (sequential, low=blue → high=yellow) "Spectral_r", "viridis", "plasma", "turbo"
HEATMAP_CMAP_DFF = "Spectral_r"
# Z-score (diverging)
HEATMAP_CMAP_Z = "RdBu_r"


# Grid & style
HEATMAP_ASPECT = "auto"         # "auto" Or "equal" for square cells
HEATMAP_SMOOTHING = 1          # Time points to smooth (1=off)
HEATMAP_TRIAL_SORTING = True   # Sort trials by peak response
HEATMAP_COLORBAR_SHRINK = 0.8  # Compact colorbars
HEATMAP_GRID_ALPHA = 0.3       #  trial separators

HEATMAP_SORT_WINDOW_S = None   # Post-event window for sorting(0, 0)

# Axes
HEATMAP_YTICKS_EVERY = 1 # Sparse labels (e.g., every 5th trial)


FIGURE_FORMAT = "png"   # Options: "png", "pdf", "svg", "tiff"
SAVE_FIGURES = False     # Save PNG to output/figures/
PREVIEW_FIGURES = True
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
# === Derived ===
print("Params loaded:")
print(f"  Output: {OUTPUT_DIR}")
print(f"  SR: {SAMPLERATE_HZ}Hz, Baseline: {BASELINE_SAMPLES}samples")
print(f"  Epoch window: {T_PRE_EVENT_S}-{T_POST_EVENT_S}s")
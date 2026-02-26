# QUICK FIX params.py - Auto-project root + explicit check
from pathlib import Path
import numpy as np

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
SKIPROWS_FIRST_EVENT = 1
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

# === EVENTS ===
EVENT_GAP_MS = 5000  # Cluster threshold
T_PRE_EVENT_S = 2.0
T_POST_EVENT_S = 48.0
SKIP_FIRST_EVENT = True
# === EPOCHS ===
EPOCH_MIN_LENGTH_SAMPLES = 100

# === PLOT PARAMS ===
FIGURE_DPI = 150
FIGURE_SIZE_TRACE = (14, 6)    # Full session trace
FIGURE_SIZE_PERI = (10, 6)     # Peri-event average

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

FIGURE_FORMAT = "png"   # Options: "png", "pdf", "svg", "tiff"
SAVE_FIGURES = False     # Save PNG to output/figures/
PREVIEW_FIGURES = True
# === Derived ===
print("Params loaded:")
print(f"  Output: {OUTPUT_DIR}")
print(f"  SR: {SAMPLERATE_HZ}Hz, Baseline: {BASELINE_SAMPLES}samples")
print(f"  Epoch window: {T_PRE_EVENT_S}-{T_POST_EVENT_S}s")
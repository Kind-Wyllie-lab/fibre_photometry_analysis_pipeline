import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#F is the raw fluorescence signal of interest (typically 470nm for green calcium indicators), while F0 is the baseline reference—either the median of a user-defined time window, the fitted baseline (after photobleaching correction), or control channel data (410nm or 560nm motion-corrected/fitted); dF/F normalizes changes relative to this F0 for comparing activity across trials.

df_process = pd.read_csv("/Users/Lou/Desktop/cleaned_fluorescence.csv")
ch470= df_process["CH1-470"].values
ch410= df_process["CH1-410"].values
corrected_signal = ch470 - ch410
"dF/F per OFRS manual: (F - F0)/F0, ΔF/F = (F<sub>470</sub> - F<sub>410</sub>) / F<sub>0</sub> where F<sub>0</sub> = median (or mean) of (F<sub>470</sub> - F<sub>410</sub>) over your baseline window (first N samples)."
f0 = (ch470 - ch410)[:1000]
dff = (ch470 - ch410) / np.median(f0)
zscore = ((ch470 - ch410) - np.mean(f0)) / np.std(f0)


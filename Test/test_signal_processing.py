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


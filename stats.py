import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def _annotate_two_group_stars(ax: plt.Axes, x_positions=(0, 1), y_pad_ratio=0.06, stars: str = "") -> None:
    """
    Draw a bracket with stars over two x positions on ax.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to annotate.
    x_positions : tuple, default=(0, 1)
        X positions of the two groups.
    y_pad_ratio : float, default=0.06
        Extra headroom (fraction of y-range).
    stars : str, default=""
        Star string, e.g. "*", "**", "***". If empty, nothing is drawn.
    """
    if not stars:
        return

    x0, x1 = x_positions
    y_min, y_max = ax.get_ylim()
    h = (y_max - y_min)

    y = y_max - 0.02 * h
    y_bracket = y_max - 0.03 * h

    ax.plot(
        [x0, x0, x1, x1],
        [y_bracket, y_bracket + 0.01 * h, y_bracket + 0.01 * h, y_bracket],
        color="k",
        lw=1,
    )
    ax.text(
        (x0 + x1) / 2,
        y,
        stars,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_ylim(y_min, y_max + y_pad_ratio * h)


def _p_to_stars(p: float) -> str:
    """
    Convert p-value to stars.
    """
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def _two_group_independent_test(x, y, alpha=0.05, normality_max_n=5000):
    """
    Independent two-sample comparison:
      - If both normal (Shapiro) -> Levene (median) to check equal variances:
          * equal -> Student t-test
          * unequal -> Welch t-test
      - Else -> Mann–Whitney U (two-sided) with method='auto'

    Returns dict with: test, p, normal, equal_var, n1, n2
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return dict(test="NA", p=np.nan, normal=False, equal_var=False, n1=n1, n2=n2)

    # Normality per group
    try:
        p_norm1 = stats.shapiro(x).pvalue if n1 <= normality_max_n else 1.0
        p_norm2 = stats.shapiro(y).pvalue if n2 <= normality_max_n else 1.0
        normal = (p_norm1 > alpha) and (p_norm2 > alpha)
    except Exception:
        normal = False

    equal_var = False
    if normal:
        # Homoscedasticity (Brown–Forsythe via Levene with median)
        try:
            p_lev = stats.levene(x, y, center="median").pvalue
            equal_var = (p_lev > alpha)
        except Exception:
            equal_var = False

    # Choose test
    if normal:
        t_res = stats.ttest_ind(x, y, equal_var=equal_var)
        p = float(t_res.pvalue)
        test_name = "t (Student)" if equal_var else "t (Welch)"
    else:
        try:
            u_res = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
            p = float(u_res.pvalue)
            test_name = "Mann–Whitney"
        except Exception:
            t_res = stats.ttest_ind(x, y, equal_var=False)
            p = float(t_res.pvalue)
            test_name = "t (Welch fallback)"

    return dict(test=test_name, p=p, normal=normal, equal_var=equal_var, n1=n1, n2=n2)

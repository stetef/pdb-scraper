"""Robust linear fit helper shared across the plotting scripts.

We use Theil-Sen (median of pairwise slopes) instead of OLS so a single
outlier point can't drag the line away from the bulk of the data --
visible most clearly in HisNE: CG-CB and similar tight clusters with
one stray observation. See notes_robust_fits.md for the why.

We report two complementary numbers per fit:

  pseudo-R^2 (Efron's R^2) -- 1 - SS_res / SS_tot against the
    Theil-Sen predictions. Same units as OLS R^2 but can go negative
    when the line is a worse predictor than mean(y) (e.g., when a
    leverage outlier dominates SS_res). Negative values are NOT a
    bug; they are the diagnostic that "the line is overkill, just
    use the mean."
  Kendall's tau + p-value -- rank-based monotonic association
    measure, robust to outliers. Provides the "is there a real
    monotonic trend?" question independent of the line. Theil-Sen
    is mathematically the slope that drives tau between residuals
    and x to zero (Sen 1968), so tau is a natural companion.

Together: tau says whether monotonic structure exists; pseudo-R^2
says whether the *line* is a useful predictor. They can disagree --
e.g., HisNE: CG-CB has tau ~ -0.7 (real trend) but pseudo-R^2 ~ -13
(line worse than mean). Both are publishable and informative.
"""

import numpy as np
from scipy.stats import kendalltau, theilslopes


def plot_slope_ci(ax, fit, xs, color="lightgray", alpha=0.35, zorder=0):
    """Shade the 95% slope CI from theilslopes as a wedge band that
    pivots around the central line's value at median(x). Picks pivot at
    the centroid so the band fans symmetrically across the data range
    rather than rotating around x=0 (which would put the visible band
    far from the data)."""
    xs = np.asarray(xs, dtype=float)
    if (
        xs.size < 2
        or not np.isfinite(fit["slope_lo"])
        or not np.isfinite(fit["slope_hi"])
    ):
        return
    x_pad = np.array([float(xs.min()), float(xs.max())])
    med_x = float(np.median(xs))
    y_pivot = fit["slope"] * med_x + fit["intercept"]
    int_lo = y_pivot - fit["slope_lo"] * med_x
    int_hi = y_pivot - fit["slope_hi"] * med_x
    y1 = fit["slope_lo"] * x_pad + int_lo
    y2 = fit["slope_hi"] * x_pad + int_hi
    ax.fill_between(
        x_pad, y1, y2, color=color, alpha=alpha,
        edgecolor="none", zorder=zorder,
    )


def theilsen_linfit(xs, ys):
    """Theil-Sen slope + intercept + fit diagnostics.

    Parameters
    ----------
    xs, ys : array-like
        Paired observations.

    Returns
    -------
    Dict with keys:
        slope, intercept       -- Theil-Sen line params
        slope_lo, slope_hi     -- 95% CI on slope from theilslopes
        r2                     -- Efron's pseudo-R^2 against the line
        tau, tau_p             -- Kendall's tau and its two-sided p-value
    Returns None if input is too degenerate (fewer than 2 points or
    all xs equal).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2 or np.unique(xs).size < 2:
        return None
    slope, intercept, slope_lo, slope_hi = theilslopes(ys, xs)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    tau, tau_p = kendalltau(xs, ys)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_lo": float(slope_lo),
        "slope_hi": float(slope_hi),
        "r2": r2,
        "tau": float(tau),
        "tau_p": float(tau_p),
    }

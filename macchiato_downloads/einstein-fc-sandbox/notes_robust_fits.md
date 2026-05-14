# Robust linear fits: loss functions vs. optimizers

Notes for choosing how to fit the per-bond `(bond_length, fc_n_per_m)` lines in
[plot_hess_bond_fcs.py](plot_hess_bond_fcs.py) and
[plot_hess_bond_fcs_heatmap.py](plot_hess_bond_fcs_heatmap.py).

The current code uses `scipy.stats.linregress`, which is ordinary least squares
(OLS). OLS is being yanked around by single outlier points (clearest case:
*HisNE: CG-CB*). This doc is a reference for picking a robust replacement.

---

## 1. The two-axis split: loss function vs. optimizer

When you "fit a line to data," there are *two* independent choices:

| Choice | Examples | What it controls |
|---|---|---|
| **Loss function** | squared residuals (MSE), absolute residuals (MAE), Huber, Tukey | *What does "best fit" mean?* Which deviations are penalized how harshly. |
| **Optimizer** | closed-form normal equations, L-BFGS-B, IRLS, gradient descent | *How do we find the parameters that minimize the loss?* |

These are orthogonal. Same loss + different optimizers → same answer (up to
numerical precision). Different loss → different answer, no matter what
optimizer you use.

### Why "L-BFGS-B minimizing MSE" is the same as OLS

OLS is, by definition, the slope/intercept that minimize

$$ L_{\text{OLS}}(m, b) = \sum_i (y_i - (m x_i + b))^2 . $$

This loss is convex and quadratic, so its minimum has a closed-form solution
(the normal equations — what `linregress` uses internally). You *can* feed the
exact same loss into `scipy.optimize.minimize(..., method="L-BFGS-B")` and
you'll converge to the same `(m, b)` — just slower, and at the mercy of
numerical tolerance.

So: **changing the optimizer does not change the fit**. To change the fit, you
must change the loss (or use an algorithm that isn't loss-minimization at all,
like Theil–Sen).

The optimizer choice matters only when your loss has no closed-form solution
(Huber, Tukey, anything non-convex) — *that* is when L-BFGS-B earns its keep.

---

## 2. Why squared residuals make OLS non-robust

The squaring in the OLS loss means a residual of size $r$ contributes $r^2$ to
the loss. A point that's 10× further from the line than a typical inlier
contributes **100×** the loss. The optimizer's job is to drive the loss down,
so it will rotate the line toward that one outlier far more than any single
inlier would justify.

For *HisNE: CG-CB*, you can see this directly: there's a tight cluster of ~25
points near $\ell = 1.49$ Å and one point near $\ell = 1.46$ Å. The lone point's
residual under the cluster's natural slope is large; squared, it dominates.
The OLS line rotates to split the difference, which looks visually wrong
because the eye intuitively weights points by *count*, not by squared
distance.

A 50% breakdown point is the gold standard for "robust": you can corrupt up to
half the data and the estimator still tracks the bulk. OLS has a breakdown
point of $1/n$ — *one* bad point is enough to move it arbitrarily far.

---

## 3. Robust loss functions

### L1 / least absolute deviations (LAD)

$$ L_1 = \sum_i |y_i - (m x_i + b)| $$

A single outlier of magnitude $r$ contributes $r$, not $r^2$. Much harder for
one point to dominate. Solved by linear programming or iteratively reweighted
least squares (IRLS); not differentiable at zero, so vanilla L-BFGS doesn't
love it.

- Breakdown point: 50% in $y$, but 0% if outliers are in $x$ (high-leverage
  points still hurt).
- Available via `scipy.optimize.minimize` with `method="Nelder-Mead"` or
  `statsmodels.regression.quantile_regression.QuantReg(y, X).fit(q=0.5)`.

### Huber loss

Quadratic for small residuals, linear for large ones. Best of both worlds:
behaves like OLS in the bulk (efficient), behaves like L1 in the tails
(robust).

$$ L_\delta(r) = \begin{cases} \tfrac{1}{2} r^2 & |r| \le \delta \\ \delta(|r| - \tfrac{1}{2}\delta) & |r| > \delta \end{cases} $$

- Has a tuning knob $\delta$ (the transition point). `sklearn.linear_model.HuberRegressor` defaults to $\delta = 1.35$ in standardized units, which gives ~95% of OLS's efficiency at Gaussian data while bounding the influence of outliers.
- Convex, smooth → L-BFGS-B works great. **This is the case where the user's "L-BFGS-B on the loss" intuition is correct.**
- Breakdown point: still 0% in the worst case (high-leverage), but in practice excellent for the kind of point-outliers we see in the bond-FC plots.

### Tukey biweight (redescending M-estimator)

Quadratic in the bulk, then *flat* (zero gradient) past some threshold. Outliers
beyond the threshold contribute *nothing* to the loss — they're ignored
entirely. Maximally robust, but the loss is non-convex, so optimization is
sensitive to the starting point.

- Useful for "burn the outliers" cases, but Theil–Sen gives most of the same
  benefit without the optimization headaches.
- Available via `statsmodels.robust.norms.TukeyBiweight` paired with
  `RLM(...).fit()`.

---

## 4. Algorithms that aren't loss-minimization

### Theil–Sen

Compute the slope between every pair of points $(i, j)$:

$$ s_{ij} = \frac{y_j - y_i}{x_j - x_i} $$

Take the median of all the $s_{ij}$ — that's your slope. Compute the intercept
from a median of $y_i - \hat{m} x_i$.

- No loss function. No optimizer. Just two medians.
- Breakdown point: ~29.3% (you'd need a coordinated outlier campaign to corrupt the slope estimate).
- Deterministic — same data always gives the same line.
- $\mathcal{O}(n^2)$ pairs, but $n$ is tiny here (≤ 119 per panel), so this is free.
- In scipy already: `scipy.stats.theilslopes`.

### RANSAC (RANdom SAmple Consensus)

Repeatedly: pick a random minimal subset (2 points for a line), fit, count
inliers (points within some threshold), keep the model with the most inliers.

- Designed for the "lots of outliers, want only inliers" regime — common in
  computer vision, less so for our use case.
- *Stochastic*: same data gives slightly different lines on different runs
  unless you fix the random seed.
- Has tuning knobs (inlier threshold, number of iterations, min sample size).
- `sklearn.linear_model.RANSACRegressor`.

---

## 5. Pros/cons comparison

| Method | Outlier resistance | Deterministic | Dependencies | Tuning knobs | Complexity |
|---|---|---|---|---|---|
| **OLS** | None ($1/n$ breakdown) | Yes | scipy | None | $\mathcal{O}(n)$, closed-form |
| **L1 / LAD** | Good in $y$, none in $x$ | Yes | scipy or statsmodels | None (or quantile $q$) | LP, fast for small $n$ |
| **Huber** | Good (in practice) | Yes | sklearn or statsmodels | Threshold $\delta$ | Convex; L-BFGS-B in seconds |
| **Theil–Sen** | Excellent (~29% breakdown) | Yes | scipy (already there) | None | $\mathcal{O}(n^2)$ pair slopes |
| **RANSAC** | Excellent (any %) | **No** (unless seeded) | sklearn | Inlier threshold, iterations | Iterative sampling |

---

## 6. What to use when, for the bond-FC plots

The situation in this sandbox:

- $N$ per panel: 27 to 119. Small.
- Outliers: point outliers in $y$, no obvious leverage problem (the $x$ range
  per bond is tight).
- Goal: "dominated by the majority", deterministic, simple.
- Constraint: prefer not to add new dependencies.

Recommendation order:

1. **Theil–Sen.** Free (`scipy.stats.theilslopes`), zero knobs, deterministic, and the breakdown-point story is unbeatable for "ignore the lone outlier and follow the cluster." This is what I'd reach for first.

2. **Huber.** If you want a smooth, continuous downweighting (rather than the median's hard-cut feel) and you're OK adding sklearn. The default $\delta$ is fine for almost everything.

3. **L1 / LAD.** A reasonable middle ground if you want no new deps but want a loss-based method. Less robust than Theil–Sen in practice; not really worth choosing over it for this use case.

4. **RANSAC.** Overkill — designed for situations with many outliers, and the stochasticity is annoying for a figure that should look the same every time you regenerate it.

5. **OLS.** Keep only if you want to compare against the robust line as a sanity check.

### Plotting two lines for comparison

For the writeup figure, it can be useful to plot OLS as a faint dashed line
and the robust fit as the solid line — readers see at a glance how much the
outlier moved the OLS estimate. Cheap to add.

---

## 7. Code snippets

All of these are drop-in replacements for the current
`fit = linregress(xs, ys)` block in the plotting scripts.

### OLS (current behavior, for reference)

```python
from scipy.stats import linregress
fit = linregress(xs, ys)
m, b, r2 = fit.slope, fit.intercept, fit.rvalue ** 2
```

### Theil–Sen (recommended)

```python
import numpy as np
from scipy.stats import theilslopes

slope, intercept, lo_slope, hi_slope = theilslopes(ys, xs)
pred = slope * xs + intercept
ss_res = np.sum((ys - pred) ** 2)
ss_tot = np.sum((ys - np.mean(ys)) ** 2)
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
```

`r2` here is "fraction of variance explained by the *robust* line" — same
threshold semantics as before, but the value will differ from OLS R². It can
go negative when the line is a poor fit; that's expected and informative.

`lo_slope` / `hi_slope` are a 95% CI on the slope, free of charge — handy if
you ever want error bars.

### Huber

```python
from sklearn.linear_model import HuberRegressor

huber = HuberRegressor().fit(xs.reshape(-1, 1), ys)
m, b = huber.coef_[0], huber.intercept_
pred = m * xs + b
r2 = 1 - np.sum((ys - pred) ** 2) / np.sum((ys - ys.mean()) ** 2)
```

`HuberRegressor` has an `epsilon` parameter (default 1.35); smaller = more
aggressive against outliers.

### L1 via L-BFGS (just to show the optimizer-vs-loss split)

```python
from scipy.optimize import minimize

def l1_loss(params, xs, ys):
    m, b = params
    return np.sum(np.abs(ys - (m * xs + b)))

# OLS as a starting point
ols = np.polyfit(xs, ys, 1)
res = minimize(l1_loss, x0=ols, args=(xs, ys), method="L-BFGS-B")
m, b = res.x
```

This is the "L-BFGS-B on a non-MSE loss" version. Note that L1 is not
differentiable at zero, so L-BFGS-B is technically wrong here — it works in
practice for small problems but a real implementation should use IRLS or LP.
This snippet is mostly to make the loss-vs-optimizer point concrete.

### RANSAC (for completeness)

```python
from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(random_state=0).fit(xs.reshape(-1, 1), ys)
m, b = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
inlier_mask = ransac.inlier_mask_  # bool array, True for kept points
```

Pin `random_state` if you want reproducible figures.

---

## 8. A note on Bayesian regression

You asked whether a Bayesian fit would help. By itself: no.

`sklearn.linear_model.BayesianRidge` and similar models give you a posterior
*distribution* over the slope and intercept rather than a point estimate.
That's about *uncertainty*, not *robustness*. If you assume Gaussian noise (the
default), the posterior mean is still pulled by outliers exactly like OLS —
because the Gaussian likelihood penalizes large residuals quadratically, same
as the OLS loss.

Bayesian regression *can* be made robust by swapping the Gaussian likelihood
for a heavy-tailed one (Student-t with low degrees of freedom). That makes the
posterior insensitive to outliers in the same way Huber/Tukey losses do. But
it's a step up in complexity (PyMC or numpyro), and Theil–Sen gets you 90% of
the way there with two lines of scipy.

The case for Bayesian here would be if you wanted:
- Slope uncertainty bands on the figure (Theil–Sen also gives a CI, simpler).
- A principled prior on the slope (e.g., "physics says the FC-vs-length slope should be negative" → Half-Normal prior on $-m$).

Neither of those seems pressing for the current figure.

---

## 9. References

- Rousseeuw & Leroy, *Robust Regression and Outlier Detection* (1987) — the canonical reference; chapters on Theil–Sen and breakdown point are very readable.
- Huber, *Robust Statistics* (1981) — original Huber loss.
- scipy docs: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html>
- sklearn robust regression overview: <https://scikit-learn.org/stable/modules/linear_model.html#robustness-regression-outliers-and-modeling-errors>

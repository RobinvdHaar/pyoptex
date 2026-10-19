#!/usr/bin/env python3

"""
Example: central composite designs in a split-plot structure.

The wind tunnel example of Parker, Kowalski and Vining (2007): two
hard-to-change flap settings (offset gap and deflection angle) and two
easy-to-change factors (angle of attack and Reynolds number), D(2, 2).

* The VKM method (Vining, Kowalski and Montgomery, 2005) gives 10 whole
  plots of 4 runs, with any axial distances.
* The minimum whole-plot (MWP) method gives 9 whole plots of 5 runs, the
  size of a CCD in the two hard-to-change factors, with
  alpha = beta = 2.

Both are equivalent estimation designs: the ordinary least squares
estimates of the second-order model equal the generalized least squares
estimates of the split-plot model. This is checked with the split-plot
model of pyoptex.doe.fixed_structure, the same machinery as for pyoptex's
optimal split-plot designs.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.classical import WHOLE_PLOT_COLUMN, split_plot_central_composite
from pyoptex.doe.fixed_structure import Factor, RandomEffect, create_parameters, default_fn
from pyoptex.doe.fixed_structure.evaluate import evaluate_metrics
from pyoptex.doe.fixed_structure.metric import Dopt
from pyoptex.utils.design import encode_design
from pyoptex.utils.model import model2Y2X, partial_rsm_names

set_seed(42)

wp_names = ["Gap", "Deflection"]
sp_names = ["AngleOfAttack", "Reynolds"]
names = wp_names + sp_names


def evaluate(Y, ratio=1.0):
    """
    OLS and GLS estimates of the second-order model for one simulated
    response, and the D-criterion, with whole plots as a random effect.
    """
    whole_plots = RandomEffect(Y[WHOLE_PLOT_COLUMN].to_numpy() - 1, ratio=ratio)
    factors = [Factor(name, whole_plots) for name in wp_names] + [Factor(name) for name in sp_names]
    Y2X = model2Y2X(partial_rsm_names({name: "quad" for name in names}), factors)
    params = create_parameters(factors, default_fn(factors, Dopt(), Y2X), len(Y))

    # The model matrix, as in evaluate_metrics
    normalized = np.column_stack([f.normalize(Y[str(f.name)]) for f in factors]).astype(np.float64)
    X = params.fn.Y2X(encode_design(normalized, params.effect_types, params.coords))

    # A response with whole-plot and subplot errors
    plots = Y[WHOLE_PLOT_COLUMN].to_numpy() - 1
    y = X @ np.random.normal(size=X.shape[1])
    y += np.sqrt(ratio) * np.random.normal(size=plots.max() + 1)[plots] + np.random.normal(size=len(Y))

    Vinv = params.Vinv[0]
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    gls = np.linalg.solve(X.T @ Vinv @ X, X.T @ Vinv @ y)
    (d,) = evaluate_metrics(Y, params, [Dopt()])
    return ols, gls, d


wp = [Factor(name) for name in wp_names]
sp = [Factor(name) for name in sp_names]
for method in ("vkm", "mwp"):
    Y = split_plot_central_composite(wp, sp, method=method)
    m, n = Y[WHOLE_PLOT_COLUMN].nunique(), len(Y) // Y[WHOLE_PLOT_COLUMN].nunique()
    alpha, beta = Y.attrs["alpha"], Y.attrs["beta"]
    print(f"=== {method.upper()}: {m} whole plots of {n} runs, alpha = {alpha:.4f}, beta = {beta:.4f} ===")
    print(Y.round(3).to_string())

    ols, gls, d = evaluate(Y)
    print(f"Largest difference between the OLS and GLS estimates: {np.max(np.abs(ols - gls)):.1e}")
    print(f"D-criterion with the split-plot model: {d:.3f}\n")

    log_checkpoint(f"{method}_shape", list(Y.shape))
    log_checkpoint(f"{method}_ols_equals_gls", bool(np.allclose(ols, gls)))

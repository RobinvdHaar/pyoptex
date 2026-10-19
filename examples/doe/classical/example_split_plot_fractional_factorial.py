#!/usr/bin/env python3

"""
Example: a minimum aberration fractional factorial split-plot design.

The ball mill example of Bingham and Sitter (1999), Example 1: four factors
are hard to change (motor speed, feed mode, feed sizing and material type)
and three are easy to change (gain, screen angle and screen vibration), with
16 runs in 8 whole plots. The minimum aberration design is a 2^((4+3)-(1+2))
fractional factorial split-plot design of resolution IV.

The whole plots are run in random order, and so are the runs within each
whole plot. The design is then evaluated with the split-plot model, in which
the whole plots are a random effect, using the same machinery as pyoptex's
optimal split-plot designs.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

import numpy as np

from pyoptex._seed import set_seed
from pyoptex.doe.classical import (
    WHOLE_PLOT_COLUMN,
    min_aberration_split_plot_generators,
    randomize,
    split_plot_fractional_factorial,
)
from pyoptex.doe.fixed_structure import Factor, RandomEffect, create_parameters, default_fn
from pyoptex.doe.fixed_structure.evaluate import evaluate_metrics
from pyoptex.doe.fixed_structure.metric import Dopt
from pyoptex.utils.model import model2Y2X, partial_rsm_names

set_seed(42)

n_runs, n_whole_plots = 16, 8

# The whole-plot factors get a random effect for the whole plots, the subplot factors none.
# The design is sorted by whole plot, so whole plot i holds runs 2i and 2i + 1.
whole_plots = RandomEffect(np.repeat(np.arange(n_whole_plots), n_runs // n_whole_plots), ratio=1.0)
wp_factors = [
    Factor("MotorSpeed", whole_plots, min=1000, max=1500),
    Factor("FeedMode", whole_plots, min=1, max=2),
    Factor("FeedSizing", whole_plots, min=10, max=20),
    Factor("MaterialType", whole_plots, min=1, max=2),
]
sp_factors = [
    Factor("Gain", min=1, max=3),
    Factor("ScreenAngle", min=20, max=40),
    Factor("ScreenVibration", min=5, max=10),
]
factors = wp_factors + sp_factors

generators = min_aberration_split_plot_generators(wp_factors, sp_factors, n_runs, n_whole_plots)
print("=== Generators ===")
for name, defining in generators.items():
    print(f"{name} = {' * '.join(defining)}")

Y = split_plot_fractional_factorial(wp_factors, sp_factors, n_runs, n_whole_plots)
print("\n=== Design in run order: whole plots and runs within them randomized ===")
print(randomize(Y, block_column=WHOLE_PLOT_COLUMN, shuffle_blocks=True).to_string())

log_checkpoint("split_plot_shape", list(Y.shape))

# Evaluate the main effects model with the split-plot model (whole-plot to subplot
# variance ratio 1) and as if the runs were completely randomized
model = partial_rsm_names({str(f.name): "lin" for f in factors})
Y2X = model2Y2X(model, factors)
params = create_parameters(factors, default_fn(factors, Dopt(), Y2X), len(Y))
(d_split_plot,) = evaluate_metrics(Y, params, [Dopt()])
plain = [Factor(f.name, min=f.min, max=f.max) for f in factors]
params = create_parameters(plain, default_fn(plain, Dopt(), Y2X), len(Y))
(d_random,) = evaluate_metrics(Y, params, [Dopt()])
print("\n=== D-criterion of the main effects model ===")
print(f"Split-plot model: {d_split_plot:.3f}")
print(f"Completely randomized: {d_random:.3f}")

log_checkpoint("split_plot_dopt", float(np.round(d_split_plot, 6)))

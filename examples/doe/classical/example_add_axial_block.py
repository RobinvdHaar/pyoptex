#!/usr/bin/env python3

"""
Example: sequential experimentation from a factorial to a central
composite design.

Stage 1 is a 2^4 factorial with center points. If the center points show
curvature, stage 2 adds the axial block: 8 axial runs and more center runs.
With the default alpha the two stages are orthogonal blocks, so a shift
between them does not bias the second-order model.
"""

try:
    from examples._log_checkpoint import log_checkpoint
except ImportError:
    log_checkpoint = lambda *args, **kwargs: None

from pyoptex.doe.classical import add_axial_block, full_factorial
from pyoptex.utils import Factor

factors = [
    Factor("Temperature", min=160, max=190),
    Factor("Pressure", min=2, max=4),
    Factor("Time", min=25, max=45),
    Factor("Speed", min=100, max=200),
]

# Stage 1: the factorial with 4 center runs
Y1 = full_factorial(factors, n_center=4)
print("=== Stage 1: 2^4 factorial with center points ===")
print(f"Runs: {len(Y1)}")

# Stage 2: the axial block, with 2 center runs, at the alpha for orthogonal blocking
Y = add_axial_block(Y1, factors, n_center=2)
alpha = factors[0].normalize(Y["Temperature"].max())
print("=== Stages 1 + 2: central composite design in 2 blocks ===")
print(f"Runs: {len(Y)} ({(Y['Block'] == 1).sum()} + {(Y['Block'] == 2).sum()}), alpha = {alpha:.4f}")
print(Y[Y["Block"] == 2].round(2))

log_checkpoint("add_axial_block_shape", list(Y.shape))

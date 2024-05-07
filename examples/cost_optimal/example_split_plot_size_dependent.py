#!/usr/bin/env python3

# Normal imports
import numpy as np
import pandas as pd
import numba
import os
import time

# Library imports
from pyoptex.doe.cost_optimal import create_cost_optimal_design, default_fn
from pyoptex.doe.utils.model import partial_rsm_names
from pyoptex.doe.utils.design import obs_var_from_Zs
from pyoptex.doe.cost_optimal.metric import Dopt, Aopt, Iopt

np.random.seed(42)

# Define parameters
effects = {
    # Define effect type, model type
    'A': (1, 'tfi'),
    'B': (1, 'tfi'),
    'C': (1, 'quad'),
    'D': (1, 'quad'),
    'E': (1, 'quad'),
    'F': (1, 'quad'),
    'G': (1, 'quad'),
    'H': (1, 'tfi'),
    'I': (1, 'tfi'),
}

# Derived parameters
effect_types = {key: value[0] for key, value in effects.items()}
model = partial_rsm_names({key: value[1] for key, value in effects.items()})
grouped_cols = np.zeros(len(effects))

#########################################################################

# Cost function
nb_plots = 6
runs_per_plot_low = 7
runs_per_plot_high = 14
def cost_fn(Y):
    # Short-circuit
    if len(Y) == 0:
        return []

    # Determine the number of resets
    resets = np.zeros(len(Y))
    resets[1:] = np.any(np.diff(Y[:, :2], axis=0) != 0, axis=1).astype(np.int64)
    for i in range(0, len(resets)):
        # Reset when factor does not change for a period depending on the first factor
        if (Y[i-1, 0] == -1 and i >= runs_per_plot_low and np.all(resets[i-(runs_per_plot_low-1):i] == 0))\
                or (Y[i-1, 0] == 1 and i >= runs_per_plot_high and np.all(resets[i-(runs_per_plot_high-1):i] == 0)):
            resets[i] = 1

    # Determine number of runs per plot
    idx = np.concatenate([[0], np.flatnonzero(resets), [len(Y)]])
    plot_costs = [None] * (len(idx) - 1)
    for i in range(len(idx)-1):
        if Y[i, 0] == -1:
            rp = runs_per_plot_low
        else:
            rp = runs_per_plot_high

        plot_costs[i] = (np.ones(idx[i+1] - idx[i]), rp, np.arange(idx[i], idx[i+1]))
    
    return [
        (resets, nb_plots - 1, np.arange(len(Y))),
        *plot_costs
    ]

# Split plot covariance matrix
def cov(Y, X, Zs, Vinv, costs, random=False):
    resets = costs[0][0].astype(np.int64)
    Z1 = np.cumsum(resets)
    Zs = [Z1]
    V = obs_var_from_Zs(Zs, len(Y), ratios=np.array([1.]))
    Vinv = np.expand_dims(np.linalg.inv(V), 0)
    return Y, X, Zs, Vinv

# Define the metric
metric = Dopt(cov=cov)

coords = [
    np.array([-1, 1]).reshape(-1, 1),
    np.array([-1, 1]).reshape(-1, 1),
    np.array([-1, -1/3, 1/3, 1]).reshape(-1, 1),
    None, None, None, None, None, None
]

#########################################################################

# Parameter initialization
nsims = 500
nreps = 1

# Create the set of operators
fn = default_fn(nsims, cost_fn, metric)

# Create design
start_time = time.time()
Y, state = create_cost_optimal_design(
    effect_types, fn, model=model, coords=coords,
    nsims=nsims, nreps=nreps, grouped_cols=grouped_cols, 
    validate=True
)
end_time = time.time()

#########################################################################

# Write design to storage
root = os.path.split(__file__)[0]
Y.to_csv(os.path.join(root, 'results', 'example_split_plot_size_dependent.csv'), index=False)

print('Completed optimization')
print(f'Metric: {state.metric:.3f}')
print(f'Cost: {state.cost_Y}')
print(f'Number of experiments: {len(state.Y)}')
print(f'Execution time: {end_time - start_time:.3f}')


"""
Module containing all the generic initialization functions
"""

import numpy as np

from ._init_cy import init_single_unconstrained_cython_impl


def init_single_unconstrained(colstart, coords, run, effect_types, low=None, high=None):
    """
    Initializes a run at random. There are three possibilities:
    'continuous' which is random between `low` and `high` (by default
    -1 and 1), categorical which is a random level, or from coords which
    selects a random coordinate from `coords`.

    Parameters
    ----------
    colstart : np.array(1d)
        The starting column of each factor.
    coords : list(np.array(2d) or None)
        The coordinates to sample from.
    run : np.array(2d)
        Output buffer of the function. Also returned at the end.
    effect_types : np.array(1d)
        The type of each effect in case no coordinates are specified.
    low : None or np.array(1d)
        The lower bound of each continuous factor without coordinates. Defaults to -1.
    high : None or np.array(1d)
        The upper bound of each continuous factor without coordinates. Defaults to 1.

    Returns
    -------
    run : np.array(2d)
        The randomly sampled run.
    """
    n_factors = len(colstart) - 1
    low = np.full(n_factors, -1.0) if low is None else np.ascontiguousarray(low, dtype=np.float64)
    high = np.full(n_factors, 1.0) if high is None else np.ascontiguousarray(high, dtype=np.float64)
    return init_single_unconstrained_cython_impl(colstart, coords, np.ascontiguousarray(run), effect_types, low, high)


def design_region_bounds(factors: list) -> tuple[np.ndarray, np.ndarray]:
    """
    The bounds to draw each continuous factor from when sampling the complete
    design region: -1 and 1, except for mixture components. A mixture
    component is drawn between its lowest level L and the highest share it
    can reach, U' = min(highest level, L + R), with R = 1 - sum(L) what the
    lowest levels of the components leave (the implied component's lowest
    share is 0). From -1 to 1 would mean a share from -100% to 100%. Capping
    at U' removes no feasible mixture, but keeps the rejection on the mixture
    sum efficient when the lowest levels sum to almost 1.

    Parameters
    ----------
    factors : list(:py:class:`Factor <pyoptex.utils.factor.FactorMixin>`)
        The factors.

    Returns
    -------
    low : np.array(1d)
        The lower bound of each factor.
    high : np.array(1d)
        The upper bound of each factor.
    """
    low, high = np.full(len(factors), -1.0), np.full(len(factors), 1.0)
    mixture = [i for i, f in enumerate(factors) if f.is_mixture]
    if mixture:
        lower = np.array([np.min(factors[i].levels) for i in mixture], dtype=np.float64)
        upper = np.array([np.max(factors[i].levels) for i in mixture], dtype=np.float64)
        rest = 1 - np.sum(lower)
        if rest < -1e-6:  # The tolerance eps of the mixture constraint
            raise ValueError(
                f"The lowest levels of the mixture components sum to {1 - rest:.10g} > 1: no mixture is possible"
            )
        low[mixture] = lower
        high[mixture] = np.minimum(upper, lower + max(rest, 0))
    return low, high


def full_factorial(colstart, coords, Y=None):
    """
    Generates a full factorial design.

    Parameters
    ----------
    colstart : np.array(1d)
        The starting columns of each factor
    coords : list(np.array(2d))
        The list of possible coordinates for each factor.
    Y : np.array(2d) or None
        The output array for the full factorial design.

    Returns
    -------
    Y : np.array(2d)
        The full factorial design.
    """
    # Initialize Y
    if Y is None:
        n = np.prod([coords[i].shape[0] for i in range(len(coords))])
        Y = np.zeros((n, colstart[-1]), dtype=np.float64)

    # Create the full factorial matrix
    tile = 1
    rep = len(Y)
    for i in range(colstart.size - 1):
        rep = int(rep / coords[i].shape[0])
        Y[:, colstart[i] : colstart[i + 1]] = np.tile(np.repeat(coords[i], rep, axis=0), (tile, 1))
        tile *= coords[i].shape[0]

    return Y

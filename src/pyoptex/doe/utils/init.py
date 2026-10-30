"""
Module containing all the generic initialization functions
"""

from math import factorial

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


def design_region_bounds(factors: list) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    The bounds to draw each continuous factor from when sampling the complete
    design region: -1 and 1, except for mixture components. A mixture
    component is drawn between its lowest level L and the highest share it
    can reach, U' = min(highest level, L + R), with R = 1 - sum(L) what the
    lowest levels of the components leave (the implied component's lowest
    share is 0). From -1 to 1 would mean a share from -100% to 100%. Capping
    at U' removes no feasible mixture, but keeps the rejection on the mixture
    sum efficient when the lowest levels sum to almost 1.

    For broad mixtures this box of the q - 1 explicit components is still
    much larger than the feasible mixtures: with 8 mixture factors
    (9 components) of 0 to 1 each, only 1 in 8! = 40,320 draws is feasible
    (28 s for Iopt), with 10 components 1 in 9! = 362,880. The mixtures are
    then drawn from the simplex
    L + R z instead (z uniform over all q components, see
    :py:func:`resample_mixture_simplex`), whenever its volume
    R^(q-1) / (q-1)! is smaller than that of the box, prod(U' - L).

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
    simplex : bool
        Whether to draw the mixture components from the simplex.
    """
    low, high = np.full(len(factors), -1.0), np.full(len(factors), 1.0)
    mixture = [i for i, f in enumerate(factors) if f.is_mixture]
    if not mixture:
        return low, high, False

    lower = np.array([np.min(factors[i].levels) for i in mixture], dtype=np.float64)
    upper = np.array([np.max(factors[i].levels) for i in mixture], dtype=np.float64)
    rest = 1 - np.sum(lower)
    if rest < -1e-6:  # The tolerance eps of the mixture constraint
        raise ValueError(
            f"The lowest levels of the mixture components sum to {1 - rest:.10g} > 1: no mixture is possible"
        )
    low[mixture] = lower
    if rest <= 1e-6:
        # Only the lowest levels themselves remain
        high[mixture] = lower
        return low, high, False

    high[mixture] = np.minimum(upper, lower + rest)
    simplex = rest ** len(mixture) / factorial(len(mixture)) < np.prod(high[mixture] - lower)
    return low, high, simplex


def resample_mixture_simplex(
    run: np.ndarray, colstart: np.ndarray, factors: list, low: np.ndarray, high: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Redraws the mixture components uniformly from the simplex x = L + R z,
    with z ~ Dirichlet(1, ..., 1) over all q components (the implied one
    included), so every component is at least L and the shares sum to 1.
    A component can exceed its upper bound, so these runs are returned for
    rejection.

    Parameters
    ----------
    run : np.array(2d)
        The sampled runs. Changed in place and returned.
    colstart : np.array(1d)
        The starting column of each factor.
    factors : list(:py:class:`Factor <pyoptex.utils.factor.FactorMixin>`)
        The factors.
    low : np.array(1d)
        The lower bound of each factor, see :py:func:`design_region_bounds`.
    high : np.array(1d)
        The upper bound of each factor, see :py:func:`design_region_bounds`.

    Returns
    -------
    run : np.array(2d)
        The runs with the redrawn mixture components.
    too_high : np.array(1d)
        Whether a mixture component of the run exceeds its upper bound.
    """
    mixture = [i for i, f in enumerate(factors) if f.is_mixture]
    rest = 1 - np.sum(low[mixture])
    shares = low[mixture] + rest * np.random.dirichlet(np.ones(len(mixture) + 1), size=len(run))[:, :-1]
    run[:, colstart[mixture]] = shares
    return run, np.any(shares > high[mixture], axis=1)


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

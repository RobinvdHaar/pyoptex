"""
Module for classical split-plot designs:

* minimum aberration two-level fractional factorial split-plot designs
  (Bingham, D. and Sitter, R. R. (1999), "Minimum-Aberration Two-Level
  Fractional Factorial Split-Plot Designs", Technometrics 41(1), 62-70);
* central composite designs in a split-plot structure with equivalent
  estimation (Vining, G. G., Kowalski, S. M. and Montgomery, D. C. (2005),
  "Response Surface Designs within a Split-Plot Structure", Journal of
  Quality Technology 37(2), 115-129; Parker, P. A., Kowalski, S. M. and
  Vining, G. G. (2007), "Construction of Balanced Equivalent Estimation
  Second-Order Split-Plot Designs", Technometrics 49(1), 56-65).
"""

from itertools import product

import numpy as np
import pandas as pd

from ._split_plot_aberration import _SPLIT_PLOT_GENERATORS
from .fractional_factorial import fractional_factorial
from .utils import (
    WHOLE_PLOT_COLUMN,
    _assert_no_categorical_center,
    _center_points,
    _levels,
    _to_dataframe,
    _validate_factors,
)


def _log2(n, name):
    """Base-2 logarithm of a power of 2."""
    r = int(n).bit_length() - 1
    assert n >= 1 and 2**r == n, f"{name} must be a power of 2 (got {n})"
    return r


def _validate_split_plot_factors(wp_factors, sp_factors):
    """At least one whole-plot and one subplot factor, all two-level, with distinct names."""
    assert len(wp_factors) > 0, "At least one whole-plot factor is required"
    assert len(sp_factors) > 0, "At least one subplot factor is required"
    factors = list(wp_factors) + list(sp_factors)
    _validate_factors(factors)
    names = [str(f.name) for f in factors]
    assert len(set(names)) == len(names), f"The factor names must be distinct (got {names})"
    assert WHOLE_PLOT_COLUMN not in names, f"A factor cannot be named '{WHOLE_PLOT_COLUMN}' in a split-plot design"
    for f in factors:
        nlev = len(_levels(f, 2))
        assert nlev == 2, f"Factor {f.name} must have exactly 2 levels for a two-level split-plot design (has {nlev})"


def min_aberration_split_plot_generators(wp_factors, sp_factors, n_runs, n_whole_plots):
    """
    Generators of the minimum aberration two-level fractional factorial
    split-plot (FFSP) design.

    A 2^((k1+k2)-(p1+p2)) FFSP design for k1 whole-plot (hard-to-change)
    and k2 subplot (easy-to-change) factors has ``r1 = log2(n_whole_plots)``
    whole-plot base factors and ``r2 = log2(n_runs) - r1`` subplot base
    factors.  Following Bingham and Sitter (1999), the generators of the
    whole-plot factors contain whole-plot base factors only, so the
    whole-plot factors are constant within each whole plot, and the
    generators of the subplot factors contain at least one subplot base
    factor (and possibly whole-plot base factors).  Among all such designs,
    the catalogue holds the one with minimum aberration (the
    lexicographically smallest word length pattern of the full design,
    B&S Definition 1), found by exhaustive search for 4 runs up to 3
    factors, 8 runs up to 7 factors, 16 runs up to 15 factors and 32 runs
    up to 12 factors.  The full factorial (no generators) is always
    available.

    The first r1 whole-plot factors and the first r2 subplot factors are
    the base factors, so put the factors in the order you want them
    assigned.  The result can be passed to
    :py:func:`fractional_factorial <pyoptex.doe.classical.fractional_factorial>`
    with the whole-plot factors followed by the subplot factors, or use
    :py:func:`split_plot_fractional_factorial` directly.

    Parameters
    ----------
    wp_factors : list(Factor)
        The two-level whole-plot factors.
    sp_factors : list(Factor)
        The two-level subplot factors.
    n_runs : int
        Number of runs (a power of 2).
    n_whole_plots : int
        Number of whole plots (a power of 2, smaller than *n_runs*).

    Returns
    -------
    generators : dict(str, list(str))
        Maps each generated factor name to its defining base factors.
    """
    _validate_split_plot_factors(wp_factors, sp_factors)
    k1, k2 = len(wp_factors), len(sp_factors)
    r = _log2(n_runs, "n_runs")
    r1 = _log2(n_whole_plots, "n_whole_plots")
    r2 = r - r1

    full = (2 ** (k1 + k2), 2**k1)
    available = sorted({(n, w) for (a, b, n, w) in _SPLIT_PLOT_GENERATORS if (a, b) == (k1, k2)} | {full})
    if (n_runs, n_whole_plots) == full:
        return {}
    key = (k1, k2, n_runs, n_whole_plots)
    assert key in _SPLIT_PLOT_GENERATORS, (
        f"No catalogued fractional factorial split-plot design for {k1} whole-plot and {k2} subplot factors "
        f"in {n_runs} runs and {n_whole_plots} whole plots; available (n_runs, n_whole_plots): {available}. "
        f"Alternatives: (a) build the split-plot fraction with fractional_factorial and your own generators "
        f"(whole-plot generators may contain only whole-plot factors), add a whole-plot column and run it with "
        f"randomize(..., shuffle_blocks=True); only the minimum aberration guarantee is missing; "
        f"(b) create an optimal split-plot design with pyoptex.doe.fixed_structure"
    )

    _, gens = _SPLIT_PLOT_GENERATORS[key]
    wp_names = [str(f.name) for f in wp_factors]
    sp_names = [str(f.name) for f in sp_factors]
    base = wp_names[:r1] + sp_names[:r2]
    added = wp_names[r1:] + sp_names[r2:]
    return {name: [base[i] for i in gen] for name, gen in zip(added, gens, strict=True)}


def split_plot_fractional_factorial(wp_factors, sp_factors, n_runs, n_whole_plots, n_center=0):
    """
    Generate a minimum aberration two-level fractional factorial split-plot
    design.

    The generators are those of
    :py:func:`min_aberration_split_plot_generators` (Bingham and Sitter,
    1999).  The design is sorted by whole plot: the whole-plot factors are
    constant within each whole plot, and the ``'WholePlot'`` column
    numbers the whole plots from 1.  Each of the ``n_whole_plots`` whole
    plots holds ``n_runs / n_whole_plots`` runs.

    The experiment is run whole plot by whole plot: randomize the order of
    the whole plots and the order of the runs within each whole plot with
    ``randomize(Y, block_column='WholePlot', shuffle_blocks=True)``.  The
    design must then be analysed with the split-plot (GLS) model, in which
    the whole-plot factors are compared with the whole-plot error.  To
    evaluate the design with that model, create the factors with
    :py:class:`pyoptex.doe.fixed_structure.Factor` and give the
    whole-plot factors a random effect for the whole plots, e.g. for 16
    runs in 8 whole plots:

    .. code-block:: python

        re = RandomEffect(np.repeat(np.arange(8), 2), ratio=1.0)
        wp = [Factor(name, re) for name in "ABCD"]
        sp = [Factor(name) for name in "PQR"]
        Y = split_plot_fractional_factorial(wp, sp, 16, 8)
        params = create_parameters(wp + sp, default_fn(wp + sp, Dopt(), Y2X), len(Y))
        evaluate_metrics(Y, params, [Dopt()])

    These are the same factors and parameters as for an optimal
    split-plot design with
    :py:func:`create_fixed_structure_design <pyoptex.doe.fixed_structure.create_fixed_structure_design>`,
    so the two designs can be compared directly.

    Parameters
    ----------
    wp_factors : list(Factor)
        The two-level whole-plot (hard-to-change) factors.
    sp_factors : list(Factor)
        The two-level subplot (easy-to-change) factors.
    n_runs : int
        Number of runs (a power of 2).
    n_whole_plots : int
        Number of whole plots (a power of 2, smaller than *n_runs*).
    n_center : int
        Number of center runs (default 0).  They form one extra whole plot
        with all factors at the middle of their range, so the design then
        has ``n_whole_plots + 1`` whole plots.  Not allowed with
        categorical factors.

    Returns
    -------
    Y : pd.DataFrame
        The design (denormalized): the whole-plot factors, the subplot
        factors and the ``'WholePlot'`` column.
    """
    generators = min_aberration_split_plot_generators(wp_factors, sp_factors, n_runs, n_whole_plots)
    factors = list(wp_factors) + list(sp_factors)
    assert n_center >= 0, f"n_center must be non-negative (got {n_center})"
    _assert_no_categorical_center(factors, n_center, "split-plot design")

    # The base factors form a full factorial with the whole-plot base factors
    # varying slowest, so the runs are already grouped by whole plot.
    Y = fractional_factorial(factors, generators)
    Y[WHOLE_PLOT_COLUMN] = np.repeat(np.arange(1, n_whole_plots + 1), n_runs // n_whole_plots)

    if n_center > 0:
        center = _to_dataframe(_center_points(factors, 2, n_center), factors)
        center[WHOLE_PLOT_COLUMN] = n_whole_plots + 1
        Y = pd.concat([Y, center], ignore_index=True)
    return Y


# PKV (2007), Table 1: the numbers of whole-plot and subplot factors (w, k) for which
# the factorial part of the CCD is a half fraction (q = 1); otherwise it is the full 2^(w+k)
_HALF_FRACTION_CCD = {(1, 4), (2, 3), (2, 4), (3, 3), (3, 4)}


def _subplot_factorial(z, k, half):
    """
    Subplot factorial runs of the whole plot with whole-plot levels z: the
    full 2^k, or, for a half fraction, the runs in which the last subplot
    factor is the product of all other (whole-plot and subplot) factors.
    """
    if not half:
        return np.array(list(product((-1.0, 1.0), repeat=k)))
    base = np.array(list(product((-1.0, 1.0), repeat=k - 1)))
    return np.column_stack([base, np.prod(z) * np.prod(base, axis=1)])


def _axial_runs(k, a):
    """The 2k axial runs at distance a: -a and +a on each factor in turn, the others at 0."""
    runs = np.zeros((2 * k, k))
    for j in range(k):
        runs[2 * j, j], runs[2 * j + 1, j] = -a, a
    return runs


def split_plot_central_composite(
    wp_factors, sp_factors, method="mwp", alpha=None, beta=None, n_center_whole_plots=None
):
    """
    Generate a central composite design in a split-plot structure with
    equivalent estimation.

    The runs of a central composite design in the w whole-plot
    (hard-to-change) and k subplot (easy-to-change) factors are arranged
    in balanced whole plots (all of n runs) such that the ordinary least
    squares estimates of the full second-order model equal the
    generalized least squares estimates of the split-plot model, whatever
    the variance components: the model can be fitted with OLS (Vining,
    Kowalski and Montgomery, 2005; Parker, Kowalski and Vining, 2007).

    The factorial part is the full 2^(w+k) factorial, or the half fraction
    of PKV Table 1 for (w, k) = (1, 4), (2, 3), (2, 4), (3, 3) and (3, 4),
    in which the last subplot factor is the product of all other factors.
    The whole-plot factors then still form a full factorial, and each
    factorial whole plot holds ``n_f = 2^(w+k-q) / 2^w`` subplot runs.

    * ``method='vkm'`` (PKV section 7.1): the 2^w factorial whole plots,
      2w whole plots with one whole-plot factor at +-beta and the subplot
      factors at 0, the subplot axial runs with the whole-plot factors at
      0 (one whole plot with all 2k axial runs when k is a power of 2,
      otherwise one whole plot per subplot factor with its two axial runs
      replicated), and *n_center_whole_plots* whole plots with all factors
      at 0; n = n_f.  Equivalence holds for any alpha and beta, and for
      any submodel of the second-order model.
    * ``method='mwp'`` (minimum whole-plot, PKV section 7.2): the 2^w
      factorial whole plots, the 2w whole-plot axial whole plots and one
      whole plot with the whole-plot factors at 0 that holds the 2k
      subplot axial runs, every whole plot completed to
      ``n = max(n_f, 2k + 1)`` runs with subplot center runs.  This uses
      the minimum number of whole plots, 2^w + 2w + 1, the size of a CCD
      in the whole-plot factors.  Equivalence requires the full
      second-order model and ties alpha to beta (PKV eq. 16):
      ``alpha^2 = n_f / (2 (1 - w / beta^2))``; by default alpha = beta
      with ``alpha^2 = n_f / 2 + w`` (PKV eq. 17).  For one whole-plot
      factor, ``beta=1`` merges the whole-plot axial points with the
      factorial levels: 3 whole plots of ``n = max(n_f + 1, 2k + 1)``
      runs, the factorial whole plots completed with subplot center runs,
      and alpha free.  A whole plot with all factors at 0 would break the
      equivalence (PKV section 7.2.3), so there are no center whole plots.

    The numbers of whole plots m and runs per whole plot n are those of
    PKV Table B.1.  The range is 1 to 3 whole-plot and 1 to 4 subplot
    factors; other split-plot designs can be created with pyoptex's
    optimal designs (:py:mod:`pyoptex.doe.fixed_structure`).

    The factors' ``min`` and ``max`` define the cube (coded -1 and +1), as
    for :py:func:`central_composite <pyoptex.doe.classical.central_composite>`;
    the axial runs lie alpha (subplot factors) or beta (whole-plot
    factors) half-ranges from the center.  The design is sorted by whole
    plot; run it with
    ``randomize(Y, block_column='WholePlot', shuffle_blocks=True)``.

    Parameters
    ----------
    wp_factors : list(Factor)
        The continuous whole-plot (hard-to-change) factors, 1 to 3.
    sp_factors : list(Factor)
        The continuous subplot (easy-to-change) factors, 1 to 4.
    method : str
        ``'mwp'`` (default) or ``'vkm'``.
    alpha : float or None
        Coded axial distance of the subplot factors.  For ``'vkm'`` and
        for ``'mwp'`` with ``beta=1`` the default is the rotatable value
        of the completely randomized CCD, ``(2^(w+k-q))^(1/4)``.  For
        ``'mwp'`` with ``beta != 1`` it follows from beta and cannot be
        specified.
    beta : float or None
        Coded axial distance of the whole-plot factors.  For ``'vkm'``
        the default is the rotatable value; for ``'mwp'`` the default is
        alpha = beta from PKV eq. (17), and a given beta must exceed
        ``sqrt(w)`` (or equal 1 for one whole-plot factor).
    n_center_whole_plots : int or None
        Number of whole plots with all factors at 0, for ``'vkm'`` only
        (default 1).

    Returns
    -------
    Y : pd.DataFrame
        The design (denormalized), with the whole-plot factors, the
        subplot factors and the ``'WholePlot'`` column.  The coded axial
        distances are stored in ``Y.attrs['alpha']`` and
        ``Y.attrs['beta']``.
    """
    assert len(wp_factors) > 0, "At least one whole-plot factor is required"
    assert len(sp_factors) > 0, "At least one subplot factor is required"
    factors = list(wp_factors) + list(sp_factors)
    _validate_factors(factors)
    names = [str(f.name) for f in factors]
    assert len(set(names)) == len(names), f"The factor names must be distinct (got {names})"
    assert WHOLE_PLOT_COLUMN not in names, f"A factor cannot be named '{WHOLE_PLOT_COLUMN}' in a split-plot design"
    for f in factors:
        assert f.is_continuous and not f.is_mixture, f"Factor {f.name} must be continuous for a CCD"

    w, k = len(wp_factors), len(sp_factors)
    assert 1 <= w <= 3 and 1 <= k <= 4, (
        f"Split-plot central composite designs are tabulated for 1 to 3 whole-plot and 1 to 4 subplot factors "
        f"(got {w} and {k}); create an optimal split-plot design with pyoptex.doe.fixed_structure instead"
    )
    assert method in ("vkm", "mwp"), f"method must be 'vkm' or 'mwp' (got {method!r})"

    half = (w, k) in _HALF_FRACTION_CCD
    n_cube = 2 ** (w + k - half)
    n_f = n_cube // 2**w
    rotatable = n_cube**0.25
    wp_cube = np.array(list(product((-1.0, 1.0), repeat=w)))
    center = np.zeros(w)

    def wp_axial_plots(beta, n):
        plots = []
        for i in range(w):
            for s in (-beta, beta):
                z = np.zeros(w)
                z[i] = s
                plots.append((z, np.zeros((n, k))))
        return plots

    if method == "vkm":
        n_center_whole_plots = 1 if n_center_whole_plots is None else n_center_whole_plots
        assert n_center_whole_plots >= 0, f"n_center_whole_plots must be non-negative (got {n_center_whole_plots})"
        alpha = rotatable if alpha is None else float(alpha)
        beta = rotatable if beta is None else float(beta)
        n = n_f
        plots = [(z, _subplot_factorial(z, k, half)) for z in wp_cube]
        plots += wp_axial_plots(beta, n)
        if k & (k - 1) == 0:
            # k a power of 2: all subplot axial runs in one whole plot (2k = n in the tabulated range)
            plots.append((center, np.tile(_axial_runs(k, alpha), (n // (2 * k), 1))))
        else:
            # One whole plot per subplot factor, its two axial runs replicated
            axial = _axial_runs(k, alpha)
            plots += [(center, np.tile(axial[2 * j : 2 * j + 2], (n // 2, 1))) for j in range(k)]
        plots += [(center, np.zeros((n, k))) for _ in range(n_center_whole_plots)]
    else:
        assert not n_center_whole_plots, (
            "A whole plot with all factors at 0 breaks the equivalence of the 'mwp' design (Parker, Kowalski and "
            "Vining, 2007, section 7.2.3); use method='vkm' for center whole plots"
        )
        if beta is not None and np.isclose(beta, 1):
            assert w == 1, (
                f"beta=1 (whole-plot axial points at the factorial levels) requires 1 whole-plot factor (got {w})"
            )
            alpha = rotatable if alpha is None else float(alpha)
            beta = 1.0
            n = max(n_f + 1, 2 * k + 1)
            plots = [(z, _subplot_factorial(z, k, half)) for z in wp_cube]
        else:
            assert alpha is None, (
                "For method='mwp', alpha follows from beta (Parker, Kowalski and Vining, 2007, eq. 16); "
                "specify beta only (or beta=1 for one whole-plot factor, with alpha free)"
            )
            if beta is None:
                alpha = beta = np.sqrt(n_f / 2 + w)
            else:
                beta = float(beta)
                assert beta > np.sqrt(w), f"For method='mwp', beta must exceed sqrt(w) = {np.sqrt(w):g} (got {beta:g})"
                alpha = np.sqrt(n_f / (2 * (1 - w / beta**2)))
            n = max(n_f, 2 * k + 1)
            plots = [(z, _subplot_factorial(z, k, half)) for z in wp_cube]
            plots += wp_axial_plots(beta, n)
        plots.append((center, _axial_runs(k, alpha)))
    assert alpha > 0 and beta > 0, f"alpha and beta must be positive (got {alpha:g} and {beta:g})"

    # Complete every whole plot to n runs with subplot center runs
    design = np.vstack(
        [np.column_stack([np.tile(z, (n, 1)), np.vstack([runs, np.zeros((n - len(runs), k))])]) for z, runs in plots]
    )
    Y = _to_dataframe(design, factors)
    Y[WHOLE_PLOT_COLUMN] = np.repeat(np.arange(1, len(plots) + 1), n)
    Y.attrs["alpha"], Y.attrs["beta"] = float(alpha), float(beta)
    return Y

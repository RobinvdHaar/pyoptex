"""
Module for run orders with few level changes, and for counting level changes.
"""

import numpy as np
import pandas as pd

from .utils import BLOCK_COLUMN, _validate_factors


def level_changes(Y, factors):
    """
    Count how often each factor changes level between consecutive runs.

    Parameters
    ----------
    Y : pd.DataFrame
        The design, in run order.
    factors : list(Factor)
        The factors to count.

    Returns
    -------
    changes : dict(str, int)
        The number of level changes per factor name.
    """
    assert isinstance(Y, pd.DataFrame), "Y must be a DataFrame"
    changes = {}
    for f in factors:
        values = Y[str(f.name)].to_numpy()
        changes[str(f.name)] = int(np.sum(values[1:] != values[:-1]))
    return changes


def _gray_mixed_radix(radices):
    """Reflected mixed-radix Gray code: digit tuples where exactly one digit changes per step, the first slowest."""
    if len(radices) == 0:
        return [()]
    rest = _gray_mixed_radix(radices[1:])
    sequence = []
    for d in range(radices[0]):
        sequence += [(d, *r) for r in (rest if d % 2 == 0 else reversed(rest))]
    return sequence


def _level_index(Y, factors):
    """Level index of every run for every factor (continuous levels sorted, categorical in the order of levels)."""
    index, levels = [], []
    for f in factors:
        values = Y[str(f.name)].to_numpy()
        lv = sorted(set(values)) if f.is_continuous else [lv for lv in f.levels if lv in set(values)]
        position = {v: i for i, v in enumerate(lv)}
        index.append(np.array([position[v] for v in values], dtype=np.int64))
        levels.append(len(lv))
    return np.column_stack(index), levels


def _full_factorial_order(idx, levels):
    """Positions of the runs in the Gray code order of a full factorial, or None if the runs are no full factorial."""
    rows = [tuple(r) for r in idx]
    if len(set(rows)) != len(rows) or len(rows) != int(np.prod(levels)):
        return None
    rank = {code: i for i, code in enumerate(_gray_mixed_radix(levels))}
    return np.argsort([rank[r] for r in rows], kind="stable")


def _regular_fraction_order(idx, levels):
    """
    Positions of the runs in a binary Gray code order on base factors chosen greedily (factors in priority
    order), or None if the runs are no regular two-level fraction (an affine subspace of GF(2)^k).
    """
    n, k = idx.shape
    if any(lv != 2 for lv in levels) or n & (n - 1) or len({tuple(r) for r in idx}) != n:
        return None
    rows = {tuple(r) for r in (idx ^ idx[0])}
    if any(tuple(a ^ b for a, b in zip(r1, r2, strict=True)) not in rows for r1 in rows for r2 in rows):
        return None

    # Greedy choice of base factors: the next factor (in priority order) whose column is independent
    # of the chosen ones gets the next Gray bit (most significant first, so it changes least often).
    r = n.bit_length() - 1
    cols = idx.T
    basis, reduced = [], []  # chosen factor indices; their columns in reduced (echelon) form
    for j in range(k):
        v = cols[j] ^ cols[j][0]
        for b in reduced:
            pivot = int(np.argmax(b))
            if v[pivot]:
                v = v ^ b
        if v.any():
            basis.append(j)
            reduced.append(v)
        if len(basis) == r:
            break

    words = idx[:, basis]  # Gray word of every run: bit p is base factor basis[p]
    binary = np.cumsum(words, axis=1) % 2  # inverse Gray code, most significant bit first
    rank = binary @ (1 << np.arange(r - 1, -1, -1))
    return np.argsort(rank, kind="stable")


def gray_code_order(Y, factors, priority=None):
    """
    Order the runs so that factors change level as rarely as possible.

    The runs are put in Gray code order, in which exactly one base factor
    changes between consecutive runs.  The factors in *priority* (by
    default the order of *factors*) get the slowest-changing positions:

    * Full factorials, with any numbers of levels: a reflected
      mixed-radix Gray code, so exactly one factor changes between
      consecutive runs; the first factor in *priority* changes L - 1
      times for L levels, the next one the next fewest, and so on.
    * Regular two-level fractional factorials: log2(N) independent
      factors are chosen as base factors, greedily in *priority* order,
      and the runs are put in binary Gray code order on them; the first
      base factor changes once, the next twice, then 4, 8, ... times.  A
      generated factor changes whenever one of its defining base factors
      changes, so as often as the sum of their changes.

    Center runs (every continuous factor in the middle of its levels) are
    placed at the end.  In a blocked design (with a ``'Block'`` column) the
    runs are ordered within each block and the blocks keep their order.
    Other designs (e.g. Plackett-Burman designs of 12 runs) are not
    supported.

    .. warning::
        A systematic run order is not randomized, so time trends can bias
        the effects of the frequently changing factors.  For
        hard-to-change factors the randomized alternative is a split-plot
        design, e.g. with
        :py:func:`randomize(Y, block_column=[...], shuffle_blocks=True)
        <pyoptex.doe.classical.randomize>`.

    Parameters
    ----------
    Y : pd.DataFrame
        The design.
    factors : list(Factor)
        The factors of the design.
    priority : list(str) or None
        Factor names, most hard-to-change first.  Factors that are not
        listed follow in the order of *factors*.  ``None`` uses the order
        of *factors*.

    Returns
    -------
    Y : pd.DataFrame
        The same runs in the new order.  The index is kept, so it still
        gives the original position of each run.
    """
    assert isinstance(Y, pd.DataFrame), "Y must be a DataFrame"
    _validate_factors(factors)
    names = [str(f.name) for f in factors]
    priority = [] if priority is None else [str(p) for p in priority]
    for p in priority:
        assert p in names, f"Factor {p} in priority is not one of the factors"
    by_name = {str(f.name): f for f in factors}
    ordered = [by_name[n] for n in priority] + [f for f in factors if str(f.name) not in priority]

    if BLOCK_COLUMN in Y.columns:
        blocks = Y[BLOCK_COLUMN].to_numpy()
        groups = [np.flatnonzero(blocks == b) for b in pd.unique(blocks)]
    else:
        groups = [np.arange(len(Y))]

    order = []
    for rows in groups:
        order.append(rows[_order_block(Y.iloc[rows], ordered)])
    return Y.iloc[np.concatenate(order)]


def _order_block(Y, factors):
    """Positions of the runs of one block in Gray code order, with the center runs at the end."""
    continuous = [f for f in factors if f.is_continuous]
    center = np.zeros(len(Y), dtype=bool)
    if continuous:
        center = np.all(
            [np.isclose(Y[str(f.name)], (Y[str(f.name)].min() + Y[str(f.name)].max()) / 2) for f in continuous],
            axis=0,
        )

    # Try with the center candidates set apart, then without (e.g. the middle level of a 3^k factorial)
    for centers in ([center] if center.any() else []) + [np.zeros(len(Y), dtype=bool)]:
        runs = np.flatnonzero(~centers)
        idx, levels = _level_index(Y.iloc[runs], factors)
        for method in (_full_factorial_order, _regular_fraction_order):
            positions = method(idx, levels)
            if positions is not None:
                return np.concatenate([runs[positions], np.flatnonzero(centers)])
    raise ValueError(
        "gray_code_order supports only full factorials and regular two-level fractions (with center runs and blocks)"
    )

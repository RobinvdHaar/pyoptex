from collections import Counter

import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import (
    Factor,
    fractional_factorial,
    full_factorial,
    gray_code_order,
    level_changes,
    plackett_burman,
)


def _same_runs(Y1, Y2):
    return Counter(map(tuple, Y1.to_numpy())) == Counter(map(tuple, Y2.to_numpy()))


def _changes_per_step(Y):
    X = Y.to_numpy()
    return [int(np.sum(X[i] != X[i + 1])) for i in range(len(X) - 1)]


class TestLevelChanges:
    def test_known_design(self):
        factors = [Factor("A"), Factor("B")]
        Y = pd.DataFrame({"A": [-1, -1, 1, 1, -1], "B": [-1, 1, 1, -1, 1]})
        assert level_changes(Y, factors) == {"A": 2, "B": 3}

    def test_standard_order(self):
        """In standard order the last factor changes at every step."""
        factors = [Factor(n) for n in "ABC"]
        assert level_changes(full_factorial(factors), factors) == {"A": 1, "B": 3, "C": 7}


class TestGrayCodeOrder:
    def test_two_cubed(self):
        """7 changes in total: the priority factor changes once, the others 2 and 4 times."""
        factors = [Factor(n) for n in "ABC"]
        Y = full_factorial(factors)
        Yg = gray_code_order(Y, factors, priority=["C"])
        assert level_changes(Yg, factors) == {"C": 1, "A": 2, "B": 4}
        assert _changes_per_step(Yg) == [1] * 7
        assert _same_runs(Yg, Y)

    def test_mixed_levels(self):
        """3 x 2 full factorial: exactly one factor changes per step, the 3-level factor twice in total."""
        factors = [Factor("T", levels=[1, 2, 3]), Factor("Cat", type="categorical", levels=["x", "y"])]
        Y = full_factorial(factors)
        Yg = gray_code_order(Y, factors)
        assert _changes_per_step(Yg) == [1] * 5
        assert level_changes(Yg, factors) == {"T": 2, "Cat": 3}
        assert _same_runs(Yg, Y)

    def test_three_level_factorial(self):
        """The middle run of a 3^2 factorial is a factorial run, not a center run."""
        factors = [Factor("A"), Factor("B")]
        Y = full_factorial(factors, n_levels=3)
        Yg = gray_code_order(Y, factors)
        assert _changes_per_step(Yg) == [1] * 8
        assert level_changes(Yg, factors)["A"] == 2

    def test_regular_fraction(self):
        """2^(4-1) with D = ABC and priority D, A, B, C: D changes exactly once."""
        factors = [Factor(n) for n in "ABCD"]
        Y = fractional_factorial(factors, {"D": ["A", "B", "C"]})
        Yg = gray_code_order(Y, factors, priority=["D", "A", "B", "C"])
        changes = level_changes(Yg, factors)
        assert changes["D"] == 1
        assert changes == {"D": 1, "A": 2, "B": 4, "C": 7}  # C = D + A + B: 1 + 2 + 4
        assert _same_runs(Yg, Y)

    def test_center_runs_at_the_end(self):
        factors = [Factor(n) for n in "ABC"]
        Y = full_factorial(factors, n_center=3)
        Yg = gray_code_order(Y, factors)
        np.testing.assert_array_equal(Yg.iloc[-3:].to_numpy(), 0.0)
        assert _changes_per_step(Yg.iloc[:-3]) == [1] * 7
        assert _same_runs(Yg, Y)

    def test_blocks_ordered_within(self):
        """Each block of a blocked factorial (a regular fraction) is ordered on its own, blocks kept in order."""
        factors = [Factor(n) for n in "ABCD"]
        Y = full_factorial(factors, n_blocks=2)
        Yg = gray_code_order(Y, factors)
        assert Yg["Block"].tolist() == Y["Block"].tolist()
        for _, block in Yg.groupby("Block", sort=False):
            # each block is a regular 2^(4-1): A changes once, and per step one base factor
            # changes, plus the generated factor when it depends on it
            assert level_changes(block, factors)["A"] == 1
            assert max(_changes_per_step(block[list("ABCD")])) <= 2
        assert _same_runs(Yg, Y)

    def test_index_kept(self):
        factors = [Factor(n) for n in "AB"]
        Y = full_factorial(factors)
        Yg = gray_code_order(Y, factors)
        pd.testing.assert_frame_equal(Yg.sort_index(), Y)

    def test_plackett_burman_raises(self):
        factors = [Factor(f"X{i}") for i in range(11)]
        with pytest.raises(ValueError, match="only full factorials and regular two-level fractions"):
            gray_code_order(plackett_burman(factors), factors)

    def test_unknown_priority_raises(self):
        factors = [Factor(n) for n in "AB"]
        with pytest.raises(AssertionError, match="not one of the factors"):
            gray_code_order(full_factorial(factors), factors, priority=["Z"])

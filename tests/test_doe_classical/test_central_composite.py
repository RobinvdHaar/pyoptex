from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import (
    Factor,
    add_axial_block,
    central_composite,
    fractional_factorial,
    full_factorial,
    min_aberration_generators,
)


def _sorted_rows(Y, columns):
    """The runs as a sorted list of rounded tuples, to compare designs up to run order."""
    return sorted(tuple(np.round(row, 10)) for row in Y[columns].to_numpy(dtype=np.float64))


class TestAddAxialBlock:
    def test_completes_the_blocked_ccd(self):
        """A 2^3 with 4 center runs plus the axial block is the CCD with the same blocking (up to run order)."""
        factors = [Factor(n) for n in "ABC"]
        Y = add_axial_block(full_factorial(factors, n_center=4), factors, n_center=2)
        ccd = central_composite(factors, alpha="orthogonal", n_blocks=2, n_center=(4, 2))
        columns = ["A", "B", "C", "Block"]
        assert _sorted_rows(Y, columns) == _sorted_rows(ccd, columns)

    def test_orthogonal_blocking(self):
        """Every second-order model term has the same mean in both blocks."""
        factors = [Factor(n) for n in "ABCD"]
        Y = add_axial_block(full_factorial(factors, n_center=3), factors, n_center=2)
        X = Y[list("ABCD")].to_numpy(dtype=np.float64)
        terms = [*X.T, *(X[:, i] * X[:, j] for i, j in combinations(range(4), 2)), *(X**2).T]
        blocks = Y["Block"].to_numpy()
        for t in terms:
            np.testing.assert_allclose(t[blocks == 1].mean(), t[blocks == 2].mean(), atol=1e-10)

    def test_rotatable_alpha_half_fraction(self):
        """A 2^(5-1) cube has 16 runs, so the rotatable alpha is 16^(1/4) = 2."""
        factors = [Factor(n) for n in "ABCDE"]
        cube = fractional_factorial(factors, {"E": ["A", "B", "C", "D"]}, n_center=2)
        Y = add_axial_block(cube, factors, alpha="rotatable")
        np.testing.assert_allclose(Y["A"].max(), 2.0)
        assert len(Y) == 16 + 2 + 10 + 2  # default: as many center runs as the existing block

    def test_real_units(self):
        """Alpha is measured in half the distance between the two levels, around their middle."""
        factors = [Factor("A", min=0, max=10, levels=[2, 4]), Factor("B", min=100, max=200)]
        Y = add_axial_block(full_factorial(factors), factors, alpha=2.0, n_center=1)
        axial = Y[Y["Block"] == 2]
        assert sorted(axial["A"].round(10).unique()) == [1.0, 3.0, 5.0]
        assert sorted(axial["B"].round(10).unique()) == [50.0, 150.0, 250.0]

    def test_existing_blocks(self):
        factors = [Factor(n) for n in "ABC"]
        Y = add_axial_block(full_factorial(factors, n_blocks=2, n_center=1), factors, n_center=2)
        assert Y["Block"].tolist() == [1] * 5 + [2] * 5 + [3] * 8

    def test_resolution_iv_cube_raises(self):
        factors = [Factor(n) for n in "ABCD"]
        cube = fractional_factorial(factors, {"D": ["A", "B", "C"]})
        with pytest.raises(AssertionError, match="needs resolution V"):
            add_axial_block(cube, factors)

    def test_categorical_factor_raises(self):
        factors = [Factor("A"), Factor("B"), Factor("C", type="categorical", levels=["X", "Y"])]
        with pytest.raises(AssertionError, match="must be continuous"):
            add_axial_block(full_factorial(factors), factors)

    def test_run_neither_cube_nor_center_raises(self):
        factors = [Factor("A"), Factor("B")]
        Y = pd.concat([full_factorial(factors), pd.DataFrame({"A": [0.5], "B": [0.0]})], ignore_index=True)
        with pytest.raises(AssertionError, match="Run 4 is neither a cube run"):
            add_axial_block(Y, factors)


class TestCentralComposite:
    def test_face_centered(self):
        factors = [Factor("A"), Factor("B")]
        Y = central_composite(factors, alpha="face-centered", n_center=1)
        # 2^2 + 2*2 + 1 = 4 + 4 + 1 = 9
        assert len(Y) == 9
        assert Y["A"].min() >= -1.0
        assert Y["A"].max() <= 1.0

    def test_rotatable(self):
        factors = [Factor("A"), Factor("B")]
        Y = central_composite(factors, alpha="rotatable", n_center=1)
        assert len(Y) == 9
        alpha = 4**0.25  # (2^2)^(1/4)
        np.testing.assert_allclose(Y["A"].max(), alpha, rtol=1e-10)

    def test_inscribed(self):
        factors = [Factor("A"), Factor("B")]
        Y = central_composite(factors, alpha="inscribed", n_center=1)
        assert len(Y) == 9
        np.testing.assert_allclose(Y["A"].max(), 1.0, rtol=1e-10)
        np.testing.assert_allclose(Y["A"].min(), -1.0, rtol=1e-10)

    def test_custom_alpha(self):
        factors = [Factor("A"), Factor("B")]
        Y = central_composite(factors, alpha=1.5, n_center=2)
        assert len(Y) == 10
        np.testing.assert_allclose(Y["A"].max(), 1.5, rtol=1e-10)

    def test_three_factors(self):
        factors = [Factor("A"), Factor("B"), Factor("C")]
        Y = central_composite(factors, alpha="face-centered", n_center=2)
        # 2^3 + 2*3 + 2 = 8 + 6 + 2 = 16
        assert len(Y) == 16

    def test_custom_range(self):
        factors = [Factor("A", min=10, max=20), Factor("B", min=0, max=5)]
        Y = central_composite(factors, alpha="face-centered", n_center=1)
        assert Y["A"].min() >= 10.0
        assert Y["A"].max() <= 20.0
        assert Y["B"].min() >= 0.0
        assert Y["B"].max() <= 5.0

    @pytest.mark.parametrize("k, expected_center", [(2, 5), (3, 6), (4, 7), (5, 10), (6, 15), (7, 21)])
    def test_default_center_points_full_cube(self, k, expected_center):
        """Default center points follow the Box & Hunter (1957) uniform-precision table."""
        factors = [Factor(f"X{i}") for i in range(k)]
        Y = central_composite(factors)
        assert len(Y) == 2**k + 2 * k + expected_center

    @pytest.mark.parametrize("k, expected_center", [(5, 6), (6, 9), (7, 14)])
    def test_default_center_points_half_cube(self, k, expected_center):
        factors = [Factor(f"X{i}") for i in range(k)]
        Y = central_composite(factors, cube="half")
        assert len(Y) == 2 ** (k - 1) + 2 * k + expected_center

    def test_explicit_center_points_override_default(self):
        Y = central_composite([Factor("A"), Factor("B")], n_center=2)
        assert len(Y) == 4 + 4 + 2

    def test_no_tabulated_default_raises(self):
        factors = [Factor(f"X{i}") for i in range(9)]
        with pytest.raises(AssertionError, match="specify n_center explicitly"):
            central_composite(factors)
        assert len(central_composite(factors, n_center=4)) == 512 + 18 + 4

    @pytest.mark.parametrize("cube, n_runs, expected_center", [("full", 256, 28), ("half", 128, 20), (64, 64, 13)])
    def test_default_center_points_eight_factors(self, cube, n_runs, expected_center):
        """Box & Hunter (1957), Table 3: full, half and quarter (resolution V) cube for 8 factors."""
        factors = [Factor(f"X{i}") for i in range(8)]
        if not isinstance(cube, str):
            cube = min_aberration_generators(factors, n_runs=cube)
        Y = central_composite(factors, cube=cube)
        assert len(Y) == n_runs + 16 + expected_center

    def test_half_cube_five_factors(self):
        """k = 5 with E = ABCD: 16 cube + 10 axial + center runs, rotatable alpha = 16^(1/4) = 2."""
        factors = [Factor(n) for n in "ABCDE"]
        Y = central_composite(factors, cube="half", n_center=3)
        assert len(Y) == 16 + 10 + 3
        np.testing.assert_allclose(Y["A"].max(), 2.0, rtol=1e-10)
        cube = Y.iloc[:16]
        np.testing.assert_allclose(cube["E"].values, cube[["A", "B", "C", "D"]].prod(axis=1).values)

    def test_half_cube_too_few_factors_raises(self):
        factors = [Factor(n) for n in "ABCD"]
        with pytest.raises(AssertionError, match="at least 5 factors"):
            central_composite(factors, cube="half")

    def test_custom_generators(self):
        factors = [Factor(n) for n in "ABCDE"]
        Y1 = central_composite(factors, cube={"E": ["A", "B", "C", "D"]})
        Y2 = central_composite(factors, cube="half")
        np.testing.assert_allclose(Y1.values, Y2.values)

    def test_custom_generators_without_table_entry_require_n_center(self):
        """A 2^(6-2) cube has no tabulated center points."""
        factors = [Factor(f"X{i}") for i in range(6)]
        generators = {"X4": ["X0", "X1", "X2"], "X5": ["X1", "X2", "X3"]}
        with pytest.raises(AssertionError, match="specify n_center explicitly"):
            central_composite(factors, cube=generators)
        Y = central_composite(factors, cube=generators, n_center=4)
        assert len(Y) == 16 + 12 + 4
        np.testing.assert_allclose(Y["X0"].max(), 2.0, rtol=1e-10)

    def test_unknown_cube_raises(self):
        with pytest.raises(ValueError, match="Unknown cube type"):
            central_composite([Factor("A"), Factor("B")], cube="quarter")

    def test_categorical_raises(self):
        factors = [Factor("A"), Factor("B", type="categorical", levels=["X", "Y"])]
        with pytest.raises(AssertionError, match="must be continuous"):
            central_composite(factors)

    def test_one_factor_raises(self):
        factors = [Factor("A")]
        with pytest.raises(AssertionError, match="at least 2 factors"):
            central_composite(factors)

    def test_nonpositive_alpha_raises(self):
        factors = [Factor("A"), Factor("B")]
        with pytest.raises(AssertionError, match="alpha must be strictly positive"):
            central_composite(factors, alpha=0.0)

    def test_unknown_alpha_raises(self):
        factors = [Factor("A"), Factor("B")]
        with pytest.raises(ValueError, match="Unknown alpha type"):
            central_composite(factors, alpha="bogus")

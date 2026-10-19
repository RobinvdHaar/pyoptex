import numpy as np
import pandas as pd
import pytest

from pyoptex.doe.classical import (
    WHOLE_PLOT_COLUMN,
    Factor,
    fractional_factorial,
    min_aberration_split_plot_generators,
    randomize,
    split_plot_fractional_factorial,
)
from pyoptex.doe.classical._min_aberration import _MA_GENERATORS
from pyoptex.doe.classical._split_plot_aberration import _SPLIT_PLOT_GENERATORS


def _split_factors(k1, k2):
    return [Factor(f"W{i}") for i in range(k1)], [Factor(f"S{i}") for i in range(k2)]


def _wlp(Y, names):
    """Word length pattern (A1, A2, ...) of a two-level design, from its columns."""
    X = Y[names].to_numpy(dtype=np.float64)
    bits = X > (X.min(axis=0) + X.max(axis=0)) / 2
    masks = [int(sum(1 << i for i in np.flatnonzero(col))) for col in bits.T]
    full = (1 << len(X)) - 1
    xor = np.zeros(1, dtype=object)
    for m in masks:
        xor = np.concatenate([xor, xor ^ m])
    sizes = np.array([bin(s).count("1") for s in range(len(xor))])
    words = sizes[(xor == 0) | (xor == full)]
    return tuple(int(np.sum(words == length)) for length in range(1, len(names) + 1))


def _design(label):
    """The catalogued design for a B&S label k1.k2.p1.p2, with its factor names."""
    k1, k2, p1, p2 = map(int, label.split("."))
    wp, sp = _split_factors(k1, k2)
    Y = split_plot_fractional_factorial(wp, sp, 2 ** (k1 + k2 - p1 - p2), 2 ** (k1 - p1))
    return Y, [str(f.name) for f in wp + sp]


# Bingham and Sitter (1999), Table 1 (8 runs): the first, minimum aberration, design of each k1.k2.p1.p2
_BS_TABLE_1 = {
    "1.3.0.1": (0, 1), "2.2.0.1": (0, 1),
    "1.4.0.2": (2, 1), "2.3.0.2": (2, 1), "3.2.1.1": (2, 1),
    "1.5.0.3": (4, 3), "2.4.0.3": (4, 3), "3.3.1.2": (4, 3),
    "1.6.0.4": (7, 7, 0, 0, 1), "3.4.1.3": (7, 7, 0, 0, 1),
}  # fmt: skip

# Bingham and Sitter (1999), Table 2 (16 runs). Some patterns are printed truncated.
_BS_TABLE_2 = {
    "1.4.0.1": (0, 0, 1), "2.3.0.1": (0, 0, 1), "3.2.0.1": (0, 0, 1),
    "1.5.0.2": (0, 3), "2.4.0.2": (0, 3), "3.3.0.2": (0, 3), "3.3.1.1": (1, 1, 1), "4.2.1.1": (0, 3),
    "1.6.0.3": (0, 7), "2.5.0.3": (0, 7), "3.4.0.3": (0, 7), "3.4.1.2": (2, 3, 2), "4.3.1.2": (0, 7),
    "5.2.2.1": (2, 3, 2),
    "1.7.0.4": (0, 14, 0, 0, 0, 1), "2.6.0.4": (0, 14, 0, 0, 0, 1), "3.5.0.4": (3, 7, 4, 0, 1),
    "3.5.1.3": (3, 7, 4, 0, 1), "4.4.1.3": (0, 14, 0, 0, 0, 1), "5.3.2.2": (3, 7, 4, 0, 1),
    "6.2.3.1": (4, 6, 4, 0, 0, 1),
    "1.8.0.5": (4, 14, 8, 0, 4, 1), "2.7.0.5": (4, 14, 8, 0, 4, 1), "3.6.0.5": (6, 9, 9, 6, 0, 0, 1),
    "3.6.1.4": (4, 14, 8, 0, 4, 1), "4.5.1.4": (4, 14, 8, 0, 4, 1), "5.4.2.3": (4, 14, 8, 0, 4, 1),
    "6.3.3.2": (6, 10, 8, 4, 2, 1), "7.2.4.1": (8, 10, 4, 4, 4, 1),
    "1.9.0.6": (8, 18, 16, 8, 8, 5), "2.8.0.6": (8, 18, 16, 8, 8, 5), "3.7.0.6": (9, 16, 15, 12, 7, 3, 1),
    "3.7.1.5": (8, 18, 16, 8, 8, 5), "4.6.1.5": (8, 18, 16, 8, 8, 5), "5.5.2.4": (8, 18, 16, 8, 8, 5),
    "6.4.3.3": (8, 18, 16, 8, 8, 5), "7.3.4.2": (10, 16, 12, 12, 10, 3),
    "1.10.0.7": (12, 26, 28, 24, 20, 13, 4), "2.9.0.7": (12, 26, 28, 24, 20, 13, 4),
    "3.8.0.7": (12, 26, 28, 24, 20, 13, 4), "3.8.1.6": (12, 26, 28, 24, 20, 13, 4),
    "4.7.1.6": (12, 26, 28, 24, 20, 13, 4), "5.6.2.5": (12, 26, 28, 24, 20, 13, 4),
    "6.5.3.4": (12, 26, 28, 24, 20, 13, 4), "7.4.4.3": (13, 25, 25, 27, 23, 10, 3, 1),
    "1.11.0.8": (16, 39, 48, 48, 48, 39, 16, 0, 0, 1), "2.10.0.8": (16, 39, 48, 48, 48, 39, 16, 0, 0, 1),
    "3.9.1.7": (16, 39, 48, 48, 48, 39, 16, 0, 0, 1), "4.8.1.7": (16, 39, 48, 48, 48, 39, 16, 0, 0, 1),
    # B&S print (16, 39, 48, 48, 48, 39, 16, 0, 0, 1), a misprint: their own design has this pattern
    "5.7.2.6": (17, 38, 44, 52, 54, 33, 12, 4, 1),
    "6.6.3.5": (16, 39, 48, 48, 48, 39, 16, 0, 0, 1),
    "7.5.4.4": (17, 38, 44, 52, 54, 33, 12, 4, 1),
    "1.12.0.9": (22, 55, 72, 96, 116, 87, 40, 16, 6, 1), "2.11.0.9": (22, 55, 72, 96, 116, 87, 40, 16, 6, 1),
    "3.10.1.8": (22, 55, 72, 96, 116, 87, 40, 16, 6, 1), "5.8.2.7": (22, 55, 72, 96, 116, 87, 40, 16, 6, 1),
    "6.7.3.6": (22, 55, 72, 96, 116, 87, 40, 16, 6, 1), "7.6.4.5": (22, 55, 72, 96, 116, 87, 40, 16, 6, 1),
    "1.13.0.10": (28, 77, 112, 168, 232, 203, 112, 56, 28, 7),
    "2.12.0.10": (28, 77, 112, 168, 232, 203, 112, 56, 28, 7),
    "3.11.1.9": (28, 77, 112, 168, 232, 203, 112, 56, 28, 7),
    "6.8.3.7": (28, 77, 112, 168, 232, 203, 112, 56, 28, 7),
    "7.7.4.6": (28, 77, 112, 168, 232, 203, 112, 56, 28, 7),
    "1.14.0.11": (35, 105, 168, 280, 435, 435, 280, 168, 105, 35, 0, 0, 1),
    "3.12.1.10": (35, 105, 168, 280, 435, 435, 280, 168, 105, 35, 0, 0, 1),
    "7.8.4.7": (35, 105, 168, 280, 435, 435, 280, 168, 105, 35, 0, 0, 1),
}  # fmt: skip


class TestBinghamSitter:
    @pytest.mark.parametrize("label, expected", list(_BS_TABLE_1.items()) + list(_BS_TABLE_2.items()))
    def test_word_length_pattern(self, label, expected):
        """The word length pattern from A3 onward (printed truncated for some designs: compare the prefix)."""
        Y, names = _design(label)
        wlp = _wlp(Y, names)
        assert wlp[:2] == (0, 0)  # resolution III or higher
        assert wlp[2 : 2 + len(expected)] == expected

    def test_misprint_5_7_2_6(self):
        """B&S's own design for 5.7.2.6 (WP columns 3 5, SP columns 9-14 of their Table 3) has (17, 38, ...)."""
        wp, sp = _split_factors(5, 7)
        # Yates column c: bit i is base factor i (W0, W1, W2, then S0)
        base = ["W0", "W1", "W2", "S0"]
        columns = {"W3": 3, "W4": 5, "S1": 9, "S2": 10, "S3": 11, "S4": 12, "S5": 13, "S6": 14}
        generators = {name: [base[i] for i in range(4) if c >> i & 1] for name, c in columns.items()}
        names = [str(f.name) for f in wp + sp]
        Y = fractional_factorial(wp + sp, generators)
        assert _wlp(Y, names)[2:] == _BS_TABLE_2["5.7.2.6"] + (0,)

    def test_ball_mill_example(self):
        """B&S Example 1: 4 hard-to-change and 3 easy-to-change factors in 16 runs and 8 whole plots."""
        wp = [Factor(name) for name in "ABCD"]
        sp = [Factor(name) for name in "PQR"]
        Y = split_plot_fractional_factorial(wp, sp, 16, 8)
        assert _wlp(Y, list("ABCDPQR"))[2:4] == (0, 7)  # resolution IV, as 4.3.1.2 in Table 2
        assert Y[WHOLE_PLOT_COLUMN].nunique() == 8


@pytest.mark.parametrize("key", sorted(_SPLIT_PLOT_GENERATORS), ids=lambda key: "k1={}-k2={}-N={}-WP={}".format(*key))
class TestCatalogue:
    def test_generator_rules(self, key):
        """WP generators contain WP base factors only, SP generators at least one SP base factor, no duplicates."""
        k1, k2, n_runs, n_whole_plots = key
        r, r1 = n_runs.bit_length() - 1, n_whole_plots.bit_length() - 1
        wlp, gens = _SPLIT_PLOT_GENERATORS[key]
        assert len(gens) == k1 + k2 - r
        for i, gen in enumerate(gens):
            assert len(gen) >= 2 and len(set(gen)) == len(gen) and all(0 <= b < r for b in gen)
            if i < k1 - r1:
                assert all(b < r1 for b in gen)
            else:
                assert any(b >= r1 for b in gen)
        assert len({tuple(sorted(g)) for g in gens}) == len(gens)
        assert len(wlp) == k1 + k2 - 2

    def test_design(self, key):
        """The stored pattern is that of the design; the whole plots hold the WP factors constant."""
        k1, k2, n_runs, n_whole_plots = key
        wp, sp = _split_factors(k1, k2)
        Y = split_plot_fractional_factorial(wp, sp, n_runs, n_whole_plots)
        names = [str(f.name) for f in wp + sp]
        assert len(Y) == n_runs
        assert len(Y.drop_duplicates(names)) == n_runs
        assert _wlp(Y, names)[2:] == _SPLIT_PLOT_GENERATORS[key][0]

        wp_names = names[:k1]
        plots = np.repeat(np.arange(1, n_whole_plots + 1), n_runs // n_whole_plots)
        np.testing.assert_array_equal(Y[WHOLE_PLOT_COLUMN], plots)
        assert (Y.groupby(WHOLE_PLOT_COLUMN)[wp_names].nunique() == 1).all().all()
        assert len(Y.drop_duplicates(wp_names)) == n_whole_plots


def test_no_subplot_generators_match_min_aberration_fraction():
    """With p2 = 0, the design has the pattern of the minimum aberration fraction of the WP factors (B&S p. 66)."""
    checked = 0
    for (k1, k2, n_runs, n_whole_plots), (wlp, _) in _SPLIT_PLOT_GENERATORS.items():
        r1 = n_whole_plots.bit_length() - 1
        if k1 + k2 - (n_runs.bit_length() - 1) != k1 - r1 or (k1, n_whole_plots) not in _MA_GENERATORS:
            continue
        factors = [Factor(f"W{i}") for i in range(k1)]
        _, gens = _MA_GENERATORS[(k1, n_whole_plots)]
        generators = {f"W{r1 + i}": [f"W{b}" for b in gen] for i, gen in enumerate(gens)}
        ma = _wlp(fractional_factorial(factors, generators), [str(f.name) for f in factors])
        assert wlp[: k1 - 2] == ma[2:]
        assert not any(wlp[k1 - 2 :])
        checked += 1
    assert checked > 10


class TestMinAberrationSplitPlotGenerators:
    def test_names(self):
        """3.3.1.1: C = AB and R = APQ, the design of B&S p. 66 (base factors A, B, P and Q)."""
        wp = [Factor(name) for name in "ABC"]
        sp = [Factor(name) for name in "PQR"]
        generators = min_aberration_split_plot_generators(wp, sp, 16, 4)
        assert generators == {"C": ["A", "B"], "R": ["A", "P", "Q"]}

    def test_full_factorial(self):
        wp, sp = _split_factors(2, 3)
        assert min_aberration_split_plot_generators(wp, sp, 32, 4) == {}
        Y = split_plot_fractional_factorial(wp, sp, 32, 4)
        assert len(Y.drop_duplicates([str(f.name) for f in wp + sp])) == 32

    def test_not_catalogued_raises(self):
        wp, sp = _split_factors(4, 3)
        with pytest.raises(AssertionError, match=r"available \(n_runs, n_whole_plots\): \[\(16, 8\), .*\(128, 16\)\]"):
            min_aberration_split_plot_generators(wp, sp, 16, 2)

    def test_invalid_inputs_raise(self):
        wp, sp = _split_factors(2, 2)
        with pytest.raises(AssertionError, match="n_runs must be a power of 2"):
            min_aberration_split_plot_generators(wp, sp, 12, 4)
        with pytest.raises(AssertionError, match="At least one subplot factor"):
            min_aberration_split_plot_generators(wp, [], 8, 4)
        with pytest.raises(AssertionError, match="exactly 2 levels"):
            min_aberration_split_plot_generators(wp, [Factor("S", levels=[0, 1, 2])], 8, 4)
        with pytest.raises(AssertionError, match="cannot be named"):
            min_aberration_split_plot_generators(wp, [Factor(WHOLE_PLOT_COLUMN)], 8, 4)


class TestSplitPlotFractionalFactorial:
    def test_center_whole_plot(self):
        wp, sp = _split_factors(3, 3)
        Y = split_plot_fractional_factorial(wp, sp, 16, 8, n_center=3)
        assert len(Y) == 19
        center = Y[Y[WHOLE_PLOT_COLUMN] == 9]
        assert len(center) == 3
        np.testing.assert_allclose(center.drop(columns=WHOLE_PLOT_COLUMN).to_numpy(), 0)

    def test_center_with_categorical_factor_raises(self):
        wp = [Factor("W0", type="categorical", levels=["a", "b"]), Factor("W1")]
        sp = [Factor("S0"), Factor("S1")]
        assert len(split_plot_fractional_factorial(wp, sp, 8, 4)) == 8
        with pytest.raises(AssertionError, match="categorical factor W0"):
            split_plot_fractional_factorial(wp, sp, 8, 4, n_center=1)

    def test_real_units(self):
        wp = [Factor("Temp", min=150, max=200), Factor("Speed", levels=[10, 20])]
        sp = [Factor("Time", min=1, max=3), Factor("Load", min=0, max=5)]
        Y = split_plot_fractional_factorial(wp, sp, 8, 4)
        assert set(Y["Temp"]) == {150, 200} and set(Y["Speed"]) == {10, 20}
        assert set(Y["Time"]) == {1, 3} and set(Y["Load"]) == {0, 5}

    def test_randomize_whole_plots(self):
        """randomize keeps each whole plot together, shuffles the whole plots and the runs within them."""
        wp, sp = _split_factors(4, 3)
        Y = split_plot_fractional_factorial(wp, sp, 16, 8)
        R = randomize(Y, block_column=WHOLE_PLOT_COLUMN, shuffle_blocks=True)
        assert sorted(R.index) == list(range(16))
        plots = R[WHOLE_PLOT_COLUMN].to_numpy()
        assert len(np.flatnonzero(np.diff(plots))) == 7  # contiguous whole plots
        pd.testing.assert_frame_equal(R.sort_index(), Y)

from itertools import combinations, product

import numpy as np
import pytest

from pyoptex.doe.classical import WHOLE_PLOT_COLUMN, Factor, randomize, split_plot_central_composite


def _factors(w, k):
    return [Factor(f"z{i + 1}") for i in range(w)], [Factor(f"x{i + 1}") for i in range(k)]


def _ccd(w, k, **kwargs):
    wp, sp = _factors(w, k)
    return split_plot_central_composite(wp, sp, **kwargs)


def _second_order_matrix(D):
    """Intercept, main effects, two-factor interactions and pure quadratics."""
    f = D.shape[1]
    cols = [np.ones(len(D))] + [D[:, i] for i in range(f)]
    cols += [D[:, i] * D[:, j] for i, j in combinations(range(f), 2)]
    cols += [D[:, i] ** 2 for i in range(f)]
    return np.column_stack(cols)


def _coded(Y):
    return Y.drop(columns=WHOLE_PLOT_COLUMN).to_numpy(dtype=np.float64)


def _equivalent(Y):
    """PKV eq. (4): X (X'X)^-1 X' J_b X == J_b X, with J_b the block-diagonal matrix of ones per whole plot."""
    X = _second_order_matrix(_coded(Y))
    plots = Y[WHOLE_PLOT_COLUMN].to_numpy()
    JX = (plots[:, None] == plots[None, :]).astype(np.float64) @ X
    return np.allclose(X @ np.linalg.solve(X.T @ X, X.T @ JX), JX, atol=1e-8)


_RANGE = [(w, k) for w in (1, 2, 3) for k in (1, 2, 3, 4)]

# PKV Table B.1 (p. 64): (m, n) for VKM, MWP with beta != 1 and MWP with beta = 1
_TABLE_B1 = {
    (1, 1): ((6, 2), (5, 3), (3, 3)),
    (1, 2): ((6, 4), (5, 5), (3, 5)),
    (1, 3): ((8, 8), (5, 8), (3, 9)),
    (1, 4): ((6, 8), (5, 9), (3, 9)),
    (2, 1): ((10, 2), (9, 3), None),
    (2, 2): ((10, 4), (9, 5), None),
    (2, 3): ((12, 4), (9, 7), None),
    (2, 4): ((10, 8), (9, 9), None),
    (3, 1): ((16, 2), (15, 3), None),
    (3, 2): ((16, 4), (15, 5), None),
    (3, 3): ((18, 4), (15, 7), None),
    (3, 4): ((16, 8), (15, 9), None),
}

# PKV Table 1 (p. 61): alpha = beta for the MWP method
_TABLE_1 = {
    (1, 1): 2, (1, 2): 3, (1, 3): 5, (1, 4): 5,
    (2, 1): 3, (2, 2): 4, (2, 3): 4, (2, 4): 6,
    (3, 1): 4, (3, 2): 5, (3, 3): 5, (3, 4): 7,
}  # fmt: skip

_VARIANTS = [("vkm", {}), ("mwp", {})]


def _variants(w):
    return _VARIANTS + ([("mwp", {"beta": 1})] if w == 1 else [])


class TestEquivalence:
    @pytest.mark.parametrize("w, k", _RANGE)
    def test_equivalent_estimation_and_full_rank(self, w, k):
        for method, kwargs in _variants(w):
            Y = _ccd(w, k, method=method, **kwargs)
            X = _second_order_matrix(_coded(Y))
            assert np.linalg.matrix_rank(X) == X.shape[1], (method, kwargs)
            assert _equivalent(Y), (method, kwargs)

    @pytest.mark.parametrize("w, k", _RANGE)
    def test_vkm_any_alpha_and_beta(self, w, k):
        for alpha, beta in [(1, 1), (1.5, 2), (2.2, 0.7)]:
            assert _equivalent(_ccd(w, k, method="vkm", alpha=alpha, beta=beta))

    @pytest.mark.parametrize("w, k", [(1, 2), (2, 2), (2, 3), (3, 4)])
    def test_mwp_given_beta(self, w, k):
        """alpha from PKV eq. (16) for a beta other than the default."""
        Y = _ccd(w, k, method="mwp", beta=np.sqrt(w) + 1)
        assert _equivalent(Y)
        assert Y.attrs["alpha"] != pytest.approx(Y.attrs["beta"])

    @pytest.mark.parametrize("w, k", [(1, 2), (2, 2), (3, 3)])
    def test_mwp_wrong_alpha_breaks_equivalence(self, w, k):
        """Sensitivity of the check: the MWP design with alpha away from eq. (16) is not equivalent."""
        Y = _ccd(w, k, method="mwp")
        assert _equivalent(Y)
        D = _coded(Y)
        sp = slice(w, w + k)
        D[:, sp] = np.where(np.isclose(np.abs(D[:, sp]), Y.attrs["alpha"]), np.sign(D[:, sp]) * 1.1, D[:, sp])
        Z = Y.copy()
        Z.iloc[:, : w + k] = D
        assert not _equivalent(Z)


class TestPKVTables:
    @pytest.mark.parametrize("w, k", _RANGE)
    def test_whole_plots_and_sizes(self, w, k):
        """m whole plots of n runs, PKV Table B.1."""
        for (method, kwargs), expected in zip(_variants(w), _TABLE_B1[(w, k)], strict=False):
            Y = _ccd(w, k, method=method, **kwargs)
            sizes = Y[WHOLE_PLOT_COLUMN].value_counts()
            assert (len(sizes), sizes.iloc[0]) == expected, (method, kwargs)
            assert sizes.nunique() == 1  # balanced

    @pytest.mark.parametrize("w, k", _RANGE)
    def test_mwp_alpha_equals_beta(self, w, k):
        """alpha = beta for the MWP method, PKV Table 1."""
        Y = _ccd(w, k, method="mwp")
        assert Y.attrs["alpha"] == pytest.approx(np.sqrt(_TABLE_1[(w, k)]))
        assert Y.attrs["beta"] == pytest.approx(np.sqrt(_TABLE_1[(w, k)]))

    @staticmethod
    def _canonical(plots):
        """Whole plots as a multiset of multisets of runs."""
        return sorted(sorted(tuple(np.round(run, 9)) for run in runs) for runs in plots)

    def _assert_design(self, Y, plots):
        observed = [_coded(Y[Y[WHOLE_PLOT_COLUMN] == p]) for p in Y[WHOLE_PLOT_COLUMN].unique()]
        assert self._canonical(observed) == self._canonical(plots)

    @staticmethod
    def _plot(z, x_runs):
        return [tuple(z) + tuple(x) for x in x_runs]

    def test_table_a1_vkm_d22(self):
        a, b = 1.3, 1.7
        sq = list(product((-1, 1), repeat=2))
        plots = [self._plot(z, sq) for z in product((-1, 1), repeat=2)]
        plots += [self._plot(z, [(0, 0)] * 4) for z in [(-b, 0), (b, 0), (0, -b), (0, b)]]
        plots += [self._plot((0, 0), [(-a, 0), (a, 0), (0, -a), (0, a)]), self._plot((0, 0), [(0, 0)] * 4)]
        self._assert_design(_ccd(2, 2, method="vkm", alpha=a, beta=b), plots)

    def test_table_a2_vkm_d12(self):
        a, b = 1.3, 1.7
        sq = list(product((-1, 1), repeat=2))
        plots = [self._plot((z,), sq) for z in (-1, 1)]
        plots += [self._plot((z,), [(0, 0)] * 4) for z in (-b, b)]
        plots += [self._plot((0,), [(-a, 0), (a, 0), (0, -a), (0, a)]), self._plot((0,), [(0, 0)] * 4)]
        self._assert_design(_ccd(1, 2, method="vkm", alpha=a, beta=b), plots)

    def test_table_a3_vkm_d13(self):
        a, b = 1.3, 1.7
        cube = list(product((-1, 1), repeat=3))
        plots = [self._plot((z,), cube) for z in (-1, 1)]
        plots += [self._plot((z,), [(0, 0, 0)] * 8) for z in (-b, b)]
        for j in range(3):
            lo, hi = [0, 0, 0], [0, 0, 0]
            lo[j], hi[j] = -a, a
            plots.append(self._plot((0,), [lo, hi] * 4))
        plots.append(self._plot((0,), [(0, 0, 0)] * 8))
        self._assert_design(_ccd(1, 3, method="vkm", alpha=a, beta=b), plots)

    def test_table_a4_mwp_beta_one_d12(self):
        a = 1.5
        sq = list(product((-1, 1), repeat=2))
        plots = [self._plot((z,), [*sq, (0, 0)]) for z in (-1, 1)]
        plots.append(self._plot((0,), [(-a, 0), (a, 0), (0, -a), (0, a), (0, 0)]))
        self._assert_design(_ccd(1, 2, method="mwp", alpha=a, beta=1), plots)

    def test_table_a5_mwp_d22(self):
        a = b = 2.0  # sqrt(4), PKV p. 61
        sq = list(product((-1, 1), repeat=2))
        plots = [self._plot(z, [*sq, (0, 0)]) for z in product((-1, 1), repeat=2)]
        plots += [self._plot(z, [(0, 0)] * 5) for z in [(-b, 0), (b, 0), (0, -b), (0, b)]]
        plots.append(self._plot((0, 0), [(-a, 0), (a, 0), (0, -a), (0, a), (0, 0)]))
        self._assert_design(_ccd(2, 2, method="mwp"), plots)

    def test_table_a6_mwp_d12(self):
        a = b = np.sqrt(3)  # PKV p. 61
        sq = list(product((-1, 1), repeat=2))
        plots = [self._plot((z,), [*sq, (0, 0)]) for z in (-1, 1)]
        plots += [self._plot((z,), [(0, 0)] * 5) for z in (-b, b)]
        plots.append(self._plot((0,), [(-a, 0), (a, 0), (0, -a), (0, a), (0, 0)]))
        self._assert_design(_ccd(1, 2, method="mwp"), plots)


class TestStructure:
    @pytest.mark.parametrize("w, k", _RANGE)
    def test_whole_plot_factors_constant(self, w, k):
        for method, kwargs in _variants(w):
            Y = _ccd(w, k, method=method, **kwargs)
            assert (Y.groupby(WHOLE_PLOT_COLUMN)[[f"z{i + 1}" for i in range(w)]].nunique() == 1).all().all()
            assert Y[WHOLE_PLOT_COLUMN].is_monotonic_increasing

    def test_half_fraction_generator(self):
        """D(2, 3): x3 = z1 z2 x1 x2, so every factorial whole plot holds a half fraction in the subplot factors."""
        Y = _ccd(2, 3, method="vkm")
        D = _coded(Y)
        factorial = np.all(np.abs(D) == 1, axis=1)
        assert factorial.sum() == 16
        np.testing.assert_array_equal(D[factorial, 4], np.prod(D[factorial, :4], axis=1))

    def test_real_units(self):
        wp = [Factor("Gap", min=0.1, max=0.3)]
        sp = [Factor("Angle", min=5, max=15), Factor("Reynolds", min=2, max=4)]
        Y = split_plot_central_composite(wp, sp, method="mwp")
        a = Y.attrs["alpha"]
        assert sorted(Y["Gap"].unique()) == pytest.approx([0.2 - 0.1 * a, 0.1, 0.2, 0.3, 0.2 + 0.1 * a])
        assert Y["Angle"].min() == pytest.approx(10 - 5 * a)
        assert set(Y.columns) == {"Gap", "Angle", "Reynolds", WHOLE_PLOT_COLUMN}

    def test_vkm_center_whole_plots(self):
        assert len(_ccd(2, 2, method="vkm", n_center_whole_plots=3)[WHOLE_PLOT_COLUMN].unique()) == 12
        assert len(_ccd(2, 2, method="vkm", n_center_whole_plots=0)[WHOLE_PLOT_COLUMN].unique()) == 9

    def test_randomize(self):
        Y = _ccd(2, 2, method="mwp")
        R = randomize(Y, block_column=WHOLE_PLOT_COLUMN, shuffle_blocks=True)
        assert len(np.flatnonzero(np.diff(R[WHOLE_PLOT_COLUMN].to_numpy()))) == 8
        assert sorted(R.index) == list(range(len(Y)))


class TestErrors:
    def test_out_of_range(self):
        with pytest.raises(AssertionError, match="optimal split-plot design"):
            _ccd(4, 2)
        with pytest.raises(AssertionError, match="optimal split-plot design"):
            _ccd(1, 5)

    def test_mwp_alpha_given(self):
        with pytest.raises(AssertionError, match="alpha follows from beta"):
            _ccd(2, 2, method="mwp", alpha=1.5)

    def test_mwp_beta_too_small(self):
        with pytest.raises(AssertionError, match="beta must exceed sqrt"):
            _ccd(2, 2, method="mwp", beta=1.2)

    def test_mwp_beta_one_requires_one_whole_plot_factor(self):
        with pytest.raises(AssertionError, match="requires 1 whole-plot factor"):
            _ccd(2, 2, method="mwp", beta=1)

    def test_mwp_center_whole_plot(self):
        with pytest.raises(AssertionError, match="breaks the equivalence"):
            _ccd(2, 2, method="mwp", n_center_whole_plots=1)

    def test_invalid_factors(self):
        with pytest.raises(AssertionError, match="must be continuous"):
            split_plot_central_composite([Factor("z", type="categorical", levels=["a", "b"])], [Factor("x")])
        with pytest.raises(AssertionError, match="method must be"):
            _ccd(1, 1, method="ccd")

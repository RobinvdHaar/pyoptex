import numpy as np
import pytest

from pyoptex._seed import set_seed
from pyoptex.doe import cost_optimal, fixed_structure
from pyoptex.doe.constraints import parse_constraints_script
from pyoptex.doe.cost_optimal.codex import create_parameters as create_codex_parameters
from pyoptex.doe.cost_optimal.codex import default_fn as codex_default_fn
from pyoptex.doe.cost_optimal.cost import parallel_worker_cost
from pyoptex.doe.cost_optimal.init import init
from pyoptex.doe.cost_optimal.metric import Dopt as CodexDopt
from pyoptex.doe.fixed_structure import init as fixed_structure_init
from pyoptex.doe.fixed_structure.init import init_random
from pyoptex.doe.fixed_structure.metric import Dopt
from pyoptex.doe.utils.init import design_region_bounds, init_single_unconstrained
from pyoptex.utils.model import model2Y2X, partial_rsm_names


def _factors(factor_class, mixture, **kwargs):
    """Dough of flour 0.5 to 0.8 and sugar 0 to 0.1 (water implied) with an oven temperature, or a design without
    mixture components."""
    if mixture:
        return [
            factor_class("flour", type="mixture", levels=[0.5, 0.8], **kwargs),
            factor_class("sugar", type="mixture", levels=[0, 0.1], **kwargs),
            factor_class("temperature", **kwargs),
        ]
    return [
        factor_class("A", **kwargs),
        factor_class("B", **kwargs),
        factor_class("C", type="categorical", levels=["x", "y", "z"], **kwargs),
    ]


def _fixed_structure(mixture, constraints):
    factors = _factors(fixed_structure.Factor, mixture)
    Y2X = model2Y2X(partial_rsm_names({str(f.name): "lin" for f in factors}), factors)
    constraints = parse_constraints_script(constraints, factors) if constraints else None
    params = fixed_structure.create_parameters(
        factors, fixed_structure.default_fn(factors, Dopt(), Y2X, constraints=constraints), 10
    )
    return params, lambda n: init_random(params, n, complete=True)


def _codex(mixture, constraints):
    factors = _factors(cost_optimal.Factor, mixture, grouped=False)
    Y2X = model2Y2X(partial_rsm_names({str(f.name): "lin" for f in factors}), factors)
    constraints = parse_constraints_script(constraints, factors) if constraints else None
    cost = parallel_worker_cost({str(f.name): 1 for f in factors}, factors, 100, 5)
    params = create_codex_parameters(
        factors, codex_default_fn(10, factors, cost, CodexDopt(), Y2X, constraints=constraints)
    )
    return params, lambda n: init(params, n, complete=True)


@pytest.mark.parametrize("design", [_fixed_structure, _codex], ids=["fixed_structure", "codex"])
def test_mixture_samples_lie_in_the_mixture_region(design):
    """Every share lies within the range of its levels and the shares sum to at most 1; the process variable still
    covers its full range [-1, 1]."""
    _, sample = design(mixture=True, constraints=None)
    Y = sample(5000)
    flour, sugar, temperature = Y[:, 0], Y[:, 1], Y[:, 2]
    assert flour.min() >= 0.5 and flour.max() <= 0.8
    assert sugar.min() >= 0 and sugar.max() <= 0.1
    assert np.all(flour + sugar <= 1 + 1e-6)
    assert temperature.min() >= -1 and temperature.max() <= 1
    assert temperature.min() < -0.99 and temperature.max() > 0.99


@pytest.mark.parametrize("design", [_fixed_structure, _codex], ids=["fixed_structure", "codex"])
def test_without_mixture_components_the_samples_are_unchanged(design):
    """Without mixture components every factor keeps the bounds -1 and 1, with the same number of draws: the samples
    equal those of the plain rejection loop on init_single_unconstrained without bounds."""
    params, sample = design(mixture=False, constraints="`A` + `B` > 0.5")
    set_seed(7)
    Y = sample(2000)

    set_seed(7)
    expected = np.zeros_like(Y)
    invalid = np.ones(len(expected), dtype=np.bool_)
    coords = [None if f.is_continuous else params.coords[i] for i, f in enumerate(params.factors)]
    while np.any(invalid):
        expected[invalid] = init_single_unconstrained(params.colstart, coords, expected[invalid], params.effect_types)
        invalid[invalid] = params.fn.constraints(expected[invalid])
    np.testing.assert_array_equal(Y, expected)


def test_mixture_with_high_lowest_levels_is_sampled_efficiently():
    """Three components of 30 to 100 % each (the fourth implied): only shares up to 40 % can occur. Sampling each up
    to 40 % gives about 1 feasible mixture in 6 proposals (instead of 1 in 2000 up to 100 %)."""
    factors = [fixed_structure.Factor(name, type="mixture", levels=[0.3, 1]) for name in "ABC"]
    Y2X = model2Y2X(partial_rsm_names({str(f.name): "lin" for f in factors}), factors)
    params = fixed_structure.create_parameters(factors, fixed_structure.default_fn(factors, Dopt(), Y2X), 10)

    # Every proposal is checked against the constraints exactly once
    proposals = 0

    def counted_constraints(Y):
        nonlocal proposals
        proposals += len(Y)
        return constraints(Y)

    constraints = params.fn.constraints
    params = params._replace(fn=params.fn._replace(constraints=counted_constraints))

    n = 5000
    Y = init_random(params, n, complete=True)
    assert Y.min() >= 0.3 and Y.max() <= 0.4
    assert np.all(Y.sum(axis=1) <= 1 + 1e-6)
    assert proposals <= 6.5 * n


def _mixture_params(levels):
    """Fixed structure parameters for mixture factors with the given levels, and a counter of the proposals: every
    proposal is checked against the constraints exactly once."""
    factors = [fixed_structure.Factor(f"X{i}", type="mixture", levels=lv) for i, lv in enumerate(levels)]
    Y2X = model2Y2X(partial_rsm_names({str(f.name): "lin" for f in factors}), factors)
    params = fixed_structure.create_parameters(factors, fixed_structure.default_fn(factors, Dopt(), Y2X), 10)
    proposals = [0]
    constraints = params.fn.constraints

    def counted_constraints(Y):
        proposals[0] += len(Y)
        return constraints(Y)

    return params._replace(fn=params.fn._replace(constraints=counted_constraints)), proposals


def test_broad_mixture_is_sampled_from_the_simplex():
    """The recipe keeps the box; 10 components of 0 to 1 each, of which the box holds only 1 feasible mixture in
    9! = 362,880, are drawn from the simplex with few proposals."""
    assert not design_region_bounds(_factors(fixed_structure.Factor, mixture=True))[2]

    params, proposals = _mixture_params([[0, 1]] * 9)
    assert design_region_bounds(params.factors)[2]
    Y = init_random(params, 10000, complete=True)
    assert Y.min() >= 0 and np.all(Y.sum(axis=1) <= 1 + 1e-6)
    assert proposals[0] < 50000


def test_box_and_simplex_give_the_same_distribution(monkeypatch):
    """Both proposals are uniform on the same mixtures, so their samples agree within Monte Carlo error."""
    params, _ = _mixture_params([[0.1, 0.6], [0, 0.5]])
    samples = []
    for simplex in (False, True):
        monkeypatch.setattr(
            fixed_structure_init,
            "design_region_bounds",
            lambda factors, simplex=simplex: (*design_region_bounds(factors)[:2], simplex),
        )
        Y = init_random(params, 20000, complete=True)
        assert Y[:, 0].min() >= 0.1 and Y[:, 0].max() <= 0.6 and Y[:, 1].max() <= 0.5
        assert np.all(Y.sum(axis=1) <= 1 + 1e-6)
        samples.append(Y)
    box, simplex = samples
    np.testing.assert_allclose(box.mean(axis=0), simplex.mean(axis=0), atol=0.005)
    np.testing.assert_allclose(
        np.quantile(box, [0.1, 0.5, 0.9], axis=0), np.quantile(simplex, [0.1, 0.5, 0.9], axis=0), atol=0.01
    )

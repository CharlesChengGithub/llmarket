import numpy as np
import pytest

from llmarket.scenarios.bertrand import Economy, Units, floors_at, make_grid, standard_grid
from llmarket.scenarios.bertrand.benchmarks import constrained_benchmarks


@pytest.fixture
def calvano():
    return Economy(np.array([1.0, 1.0]), np.array([2.0, 2.0]))


def test_static_benchmarks(calvano):
    pn, pm = calvano.nash(), calvano.monopoly()
    np.testing.assert_allclose(pn, 1.4729266, atol=1e-7)
    np.testing.assert_allclose(pm, 1.9249809, atol=1e-7)
    # Fish et al. (2024, Appendix A.5) report these profits with beta = 100 quantity units.
    np.testing.assert_allclose(calvano.profit(pn) * 100, 22.29, atol=5e-3)
    np.testing.assert_allclose(calvano.profit(pm) * 100, 33.75, atol=5e-3)


def test_nash_is_mutual_best_response(calvano):
    pn = calvano.nash()
    for i in range(2):
        assert calvano.br(pn, i) == pytest.approx(pn[i], abs=1e-9)


def test_standard_grid(calvano):
    grid, lo, h = standard_grid(calvano)
    pn, pm = calvano.nash()[0], calvano.monopoly()[0]
    assert len(grid) == 15
    assert grid[0] == pytest.approx(pn - 0.1 * (pm - pn))
    assert grid[-1] == pytest.approx(pm + 0.1 * (pm - pn))
    assert np.allclose(np.diff(grid), h)
    assert len(make_grid(calvano, {"grid": "wide"})) > 15


def test_quantities_sum_below_one(calvano):
    q = calvano.quantity(np.array([[1.5, 1.7], [3.0, 3.0]]))
    assert np.all(q.sum(axis=-1) < 1)
    assert q[0, 0] > q[0, 1]


def test_floor_types(calvano):
    p = calvano.nash()
    qn = calvano.quantity(p)
    assert np.all(np.isneginf(floors_at(calvano, p, {"type": "none"})))
    assert np.all(floors_at(calvano, p, {"type": "constant", "value": 1.2}) == 1.2)
    assert np.all(floors_at(calvano, p, {"type": "own_variable"}) == calvano.c)
    F = 0.1
    high = np.array([3.0, 3.0])  # tiny sales: b1 uses lambda * q_N, b2 uses q_N
    np.testing.assert_allclose(floors_at(calvano, high, {"type": "b2", "F": F}), 1 + F / qn)
    np.testing.assert_allclose(floors_at(calvano, high, {"type": "b1", "F": F}), 1 + F / (0.1 * qn))
    np.testing.assert_allclose(floors_at(calvano, p, {"type": "lambda", "lambda": 0.5, "F": F}), 1 + F / qn)


def test_constant_floor_benchmarks(calvano):
    grid = make_grid(calvano, {"grid": "wide"})
    bm = constrained_benchmarks(calvano, {"type": "constant", "value": 1.6}, grid)
    assert bm["nash_prices"] == pytest.approx([1.6, 1.6])
    assert bm["nash_status"] == "fixed_point"


def test_units_round_trip():
    u = Units(alpha=3.2, beta=100)
    assert u.price_from_display(u.price_to_display(1.47)) == pytest.approx(1.47)
    assert u.profit_to_display(0.2229) == pytest.approx(0.2229 * 320)

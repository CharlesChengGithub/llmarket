import numpy as np
import pytest

numba = pytest.importorskip("numba")


@numba.njit
def _draws(seed, k):
    np.random.seed(seed)
    out = np.empty(k)
    for j in range(k):
        out[j] = np.random.random()
    return out


@pytest.mark.parametrize("seed", [0, 1, 1095125285, 4294967295])
def test_numba_stream_matches_numpy_randomstate(seed):
    """The step-level agents rely on this: kernel draws == RandomState(seed).random()."""
    rs = np.random.RandomState(seed)
    expected = np.array([rs.random() for _ in range(10000)])
    assert np.array_equal(_draws(seed, 10000), expected)

import numpy as np
import pytest

from rl_agents.utils import bernoulli_kullback_leibler, d_bernoulli_kullback_leibler_dq, kl_upper_bound


def test_bernoulli_kullback_leibler():
    assert bernoulli_kullback_leibler(0, 1) == np.infty
    q = np.random.random()
    assert bernoulli_kullback_leibler(0, q) > 0
    assert bernoulli_kullback_leibler(q, q) == 0

    x = np.random.uniform(0, 1, 10)
    x.sort()
    for i in range(np.size(x) - 1):
        assert bernoulli_kullback_leibler(x[0], x[i]) < bernoulli_kullback_leibler(x[0], x[i+1])


def test_d_bernoulli_kullback_leibler_dq():
    x = np.random.uniform(0, 1, 2)
    p, q = x
    eps = 1e-6
    assert d_bernoulli_kullback_leibler_dq(p, q) == \
        pytest.approx((bernoulli_kullback_leibler(p, q+eps) - bernoulli_kullback_leibler(p, q-eps)) / (2*eps), 1e-3)


def test_kl_upper_bound():
    assert kl_upper_bound(0.5 * 1, 1, 10, c=1, eps=1e-3) == pytest.approx(0.997, abs=1e-3)
    assert kl_upper_bound(0.5 * 10, 10, 20, c=1, eps=1e-3) == pytest.approx(0.835, abs=1e-3)
    assert kl_upper_bound(0.5 * 20, 20, 40, c=1, eps=1e-3) == pytest.approx(0.777, abs=1e-3)

    rands = np.random.randint(1, 500, 2)
    rands.sort()
    mu, count, time = np.random.random(), rands[0], rands[1]
    ucb = kl_upper_bound(mu*count, count, time, c=1, eps=1e-3)
    assert not np.isnan(ucb)
    d_max = 1 * np.log(time) / count
    assert bernoulli_kullback_leibler(mu, ucb) == pytest.approx(d_max, abs=1e-2)

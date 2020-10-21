import numpy as np
import pytest

from rl_agents.utils import bernoulli_kullback_leibler, d_bernoulli_kullback_leibler_dq, kl_upper_bound, \
    max_expectation_under_constraint, kullback_leibler


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
    assert kl_upper_bound(0.5 * 1, 1, threshold=np.log(10), eps=1e-3) == pytest.approx(0.997, abs=1e-3)
    assert kl_upper_bound(0.5 * 10, 10, threshold=np.log(20), eps=1e-3) == pytest.approx(0.835, abs=1e-3)
    assert kl_upper_bound(0.5 * 20, 20, threshold=np.log(40), eps=1e-3) == pytest.approx(0.777, abs=1e-3)

    rands = np.random.randint(1, 500, 2)
    rands.sort()
    mu, count, time = np.random.random(), rands[0], rands[1]
    ucb = kl_upper_bound(mu*count, count, threshold=np.log(time), eps=1e-3)
    assert not np.isnan(ucb)
    d_max = 1 * np.log(time) / count
    assert bernoulli_kullback_leibler(mu, ucb) == pytest.approx(d_max, abs=1e-1)


def test_max_expectation_constrained():
    # Edge case 1
    q = np.array([0, 0, 1, 1], dtype='float')
    q /= q.sum()
    f = np.array([1, 1, 0, 0])
    c = 0.3
    p = max_expectation_under_constraint(f, q, c, eps=1e-3)
    kl = kullback_leibler(q, p)
    print(q @ f, p @ f, kl, c)
    assert q @ f <= p @ f
    assert c - 1e-1 <= kl <= c + 1e-1

    # Edge case 2
    q = np.array([0, 1,  1], dtype='float')
    q /= q.sum()
    f = np.array([0, 1, 1])
    c = 0.1
    p = max_expectation_under_constraint(f, q, c, eps=1e-3)
    kl = kullback_leibler(q, p)
    print(q @ f, p @ f, kl, c)
    assert q @ f <= p @ f
    assert kl <= c + 1e-1

    # Random distribution
    for _ in range(100):
        q = np.random.random(10)
        q /= q.sum()
        f = np.random.random(10)
        c = np.random.random()
        p = max_expectation_under_constraint(f, q, c, eps=1e-4)
        kl = q @ np.log(q/p)
        assert q @ f <= p @ f
        assert c - 1e-1 <= kl <= c + 1e-1

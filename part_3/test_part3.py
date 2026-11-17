"""Tests for Part 3 on synthetic data (no dataset needed). Run with:  pytest -q"""
import numpy as np
import pytest

import part3_model as p3


def synthetic(n=1000, seed=0, a=0.4, b=0.05, noise=0.2):
    rng = np.random.default_rng(seed)
    d = rng.uniform(3, 60, n)
    return d, a + b * d + rng.normal(0, noise, n)


def test_mean_model_predicts_training_mean():
    d, t = synthetic()
    c = p3.fit_model(0, d, t)
    np.testing.assert_allclose(p3.predict(c, np.array([1.0, 50.0])), t.mean())


def test_linear_recovers_line():
    d, t = synthetic(noise=0.0)
    c = p3.fit_model(1, d, t)
    np.testing.assert_allclose(c, [0.05, 0.4], atol=1e-10)


def test_scores():
    s = p3.scores(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]))
    assert s["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert s["mae"] == pytest.approx(2 / 3)
    assert s["bias"] == pytest.approx(0.0)


def test_random_split_is_a_partition():
    tr, te = p3.random_split(1000, seed=3)
    assert len(te) == 200 and len(tr) == 800
    assert len(np.intersect1d(tr, te)) == 0
    assert set(np.r_[tr, te]) == set(range(1000))


def test_short_long_split_tests_on_longest():
    d, _ = synthetic()
    tr, te = p3.short_long_split(d)
    assert len(te) == 200
    assert d[tr].max() <= d[te].min()
    np.testing.assert_array_equal(np.sort(d)[-200:], np.sort(d[te]))


def test_linear_beats_baseline_and_extrapolates_a_true_line():
    d, t = synthetic()
    rows, _, boot = p3.evaluate(d, t, *p3.short_long_split(d))
    r = {row["model"]: row for row in rows}
    assert r["linear"]["test_rmse"] < r["mean (baseline)"]["test_rmse"]
    assert abs(r["linear"]["test_bias"]) < 0.05           # unbiased when the truth is a line
    assert r["mean (baseline)"]["test_bias"] < -1.0       # mean of short passes is far too low
    med, lo, hi = p3.paired_difference(boot, "linear", "mean (baseline)")
    assert hi < 0                                          # linear better, CI excludes zero
    assert r["linear"]["rmse_ci_lo"] < r["linear"]["test_rmse"] < r["linear"]["rmse_ci_hi"]

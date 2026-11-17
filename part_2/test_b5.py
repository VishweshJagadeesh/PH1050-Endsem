"""Tests for the B5 lofted-ball comparison (small simulated table, no dataset needed)."""
import numpy as np
import pytest

import part2_lofted_comparison as b5   # also puts Part_1 on the import path

sim = b5.sim

ANGLES = (20.0, 30.0, 40.0)


@pytest.fixture(scope="module")
def table():
    return b5.flight_time_table(angles_deg=ANGLES, v0s=np.arange(15.0, 60.01, 1.0))


def test_interpolated_time_matches_root_find(table):
    for a in ANGLES:
        _, T = sim.flight_time_for_range(45.0, np.radians(a))
        assert b5.sim_flight_time(table, a, 45.0) == pytest.approx(T, abs=2e-3)


def test_compare_on_known_offset():
    T_sim = np.array([2.0, 2.5, 3.0, 3.5])
    c = b5.compare(T_sim + 0.3, T_sim)
    assert c["mean"] == pytest.approx(0.3)
    assert c["se"] == pytest.approx(0.0, abs=1e-12)
    assert c["rmse"] == pytest.approx(0.3)


def test_effective_angle_recovers_the_angle_used(table):
    D = np.linspace(32, 58, 40)
    T_obs = b5.sim_flight_time(table, 30.0, D)       # data that are exactly 30-degree lofts
    ang, (lo, hi), n_out = b5.effective_angle(table, D, T_obs, n_boot=200)
    assert ang == pytest.approx(30.0, abs=0.01)
    assert n_out == 0


def test_table_too_short_raises(table):
    with pytest.raises(ValueError):
        b5.sim_flight_time(table, 20.0, 500.0)

"""Tests for Part 1 (physics model). Run with:  pytest -q"""
import numpy as np
import pytest

import part_1.COMPHY_FOOTBALL_SIMULATOR as sim

V0, TH = 25.0, np.radians(35)


def analytic(v0, theta):
    return v0**2 * np.sin(2 * theta) / sim.G, 2 * v0 * np.sin(theta) / sim.G


def fitted_order(method, drag, ref):
    dts = np.logspace(-1, -2.5, 7)
    err = [abs(sim.simulate(V0, TH, dt, method=method, drag=drag)["range"] - ref) for dt in dts]
    return np.polyfit(np.log(dts), np.log(err), 1)[0]


# ---- B2: known no-drag case ----
@pytest.mark.parametrize("v0,deg", [(25.0, 35.0), (20.0, 45.0), (30.0, 20.0)])
def test_rk4_no_drag_matches_analytic(v0, deg):
    th = np.radians(deg)
    R_an, T_an = analytic(v0, th)
    r = sim.simulate(v0, th, 0.01, method="rk4", drag=False)
    assert abs(r["range"] - R_an) < 1e-9
    assert abs(r["flight_time"] - T_an) < 1e-9


def test_euler_no_drag_close_to_analytic():
    R_an, T_an = analytic(V0, TH)
    r = sim.simulate(V0, TH, 0.01, method="euler", drag=False)
    assert abs(r["range"] - R_an) / R_an < 1e-2
    assert abs(r["flight_time"] - T_an) < 2 * 0.01


# ---- B3: energy conserved without drag, decreasing with drag ----
def test_rk4_conserves_energy_without_drag():
    e = sim.energy(sim.simulate(V0, TH, 0.01, method="rk4", drag=False))
    assert np.max(np.abs(e - e[0])) / e[0] < 1e-12


def test_euler_energy_drift_matches_prediction():
    # each Euler step adds exactly 0.5*M*g^2*dt^2, so E - E0 = 0.5*M*g^2*dt*t
    dt = 0.01
    r = sim.simulate(V0, TH, dt, method="euler", drag=False)
    e = sim.energy(r)
    pred = 0.5 * sim.M * sim.G**2 * dt * r["t"]
    # last point is a shorter partial step, so it is left out
    np.testing.assert_allclose((e - e[0])[:-1], pred[:-1], atol=1e-9)


@pytest.mark.parametrize("method", ["euler", "rk4"])
@pytest.mark.parametrize("bounces", [False, True])
def test_energy_decreases_with_drag(method, bounces):
    r = sim.simulate(V0, TH, 0.01, method=method, drag=True,
                     stop_at_first_landing=not bounces, max_bounces=4)
    assert np.all(np.diff(sim.energy(r)) <= 1e-12)


def test_energy_lost_equals_drag_work():
    r = sim.simulate(V0, TH, 1e-3, method="rk4", drag=True)
    e = sim.energy(r)
    p = sim.K * np.hypot(r["vx"], r["vy"])**3
    W = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(r["t"]))])
    lost = e[0] - e
    assert np.max(np.abs(lost - W)) < 1e-5 * lost[-1]   # within 0.001% of the total loss


# ---- B4: measured orders of convergence ----
def test_euler_order_no_drag():
    assert fitted_order("euler", False, analytic(V0, TH)[0]) == pytest.approx(1.0, abs=0.1)


def test_orders_with_drag():
    ref = sim.simulate(V0, TH, 1e-4, method="rk4", drag=True)["range"]
    assert fitted_order("euler", True, ref) == pytest.approx(1.0, abs=0.1)
    assert fitted_order("rk4", True, ref) == pytest.approx(4.0, abs=0.2)


# ---- bounce rule ----
def test_bounce_flight_times_scale_by_e():
    r = sim.simulate(V0, TH, 0.001, method="rk4", drag=False,
                     stop_at_first_landing=False, max_bounces=4)
    t_land = r["t"][(r["y"] == 0) & (r["t"] > 0)]
    T = np.diff(np.r_[0, t_land])
    assert len(T) == 4
    np.testing.assert_allclose(T[1:] / T[:-1], sim.E, rtol=1e-9)

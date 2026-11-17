"""Tests for Part 2 on small synthetic inputs (no dataset needed). Run with:  pytest -q"""
import numpy as np
import pandas as pd
import pytest

import part2_statistics as p2


def make_passes(rows):
    """rows: (start_x, start_y, end_x, end_y, t_start, t_end) in pitch fractions and seconds."""
    return pd.DataFrame(rows, columns=["Start X", "Start Y", "End X", "End Y",
                                       "Start Time [s]", "End Time [s]"])


def test_distance_uses_pitch_size():
    p = p2.add_distance_duration(make_passes([(0, 0, 1, 0, 0, 5), (0, 0, 0, 1, 0, 4)]))
    np.testing.assert_allclose(p["distance"], [105.0, 68.0])
    np.testing.assert_allclose(p["duration"], [5.0, 4.0])


def test_cleaning_drops_each_bad_pass_once():
    rows = [
        (0.1, 0.1, 0.3, 0.1, 0.0, 1.5),       # good: 21 m in 1.5 s
        (0.5, 0.5, 0.6, 0.6, 10.0, 11.0),     # good
        (np.nan, 0.1, 0.3, 0.1, 0.0, 1.0),    # missing coordinate
        (0.1, 0.1, 0.3, 0.1, 2.0, 2.0),       # zero duration
        (0.1, 0.1, 0.3, 0.1, 3.0, 2.9),       # negative duration
        (0.2, 0.2, 0.2, 0.2, 0.0, 1.0),       # zero distance
        (0.0, 0.5, 1.0, 0.5, 0.0, 0.5),       # 105 m in 0.5 s = 210 m/s
        (np.nan, 0.1, 0.1, 0.1, 5.0, 5.0),    # missing AND zero duration: counted once
    ]
    clean, log = p2.clean_passes(make_passes(rows))
    assert len(clean) == 2
    dropped = {reason.split(" (")[0]: n for reason, n, _ in log[1:]}
    assert dropped == {"missing coordinates": 2, "zero or negative duration": 2,
                       "zero distance": 1, "average speed > 40 m/s": 1}
    assert log[-1][2] == 2


def synthetic_track(speed=10.0, n=50):
    """Ball moving along x at constant speed, 25 Hz, with a missing frame and a period change."""
    frame = np.arange(1, n + 1)
    t = frame / p2.FPS
    x = speed * t
    period = np.where(frame <= 30, 1, 2)
    x[10] = np.nan                       # tracking dropout
    x[30:] += 40.0                       # ball jumps at half-time: must not become a speed
    return pd.DataFrame({"period": period, "frame": frame, "time": t, "x": x, "y": np.zeros(n)})


def test_ball_speeds_skip_gaps_and_period_changes():
    v, skipped = p2.ball_speeds(synthetic_track(speed=10.0, n=50))
    np.testing.assert_allclose(v, 10.0)
    assert skipped == 3                  # two pairs touch the NaN frame, one crosses half-time
    assert len(v) == 49 - 3


def test_ball_speeds_skip_missing_frame_numbers():
    tr = synthetic_track(n=20).drop(index=[5]).reset_index(drop=True)   # frame 6 absent from file
    v, skipped = p2.ball_speeds(tr)
    np.testing.assert_allclose(v, 10.0)


def test_remove_glitches():
    v, n = p2.remove_glitches(np.array([5.0, 39.9, 40.0, 40.1, 120.0]))
    assert n == 2
    np.testing.assert_allclose(v, [5.0, 39.9, 40.0])


def test_fit_recovers_known_slope():
    rng = np.random.default_rng(1)
    d = rng.uniform(3, 60, 800)
    t = 0.4 + 0.05 * d + rng.normal(0, 0.2, d.size)
    fit = p2.fit_duration_vs_distance(d, t, n_boot=500)
    assert fit["slope"] == pytest.approx(0.05, abs=0.003)
    lo, hi = fit["slope_ci_t"]
    assert lo < 0.05 < hi
    blo, bhi = fit["slope_ci_boot"]
    assert blo < 0.05 < bhi

"""Part 2: pass data and ball-tracking statistics (Metrica Sports sample games 1 and 2).

Data: https://github.com/metrica-sports/sample-data  (clone it into data/sample-data)
Positions are given as fractions of the pitch (0..1 on each axis); the pitch is 105 x 68 m.

Run from anywhere:  python Part_2/part2_statistics.py [--data-dir data/sample-data/data]
Writes figures and tables to Results/Part_2/ and the cleaned passes to data/passes_clean.csv
(used by Part 3 and by the B5 lofted-ball comparison).
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project folder
OUT = os.path.join(ROOT, "Results", "Part_2")
CLEAN_CSV = os.path.join(ROOT, "data", "passes_clean.csv")

PITCH_X, PITCH_Y = 105.0, 68.0   # m
FPS = 25.0                       # tracking frame rate (Hz)
V_MAX = 40.0                     # m/s; faster than any real kick -> glitch / timing error
GAMES = (1, 2)
COORDS = ["Start X", "Start Y", "End X", "End Y"]


# ---------------------------------------------------------------- passes
def load_passes(data_dir, games=GAMES):
    """All PASS events from the given games, with a 'game' column."""
    frames = []
    for g in games:
        ev = pd.read_csv(os.path.join(data_dir, f"Sample_Game_{g}", f"Sample_Game_{g}_RawEventsData.csv"))
        frames.append(ev[ev["Type"] == "PASS"].assign(game=g))
    return pd.concat(frames, ignore_index=True)


def add_distance_duration(p):
    """Pass distance (m) on the pitch, duration (s) and implied average speed (m/s)."""
    p = p.copy()
    p["distance"] = np.hypot((p["End X"] - p["Start X"]) * PITCH_X,
                             (p["End Y"] - p["Start Y"]) * PITCH_Y)
    p["duration"] = p["End Time [s]"] - p["Start Time [s]"]
    with np.errstate(divide="ignore", invalid="ignore"):
        p["avg_speed"] = p["distance"] / p["duration"]
    return p


def clean_passes(p, v_max=V_MAX):
    """Drop passes with missing coordinates or impossible values.

    Returns (clean DataFrame, list of (reason, n_dropped, n_left)). Each rule is applied
    in turn to what is left, so each pass is counted under the first rule it fails.
    """
    p = add_distance_duration(p)
    log = [("raw PASS events", 0, len(p))]
    rules = [
        ("missing coordinates", p[COORDS + ["Start Time [s]", "End Time [s]"]].isna().any(axis=1)),
        ("zero or negative duration", p["duration"] <= 0),
        ("zero distance", p["distance"] <= 0),
        (f"average speed > {v_max:g} m/s (timing error)", p["avg_speed"] > v_max),
    ]
    keep = pd.Series(True, index=p.index)
    for reason, bad in rules:
        drop = keep & bad
        keep &= ~bad
        log.append((reason, int(drop.sum()), int(keep.sum())))
    return p[keep].reset_index(drop=True), log


# ---------------------------------------------------------------- ball tracking
def load_ball_track(data_dir, game):
    """Ball position per frame: columns period, frame, time, x, y (metres; NaN when missing).
    The ball is the last column pair of the tracking file (same in the home and away files)."""
    f = os.path.join(data_dir, f"Sample_Game_{game}", f"Sample_Game_{game}_RawTrackingData_Home_Team.csv")
    raw = pd.read_csv(f, skiprows=2)
    return pd.DataFrame({
        "period": raw["Period"].to_numpy(),
        "frame": raw["Frame"].to_numpy(),
        "time": raw["Time [s]"].to_numpy(),
        "x": raw.iloc[:, -2].to_numpy(dtype=float) * PITCH_X,
        "y": raw.iloc[:, -1].to_numpy(dtype=float) * PITCH_Y,
    })


def ball_speeds(track):
    """Speed between consecutive frames (m/s).

    A pair is used only when both frames have a ball position, are exactly one frame apart
    and are in the same period, so a gap in tracking never becomes one huge 'jump'.
    Returns (speeds, n_pairs_skipped).
    """
    fr, per = track["frame"].to_numpy(), track["period"].to_numpy()
    x, y, t = track["x"].to_numpy(), track["y"].to_numpy(), track["time"].to_numpy()
    ok = ((np.diff(fr) == 1) & (np.diff(per) == 0)
          & np.isfinite(x[1:]) & np.isfinite(x[:-1]) & np.isfinite(y[1:]) & np.isfinite(y[:-1]))
    v = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
    return v[ok], int((~ok).sum())


def remove_glitches(v, v_max=V_MAX):
    """Drop speeds above v_max. Returns (kept speeds, number removed)."""
    bad = v > v_max
    return v[~bad], int(bad.sum())


# ---------------------------------------------------------------- fit
def fit_duration_vs_distance(distance, duration, n_boot=2000, seed=0):
    """Least-squares line duration = a + b*distance with 95% CIs on the slope b:
    the usual t-interval from the standard error, and a bootstrap (resampling passes),
    which does not assume the scatter about the line is the same at every distance."""
    distance, duration = np.asarray(distance), np.asarray(duration)
    res = stats.linregress(distance, duration)
    n = len(distance)
    t = stats.t.ppf(0.975, n - 2)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([np.polyfit(distance[i], duration[i], 1)[0] for i in idx])
    return {
        "n": n, "slope": res.slope, "intercept": res.intercept, "r2": res.rvalue**2,
        "slope_se": res.stderr, "slope_ci_t": (res.slope - t * res.stderr, res.slope + t * res.stderr),
        "slope_ci_boot": tuple(np.percentile(boot, [2.5, 97.5])),
        "intercept_se": res.intercept_stderr,
    }


# ---------------------------------------------------------------- figures
def plot_pass_histograms(p, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    a1.hist(p["distance"], bins=np.arange(0, 72, 2), color="C0", edgecolor="white", linewidth=0.5)
    a1.set_xlabel("Pass distance (m)"); a1.set_ylabel("Number of passes")
    a1.set_title("Pass distances")
    # durations come in 0.04 s steps (one frame), so put bin edges between frames
    a2.hist(p["duration"], bins=np.arange(-0.02, 5.5, 0.16), color="C0", edgecolor="white", linewidth=0.5)
    a2.set_xlabel("Pass duration (s)"); a2.set_title("Pass durations")
    for a in (a1, a2):
        a.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Cleaned passes, Metrica sample games 1 and 2 (n = {len(p)})")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_speed_histogram(v, median, p99, n_removed, path):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.hist(v, bins=np.arange(0, V_MAX + 0.5, 0.5), color="C0", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.axvline(median, color="k", ls="--", lw=1.2, label=f"median = {median:.2f} m/s")
    ax.axvline(p99, color="k", ls=":", lw=1.5, label=f"99th percentile = {p99:.2f} m/s")
    ax.set_xlabel("Ball speed between frames (m/s)"); ax.set_ylabel("Number of frame pairs")
    ax.set_title(f"Ball speed from tracking (25 Hz), {n_removed} glitches > {V_MAX:g} m/s removed")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_fit(p, fit, path):
    d = np.linspace(0, p["distance"].max(), 200)
    # 95% confidence band for the mean line (t-based)
    n, x = fit["n"], p["distance"].to_numpy()
    resid = p["duration"].to_numpy() - (fit["intercept"] + fit["slope"] * x)
    s = np.sqrt(np.sum(resid**2) / (n - 2))
    band = stats.t.ppf(0.975, n - 2) * s * np.sqrt(1 / n + (d - x.mean())**2 / np.sum((x - x.mean())**2))
    line = fit["intercept"] + fit["slope"] * d
    lo, hi = fit["slope_ci_t"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(p["distance"], p["duration"], s=8, alpha=0.35, color="C0", linewidths=0, label="Passes")
    ax.plot(d, line, color="C3", lw=2,
            label=f"Fit: T = {fit['intercept']:.3f} + {fit['slope']:.4f}·D\n"
                  f"slope 95% CI [{lo:.4f}, {hi:.4f}] s/m")
    ax.fill_between(d, line - band, line + band, color="C3", alpha=0.25, linewidth=0)
    ax.set_xlabel("Pass distance D (m)"); ax.set_ylabel("Pass duration T (s)")
    ax.set_title("Pass duration vs distance")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main
def main(data_dir):
    os.makedirs(OUT, exist_ok=True)
    out = []   # lines for the summary file

    def say(s=""):
        print(s); out.append(s)

    # --- passes: clean, distance and duration
    passes, log = clean_passes(load_passes(data_dir))
    say("PASS CLEANING (games " + ", ".join(map(str, GAMES)) + ")")
    for reason, dropped, left in log:
        say(f"  {reason:45s} dropped {dropped:4d}   left {left:4d}")
    off = ((passes[COORDS] < 0) | (passes[COORDS] > 1)).any(axis=1).sum()
    say(f"  kept {off} passes with a start/end up to "
        f"{max(-passes[COORDS].min().min(), passes[COORDS].max().max() - 1) * PITCH_X:.1f} m outside the lines "
        f"(realistic near a touchline)")
    say(f"  distance: median {passes['distance'].median():.1f} m, range "
        f"{passes['distance'].min():.1f}-{passes['distance'].max():.1f} m")
    say(f"  duration: median {passes['duration'].median():.2f} s, range "
        f"{passes['duration'].min():.2f}-{passes['duration'].max():.2f} s")
    passes[["game", "Period", "Subtype", "Start Time [s]", "End Time [s]", "distance", "duration"]] \
        .to_csv(CLEAN_CSV, index=False)
    plot_pass_histograms(passes, os.path.join(OUT, "part2_pass_histograms.png"))

    # --- ball speeds from tracking
    say("\nBALL SPEEDS FROM TRACKING")
    all_v, total_removed = [], 0
    for g in GAMES:
        track = load_ball_track(data_dir, g)
        v, skipped = ball_speeds(track)
        v_ok, removed = remove_glitches(v)
        all_v.append(v_ok); total_removed += removed
        say(f"  game {g}: {len(track)} frames, {np.isnan(track['x']).sum()} without a ball position, "
            f"{len(v)} usable frame pairs, {removed} glitches > {V_MAX:g} m/s removed")
    v = np.concatenate(all_v)
    median, p99 = np.median(v), np.percentile(v, 99)
    say(f"  total: {len(v)} speeds kept, {total_removed} glitches removed "
        f"({100 * total_removed / (len(v) + total_removed):.3f}%)")
    say(f"  median speed {median:.2f} m/s, 99th percentile {p99:.2f} m/s")
    plot_speed_histogram(v, median, p99, total_removed, os.path.join(OUT, "part2_ball_speeds.png"))

    # --- duration vs distance fit
    fit = fit_duration_vs_distance(passes["distance"], passes["duration"])
    lo, hi = fit["slope_ci_t"]; blo, bhi = fit["slope_ci_boot"]
    say("\nFIT: duration = a + b * distance")
    say(f"  n = {fit['n']}, R^2 = {fit['r2']:.3f}")
    say(f"  slope b     = {fit['slope']:.4f} s/m  (SE {fit['slope_se']:.4f})")
    say(f"  95% CI (t)         [{lo:.4f}, {hi:.4f}] s/m")
    say(f"  95% CI (bootstrap) [{blo:.4f}, {bhi:.4f}] s/m")
    say(f"  intercept a = {fit['intercept']:.3f} s  (SE {fit['intercept_se']:.3f})")
    say(f"  -> each extra metre adds {fit['slope']:.3f} s; 1/b = {1 / fit['slope']:.1f} m/s "
        f"(95% CI {1 / hi:.1f}-{1 / lo:.1f} m/s) is the marginal speed of a pass")
    plot_fit(passes, fit, os.path.join(OUT, "part2_duration_vs_distance.png"))

    with open(os.path.join(OUT, "part2_summary.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    return passes, v, fit


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "sample-data", "data"),
                    help="folder containing Sample_Game_1, Sample_Game_2, ...")
    main(ap.parse_args().data_dir)

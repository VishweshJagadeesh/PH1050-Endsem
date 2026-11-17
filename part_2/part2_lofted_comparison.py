"""B5: simulated lofted-ball flight times vs the long passes in the data.

For a pass of distance D, the simulator (RK4, quadratic drag) is asked: at launch angle theta,
what launch speed lands the ball D metres away, and how long is it in the air? That flight time
is compared with the observed duration of every long pass (D > 30 m), and separately with goal
kicks, which are almost always lofted.

Error on the comparison: mean of (observed - simulated) with its standard error and a 95%
bootstrap interval. We also find the 'effective' launch angle at which the model matches the
data on average, with a bootstrap interval.

Run (after part2_statistics.py):  python Part_2/part2_lofted_comparison.py   (-> Results/Part_2/b5_*)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project folder
sys.path.insert(0, os.path.join(ROOT, "Part_1"))   # the simulator lives in Part_1
import COMPHY_FOOTBALL_SIMULATOR as sim  # noqa: E402

OUT = os.path.join(ROOT, "Results", "Part_2")
CLEAN_CSV = os.path.join(ROOT, "data", "passes_clean.csv")


LONG = 30.0                                   # m; passes longer than this are compared
ANGLES = np.arange(10.0, 60.01, 2.5)          # degrees, grid for the simulated flight times
SHOW = (20.0, 30.0, 40.0, 50.0)               # angles drawn on the figure / tabulated
DT = 0.01
N_BOOT = 2000
SEED = 0


def flight_time_table(angles_deg=ANGLES, v0s=np.arange(8.0, 90.01, 1.0), dt=DT):
    """For each angle, simulate a range of launch speeds and record (range, flight time).
    Range rises steadily with v0 at a fixed angle, so T(D) can then be read off by
    interpolation instead of a root-find for every pass. Returns {angle: (ranges, times)}."""
    table = {}
    for a in angles_deg:
        th = np.radians(a)
        rs, ts = [], []
        for v0 in v0s:
            r = sim.simulate(v0, th, dt, method="rk4", drag=True)
            rs.append(r["range"]); ts.append(r["flight_time"])
        table[a] = (np.array(rs), np.array(ts))
    return table


def sim_flight_time(table, angle, D):
    ranges, times = table[angle]
    if np.max(D) > ranges.max():
        raise ValueError(f"{angle} deg: need launch speeds above the table for D = {np.max(D):.1f} m")
    return np.interp(D, ranges, times)


def compare(T_obs, T_sim, n_boot=N_BOOT, seed=SEED):
    """Mean residual (observed - simulated) with SE and 95% bootstrap CI, plus RMSE."""
    res = T_obs - T_sim
    rng = np.random.default_rng(seed)
    boot = res[rng.integers(0, len(res), size=(n_boot, len(res)))].mean(axis=1)
    return {"n": len(res), "mean": res.mean(), "se": res.std(ddof=1) / np.sqrt(len(res)),
            "ci": tuple(np.percentile(boot, [2.5, 97.5])), "rmse": np.sqrt(np.mean(res**2))}


def effective_angle(table, D, T_obs, n_boot=N_BOOT, seed=SEED):
    """Angle at which the mean simulated flight time equals the mean observed duration.
    Mean simulated time increases with angle, so the zero of the mean residual is found by
    interpolating along the angle grid. Bootstrap over passes gives a 95% interval."""
    angles = np.array(sorted(table))
    Tsim = np.array([sim_flight_time(table, a, D) for a in angles])     # (n_angles, n_passes)

    def solve(idx):
        mean_res = T_obs[idx].mean() - Tsim[:, idx].mean(axis=1)        # decreasing in angle
        if mean_res[0] < 0 or mean_res[-1] > 0:
            return np.nan                                                # outside the grid
        return np.interp(0.0, mean_res[::-1], angles[::-1])

    rng = np.random.default_rng(seed)
    boot = np.array([solve(rng.integers(0, len(D), len(D))) for _ in range(n_boot)])
    return solve(np.arange(len(D))), tuple(np.nanpercentile(boot, [2.5, 97.5])), int(np.isnan(boot).sum())


def main(csv=CLEAN_CSV):
    os.makedirs(OUT, exist_ok=True)
    p = pd.read_csv(csv)
    groups = {
        f"all passes > {LONG:g} m": p[p["distance"] > LONG],
        f"goal kicks > {LONG:g} m": p[(p["distance"] > LONG) & (p["Subtype"] == "GOAL KICK")],
    }
    out = []

    def say(s=""):
        print(s); out.append(s)

    table = flight_time_table()

    say("LAUNCH SPEED NEEDED TO REACH D (RK4 with drag) -- plausibility check")
    say("  angle   " + "   ".join(f"D={D:2d} m: v0, T" for D in (30, 45, 60)))
    for a in SHOW:
        cells = []
        for D in (30, 45, 60):
            v0, T = sim.flight_time_for_range(D, np.radians(a), dt=DT)
            cells.append(f"{v0:5.1f} m/s {T:4.2f} s")
        say(f"  {a:4.0f}°   " + "   ".join(cells))
    say("  (hard long kicks in real football are roughly 25-35 m/s)")

    results = {}
    for name, g in groups.items():
        D, T = g["distance"].to_numpy(), g["duration"].to_numpy()
        say(f"\n{name.upper()}: n = {len(D)}, distance {D.min():.1f}-{D.max():.1f} m, "
            f"mean observed duration {T.mean():.2f} s")
        say("  angle   mean(T_obs - T_sim) ± SE    95% bootstrap CI     RMSE")
        for a in SHOW:
            c = compare(T, sim_flight_time(table, a, D))
            say(f"  {a:4.0f}°   {c['mean']:+.3f} ± {c['se']:.3f} s       "
                f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] s   {c['rmse']:.3f} s")
        ang, (lo, hi), n_out = effective_angle(table, D, T)
        note = f" ({n_out} of {N_BOOT} resamples outside {ANGLES[0]:g}-{ANGLES[-1]:g}°)" if n_out else ""
        say(f"  effective launch angle (mean residual = 0): {ang:.1f}°, 95% CI [{lo:.1f}°, {hi:.1f}°]{note}")
        results[name] = dict(D=D, T=T, angle=ang, ci=(lo, hi))

    # ---- figure 1: data vs simulated flight-time curves
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    allg, gk = list(groups.values())
    ax.scatter(allg["distance"], allg["duration"], s=12, color="C0", alpha=0.5, linewidths=0,
               label=f"Passes > {LONG:g} m (n = {len(allg)})")
    ax.scatter(gk["distance"], gk["duration"], s=30, facecolors="none", edgecolors="k", linewidths=1,
               marker="s", label=f"Goal kicks (n = {len(gk)})")
    Dg = np.linspace(LONG, allg["distance"].max(), 200)
    shades = plt.cm.Oranges(np.linspace(0.45, 0.95, len(SHOW)))
    for a, col in zip(SHOW, shades):
        ax.plot(Dg, sim_flight_time(table, a, Dg), color=col, lw=2, label=f"Simulated flight, θ = {a:g}°")
    ax.set_xlabel("Pass distance D (m)"); ax.set_ylabel("Duration / flight time (s)")
    ax.set_title("Long passes vs simulated lofted flight times (RK4, drag)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "b5_lofted_vs_passes.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    # ---- figure 2: mean residual vs launch angle, with 95% CI band
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for (name, r), col, ls in zip(results.items(), ("C0", "k"), ("-", "--")):
        means, los, his = [], [], []
        for a in ANGLES:
            c = compare(r["T"], sim_flight_time(table, a, r["D"]), n_boot=500)
            means.append(c["mean"]); los.append(c["ci"][0]); his.append(c["ci"][1])
        ax.plot(ANGLES, means, color=col, ls=ls, lw=2,
                label=f"{name}: effective θ = {r['angle']:.0f}° [{r['ci'][0]:.0f}°, {r['ci'][1]:.0f}°]")
        ax.fill_between(ANGLES, los, his, color=col, alpha=0.15, linewidth=0)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("Assumed launch angle θ (degrees)"); ax.set_ylabel("Mean (observed − simulated) (s)")
    ax.set_title("Observed minus simulated flight time (bands: 95% bootstrap CI)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "b5_residual_vs_angle.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    with open(os.path.join(OUT, "b5_summary.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    return results


if __name__ == "__main__":
    main()

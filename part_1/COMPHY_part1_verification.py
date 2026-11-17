import os
import numpy as np
import matplotlib.pyplot as plt
import part_1.COMPHY_FOOTBALL_SIMULATOR as sim

os.makedirs("results", exist_ok=True)

# ---- Parameters from the assignment ----
M = sim.M
R = sim.R
CD = sim.CD
RHO = sim.RHO
G = sim.G
E = sim.E          # coefficient of restitution (vertical velocity only)
AREA = sim.AREA
K = sim.K          # drag constant: F_drag = K * |v| * v

v0, th = 25.0, np.radians(35)


def energy_checks():
    # --- no drag: energy conserved (plot relative drift on a log scale) ---
    plt.figure()
    for m in ("euler", "rk4"):
        r = sim.simulate(v0, th, dt=0.01, method=m, drag=False)
        energies = sim.energy(r)
        drift = (energies - energies[0]) / energies[0]
        max_drift = np.max(np.abs(drift))
        plt.semilogy(r["t"], np.maximum(np.abs(drift), 1e-17), label=m.upper())
        print(f"{m:5s} no drag: E0 = {energies[0]:.5f} J, E_final = {energies[-1]:.5f} J, "
              f"max relative drift = {max_drift:.3e}")
    plt.title("No drag: relative energy drift |E - E0|/E0 (dt = 0.01 s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Relative energy drift")
    plt.legend()
    plt.savefig("results/nodrag_energy_conservation.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # --- with drag: energy only decreases (single flight, then across bounces) ---
    tol = 1e-12
    plt.figure()
    for m in ("euler", "rk4"):
        r = sim.simulate(v0, th, dt=0.01, method=m, drag=True)
        energies = sim.energy(r)
        decreasing = np.all(np.diff(energies) <= tol)
        print(f"{m:5s} with drag: E0 = {energies[0]:.5f} J, E_final = {energies[-1]:.5f} J, "
              f"monotone decrease = {decreasing}")
        plt.plot(r["t"], energies, label=m.upper())
    plt.title("Total energy with drag: Euler vs RK4 (dt = 0.01 s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.savefig("results/drag_energy_decrease.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    plt.figure()
    for m in ("euler", "rk4"):
        r = sim.simulate(v0, th, dt=0.01, method=m, drag=True,
                         stop_at_first_landing=False, max_bounces=4)
        energies = sim.energy(r)
        decreasing = np.all(np.diff(energies) <= tol)
        print(f"{m:5s} with drag + 4 bounces: monotone decrease = {decreasing}")
        plt.plot(r["t"], energies, label=m.upper())
    plt.title("Total energy with drag and bounces (e = 0.7, dt = 0.01 s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.savefig("results/drag_bounce_energy.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def convergence_study():
    # Error in range vs timestep (log-log), with measured orders from a least-squares fit.
    dts = np.logspace(-1, -2.5, 7)

    # (a) no drag: exact range known. Euler shows order 1; RK4 is exact for a quadratic
    #     trajectory, so its error is just roundoff (flat at ~1e-13) and has no meaningful order.
    analytic_range = v0**2 * np.sin(2 * th) / G
    plt.figure()
    for m in ("euler", "rk4"):
        err = np.array([abs(sim.simulate(v0, th, dt, method=m, drag=False)["range"] - analytic_range)
                        for dt in dts])
        err = np.maximum(err, 1e-16)
        label = m.upper()
        if m == "euler":
            p = np.polyfit(np.log(dts), np.log(err), 1)[0]
            print(f"no drag, euler: measured order = {p:.2f}")
            label += f" (slope {p:.2f})"
        else:
            print(f"no drag, rk4: error stays at roundoff level (max {err.max():.1e} m)")
            label += " (roundoff level)"
        plt.loglog(dts, err, "o-", label=label)
    plt.title("No-drag range error vs timestep (v0 = 25 m/s, theta = 35 deg)")
    plt.xlabel("Timestep dt (s)")
    plt.ylabel("|Range error| (m)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("results/nodrag_convergence_study.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # (b) with drag: no analytic answer, so compare against a very fine RK4 reference.
    ref = sim.simulate(v0, th, 1e-4, method="rk4", drag=True)["range"]
    plt.figure()
    for m in ("euler", "rk4"):
        err = np.array([abs(sim.simulate(v0, th, dt, method=m, drag=True)["range"] - ref)
                        for dt in dts])
        p = np.polyfit(np.log(dts), np.log(err), 1)[0]
        print(f"with drag, {m}: measured order = {p:.2f}")
        plt.loglog(dts, err, "o-", label=f"{m.upper()} (slope {p:.2f})")
    plt.title("Range error vs timestep with drag (reference: RK4, dt = 1e-4 s)")
    plt.xlabel("Timestep dt (s)")
    plt.ylabel("|Range error| (m)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("results/drag_convergence_study.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def bounce_check():
    # with no drag, successive flight durations must be in the ratio e
    r = sim.simulate(v0, th, 0.001, method="rk4", drag=False,
                     stop_at_first_landing=False, max_bounces=4)
    t_land = r["t"][(r["y"] == 0) & (r["t"] > 0)]
    T = np.diff(np.r_[0, t_land])
    print("bounce flight durations (s):", T)
    print("successive ratios:", T[1:] / T[:-1], f"(expected e = {E})")


def trajectory_plot():
    plt.figure()
    for m, ls in (("euler", "--"), ("rk4", "-")):
        for drag in (False, True):
            r = sim.simulate(v0, th, dt=0.01, method=m, drag=drag,
                             stop_at_first_landing=False, max_bounces=3)
            plt.plot(r["x"], r["y"], ls, label=f"{m.upper()}, {'drag' if drag else 'no drag'}")
    plt.title("Trajectories with bounces (e = 0.7)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.savefig("results/trajectories.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def no_drag_comparison():
    # Comparing the no drag range and flight times of both methods with the analytic results
    DT = 0.01
    CASES = [(25.0, 35.0), (20.0, 45.0), (30.0, 20.0)]   # (v0 in m/s, theta in degrees)
    METHODS = ("euler", "rk4")

    rows = []
    for v0_, deg in CASES:
        th_ = np.radians(deg)
        R_an = v0_**2 * np.sin(2 * th_) / G
        T_an = 2 * v0_ * np.sin(th_) / G
        for m in METHODS:
            r = sim.simulate(v0_, th_, DT, method=m, drag=False)
            R_num, T = r["range"], r["flight_time"]
            rows.append(dict(v0=v0_, theta=deg, method=m.upper(),
                             R=R_num, R_an=R_an, R_abs=abs(R_num - R_an), R_rel=abs(R_num - R_an) / R_an,
                             T=T, T_an=T_an, T_abs=abs(T - T_an)))

    cols = ["v0", "theta", "method", "R", "R_an", "R_abs", "R_rel", "T", "T_an", "T_abs"]

    def csv_val(c, v):
        if c == "method":
            return v
        return f"{v:.6g}" if c in ("v0", "theta") else f"{v:.6e}"

    with open("results/no_drag_comparison_table_full.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    with open("results/no_drag_comparison_table.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(csv_val(c, r[c]) for c in cols) + "\n")

    def fmt(r):
        return [f"{r['v0']:g}, {r['theta']:g}°", r["method"],
                f"{r['R']:.5f}", f"{r['R_an']:.5f}", f"{r['R_abs']:.2e}", f"{r['R_rel']:.2e}",
                f"{r['T']:.5f}", f"{r['T_an']:.5f}", f"{r['T_abs']:.2e}"]

    tex_head = [r"Case ($v_0$, $\theta$)", "Method", "$R$ num (m)", "$R$ analytic (m)",
                r"$|\Delta R|$ (m)", "rel. err.", "$T$ num (s)", "$T$ analytic (s)", r"$|\Delta T|$ (s)"]
    with open("results/no_drag_comparison_table.tex", "w") as f:
        f.write("\\begin{table}[h]\n\\centering\\small\n")
        f.write("\\begin{tabular}{" + "l" * 2 + "r" * 7 + "}\n\\hline\n")
        f.write(" & ".join(tex_head) + " \\\\\n\\hline\n")
        for r in rows:
            cells = fmt(r)
            cells[0] = cells[0].replace("°", r"$^\circ$")
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write(f"\\caption{{No-drag range and flight time, numerical vs analytic ($\\Delta t = {DT}$ s).}}\n")
        f.write("\\label{tab:no_drag_comparison}\n\\end{table}\n")


if __name__ == "__main__":
    no_drag_comparison()
    energy_checks()
    convergence_study()
    bounce_check()
    trajectory_plot()

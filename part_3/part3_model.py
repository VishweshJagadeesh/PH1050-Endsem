"""Part 3: data-driven model of pass duration from distance.

Models: mean duration (baseline), straight line, quadratic.
Test 1  random 80/20 train/test split (also repeated over many random splits).
Test 2  extrapolation: train on the shortest 80% of passes, test on the longest 20%.
Errors on test scores come from bootstrapping the test passes; model-vs-model differences
use a paired bootstrap (the same resampled passes for both models).

Run (after part2_statistics.py):  python part3_model.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = {"mean (baseline)": 0, "linear": 1, "quadratic": 2}   # name -> polynomial degree
SEED = 0
N_BOOT = 2000


# ---------------------------------------------------------------- models and scores
def fit_model(degree, d, t):
    """Least-squares polynomial of the given degree (0 = predict the mean duration)."""
    return np.polyfit(d, t, degree)


def predict(coef, d):
    return np.polyval(coef, d)


def scores(t_true, t_pred):
    err = t_pred - t_true
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err))}       # bias > 0: model predicts too long


def random_split(n, test_frac=0.2, seed=SEED):
    """Indices (train, test) for a random split."""
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(round(test_frac * n))
    return np.sort(idx[n_test:]), np.sort(idx[:n_test])


def short_long_split(d, test_frac=0.2):
    """Indices (train, test): train on the shortest (1 - test_frac), test on the longest test_frac."""
    order = np.argsort(d, kind="stable")
    n_test = int(round(test_frac * len(d)))
    return np.sort(order[:-n_test]), np.sort(order[-n_test:])


def bootstrap_rmse(t_true, preds, n_boot=N_BOOT, seed=SEED):
    """Resample test passes; returns {model: array of RMSEs}, same resamples for every model."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t_true), size=(n_boot, len(t_true)))
    return {m: np.sqrt(np.mean((p[idx] - t_true[idx])**2, axis=1)) for m, p in preds.items()}


def evaluate(d, t, train, test):
    """Fit every model on train, score on test. Returns (table rows, coefs, bootstrap RMSEs)."""
    coefs = {m: fit_model(deg, d[train], t[train]) for m, deg in MODELS.items()}
    preds = {m: predict(c, d[test]) for m, c in coefs.items()}
    boot = bootstrap_rmse(t[test], preds)
    rows = []
    for m in MODELS:
        s = scores(t[test], preds[m])
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        s_train = scores(t[train], predict(coefs[m], d[train]))
        rows.append({"model": m, "train_rmse": s_train["rmse"], "test_rmse": s["rmse"],
                     "rmse_ci_lo": lo, "rmse_ci_hi": hi, "test_mae": s["mae"], "test_bias": s["bias"]})
    return rows, coefs, boot


def paired_difference(boot, a, b):
    """RMSE(a) - RMSE(b) on the same bootstrap resamples: (median, 2.5%, 97.5%)."""
    diff = boot[a] - boot[b]
    return tuple(np.percentile(diff, [50, 2.5, 97.5]))


# ---------------------------------------------------------------- figures
STYLE = {"mean (baseline)": dict(color="0.35", ls=":"),
         "linear": dict(color="C3", ls="-"),
         "quadratic": dict(color="C2", ls="--")}


def plot_split(d, t, train, test, coefs, title, path, cut=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(d[train], t[train], s=7, color="0.75", linewidths=0, label="Training passes")
    ax.scatter(d[test], t[test], s=9, color="C0", alpha=0.6, linewidths=0, label="Test passes")
    x = np.linspace(0, d.max() * 1.02, 300)
    for m, c in coefs.items():
        ax.plot(x, predict(c, x), lw=2, label=m, **STYLE[m])
    if cut is not None:
        ax.axvline(cut, color="k", lw=0.8)
        ax.annotate(f"train | test\n{cut:.1f} m", (cut, ax.get_ylim()[1]), xytext=(4, -26),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Pass distance D (m)"); ax.set_ylabel("Pass duration T (s)")
    ax.set_title(title); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_rmse_bars(results, path):
    """Test RMSE with 95% bootstrap CI, for both tests (one axis: seconds)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    tests = list(results)
    w = 0.25
    for k, m in enumerate(MODELS):
        rows = [next(r for r in results[tst] if r["model"] == m) for tst in tests]
        y = np.array([r["test_rmse"] for r in rows])
        err = np.array([[r["test_rmse"] - r["rmse_ci_lo"] for r in rows],
                        [r["rmse_ci_hi"] - r["test_rmse"] for r in rows]])
        xs = np.arange(len(tests)) + (k - 1) * w
        ax.bar(xs, y, w * 0.9, color=STYLE[m]["color"], label=m)
        ax.errorbar(xs, y, yerr=err, fmt="none", ecolor="k", capsize=3, lw=1)
    ax.set_xticks(np.arange(len(tests)), tests)
    ax.set_ylabel("Test RMSE (s)")
    ax.set_title("Test error of each model (bars: 95% bootstrap CI)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main
def main(csv="passes_clean.csv"):
    os.makedirs("results", exist_ok=True)
    p = pd.read_csv(csv)
    d, t = p["distance"].to_numpy(), p["duration"].to_numpy()
    out = []

    def say(s=""):
        print(s); out.append(s)

    def table(rows):
        say(f"  {'model':16s} {'train RMSE':>10s} {'test RMSE [95% CI]':>24s} {'MAE':>7s} {'bias':>7s}")
        for r in rows:
            say(f"  {r['model']:16s} {r['train_rmse']:10.3f} "
                f"{r['test_rmse']:8.3f} [{r['rmse_ci_lo']:.3f}, {r['rmse_ci_hi']:.3f}] "
                f"{r['test_mae']:7.3f} {r['test_bias']:+7.3f}")

    say(f"{len(d)} cleaned passes, distance {d.min():.1f}-{d.max():.1f} m. All errors in seconds; "
        f"bias = mean(predicted - observed).")

    results, coef_store = {}, {}

    # ---- test 1: random split
    tr, te = random_split(len(d))
    rows, coefs, boot = evaluate(d, t, tr, te)
    results["random 80/20"] = rows; coef_store["random"] = coefs
    say(f"\nTEST 1: random split (train {len(tr)}, test {len(te)}, seed {SEED})")
    table(rows)
    for a, b in (("linear", "mean (baseline)"), ("quadratic", "linear")):
        med, lo, hi = paired_difference(boot, a, b)
        say(f"  RMSE({a}) - RMSE({b}) = {med:+.4f} s, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    plot_split(d, t, tr, te, coefs, "Random 80/20 split", "results/part3_random_split.png")

    # repeated random splits: how much does the score depend on which passes land in the test set?
    rep = {m: [] for m in MODELS}
    for seed in range(200):
        tr_, te_ = random_split(len(d), seed=seed)
        for m, deg in MODELS.items():
            rep[m].append(scores(t[te_], predict(fit_model(deg, d[tr_], t[tr_]), d[te_]))["rmse"])
    say("  over 200 random splits, test RMSE mean ± std: " +
        ", ".join(f"{m} {np.mean(v):.3f} ± {np.std(v):.3f}" for m, v in rep.items()))
    q_wins = np.mean(np.array(rep["quadratic"]) < np.array(rep["linear"]))
    say(f"  quadratic beats linear in {100 * q_wins:.0f}% of the 200 splits")

    # ---- test 2: extrapolation, short -> long
    tr, te = short_long_split(d)
    cut = 0.5 * (d[tr].max() + d[te].min())
    rows, coefs, boot = evaluate(d, t, tr, te)
    results["short → long\n(extrapolation)"] = rows; coef_store["extrapolation"] = coefs
    say(f"\nTEST 2: extrapolation (train on {len(tr)} passes up to {d[tr].max():.1f} m, "
        f"test on the longest {len(te)}: {d[te].min():.1f}-{d[te].max():.1f} m)")
    table(rows)
    for a, b in (("linear", "mean (baseline)"), ("quadratic", "linear")):
        med, lo, hi = paired_difference(boot, a, b)
        say(f"  RMSE({a}) - RMSE({b}) = {med:+.4f} s, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    plot_split(d, t, tr, te, coefs, "Extrapolation: train on short passes, test on the longest 20%",
               "results/part3_extrapolation.png", cut=cut)

    # coefficients, for the report
    say("\nFITTED COEFFICIENTS (T in s, D in m)")
    for name, coefs in coef_store.items():
        c1, c2 = coefs["linear"], coefs["quadratic"]
        say(f"  {name:13s} linear: T = {c1[1]:.3f} + {c1[0]:.4f} D    "
            f"quadratic: T = {c2[2]:.3f} + {c2[1]:.4f} D {c2[0]:+.6f} D^2")

    plot_rmse_bars(results, "results/part3_rmse_comparison.png")

    # results table (csv + LaTeX)
    flat = [dict(test=k.replace("\n", " "), **r) for k, rows in results.items() for r in rows]
    pd.DataFrame(flat).to_csv("results/part3_results.csv", index=False, float_format="%.4f")
    with open("results/part3_results.tex", "w") as f:
        f.write("\\begin{table}[h]\n\\centering\\small\n\\begin{tabular}{llrrrr}\n\\hline\n")
        f.write("Test & Model & Train RMSE (s) & Test RMSE (s) [95\\% CI] & MAE (s) & Bias (s) \\\\\n\\hline\n")
        for r in flat:
            test_name = r["test"].replace("→", r"$\to$")   # kept out of the f-string for Python < 3.12
            f.write(f"{test_name} & {r['model']} & {r['train_rmse']:.3f} & "
                    f"{r['test_rmse']:.3f} [{r['rmse_ci_lo']:.3f}, {r['rmse_ci_hi']:.3f}] & "
                    f"{r['test_mae']:.3f} & {r['test_bias']:+.3f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\caption{Pass duration predicted from distance. "
                "CIs from 2000 bootstrap resamples of the test passes.}\n"
                "\\label{tab:part3}\n\\end{table}\n")

    with open("results/part3_summary.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    return results


if __name__ == "__main__":
    main()

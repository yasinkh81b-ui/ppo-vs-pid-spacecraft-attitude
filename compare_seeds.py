"""
Multi-seed analysis: how stable are the PPO results across independent training runs?

Uses every PPO model it finds:
    plain PPO : ppo_attitude.zip (seed 0), ppo_attitude_s1.zip, ppo_attitude_s2.zip, ...
    PPO+I     : ppo_attitude_i.zip (seed 0), ppo_attitude_i_s1.zip, ...
(train them with train_all_seeds.py). All controllers are tested on the same 100 unseen scenarios
as compare.py. The PD/PID gains are the ones found by compare.py's grid search (set below).

Reported uncertainty:
  * PD and PID (deterministic controllers): 95% bootstrap confidence interval of the median over the scenarios.
  * PPO and PPO+I: mean +- standard deviation (and min-max) ACROSS TRAINING SEEDS of each seed's median.
  * "PID better in X% of scenarios": paired comparison against each PPO model, averaged over seeds.

Usage:
    python compare_seeds.py            # full analysis (results of each model are cached in seed_cache/)
    python compare_seeds.py --quick    # only 10 test scenarios, separate cache: a fast check
    python compare_seeds.py --fresh    # ignore the cache and recompute everything
Outputs: seed_results.csv, seed_plot.png
"""
import os
import sys
import glob
import time
import numpy as np

from compare import make_scenarios, rollout, metrics, pid_policy, N_TEST, TEST_SEED0, DT_CTRL, N_STEPS

PD_GAINS = (1.6, 0.4, 0.0)     # (kp, kd, ki) found by compare.py (grid search on 10 tuning scenarios)
PID_GAINS = (0.8, 0.2, 0.06)

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = "--quick" in sys.argv
FRESH = "--fresh" in sys.argv
CACHE = os.path.join(os.getcwd(), "seed_cache_quick" if QUICK else "seed_cache")


def find_models(base):
    """Return {seed: path} for files base.zip (seed 0) and base_s<N>.zip."""
    found = {}
    for folder in (HERE, os.getcwd()):
        p0 = os.path.join(folder, base + ".zip")
        if os.path.exists(p0):
            found.setdefault(0, p0)
        for p in glob.glob(os.path.join(folder, base + "_s*.zip")):
            tag = os.path.basename(p)[len(base) + 2:-4]
            if tag.isdigit():
                found.setdefault(int(tag), p)
    return dict(sorted(found.items()))


def evaluate(name, policy_factory, use_integral, test):
    """Run one controller on all test scenarios (cached on disk by name)."""
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path) and not FRESH:
        z = np.load(path)
        return {k: z[k] for k in z.files}
    ms, curves = [], []
    for x0, d in test:
        errs, taus = rollout(policy_factory(), x0, d, use_integral=use_integral)
        ms.append(metrics(errs, taus))
        curves.append(errs.astype(np.float32))
    out = {k: np.array([m[k] for m in ms]) for k in ms[0]}
    out["curves"] = np.array(curves)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **out)
    return out


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    meds = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return np.percentile(meds, [2.5, 97.5])


def ms_str(v, nd=2):
    v = np.asarray(v)
    return f"{v.mean():.{nd}f} +- {v.std(ddof=1) if len(v) > 1 else 0.0:.{nd}f} (range {v.min():.{nd}f}-{v.max():.{nd}f})"


if __name__ == "__main__":
    t_start = time.time()
    n_test = 10 if QUICK else N_TEST
    test = make_scenarios(n_test, TEST_SEED0)

    results = {}   # label -> list of per-run result dicts (classical: 1 run; RL: one per seed)
    seeds_of = {}

    print(f"Evaluating on {n_test} unseen scenarios (cache folder: {os.path.basename(CACHE)})")
    results["PD"] = [evaluate("PD", lambda: pid_policy(*PD_GAINS), False, test)]
    results["PID"] = [evaluate("PID", lambda: pid_policy(*PID_GAINS), False, test)]

    from stable_baselines3 import PPO
    for label, base, flag in [("PPO", "ppo_attitude", False), ("PPO+I", "ppo_attitude_i", True)]:
        models = find_models(base)
        if not models:
            print(f"  no models found for {label}; skipping")
            continue
        results[label], seeds_of[label] = [], []
        for seed, path in models.items():
            t0 = time.time()
            cached = os.path.exists(os.path.join(CACHE, f"{label}_seed{seed}.npz")) and not FRESH
            model = None if cached else PPO.load(path, device="cpu")
            pf = (lambda m=model: (lambda env: m.predict(env._obs(), deterministic=True)[0]))
            results[label].append(evaluate(f"{label}_seed{seed}", pf, flag, test))
            seeds_of[label].append(seed)
            print(f"  {label} seed {seed}: {'loaded from cache' if cached else f'evaluated in {time.time() - t0:.0f} s'}")

    # ---------------- summary table ----------------
    print("\n" + "=" * 100)
    print(f"{'Controller':<18}{'settling time to 1 deg (s)':<46}{'final error (deg)'}")
    print("-" * 100)
    csv_rows = []
    summary = {}
    for label, runs in results.items():
        settle_med = np.array([np.median(r["settle"]) for r in runs])
        final_med = np.array([np.median(r["final_err"]) for r in runs])
        rms = np.array([r["rms_torque"].mean() for r in runs])
        jerk = np.array([r["jerk"].mean() for r in runs])
        solved = np.array([np.mean((r["final_err"] < 1.0)) * 100 for r in runs])
        summary[label] = dict(settle=settle_med, final=final_med, rms=rms, jerk=jerk)
        if label in ("PD", "PID"):
            lo_s, hi_s = boot_ci(runs[0]["settle"]); lo_f, hi_f = boot_ci(runs[0]["final_err"])
            print(f"{label:<18}{f'{settle_med[0]:.2f}  [95% CI {lo_s:.2f}-{hi_s:.2f}]':<46}"
                  f"{final_med[0]:.4f}  [95% CI {lo_f:.4f}-{hi_f:.4f}]")
            summary[label]["ci_settle"] = (lo_s, hi_s); summary[label]["ci_final"] = (lo_f, hi_f)
        else:
            print(f"{f'{label} (n={len(runs)} seeds)':<18}{ms_str(settle_med):<46}{ms_str(final_med, 4)}")
        seeds = seeds_of.get(label, [None])
        for run_i, r in enumerate(runs):
            csv_rows.append([label, seeds[run_i], np.median(r["settle"]), np.percentile(r["settle"], 90),
                             np.median(r["final_err"]), r["rms_torque"].mean(), r["jerk"].mean(), solved[run_i]])
    print("-" * 100)
    print("Effort (mean RMS torque, N*m) and smoothness (jerk):")
    for label, sm in summary.items():
        print(f"  {label:<8} RMS torque {ms_str(sm['rms'], 4)}   jerk {ms_str(sm['jerk'], 5)}")

    # ---------------- paired comparison PID vs PPO ----------------
    print("\nPaired comparison on the same scenarios (PID versus each PPO model, averaged over seeds):")
    pid = results["PID"][0]
    for label in ("PPO", "PPO+I"):
        if label not in results:
            continue
        faster = np.mean([np.mean(pid["settle"] < r["settle"]) for r in results[label]]) * 100
        more_acc = np.mean([np.mean(pid["final_err"] < r["final_err"]) for r in results[label]]) * 100
        print(f"  PID vs {label:<6}: PID settles strictly faster in {faster:.0f}% of scenarios, "
              f"has lower final error in {more_acc:.0f}%")

    # ---------------- CSV ----------------
    with open("seed_results.csv", "w") as f:
        f.write("controller,seed,median_settle_s,p90_settle_s,median_final_err_deg,mean_rms_torque,mean_jerk,pct_settled_below_1deg\n")
        for row in csv_rows:
            f.write(",".join("" if v is None else (f"{v:.5f}" if isinstance(v, (float, np.floating)) else str(v)) for v in row) + "\n")

    # ---------------- plot ----------------
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        colors = {"PD": "tab:blue", "PID": "tab:orange", "PPO": "tab:green", "PPO+I": "tab:red"}
        labels = list(results)
        tt = np.arange(N_STEPS + 1) * DT_CTRL
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        for label in labels:
            runs = results[label]
            med_curves = np.array([np.median(r["curves"], axis=0) for r in runs])
            line = med_curves.mean(axis=0)
            ax[0].plot(tt, line, color=colors[label], label=label + (f" (mean of {len(runs)} seeds)" if len(runs) > 1 else ""))
            if len(runs) > 1:
                ax[0].fill_between(tt, med_curves.min(axis=0), med_curves.max(axis=0), color=colors[label], alpha=0.2)
        ax[0].set_yscale("log"); ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("attitude error (deg), median over scenarios")
        ax[0].set_title("Median error (bands: min-max over seeds)"); ax[0].legend(fontsize=7); ax[0].grid(True, which="both", alpha=0.3)

        rng = np.random.default_rng(0)
        for a, key, ylabel, title, log in [(ax[1], "settle", "settling time to 1 deg (s)", "Settling time", False),
                                           (ax[2], "final", "final error (deg)", "Steady-state error", True)]:
            for x, label in enumerate(labels):
                vals = summary[label][key]
                if label in ("PD", "PID"):
                    lo, hi = summary[label]["ci_" + key]
                    a.errorbar([x], [vals[0]], yerr=[[vals[0] - lo], [hi - vals[0]]], fmt="o", color=colors[label], capsize=4)
                else:
                    a.scatter(x + rng.uniform(-0.12, 0.12, len(vals)), vals, color=colors[label], alpha=0.8, s=28)
                    a.hlines(vals.mean(), x - 0.25, x + 0.25, color="black", lw=1.5)
            a.set_xticks(range(len(labels))); a.set_xticklabels(labels)
            a.set_ylabel(ylabel); a.set_title(title + " (dots: seeds, bar: mean; PD/PID: 95% CI)", fontsize=9)
            if log:
                a.set_yscale("log")
            a.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig("seed_plot.png", dpi=150)
        print("\nSaved seed_results.csv and seed_plot.png")
        if "--no-show" not in sys.argv:
            plt.show()
    except ImportError:
        pass
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")

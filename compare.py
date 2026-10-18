"""
Fair comparison: PD vs tuned PID vs PPO vs PPO+I on many random scenarios.

Fairness rules:
  * Same environment, same torque limit, same control period (0.1 s) for all controllers.
  * PD and PID gains are tuned by a small grid search on 10 TRAINING scenarios,
    then everything is tested on 100 DIFFERENT scenarios (seeds never used in tuning).
  * Scenarios come from the same random distribution PPO was trained on
    (start 20-150 deg, random spin, random constant disturbance torque).

Usage:
    python compare.py            # full comparison (uses ppo_attitude.zip and, if present, ppo_attitude_i.zip)
    python compare.py --no-ppo   # PD and PID only (skips PPO)
    python compare.py --quick    # fewer scenarios, for a fast test run

Outputs: compare_results.csv, compare_plot.png
"""
import os
import sys
import time
import itertools
import numpy as np

from attitude_env import AttitudeEnv, attitude_error_deg_from_state
from pid_control import PID, TAU_MAX

T_END = 60.0          # seconds simulated per scenario
DT_CTRL = 0.1         # control period (same as the PPO environment)
N_STEPS = int(T_END / DT_CTRL)
N_TRAIN, N_TEST = 10, 100
TRAIN_SEED0, TEST_SEED0 = 0, 1000


def make_scenarios(n, seed0):
    env = AttitudeEnv()
    out = []
    for i in range(n):
        env.reset(seed=seed0 + i)
        out.append((env.x.copy(), env.d.copy()))
    return out


def rollout(policy, x0, d, use_integral=False):
    """policy(env) -> action in [-1,1]^3. Returns error curve (deg) and torque history."""
    env = AttitudeEnv(fixed_start=x0, fixed_disturbance=d, use_integral=use_integral)
    env.reset()
    errs = [attitude_error_deg_from_state(env.x)]
    taus = []
    for _ in range(N_STEPS):
        u = np.clip(policy(env), -1.0, 1.0)
        env.step(u)
        errs.append(attitude_error_deg_from_state(env.x))
        taus.append(u * TAU_MAX)
    return np.array(errs), np.array(taus)


def metrics(errs, taus):
    t = np.arange(len(errs)) * DT_CTRL
    last = np.where(errs > 1.0)[0]
    settle = 0.0 if len(last) == 0 else (T_END if last[-1] >= len(errs) - 1 else t[last[-1] + 1])
    final = errs[-int(5 / DT_CTRL):].mean()                 # mean error over the last 5 s
    rms = np.sqrt((taus ** 2).sum(axis=1).mean())           # RMS torque magnitude (N*m)
    jerk = np.linalg.norm(np.diff(taus, axis=0), axis=1).mean()  # how much torque jumps per step
    return dict(final_err=final, settle=settle, rms_torque=rms, jerk=jerk)


def pid_policy(kp, kd, ki):
    ctrl = PID(kp=kp, kd=kd, ki=ki, q_desired=[1, 0, 0, 0], dt=DT_CTRL, e_on=0.1 if ki > 0 else None)
    return lambda env: ctrl(0.0, env.x) / TAU_MAX


def score(gains, scenarios):
    kp, kd, ki = gains
    ms = [metrics(*rollout(pid_policy(kp, kd, ki), x0, d)) for x0, d in scenarios]
    # lower is better: settling time + a heavy penalty for steady-state error
    return np.mean([m["settle"] + 20.0 * m["final_err"] for m in ms])


def tune(scenarios, with_integral):
    kps, kds = [0.2, 0.4, 0.8, 1.6, 3.2], [0.1, 0.2, 0.4]
    kis = [0.01, 0.03, 0.06] if with_integral else [0.0]
    best, best_s = None, np.inf
    for kp, kd, ki in itertools.product(kps, kds, kis):
        s = score((kp, kd, ki), scenarios)
        if s < best_s:
            best, best_s = (kp, kd, ki), s
    return best, best_s


def find_model(filename="ppo_attitude.zip"):
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.getcwd()):
        p = os.path.join(folder, filename)
        if os.path.exists(p):
            return p
    return None


def summarize(name, ms):
    arr = {k: np.array([m[k] for m in ms]) for k in ms[0]}
    solved = np.mean((arr["final_err"] < 1.0) & (arr["settle"] < T_END)) * 100
    print(f"{name:<6} | settle: median {np.median(arr['settle']):5.1f} s (90th pct {np.percentile(arr['settle'], 90):5.1f}) | "
          f"final err: median {np.median(arr['final_err']):.3f} deg | RMS torque {arr['rms_torque'].mean():.4f} N*m | "
          f"jerk {arr['jerk'].mean():.4f} | settled <1deg: {solved:.0f}%")
    return arr


if __name__ == "__main__":
    use_ppo = "--no-ppo" not in sys.argv
    quick = "--quick" in sys.argv
    n_train, n_test = (5, 10) if quick else (N_TRAIN, N_TEST)
    t0 = time.time()

    train = make_scenarios(n_train, TRAIN_SEED0)
    test = make_scenarios(n_test, TEST_SEED0)

    print("Tuning PD gains ...")
    pd_gains, _ = tune(train, with_integral=False)
    print("  best PD  (kp, kd, ki) =", pd_gains)
    print("Tuning PID gains ...")
    pid_gains, _ = tune(train, with_integral=True)
    print("  best PID (kp, kd, ki) =", pid_gains)

    controllers = {"PD": pid_policy(*pd_gains), "PID": pid_policy(*pid_gains)}
    models = {}   # name -> (model, uses_integral_observation)
    if use_ppo:
        from stable_baselines3 import PPO
        for name, fname, flag in [("PPO", "ppo_attitude.zip", False), ("PPO+I", "ppo_attitude_i.zip", True)]:
            path = find_model(fname)
            if path is None:
                print(f"{fname} not found next to this script or in the current folder. Skipping {name}.")
            else:
                models[name] = (PPO.load(path, device="cpu"), flag)

    results, curves = {}, {}
    print(f"\nTesting on {n_test} random scenarios ...\n")
    for name in ["PD", "PID"] + list(models):
        ms, cs = [], []
        for x0, d in test:
            flag = False
            if name in models:
                model, flag = models[name]
                pol = lambda env, model=model: model.predict(env._obs(), deterministic=True)[0]
            elif name == "PD":
                pol = pid_policy(*pd_gains)       # fresh controller (fresh integral) per scenario
            else:
                pol = pid_policy(*pid_gains)
            errs, taus = rollout(pol, x0, d, use_integral=flag)
            ms.append(metrics(errs, taus)); cs.append(errs)
        results[name] = summarize(name, ms)
        curves[name] = np.array(cs)

    # CSV
    names = list(results)
    with open("compare_results.csv", "w") as f:
        f.write("scenario," + ",".join(f"{n}_{k}" for n in names for k in ("final_err", "settle", "rms_torque", "jerk")) + "\n")
        for i in range(n_test):
            f.write(str(i) + "," + ",".join(f"{results[n][k][i]:.5f}" for n in names
                                            for k in ("final_err", "settle", "rms_torque", "jerk")) + "\n")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        tt = np.arange(N_STEPS + 1) * DT_CTRL
        for n in names:
            ax[0].plot(tt, np.median(curves[n], axis=0), label=n)
        ax[0].set_yscale("log"); ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("attitude error (deg), median")
        ax[0].set_title("Median error over scenarios"); ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)
        ax[1].boxplot([results[n]["settle"] for n in names])
        ax[1].set_xticks(range(1, len(names) + 1)); ax[1].set_xticklabels(names)
        ax[1].set_ylabel("settling time to 1 deg (s)"); ax[1].set_title("Settling time"); ax[1].grid(True, alpha=0.3)
        ax[2].boxplot([results[n]["final_err"] for n in names])
        ax[2].set_xticks(range(1, len(names) + 1)); ax[2].set_xticklabels(names)
        ax[2].set_yscale("log"); ax[2].set_ylabel("final error (deg)"); ax[2].set_title("Steady-state error")
        ax[2].grid(True, which="both", alpha=0.3)
        plt.tight_layout(); plt.savefig("compare_plot.png", dpi=150)
        print("\nSaved compare_results.csv and compare_plot.png")
        plt.show()
    except ImportError:
        pass
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")

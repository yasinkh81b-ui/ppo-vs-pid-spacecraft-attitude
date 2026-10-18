"""
Train PPO for attitude control and evaluate it on the SAME scenario used for the PID.

Usage:
    python train_ppo.py                      # train (default 300k steps), saves ppo_attitude.zip
    python train_ppo.py 500000               # custom number of steps
    python train_ppo.py --integral           # PPO+I: observation also contains the error integral
                                             #   -> saves ppo_attitude_i.zip
    python train_ppo.py eval                 # only evaluate the saved plain model
    python train_ppo.py eval --integral      # only evaluate the saved PPO+I model
"""
import sys
import time
import numpy as np
from stable_baselines3 import PPO

from attitude_env import AttitudeEnv, attitude_error_deg_from_state

MODEL_PATH = "ppo_attitude"
MODEL_PATH_I = "ppo_attitude_i"


def make_eval_env(use_integral=False):
    # Same test scenario as pid_control.py: 90 deg about (1,1,0)/sqrt2, small spin, constant disturbance
    ang = np.radians(90)
    ax = np.array([1, 1, 0]) / np.sqrt(2)
    q0 = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * ax)])
    x0 = np.concatenate([q0, [0.02, -0.01, 0.01]])
    d = np.array([0.002, -0.001, 0.0015])
    return AttitudeEnv(fixed_start=x0, fixed_disturbance=d, use_integral=use_integral)


def evaluate(model, t_end=90.0, use_integral=False):
    env = make_eval_env(use_integral)
    obs, _ = env.reset()
    dt_ctrl = 0.1
    n = int(t_end / dt_ctrl)
    errs = [attitude_error_deg_from_state(env.x)]
    for _ in range(n):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)
        errs.append(attitude_error_deg_from_state(env.x))
    errs = np.array(errs)
    t = np.arange(n + 1) * dt_ctrl
    idx = np.where(errs > 1.0)[0]
    print(f"{'PPO+I' if use_integral else 'PPO'}: final error = {errs[-1]:.3f} deg, "
          f"settling to 1 deg = {t[idx[-1]] if len(idx) else 0.0:.1f} s, "
          f"max error after t=15s = {errs[int(15 / dt_ctrl):].max():.2f} deg")
    np.save("ppo_error_curve_i.npy" if use_integral else "ppo_error_curve.npy", np.stack([t, errs]))
    return t, errs


if __name__ == "__main__":
    args = sys.argv[1:]
    use_integral = "--integral" in args
    args = [a for a in args if a != "--integral"]
    path = MODEL_PATH_I if use_integral else MODEL_PATH

    if "eval" not in args:
        steps = int(args[0]) if args else 300_000
        env = AttitudeEnv(use_integral=use_integral)
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                    gamma=0.99, verbose=1, device="cpu", seed=0)
        t0 = time.time()
        model.learn(total_timesteps=steps)
        print(f"Training took {(time.time() - t0) / 60:.1f} min")
        model.save(path)
    else:
        model = PPO.load(path, device="cpu")

    t, errs = evaluate(model, use_integral=use_integral)
    try:
        import matplotlib.pyplot as plt
        plt.plot(t, errs, label="PPO+I" if use_integral else "PPO")
        plt.xlabel("time (s)"); plt.ylabel("attitude error (deg)")
        plt.legend(); plt.grid(True); plt.show()
    except ImportError:
        pass

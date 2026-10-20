"""
Train several independent PPO runs (different random seeds) for BOTH variants
(plain PPO and PPO+I), one after another, with no plot windows.

Usage:
    python train_all_seeds.py              # seeds 1 2 3 4  (seed 0 = the models you already have)
    python train_all_seeds.py 1 2          # only these seeds
    python train_all_seeds.py 3 4          # run this in a SECOND terminal to use two CPU cores

Each model takes about 20 minutes. The script is resumable: models that already exist are skipped,
so if you stop it (or the laptop goes to sleep) just run the same command again.
Saved files: ppo_attitude_s<seed>.zip and ppo_attitude_i_s<seed>.zip
"""
import os
import sys
import time

from train_ppo import train, model_path

if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4]
    jobs = [(seed, use_integral) for seed in seeds for use_integral in (False, True)]
    done_times = []
    for k, (seed, use_integral) in enumerate(jobs, 1):
        name = model_path(use_integral, seed)
        label = f"[{k}/{len(jobs)}] {'PPO+I' if use_integral else 'PPO'} seed {seed}"
        if os.path.exists(name + ".zip"):
            print(f"{label}: {name}.zip already exists, skipping")
            continue
        t0 = time.time()
        print(f"{label}: training started (about 20 min) ...", flush=True)
        train(use_integral=use_integral, seed=seed, verbose=0)
        done_times.append(time.time() - t0)
        avg = sum(done_times) / len(done_times)
        left = sum(1 for (s_, i_) in jobs[k:] if not os.path.exists(model_path(i_, s_) + ".zip"))
        print(f"{label}: done in {done_times[-1] / 60:.1f} min -> saved {name}.zip. "
              f"Remaining: {left} model(s), about {left * avg / 60:.0f} min", flush=True)
    print("All requested seeds are trained.")

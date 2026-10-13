"""
Quick smoke test for the 6G in-factory subnetwork MARL framework.
Runs a tiny MAPPO training loop to verify the environment + algorithm work
under the current Python/numpy/torch versions.

Run from the code/ directory:
    python smoke_test.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Args
from utils import make_env, make_runner


def main():
    all_args = Args(alg='mappo', prob='channel', clutter='sparse',
                    observation='I', reward='rate', t_agg=1024,
                    n_agent=8, numDev=1, n_channel=4)

    # Tiny run so the smoke test finishes in seconds.
    all_args.episode_length = 10
    all_args.num_env_steps = 5 * all_args.episode_length

    env = make_env(all_args)
    device = torch.device('cpu')

    run_dir = Path(__file__).resolve().parent.parent / 'results' / 'smoke_test'
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "all_args": all_args,
        "envs": env,
        "eval_envs": None,
        "num_agents": all_args.n_agent,
        "device": device,
        "run_dir": run_dir,
    }

    runner = make_runner(config)
    runner.run()

    print('\nSMOKE TEST PASSED')


if __name__ == '__main__':
    main()

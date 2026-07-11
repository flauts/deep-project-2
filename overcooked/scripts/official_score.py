"""Official Overcooked competition score calculator.

Official score formula:
    official_score = 10000 * num_soups
                    + 10 * (horizon - last_soup_timestep)
                    + (horizon - first_soup_timestep)
                    - penalty

Where:
    num_soups           = count of timesteps where reward > 0 (each delivery = +20)
    horizon             = episode length (usually 250)
    first_soup_timestep = timestep of the first soup delivery
    last_soup_timestep  = timestep of the last soup delivery
    penalty             = penalties for timeouts, invalid actions, etc. (0 by default)

Usage:
    from scripts.official_score import compute_official_score, score_from_npz, score_from_pkl
"""

from __future__ import annotations

import json
import numpy as np
import pickle
from pathlib import Path


def compute_official_score(
    rewards: np.ndarray,
    timesteps: np.ndarray | None = None,
    horizon: int | None = None,
    penalty: float = 0.0,
) -> dict:
    """Compute official score from a reward array.

    Args:
        rewards: per-timestep rewards (float array). Positive = soup delivered.
        timesteps: per-timestep timestep indices. If None, uses 0..len-1.
        horizon: episode horizon. If None, uses len(rewards).
        penalty: penalty value to subtract.

    Returns:
        dict with: official_score, num_soups, first_soup_timestep, last_soup_timestep,
        horizon, penalty
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    if timesteps is None:
        timesteps = np.arange(len(rewards))
    else:
        timesteps = np.asarray(timesteps)
    if horizon is None:
        horizon = len(rewards)

    soup_mask = rewards > 0
    soup_timesteps = timesteps[soup_mask]
    num_soups = int(soup_mask.sum())

    if num_soups == 0:
        return {
            "official_score": 0 - penalty,
            "num_soups": 0,
            "first_soup_timestep": None,
            "last_soup_timestep": None,
            "horizon": int(horizon),
            "penalty": float(penalty),
        }

    first_soup = int(soup_timesteps[0])
    last_soup = int(soup_timesteps[-1])

    score = (
        10000 * num_soups
        + 10 * (horizon - last_soup)
        + (horizon - first_soup)
        - penalty
    )

    return {
        "official_score": int(score),
        "num_soups": num_soups,
        "first_soup_timestep": first_soup,
        "last_soup_timestep": last_soup,
        "horizon": int(horizon),
        "penalty": float(penalty),
    }


def score_from_npz(npz_path: str | Path) -> dict:
    """Compute official score from an .npz recording file."""
    data = np.load(npz_path, allow_pickle=True)
    rewards = data["rewards"]
    timesteps = data["timesteps"] if "timesteps" in data.files else None
    horizon = len(rewards)
    return compute_official_score(rewards, timesteps=timesteps, horizon=horizon)


def score_from_pkl(pkl_path: str | Path) -> dict:
    """Compute official score from a .pkl recording file."""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    records = payload.get("records", [])
    rewards = np.array([r["reward"] for r in records], dtype=np.float32)
    timesteps = np.array([r.get("timestep", i) for i, r in enumerate(records)])
    horizon = len(rewards)
    return compute_official_score(rewards, timesteps=timesteps, horizon=horizon)


def score_from_episode(rewards: list, timesteps: list | None = None, horizon: int | None = None) -> dict:
    """Compute official score from raw reward/timestep lists."""
    return compute_official_score(
        np.asarray(rewards, dtype=np.float32),
        timesteps=np.asarray(timesteps) if timesteps is not None else None,
        horizon=horizon,
    )


if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) < 2:
        print("Usage: python official_score.py <file.npz|file.pkl|glob_pattern>")
        print("Example: python official_score.py ../overcooked/Softmaxxing/demonstrations/*.npz")
        sys.exit(1)

    pattern = sys.argv[1]
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        files = [pattern]

    print(f"{'FILE':<60} {'SOUPS':>5} {'FIRST':>5} {'LAST':>5} {'SCORE':>10}")
    print("-" * 90)
    for f in files:
        if not Path(f).exists():
            continue
        try:
            if f.endswith(".npz"):
                result = score_from_npz(f)
            elif f.endswith(".pkl"):
                result = score_from_pkl(f)
            else:
                continue
            name = Path(f).name
            print(
                f"{name:<60} {result['num_soups']:>5} "
                f"{str(result['first_soup_timestep']):>5} "
                f"{str(result['last_soup_timestep']):>5} "
                f"{result['official_score']:>10}"
            )
        except Exception as e:
            print(f"{Path(f).name:<60} ERROR: {e}")
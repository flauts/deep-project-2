"""Build the final training dataset from the filtered recordings.

Pads observations to a fixed max shape, attaches metadata (layout, tier, weight,
role_swap), and writes a single consolidated .npz + stats .json.

Usage:
    cd overcooked
    python ..\train\build_dataset.py [--input train/data/consolidated_filtered.npz]
                                      [--output train/data/consolidated.npz]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PROJECT_ROOT = SCRIPT_DIR.parent
OVERCOOKED_DIR = PROJECT_ROOT / "overcooked"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DATA_DIR / "consolidated_filtered.npz"))
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "consolidated.npz"))
    parser.add_argument("--stats", type=str, default=str(DATA_DIR / "dataset_stats.json"))
    args = parser.parse_args()

    print(f"Loading filtered dataset: {args.input}")
    data = np.load(args.input, allow_pickle=True)

    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    weights = data["weights"]
    tiers = data["tiers"]
    layouts = data["layouts"]
    role_swaps = data["role_swaps"]
    episode_ids = data["episode_ids"]
    timesteps = data["timesteps"]

    has_next_obs = "next_obs" in data.files
    next_obs = data["next_obs"] if has_next_obs else None

    print(f"  Raw obs shape: {obs.shape}")
    print(f"  Total timesteps: {len(actions)}")

    # Determine max obs shape across all records
    max_dim2 = obs.shape[1] if obs.ndim >= 2 else 1
    max_dim3 = obs.shape[2] if obs.ndim >= 3 else 1

    # Pad obs to (N, max_dim2, max_dim3) if needed
    if obs.ndim == 2:
        # (N, D) — already flat, just record the shape
        obs_dim = obs.shape[1]
        print(f"  Obs dim: {obs_dim}")
        padded_obs = obs.astype(np.float32)
    elif obs.ndim == 3:
        # (N, H, W) — pad H and W to max
        obs_dim = (obs.shape[1], obs.shape[2])
        print(f"  Obs dim: {obs_dim}")
        padded_obs = obs.astype(np.float32)
    else:
        # Higher dim — flatten to 2D
        padded_obs = obs.reshape(obs.shape[0], -1).astype(np.float32)
        obs_dim = padded_obs.shape[1]
        print(f"  Obs dim (flattened): {obs_dim}")

    if has_next_obs and next_obs is not None:
        if next_obs.ndim == 2:
            padded_next = next_obs.astype(np.float32)
        elif next_obs.ndim == 3:
            padded_next = next_obs.astype(np.float32)
        else:
            padded_next = next_obs.reshape(next_obs.shape[0], -1).astype(np.float32)
    else:
        padded_next = None

    # Create layout name -> index mapping
    layout_names = sorted(set(layouts.tolist()))
    layout_to_idx = {name: i for i, name in enumerate(layout_names)}

    layout_indices = np.array([layout_to_idx[str(l)] for l in layouts], dtype=np.int64)

    # Save consolidated dataset
    save_dict = {
        "obs": padded_obs,
        "actions": actions.astype(np.int64),
        "rewards": rewards.astype(np.float32),
        "dones": dones.astype(np.bool_),
        "weights": weights.astype(np.float32),
        "tiers": tiers.astype(np.int64),
        "layout_indices": layout_indices,
        "layout_names": np.array(layout_names, dtype=object),
        "role_swaps": role_swaps.astype(np.bool_),
        "episode_ids": episode_ids.astype(np.int64),
        "timesteps": timesteps.astype(np.int64),
    }
    if padded_next is not None:
        save_dict["next_obs"] = padded_next

    np.savez_compressed(args.output, **save_dict)
    print(f"\nSaved consolidated dataset: {args.output}")
    print(f"  Obs shape: {padded_obs.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Layouts: {len(layout_names)}")

    # Compute stats
    stats = {
        "total_timesteps": int(len(actions)),
        "obs_dim": int(obs_dim) if isinstance(obs_dim, int) else list(obs_dim),
        "num_layouts": len(layout_names),
        "layout_names": layout_names,
        "action_distribution": {
            str(i): int(np.sum(actions == i)) for i in range(6)
        },
        "tier_distribution": {
            "gold": int(np.sum(tiers == 0)),
            "silver": int(np.sum(tiers == 1)),
            "bronze": int(np.sum(tiers == 2)),
        },
        "per_layout": {},
    }

    for layout_name in layout_names:
        mask = layouts == layout_name
        n = int(mask.sum())
        if n == 0:
            continue
        layout_actions = actions[mask]
        layout_rewards = rewards[mask]
        layout_tiers = tiers[mask]
        layout_ep_ids = episode_ids[mask]

        num_episodes = len(set(layout_ep_ids.tolist()))
        scores = []
        for eid in set(layout_ep_ids.tolist()):
            ep_mask = (layouts == layout_name) & (episode_ids == eid)
            scores.append(float(np.sum(rewards[ep_mask])))

        vals, cnts = np.unique(layout_actions, return_counts=True)
        stats["per_layout"][layout_name] = {
            "timesteps": n,
            "episodes": num_episodes,
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "max_score": float(max(scores)) if scores else 0.0,
            "min_score": float(min(scores)) if scores else 0.0,
            "action_dist": {str(int(v)): int(c) for v, c in zip(vals, cnts)},
            "tier_counts": {
                "gold": int(np.sum(layout_tiers == 0)),
                "silver": int(np.sum(layout_tiers == 1)),
                "bronze": int(np.sum(layout_tiers == 2)),
            },
        }

    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved stats: {args.stats}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total timesteps: {stats['total_timesteps']}")
    print(f"Obs dim: {stats['obs_dim']}")
    print(f"Layouts: {stats['num_layouts']}")
    print(f"\nAction distribution:")
    for a in range(6):
        names = ["north", "south", "east", "west", "stay", "interact"]
        print(f"  {a} ({names[a]:>9}): {stats['action_distribution'][str(a)]:>6} ({100*stats['action_distribution'][str(a)]/stats['total_timesteps']:.1f}%)")
    print(f"\nTier distribution:")
    for tier in ["gold", "silver", "bronze"]:
        print(f"  {tier:>6}: {stats['tier_distribution'][tier]:>6}")
    print(f"\nPer-layout (top 15 by timestep count):")
    print(f"{'LAYOUT':<35} {'STEPS':>6} {'EPS':>4} {'AVG_SC':>7} {'GOLD':>4} {'SILV':>4} {'BRNZ':>4}")
    sorted_layouts = sorted(stats["per_layout"].items(), key=lambda x: -x[1]["timesteps"])
    for name, s in sorted_layouts[:15]:
        print(f"{name:<35} {s['timesteps']:>6} {s['episodes']:>4} {s['avg_score']:>7.1f} {s['tier_counts']['gold']:>4} {s['tier_counts']['silver']:>4} {s['tier_counts']['bronze']:>4}")


if __name__ == "__main__":
    main()
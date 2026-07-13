"""Extract user (human_keyboard) recordings into a separate training dataset.

Scans user_recordings for .npz files where the recorded agent is human_keyboard.
Loads each, concatenates, appends topology features, and saves as a consolidated .npz.
Used to train a coordination partner BC model from the user's own demonstrations.

Usage:
    python train/extract_user_recordings.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "overcooked"))

from env import compute_topology_for_layout, load_dynamics_overrides

RECORDINGS_ROOT = Path("overcooked/data/user_recordings")
OUTPUT_PATH = Path("train/data/consolidated_user_only.npz")


def is_human_recording(metadata_path: str) -> bool:
    if not os.path.exists(metadata_path):
        return False
    try:
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        pols = meta.get("policies", {})
        rai = meta.get("data_collection", {}).get("record_agent_indices", [])
        if not rai:
            return False
        idx = rai[0]
        key = f"agent_{idx}"
        return pols.get(key, {}).get("name") == "human_keyboard"
    except Exception:
        return False


def main():
    dynamics_overrides = load_dynamics_overrides()
    layouts_dir = Path("overcooked/layouts")

    all_obs, all_actions, all_rewards, all_dones = [], [], [], []
    all_weights, all_tiers, all_layouts, all_role_swaps = [], [], [], []
    all_episode_ids, all_timesteps = [], []
    next_obs_list = []

    total_files = 0
    episode_id_counter = 0

    for root, dirs, files in os.walk(RECORDINGS_ROOT):
        for f in files:
            if not f.endswith(".npz"):
                continue
            npz_path = os.path.join(root, f)
            meta_path = npz_path[:-4] + ".metadata.json"

            if not is_human_recording(meta_path):
                continue

            try:
                data = np.load(npz_path, allow_pickle=True)
                n = len(data["actions"])

                # Extract layout name from metadata_json (may be 0-dim scalar)
                layout_name = "unknown"
                try:
                    mj = data.get("metadata_json")
                    if mj is not None:
                        raw = str(mj.item()) if hasattr(mj, 'item') else str(mj)
                        meta = json.loads(raw) if isinstance(raw, str) else raw
                        layout_name = meta.get("environment", {}).get("layout_name", "unknown")
                except Exception:
                    pass

                obs = data["obs"].astype(np.float32)
                if obs.ndim == 2:
                    obs = obs[:, :96]  # ensure 96-dim raw
                actions = data["actions"].astype(np.int64)
                rewards = data["rewards"].astype(np.float32)
                dones = data.get("dones", np.zeros(n, dtype=bool)).astype(bool)
                role_swaps = data.get("role_swaps", np.zeros(n, dtype=bool)).astype(bool)

                if "next_obs" in data.files:
                    next_obs = data["next_obs"].astype(np.float32)
                    if next_obs.ndim == 2:
                        next_obs = next_obs[:, :96]

                all_obs.append(obs)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_dones.append(dones)
                all_role_swaps.append(role_swaps)
                all_layouts.extend([layout_name or "unknown"] * n)
                all_weights.extend([1.0] * n)  # all gold — user's own recordings
                all_tiers.extend([0] * n)  # tier 0 = gold
                all_episode_ids.extend([episode_id_counter] * n)
                all_timesteps.extend(range(n))
                if "next_obs" in data.files:
                    next_obs_list.append(next_obs)
                else:
                    next_obs_list.append(np.zeros_like(obs))

                episode_id_counter += 1
                total_files += 1
            except Exception as e:
                print(f"  WARNING: Failed to load {npz_path}: {e}")

    if total_files == 0:
        print("No human recordings found!")
        return

    # Concatenate
    obs = np.concatenate(all_obs)
    actions = np.concatenate(all_actions)
    rewards = np.concatenate(all_rewards)
    dones = np.concatenate(all_dones)
    role_swaps = np.concatenate(all_role_swaps)
    weights = np.array(all_weights, dtype=np.float32)
    tiers = np.array(all_tiers, dtype=np.int64)
    layouts = np.array(all_layouts, dtype=object)
    episode_ids = np.array(all_episode_ids, dtype=np.int64)
    timesteps = np.array(all_timesteps, dtype=np.int64)
    next_obs = np.concatenate(next_obs_list) if next_obs_list else None

    print(f"Loaded {total_files} recordings, {len(actions)} timesteps")
    print(f"Layouts: {len(set(all_layouts))}")

    # Append topology features
    topo_feat_map = {}
    topo_dim = 0
    for lname in sorted(set(all_layouts)):
        try:
            feats = compute_topology_for_layout(lname, layouts_dir, dynamics_overrides)
            topo_feat_map[lname] = feats
            if topo_dim == 0:
                topo_dim = len(feats)
        except Exception as e:
            print(f"  WARNING: Cannot compute topo for '{lname}': {e}. Using zeros.")
    if topo_dim == 0:
        topo_dim = 25
    topo_default = np.zeros(topo_dim, dtype=np.float32)
    topo_arr = np.array([topo_feat_map.get(str(l), topo_default) for l in layouts], dtype=np.float32)

    padded_obs = np.concatenate([obs.astype(np.float32), topo_arr], axis=1)
    if next_obs is not None:
        padded_next = np.concatenate([next_obs.astype(np.float32), topo_arr], axis=1)
    else:
        padded_next = None
    print(f"Added {topo_dim}-dim topology features: obs shape {padded_obs.shape}")

    save_dict = {
        "obs": padded_obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "weights": weights,
        "tiers": tiers,
        "layout_names": np.array(sorted(set(all_layouts)), dtype=object),
        "role_swaps": role_swaps,
        "episode_ids": episode_ids,
        "timesteps": timesteps,
    }
    if padded_next is not None:
        save_dict["next_obs"] = padded_next

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, **save_dict)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""Generate greedy vs greedy demonstration recordings for BC training.

Runs N episodes per layout using two GreedyFullTaskPolicy agents.
Records BOTH agents' perspectives for position-independent learning.

Usage:
    python train/generate_demos.py [--episodes 10] [--all-maps]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "overcooked"))

from env import build_env_for_layout, load_dynamics_overrides
from policies.basic_policies import GreedyFullTaskPolicy
from src.constants import overcooked_action_to_index

ALL_BUILTINS = [
    "cramped_room", "asymmetric_advantages", "coordination_ring",
    "counter_circuit", "forced_coordination", "simple_o", "simple_tomato",
    "large_room", "small_corridor", "soup_coordination",
    "tutorial_0", "tutorial_1", "tutorial_2", "tutorial_3",
    "scenario1_s", "scenario2", "scenario2_s", "scenario3", "scenario4",
    "m_shaped_s", "five_by_five", "mdp_test",
]


def generate_demos(layout_name, n_episodes, output_base):
    overrides = load_dynamics_overrides()
    layouts_dir = Path("overcooked/layouts")

    try:
        env = build_env_for_layout(layout_name, layouts_dir, overrides, shaped_reward_scale=0.0)
    except Exception as e:
        print(f"    ERROR building env: {e}")
        return 0, 0

    output_dir = Path(output_base) / layout_name / "greedy_full_task"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_soups = 0
    for ep in range(n_episodes):
        seed = 42 + ep
        np.random.seed(seed)

        obs = env.reset()

        g0 = GreedyFullTaskPolicy(ingredient="onion", avoid_teammate=True)
        g1 = GreedyFullTaskPolicy(ingredient="onion", avoid_teammate=True)
        g0.set_mdp(env.mdp)
        g1.set_mdp(env.mdp)
        g0.agent_index = 0
        g1.agent_index = 1

        # Separate buffers for each agent
        data0 = {"obs": [], "actions": [], "rewards": [], "next_obs": []}
        data1 = {"obs": [], "actions": [], "rewards": [], "next_obs": []}

        for step in range(400):
            state = env.env.state
            obs_pair = env.env.featurize_state_mdp(state)

            act0, _ = g0.action(state)
            act1, _ = g1.action(state)
            act0_idx = overcooked_action_to_index(act0)
            act1_idx = overcooked_action_to_index(act1)

            next_obs, rewards, dones, info = env.step([act0_idx, act1_idx])
            sparse = info.get("sparse_r_by_agent", [0.0, 0.0])

            data0["obs"].append(np.asarray(obs_pair[0], dtype=np.float32).flatten())
            data0["actions"].append(act0_idx)
            data0["rewards"].append(float(sparse[0]))

            data1["obs"].append(np.asarray(obs_pair[1], dtype=np.float32).flatten())
            data1["actions"].append(act1_idx)
            data1["rewards"].append(float(sparse[1]))

            next_state = env.env.state
            next_pair = env.env.featurize_state_mdp(next_state)
            data0["next_obs"].append(np.asarray(next_pair[0], dtype=np.float32).flatten())
            data1["next_obs"].append(np.asarray(next_pair[1], dtype=np.float32).flatten())

            obs = next_obs
            if dones[0]:
                break

        n_steps = len(data0["actions"])
        soups = int(sum(data0["rewards"]) / 20)
        total_soups += soups

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save agent 0
        npz0 = output_dir / f"{layout_name}_{timestamp}_{ep:03d}_a0.npz"
        np.savez_compressed(
            npz0,
            obs=np.array(data0["obs"], dtype=np.float32),
            actions=np.array(data0["actions"], dtype=np.int64),
            rewards=np.array(data0["rewards"], dtype=np.float32),
            dones=np.full(n_steps, False, dtype=bool),
            episode_ids=np.full(n_steps, ep, dtype=np.int64),
            episode_seeds=np.full(n_steps, seed, dtype=np.int64),
            timesteps=np.arange(n_steps, dtype=np.int64),
            agent_indices=np.zeros(n_steps, dtype=np.int64),
            role_swaps=np.zeros(n_steps, dtype=bool),
            next_obs=np.array(data0["next_obs"], dtype=np.float32),
        )
        meta0 = {"policies": {"agent_0": {"type": "builtin", "name": "greedy_full_task"},
                             "agent_1": {"type": "builtin", "name": "greedy_full_task"}},
                 "data_collection": {"enabled": True, "record_agent_indices": [0]},
                 "environment": {"layout_name": layout_name, "layout_file": None, "horizon": 400}}
        with open(npz0.with_suffix(".metadata.json"), "w") as f:
            json.dump(meta0, f, indent=2)

        # Save agent 1
        npz1 = output_dir / f"{layout_name}_{timestamp}_{ep:03d}_a1.npz"
        np.savez_compressed(
            npz1,
            obs=np.array(data1["obs"], dtype=np.float32),
            actions=np.array(data1["actions"], dtype=np.int64),
            rewards=np.array(data1["rewards"], dtype=np.float32),
            dones=np.full(n_steps, False, dtype=bool),
            episode_ids=np.full(n_steps, ep, dtype=np.int64),
            episode_seeds=np.full(n_steps, seed, dtype=np.int64),
            timesteps=np.arange(n_steps, dtype=np.int64),
            agent_indices=np.ones(n_steps, dtype=np.int64),
            role_swaps=np.zeros(n_steps, dtype=bool),
            next_obs=np.array(data1["next_obs"], dtype=np.float32),
        )
        meta1 = {"policies": {"agent_0": {"type": "builtin", "name": "greedy_full_task"},
                             "agent_1": {"type": "builtin", "name": "greedy_full_task"}},
                 "data_collection": {"enabled": True, "record_agent_indices": [1]},
                 "environment": {"layout_name": layout_name, "layout_file": None, "horizon": 400}}
        with open(npz1.with_suffix(".metadata.json"), "w") as f:
            json.dump(meta1, f, indent=2)

    avg_soups = total_soups / n_episodes if n_episodes > 0 else 0
    return total_soups, avg_soups


def main():
    parser = argparse.ArgumentParser(description="Generate greedy vs greedy demos")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per layout")
    parser.add_argument("--maps", type=str, default=None, help="Comma-separated maps (default: all 22 built-ins)")
    parser.add_argument("--output", type=str, default="overcooked/data/user_recordings")
    args = parser.parse_args()

    maps = args.maps.split(",") if args.maps else ALL_BUILTINS
    print(f"Generating {args.episodes} episodes x 2 agents on {len(maps)} layouts")

    grand_soups = 0
    for layout_name in maps:
        print(f"--- {layout_name} ---", flush=True)
        total, avg = generate_demos(layout_name, args.episodes, args.output)
        print(f"  avg {avg:.1f} soups, total {total}", flush=True)
        grand_soups += total

    print(f"\nDone! {grand_soups} total soups across {len(maps)} layouts")
    print("Next: re-run filter_recordings.py then build_dataset.py")


if __name__ == "__main__":
    main()

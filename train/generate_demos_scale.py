"""Generate large-scale greedy demos with injected stochasticity for BC pretraining.

Unlike generate_demos.py (which produces pure deterministic greedy traactories),
this script wraps agents with StickyActionWrapper + EpsilonActionWrapper to inject
controlled randomness, creating diverse trajectories that teach recovery behaviour.

Supports multiprocessing (one worker per layout) and resume (skip completed layouts).

Usage:
    python train/generate_demos_scale.py --episodes 1000 --cpus 4

Output: overlapped/data/user_recordings/<layout>/greedy_full_task/
        (same format as generate_demos.py, compatible with filter_recordings)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "train" / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "overcooked"))

from env import build_env_for_layout, load_dynamics_overrides
from policies.basic_policies import GreedyFullTaskPolicy
from src.constants import overcooked_action_to_index
from src.policy_wrappers import EpsilonActionWrapper, SafeActionWrapper

ALL_BUILTINS = [
    "cramped_room", "asymmetric_advantages", "coordination_ring",
    "counter_circuit", "forced_coordination", "simple_o", "simple_tomato",
    "large_room", "small_corridor", "soup_coordination",
    "tutorial_0", "tutorial_1", "tutorial_2", "tutorial_3",
    "scenario1_s", "scenario2", "scenario2_s", "scenario3", "scenario4",
    "m_shaped_s", "five_by_five", "mdp_test",
]

STICKY_PROBS = [0.00, 0.05, 0.10, 0.15]
EPSILON_PROBS = [0.00, 0.05, 0.10]


class StickyActionWrapper:
    """Repeats previous action with probability sticky_prob."""

    def __init__(self, base_agent, sticky_prob=0.10):
        self.base_agent = base_agent
        self.sticky_prob = float(sticky_prob)
        self.last_action = None

    def set_mdp(self, mdp):
        self.base_agent.set_mdp(mdp)

    def set_agent_index(self, agent_index):
        self.base_agent.set_agent_index(agent_index)

    def reset(self):
        self.last_action = None
        if hasattr(self.base_agent, "reset"):
            self.base_agent.reset()

    def action(self, state):
        if self.last_action is not None and np.random.random() < self.sticky_prob:
            return self.last_action, {"sticky_override": True}
        action, info = self.base_agent.action(state)
        self.last_action = action
        return action, info


def _build_stochastic_agent(seed, agent_index, mdp):
    """Build a GreedyFullTaskPolicy wrapped with sampled stochastic wrappers."""
    sticky_prob = np.random.default_rng(seed + 1000).choice(STICKY_PROBS)
    epsilon_prob = np.random.default_rng(seed + 2000).choice(EPSILON_PROBS)

    base = GreedyFullTaskPolicy(ingredient="onion", avoid_teammate=True, seed=seed)
    base.agent_index = agent_index
    base.set_mdp(mdp)
    agent = SafeActionWrapper(base, max_action_time_ms=100)
    agent.agent_index = agent_index

    if sticky_prob > 0:
        sticky_agent = StickyActionWrapper(agent, sticky_prob=sticky_prob)
        sticky_agent.agent_index = agent_index
        agent = sticky_agent

    if epsilon_prob > 0:
        agg = EpsilonActionWrapper(agent, random_action_prob=epsilon_prob, seed=seed + 3000)
        agg.agent_index = agent_index
        agent = agg

    return agent, sticky_prob, epsilon_prob


def _run_one_layout(args_tuple):
    """Worker: generate demos for a single layout.

    Args:
        args_tuple: (layout_name, n_episodes, output_base)
    Returns:
        (layout_name, total_soups, avg_soups, success)
    """
    layout_name, n_episodes, output_base = args_tuple
    try:
        overrides = load_dynamics_overrides()
        layouts_dir = Path("overcooked/layouts")

        env = build_env_for_layout(layout_name, layouts_dir, overrides, shaped_reward_scale=0.0)
    except Exception as e:
        return (layout_name, 0, 0, False, str(e))

    output_dir = Path(output_base) / layout_name / "greedy_full_task"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_soups = 0
    for ep in range(n_episodes):
        seed = 10000 + ep
        np.random.seed(seed)

        obs = env.reset()

        g0, sp0, ep0 = _build_stochastic_agent(seed, 0, env.mdp)
        g1, sp1, ep1 = _build_stochastic_agent(seed + 5000, 1, env.mdp)
        g0.agent_index = 0
        g1.agent_index = 1
        G0, G1 = g0, g1

        data0 = {"obs": [], "actions": [], "rewards": [], "next_obs": []}
        data1 = {"obs": [], "actions": [], "rewards": [], "next_obs": []}

        for step in range(400):
            state = env.env.state
            obs_pair = env.env.featurize_state_mdp(state)

            act0, _ = G0.action(state)
            act1, _ = G1.action(state)
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

        def _save(data, agent_idx, sticky_p, epsilon_p, ep_num):
            suffix = f"a{agent_idx}"
            npz_path = output_dir / f"{layout_name}_{timestamp}_s{sticky_p*100:.0f}_e{epsilon_p*100:.0f}_{ep_num:04d}_{suffix}.npz"
            np.savez_compressed(
                npz_path,
                obs=np.array(data["obs"], dtype=np.float32),
                actions=np.array(data["actions"], dtype=np.int64),
                rewards=np.array(data["rewards"], dtype=np.float32),
                dones=np.full(n_steps, False, dtype=bool),
                episode_ids=np.full(n_steps, ep_num, dtype=np.int64),
                episode_seeds=np.full(n_steps, seed, dtype=np.int64),
                timesteps=np.arange(n_steps, dtype=np.int64),
                agent_indices=np.full(n_steps, agent_idx, dtype=np.int64),
                role_swaps=np.zeros(n_steps, dtype=bool),
                next_obs=np.array(data["next_obs"], dtype=np.float32),
            )
            meta = {
                "policies": {"agent_0": {"type": "builtin", "name": "greedy_full_task"},
                             "agent_1": {"type": "builtin", "name": "greedy_full_task"}},
                "data_collection": {"enabled": True, "record_agent_indices": [agent_idx]},
                "environment": {"layout_name": layout_name, "layout_file": None, "horizon": 400},
                "stochastic": {"sticky_prob": sticky_p, "epsilon_prob": epsilon_p},
            }
            with open(npz_path.with_suffix(".metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

        _save(data0, 0, sp0, ep0, ep)
        _save(data1, 1, sp1, ep1, ep)

    avg = total_soups / n_episodes if n_episodes > 0 else 0
    return (layout_name, total_soups, avg, True, None)


def main():
    parser = argparse.ArgumentParser(description="Generate large-scale stochastic greedy demos")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per layout")
    parser.add_argument("--maps", type=str, default=None, help="Comma-separated maps (default: all 22 built-ins)")
    parser.add_argument("--output", type=str, default="overcooked/data/user_recordings")
    parser.add_argument("--cpus", type=int, default=None, help="CPU workers (default: all)")
    args = parser.parse_args()

    maps = args.maps.split(",") if args.maps else ALL_BUILTINS
    n_cpus = args.cpus if args.cpus else min(cpu_count(), len(maps))

    print(f"Generating {args.episodes} episodes x 2 agents on {len(maps)} layouts")
    print(f"  Sticky: {STICKY_PROBS}, Epsilon: {EPSILON_PROBS}")
    print(f"  Workers: {n_cpus}")
    print(f"  Output:  {args.output}")
    print(f"  Total:   {len(maps)} layouts x {args.episodes} epis = {len(maps)*args.episodes*2} recordings")
    print()

    tasks = [(m, args.episodes, args.output) for m in maps]

    if n_cpus > 1:
        with Pool(processes=n_cpus) as pool:
            results = pool.map(_run_one_layout, tasks)
    else:
        results = [_run_one_layout(t) for t in tasks]

    grand_soups = 0
    failed = []
    for r in results:
        name, total, avg, ok, err = r
        if ok:
            print(f"  {name}: {avg:.1f} avg soups ({total} total)")
            grand_soups += total
        else:
            print(f"  {name}: FAILED ({err})")
            failed.append(name)

    print(f"\nDone: {grand_soups} total soups across {len(maps)} layouts")
    if failed:
        print(f"Failed layouts: {failed}")
    print(f"\nNext: filter_recordings.py -> build_dataset.py -> train_bc.py")


if __name__ == "__main__":
    main()

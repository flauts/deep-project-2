"""Benchmark and Watch tool for Overcooked models (BC / PPO).

Allows you to quantitatively benchmark your model across the 5 competition layouts
(or all 56 layouts) while simultaneously WATCHING the AI agents play live in Pygame!

Usage:
    # Watch and evaluate PPO across the 5 competition maps:
    python overcooked/scripts/benchmark_and_watch.py --model train/models/ppo_agent.pt --watch

    # Watch and evaluate BC across the 5 competition maps:
    python overcooked/scripts/benchmark_and_watch.py --model train/models/bc_agent.pt --watch

    # Evaluate across ALL 56 maps without watching:
    python overcooked/scripts/benchmark_and_watch.py --model train/models/ppo_agent.pt --all-maps
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Add project root and overcooked/src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OVERCOOKED_ROOT = PROJECT_ROOT / "overcooked"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(OVERCOOKED_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "train" / "training"))

try:
    import torch
    from policies.trained_agent import TrainedAgent
    from src.rendering import Renderer
    from env import SelfPlayEnv
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

COMPETITION_LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "simple_o",
    "forced_coordination",
]

def get_layout_map() -> dict[str, Path | None]:
    """Find all built-in and custom layouts across the repo."""
    layout_map: dict[str, Path | None] = {name: None for name in COMPETITION_LAYOUTS}
    # Scan all .layout files inside overcooked/
    for f in sorted(OVERCOOKED_ROOT.glob("**/*.layout")):
        if f.stem not in layout_map:
            layout_map[f.stem] = f
        elif layout_map[f.stem] is None:
            layout_map[f.stem] = f
    return layout_map

def get_all_layouts() -> list[str]:
    return sorted(get_layout_map().keys())

def main():
    parser = argparse.ArgumentParser(description="Benchmark & Watch PPO / BC agents")
    parser.add_argument("--model", type=str, default="train/models/ppo_agent.pt", help="Path to .pt model")
    parser.add_argument("--watch", action="store_true", help="Pop up a live Pygame window to watch the agents play")
    parser.add_argument("--all-maps", action="store_true", help="Evaluate across all 56 maps instead of just the 5 competition maps")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes per layout")
    parser.add_argument("--horizon", type=int, default=400, help="Timesteps per episode")
    parser.add_argument("--fps", type=int, default=12, help="Framerate when watching in Pygame")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling (default false)")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    layout_map = get_layout_map()
    layouts = get_all_layouts() if args.all_maps else COMPETITION_LAYOUTS
    print(f"\n{'='*80}")
    print(f"BENCHMARKING MODEL: {args.model}")
    print(f"Layouts: {len(layouts)} maps | Episodes per map: {args.episodes} | Horizon: {args.horizon}")
    print(f"Mode: {'WATCHING LIVE (Pygame Window)' if args.watch else 'FAST EVALUATION (No Window)'}")
    print(f"{'='*80}\n")

    agent_cfg = {"model_path": str(model_path), "deterministic": args.deterministic}
    agent0 = TrainedAgent(agent_cfg)
    agent1 = TrainedAgent(agent_cfg)

    renderer = Renderer({"mode": "window" if args.watch else "none", "fps": args.fps, "window_caption": f"Benchmarking {args.model}"})

    results = {}
    total_soups_all_maps = []

    for l_idx, layout_name in enumerate(layouts, 1):
        print(f"[{l_idx}/{len(layouts)}] Evaluating: {layout_name:<25}", end="", flush=True)
        scores = []
        for ep in range(args.episodes):
            try:
                # Check if built-in
                built_in = False
                try:
                    OvercookedGridworld.from_layout_name(layout_name)
                    built_in = True
                except Exception:
                    pass

                if built_in:
                    env = SelfPlayEnv(layout_name=layout_name, horizon=args.horizon)
                else:
                    layout_file = layout_map.get(layout_name)
                    if not layout_file or not layout_file.exists():
                        raise FileNotFoundError(f"Layout file not found for {layout_name}")
                    env = SelfPlayEnv(layout_file=str(layout_file), horizon=args.horizon)
            except Exception as e:
                print(f" [SKIP: {e}]")
                break

            obs = env.reset()
            if args.watch:
                # Get inner overcooked env for rendering
                inner_env = env.env if hasattr(env, "env") else env
                renderer.reset()

            total_r = 0
            for step in range(args.horizon):
                a0 = agent0.act(obs[0])
                a1 = agent1.act(obs[1])
                acts = (a0, a1)

                obs, rews, dones, infos = env.step(acts)
                step_r = rews[0]
                total_r += step_r

                if args.watch:
                    inner_env = env.env if hasattr(env, "env") else env
                    renderer.maybe_render(inner_env, step, joint_action=acts, reward=step_r)
                    if renderer.closed_by_user:
                        print("\n[INFO] Pygame window closed by user. Stopping watch loop.")
                        return

                if any(dones):
                    break

            scores.append(total_r)

        if scores:
            mean_s = np.mean(scores)
            max_s = np.max(scores)
            soups_mean = mean_s / 20.0
            soups_max = max_s / 20.0
            results[layout_name] = {"mean_score": mean_s, "max_score": max_s, "mean_soups": soups_mean}
            total_soups_all_maps.append(soups_mean)
            print(f" -> Mean: {mean_s:>6.1f} pts ({soups_mean:>4.1f} soups) | Max: {max_s:>6.1f} pts ({soups_max:>4.1f} soups)")

    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY ({len(results)} MAPS BENCHMARKED)")
    print(f"{'='*80}")
    if total_soups_all_maps:
        avg_all = np.mean(total_soups_all_maps)
        print(f"Average Performance across all benchmarked maps: {avg_all:.2f} soups/round ({avg_all*20:.1f} pts)")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

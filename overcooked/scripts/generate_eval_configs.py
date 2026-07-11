"""Generate per-layout evaluation YAML configs for the trained agent.

Reads the recording quality TSV to determine which layouts are eligible (>= min_recordings_for_eval).
Generates configs/eval/<layout>.yaml for each, with correct layout_name/layout_file and old_dynamics.

Usage:
    cd overcooked
    python scripts/generate_eval_configs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
OVERCOOKED_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = OVERCOOKED_DIR.parent
LAYOUTS_DIR = OVERCOOKED_DIR / "layouts"
EVAL_DIR = OVERCOOKED_DIR / "configs" / "eval"

BUILTIN_LAYOUTS = [
    "asymmetric_advantages", "coordination_ring", "counter_circuit",
    "cramped_room", "forced_coordination", "large_room",
    "simple_o", "simple_tomato", "small_corridor", "soup_coordination",
    "tutorial_0", "tutorial_2", "tutorial_3",
]

MODEL_PATH_BC = "../train/models/bc_agent.pt"
MODEL_PATH_PPO = "../train/models/ppo_agent.pt"


def is_builtin(layout_name: str) -> bool:
    return layout_name in BUILTIN_LAYOUTS


def find_custom_layout_file(layout_name: str, layouts_dir: Path) -> str | None:
    """Find a custom .layout file matching this layout name across all directories."""
    # First check layouts_dir exact or prefix match
    if layouts_dir.exists():
        for f in layouts_dir.glob("*.layout"):
            if f.stem == layout_name or f.stem.startswith(layout_name + "_") or layout_name.startswith(f.stem + "_"):
                return os.path.relpath(f, OVERCOOKED_DIR).replace("\\", "/")
    # Search all of overcooked directory
    for f in OVERCOOKED_DIR.glob("**/*.layout"):
        if f.stem == layout_name or f.stem.startswith(layout_name + "_") or layout_name.startswith(f.stem + "_"):
            return os.path.relpath(f, OVERCOOKED_DIR).replace("\\", "/")
    return None


def generate_config(layout_name: str, layout_file: str | None, old_dynamics: bool,
                    model_path: str, output_path: Path):
    """Generate a single eval YAML config."""

    if layout_file:
        env_config = {
            "layout_name": None,
            "layout_file": layout_file,
            "horizon": 400,
            "old_dynamics": old_dynamics,
        }
    else:
        env_config = {
            "layout_name": layout_name,
            "layout_file": None,
            "horizon": 400,
            "old_dynamics": old_dynamics,
        }

    config = {
        "seed": 42,
        "mode": "evaluation",
        "environment": env_config,
        "policies": {
            "agent_0": {
                "type": "python_class",
                "path": "policies/trained_agent.py",
                "class_name": "TrainedAgent",
                "name": "trained_agent",
                "config": {
                    "model_path": model_path,
                    "deterministic": True,
                },
                "random_action_prob": 0.0,
                "max_action_time_ms": 100,
                "invalid_action": "stay",
                "timeout_action": "stay",
            },
            "agent_1": {
                "type": "builtin",
                "name": "greedy_full_task",
                "ingredient": "onion",
                "avoid_teammate": True,
                "random_action_prob": 0.0,
                "max_action_time_ms": 100,
                "invalid_action": "stay",
                "timeout_action": "stay",
            },
        },
        "execution": {
            "num_episodes": 3,
            "episode_seeds": [42, 43, 44],
            "swap_agent_positions": True,
        },
        "observation": {
            "type": "featurized",
            "include_agent_index": True,
        },
        "rendering": {
            "mode": "none",
            "fps": 0,
            "save_gif": False,
        },
        "logging": {
            "output_dir": f"outputs/eval_{layout_name}",
            "save_step_log": True,
            "save_episode_summary": True,
            "save_trajectory_pickle": False,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bc", "ppo"], default="bc")
    parser.add_argument("--min-recordings", type=int, default=3)
    args = parser.parse_args()

    model_path = MODEL_PATH_PPO if args.model == "ppo" else MODEL_PATH_BC

    tsv_path = PROJECT_ROOT / "train" / "data" / "recording_quality.tsv"
    if not tsv_path.exists():
        print(f"recording_quality.tsv not found at {tsv_path}")
        print("Run scripts/filter_recordings.py first.")
        return

    layout_counts = defaultdict(int)
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["keep"] == "True":
                layout_counts[row["layout"]] += 1

    dynamics_path = LAYOUTS_DIR / "dynamics_overrides.json"
    dynamics_overrides = {}
    if dynamics_path.exists():
        with open(dynamics_path, encoding="utf-8") as f:
            dynamics_overrides = json.load(f)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    eligible = []
    skipped = []

    for layout_name in sorted(layout_counts):
        n = layout_counts[layout_name]
        if n < args.min_recordings:
            skipped.append((layout_name, n))
            continue

        layout_file = None
        old_dynamics = True

        if is_builtin(layout_name):
            pass
        else:
            layout_file = find_custom_layout_file(layout_name, LAYOUTS_DIR)
            if layout_file:
                stem = Path(layout_file).stem
                if stem in dynamics_overrides:
                    old_dynamics = dynamics_overrides[stem].get("old_dynamics", True)
            else:
                old_dynamics = True

        output_path = EVAL_DIR / f"{layout_name}.yaml"
        generate_config(layout_name, layout_file, old_dynamics, model_path, output_path)
        eligible.append(layout_name)
        print(f"  Generated: {output_path.name} ({n} recordings)")

    print(f"\nGenerated {len(eligible)} eval configs (model: {args.model})")
    print(f"Skipped {len(skipped)} low-coverage layouts (< {args.min_recordings} recordings):")
    for name, n in skipped:
        print(f"  {name} ({n} recordings)")

    print(f"\nEval configs in: {EVAL_DIR}")
    print(f"\nTo evaluate:")
    print(f"  cd overcooked")
    for name in eligible[:5]:
        print(f"  python -m src.evaluate --config configs/eval/{name}.yaml")
    if len(eligible) > 5:
        print(f"  ... and {len(eligible)-5} more")


if __name__ == "__main__":
    main()
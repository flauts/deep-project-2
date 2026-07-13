#!/usr/bin/env python3
"""
watch_agent.py - Watch the trained TA agent play live in a pygame window.

Works exactly like collect_demonstrations.py -- just reads a YAML config
and calls run_from_config. The trained_ppo policy type uses featurize_state_mdp
directly (same obs pipeline as training) so you see real behaviour.

Usage (from project root):
    python watch_agent.py
    python watch_agent.py --config overcooked/configs/watch_ta_agent.yaml

Controls:
    Close the pygame window or press Q / ESC to stop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── make the overcooked package importable from the project root ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "overcooked"))
sys.path.insert(0, str(PROJECT_ROOT / "train"))
sys.path.insert(0, str(PROJECT_ROOT / "train" / "training"))

from src.config import load_yaml
from src.runner import run_from_config


def main():
    parser = argparse.ArgumentParser(description="Watch the trained TA agent play.")
    parser.add_argument(
        "--config",
        default="overcooked/configs/watch_ta_agent.yaml",
        help="Path to config YAML (default: overcooked/configs/watch_ta_agent.yaml)",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_yaml(str(config_path))

    result = run_from_config(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

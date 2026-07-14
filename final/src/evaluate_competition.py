"""CLI for the official Overcooked-AI competition evaluator.

Usage:
    python -m src.evaluate_competition --config configs/competition.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.competition_evaluation import (
    CompetitionConfigError,
    evaluate_competition,
    format_final_score_report,
    select_competition_scenario,
)
from src.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/competition.yaml")
    scenario_group = parser.add_mutually_exclusive_group()
    scenario_group.add_argument(
        "--scenario",
        help="Run only one enabled scenario, selected by its ID or name.",
    )
    scenario_group.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run every enabled scenario instead of the default first scenario.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show each competition rollout in a pygame window.",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=10,
        help="Frames per second when --render is enabled (default: 10).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)
    selected_config = not args.all_scenarios
    if selected_config:
        try:
            config = select_competition_scenario(config, args.scenario)
        except CompetitionConfigError as exc:
            parser.error(str(exc))
    if args.render:
        rendering = dict(config.get("rendering", {}) or {})
        rendering.update({"mode": "window", "fps": args.render_fps})
        config["rendering"] = rendering
    result = evaluate_competition(config, config_path=None if selected_config else config_path)
    print(format_final_score_report(result["score_reports"]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

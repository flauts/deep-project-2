"""Official competition evaluation protocol for Overcooked-AI submissions."""

from __future__ import annotations

import copy
import csv
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml
from overcooked_ai_py.agents.agent import AgentPair

from src.environment import build_env
from src.observations import ObservationBuilder
from src.policy_loader import build_policy
from src.rendering import Renderer
from src.runner import set_global_seed
from src.scoring import aggregate_attempts, compute_attempt_score, summarize_group_scores


PER_ATTEMPT_FIELDS = [
    "group_name",
    "scenario_id",
    "scenario_name",
    "layout",
    "teammate",
    "seed",
    "role",
    "horizon",
    "soups",
    "first_soup_timestep",
    "last_soup_timestep",
    "student_timeouts",
    "score",
]

PER_SCENARIO_FIELDS = [
    "group_name",
    "scenario_id",
    "scenario_name",
    "avg_score",
    "avg_soups",
    "avg_student_timeouts",
    "num_rollouts",
    "rank",
]

ALLOWED_BUILTIN_TEAMMATES = {"random_motion", "greedy_full_task"}


class CompetitionConfigError(ValueError):
    """Raised when the official competition config is malformed."""


def select_competition_scenario(
    config: dict[str, Any], selector: str | int | None = None
) -> dict[str, Any]:
    """Return a config restricted to one enabled scenario ID or name.

    When no selector is provided, the first enabled scenario in YAML order is
    selected. This makes local student runs quick by default.
    """
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list):
        raise CompetitionConfigError("scenarios must be a list before selecting one")

    if selector is None:
        matches = [
            scenario
            for scenario in scenarios
            if isinstance(scenario, dict) and bool(scenario.get("enabled", True))
        ]
        if not matches:
            raise CompetitionConfigError("No enabled scenario found in the competition config")
        selected = copy.deepcopy(config)
        selected["scenarios"] = [copy.deepcopy(matches[0])]
        return selected

    selector_text = str(selector).strip()
    matches = [
        scenario
        for scenario in scenarios
        if isinstance(scenario, dict)
        and (
            str(scenario.get("id")) == selector_text
            or str(scenario.get("name", "")).lower() == selector_text.lower()
        )
    ]
    if not matches:
        raise CompetitionConfigError(f"Scenario not found: {selector!r}")
    if len(matches) > 1:
        raise CompetitionConfigError(f"Scenario selector is ambiguous: {selector!r}")
    if not bool(matches[0].get("enabled", True)):
        raise CompetitionConfigError(
            f"Scenario {matches[0].get('id')} is disabled in the competition config"
        )

    selected = copy.deepcopy(config)
    selected["scenarios"] = [copy.deepcopy(matches[0])]
    return selected


def _resolve_path(path_like: str | Path, base_dir: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else base_dir / path


def _require_file(
    mapping: dict[str, Any],
    key: str,
    *,
    label: str,
    base_dir: Path,
    required: bool = False,
) -> None:
    value = mapping.get(key)
    if not value:
        if required:
            raise CompetitionConfigError(f"{label} requires '{key}'")
        return
    path = _resolve_path(value, base_dir)
    if not path.is_file():
        raise CompetitionConfigError(f"{label} {key} not found: {path}")
    mapping[key] = str(path.resolve())


def _validate_probability(mapping: dict[str, Any], key: str, label: str) -> None:
    value = float(mapping.get(key, 0.0) or 0.0)
    if not 0.0 <= value <= 1.0:
        raise CompetitionConfigError(f"{label}.{key} must be in [0, 1]")
    mapping[key] = value


def validate_competition_config(
    config: dict[str, Any], *, base_dir: str | Path | None = None
) -> dict[str, Any]:
    """Validate and normalize an official config without mutating the input."""
    if not isinstance(config, dict):
        raise CompetitionConfigError("Competition config must be a YAML mapping")
    normalized = copy.deepcopy(config)
    root = Path.cwd() if base_dir is None else Path(base_dir)

    rendering = normalized.get("rendering", {}) or {}
    if not isinstance(rendering, dict):
        raise CompetitionConfigError("rendering must be a mapping when provided")
    normalized["rendering"] = rendering

    soup_reward = float(normalized.get("soup_reward", 20))
    if soup_reward <= 0:
        raise CompetitionConfigError("soup_reward must be positive")
    normalized["soup_reward"] = soup_reward
    internal_only = bool(normalized.get("internal_only", False))

    submissions = normalized.get("submissions")
    if not isinstance(submissions, list) or not submissions:
        raise CompetitionConfigError("submissions must be a non-empty list")
    seen_names: set[str] = set()
    for index, submission in enumerate(submissions):
        label = f"submissions[{index}]"
        if not isinstance(submission, dict):
            raise CompetitionConfigError(f"{label} must be a mapping")
        if not submission.get("name"):
            raise CompetitionConfigError(f"{label} requires 'name'")
        name = str(submission["name"])
        if name in seen_names:
            raise CompetitionConfigError(f"Duplicate submission name: {name}")
        seen_names.add(name)
        submission_type = str(submission.get("type", "python_class")).lower()
        submission["type"] = submission_type
        if submission_type == "python_class":
            _require_file(submission, "path", label=label, base_dir=root, required=True)
            _require_file(submission, "config_path", label=label, base_dir=root)
            _require_file(submission, "model_path", label=label, base_dir=root)
            inline = submission.get("config", {}) or {}
            if not isinstance(inline, dict):
                raise CompetitionConfigError(f"{label}.config must be a mapping")
        elif submission_type == "builtin":
            if not internal_only:
                raise CompetitionConfigError(
                    "Builtin submissions are allowed only when internal_only: true"
                )
            policy_name = str(submission.get("policy_name", "")).lower()
            if policy_name != "greedy_full_task":
                raise CompetitionConfigError(
                    f"{label}.policy_name must be 'greedy_full_task' for the internal baseline"
                )
            submission["policy_name"] = policy_name
        else:
            raise CompetitionConfigError(f"{label}.type must be python_class or builtin")

    scenarios = normalized.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise CompetitionConfigError("scenarios must be a non-empty list")
    enabled_count = 0
    seen_ids: set[Any] = set()
    for index, scenario in enumerate(scenarios):
        label = f"scenarios[{index}]"
        if not isinstance(scenario, dict):
            raise CompetitionConfigError(f"{label} must be a mapping")
        if not bool(scenario.get("enabled", True)):
            continue
        enabled_count += 1
        if "id" not in scenario or not scenario.get("name"):
            raise CompetitionConfigError(f"{label} requires 'id' and 'name'")
        if scenario["id"] in seen_ids:
            raise CompetitionConfigError(f"Duplicate enabled scenario id: {scenario['id']}")
        seen_ids.add(scenario["id"])

        layout_name = scenario.get("layout_name")
        layout_file = scenario.get("layout_file")
        if not layout_name and not layout_file:
            raise CompetitionConfigError(f"{label} requires layout_name or layout_file")
        scenario["_layout_display"] = str(layout_file or layout_name)
        if layout_file:
            _require_file(scenario, "layout_file", label=label, base_dir=root, required=True)

        seeds = scenario.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise CompetitionConfigError(f"{label}.seeds must be a non-empty list")
        try:
            scenario["seeds"] = [int(seed) for seed in seeds]
        except (TypeError, ValueError) as exc:
            raise CompetitionConfigError(f"{label}.seeds must contain integers") from exc

        horizon = int(scenario.get("horizon", 400))
        if horizon <= 0:
            raise CompetitionConfigError(f"{label}.horizon must be positive")
        scenario["horizon"] = horizon
        max_action_time_ms = int(scenario.get("max_action_time_ms", 100))
        if max_action_time_ms < 0:
            raise CompetitionConfigError(f"{label}.max_action_time_ms cannot be negative")
        scenario["max_action_time_ms"] = max_action_time_ms

        teammate = scenario.get("teammate")
        if not isinstance(teammate, dict):
            raise CompetitionConfigError(f"{label} requires a teammate mapping")
        teammate_type = str(teammate.get("type", "builtin")).lower()
        teammate["type"] = teammate_type
        if teammate_type == "builtin":
            teammate_name = str(teammate.get("name", "")).lower()
            if teammate_name == "human_keyboard":
                raise CompetitionConfigError("human_keyboard is not allowed as a competition teammate")
            if teammate_name not in ALLOWED_BUILTIN_TEAMMATES:
                raise CompetitionConfigError(
                    f"{label}.teammate builtin must be one of {sorted(ALLOWED_BUILTIN_TEAMMATES)}"
                )
            teammate["_display"] = teammate_name
        elif teammate_type == "python_class":
            teammate["_display"] = str(teammate.get("name") or teammate.get("path") or "python_class")
            _require_file(teammate, "path", label=f"{label}.teammate", base_dir=root, required=True)
            _require_file(teammate, "config_path", label=f"{label}.teammate", base_dir=root)
            _require_file(teammate, "model_path", label=f"{label}.teammate", base_dir=root)
            inline = teammate.get("config", {}) or {}
            if not isinstance(inline, dict):
                raise CompetitionConfigError(f"{label}.teammate.config must be a mapping")
        else:
            raise CompetitionConfigError(f"{label}.teammate.type must be builtin or python_class")
        _validate_probability(teammate, "random_action_prob", f"{label}.teammate")
        _validate_probability(teammate, "sticky_action_prob", f"{label}.teammate")

    if enabled_count == 0:
        raise CompetitionConfigError("At least one scenario must be enabled")
    return normalized


def _submission_policy_config(submission: dict[str, Any], max_action_time_ms: int) -> dict[str, Any]:
    if submission["type"] == "builtin":
        policy = {
            "type": "builtin",
            "name": submission["policy_name"],
            "ingredient": submission.get("ingredient", "onion"),
            "avoid_teammate": submission.get("avoid_teammate", True),
        }
        policy.update(
            {
                "random_action_prob": 0.0,
                "sticky_action_prob": 0.0,
                "max_action_time_ms": max_action_time_ms,
                "invalid_action": "stay",
                "timeout_action": "stay",
            }
        )
        return policy
    return {
        "type": "python_class",
        "name": submission["name"],
        "path": submission["path"],
        "class_name": submission.get("class_name", "StudentAgent"),
        "config_path": submission.get("config_path"),
        "model_path": submission.get("model_path"),
        "config": submission.get("config", {}) or {},
        "random_action_prob": 0.0,
        "sticky_action_prob": 0.0,
        "max_action_time_ms": max_action_time_ms,
        "invalid_action": "stay",
        "timeout_action": "stay",
    }


def _teammate_policy_config(teammate: dict[str, Any], max_action_time_ms: int) -> dict[str, Any]:
    policy = {key: value for key, value in teammate.items() if not key.startswith("_")}
    policy["max_action_time_ms"] = max_action_time_ms
    policy["invalid_action"] = "stay"
    policy["timeout_action"] = "stay"
    return policy


def _wrapper_timeout_count(agent) -> int:
    """Find SafeActionWrapper's counter through any outer wrappers."""
    current = agent
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "timeout_count"):
            return int(current.timeout_count)
        current = getattr(current, "base_agent", None)
    return 0


def _count_truthy_events(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_count_truthy_events(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_count_truthy_events(item) for item in value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0


def soups_delivered_in_step(reward: float, info: dict[str, Any] | None, soup_reward: float) -> int:
    """Detect deliveries from explicit event info, falling back to sparse reward."""
    if isinstance(info, dict):
        event_infos = info.get("event_infos")
        if isinstance(event_infos, dict):
            for key in ("soup_delivery", "soup_deliveries"):
                if key in event_infos:
                    return _count_truthy_events(event_infos[key])

    positive_reward = max(0.0, float(reward))
    if positive_reward == 0:
        return 0
    return max(0, int(round(positive_reward / float(soup_reward))))


def _run_attempt(
    *,
    env,
    obs_builder: ObservationBuilder,
    submission: dict[str, Any],
    scenario: dict[str, Any],
    seed: int,
    student_agent_index: int,
    soup_reward: float,
    renderer: Renderer | None = None,
) -> dict[str, Any]:
    set_global_seed(seed)
    student_config = _submission_policy_config(submission, scenario["max_action_time_ms"])
    teammate_config = _teammate_policy_config(scenario["teammate"], scenario["max_action_time_ms"])

    configs = [student_config, teammate_config]
    if student_agent_index == 1:
        configs.reverse()
    agents = [
        build_policy(configs[0], env, obs_builder, seed=seed + 1000),
        build_policy(configs[1], env, obs_builder, seed=seed + 2000),
    ]
    student_agent = agents[student_agent_index]
    agent_pair = AgentPair(agents[0], agents[1])

    env.reset(regen_mdp=False)
    agent_pair.reset()
    agent_pair.set_mdp(env.mdp)
    if renderer is not None and not renderer.closed_by_user:
        renderer.reset()
        renderer.maybe_render(env, timestep=0)

    done = False
    timestep = 0
    soups = 0
    first_soup_timestep: int | None = None
    last_soup_timestep: int | None = None
    while not done:
        joint_action_and_infos = agent_pair.joint_action(env.state)
        joint_action, joint_infos = zip(*joint_action_and_infos)
        _, reward, done, info = env.step(joint_action, joint_infos)
        timestep += 1
        delivered = soups_delivered_in_step(reward, info, soup_reward)
        if delivered > 0:
            soups += delivered
            if first_soup_timestep is None:
                first_soup_timestep = timestep
            last_soup_timestep = timestep
        if renderer is not None and not renderer.closed_by_user:
            renderer.maybe_render(env, timestep=timestep, reward=float(reward))

    student_timeouts = _wrapper_timeout_count(student_agent)
    score = compute_attempt_score(
        soups=soups,
        horizon=scenario["horizon"],
        first_soup_timestep=first_soup_timestep,
        last_soup_timestep=last_soup_timestep,
        student_timeouts=student_timeouts,
    )
    return {
        "group_name": submission["name"],
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "layout": scenario["_layout_display"],
        "teammate": scenario["teammate"]["_display"],
        "seed": seed,
        "role": f"student_agent_{student_agent_index}",
        "horizon": scenario["horizon"],
        "soups": soups,
        "first_soup_timestep": first_soup_timestep,
        "last_soup_timestep": last_soup_timestep,
        "student_timeouts": student_timeouts,
        "score": score,
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_final_score_report(score_reports: Iterable[dict[str, Any]]) -> str:
    """Format the per-student score summary printed by the CLI."""
    lines = ["\nFinal score report (equal weight per scenario):"]
    for report in score_reports:
        lines.append(
            f"  {report['group_name']}: mean_score = {report['mean_score']:.2f} | "
            f"mean_soups = {report['mean_soups']:.2f}"
        )
    return "\n".join(lines)


def _configure_local_planner_cache(cache_dir: Path) -> None:
    """Keep Overcooked-AI's generated planners out of site-packages.

    Some official layouts do not ship with a precomputed motion planner. The
    upstream package normally writes it into its installation directory, which
    is often read-only in a managed Conda environment. This process-local
    redirection keeps those generated files under the evaluation output.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    from overcooked_ai_py.data import planners as data_planners
    from overcooked_ai_py.planning import planners as planning_planners

    cache = str(cache_dir)
    data_planners.PLANNERS_DIR = cache
    planning_planners.PLANNERS_DIR = cache


def evaluate_competition(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the configured official competition and write its CSV reports."""
    original_config = copy.deepcopy(config)
    normalized = validate_competition_config(config, base_dir=base_dir)
    root = Path.cwd() if base_dir is None else Path(base_dir)
    output_dir = _resolve_path(normalized.get("output_dir", "results/competition_eval"), root)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_local_planner_cache(output_dir / "_planner_cache")
    renderer = Renderer(normalized["rendering"])

    used_config_path = output_dir / "competition_config_used.yaml"
    if config_path is not None:
        shutil.copyfile(Path(config_path), used_config_path)
    else:
        with used_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(original_config, handle, sort_keys=False, allow_unicode=True)

    attempts: list[dict[str, Any]] = []
    enabled_scenarios = [scenario for scenario in normalized["scenarios"] if scenario.get("enabled", True)]
    try:
        for submission in normalized["submissions"]:
            for scenario in enabled_scenarios:
                environment_config = {
                    "layout_name": scenario.get("layout_name"),
                    "layout_file": scenario.get("layout_file"),
                    "horizon": scenario["horizon"],
                    "old_dynamics": bool(scenario.get("old_dynamics", True)),
                    "mdp_overrides": scenario.get("mdp_overrides", {}) or {},
                }
                env = build_env(environment_config)
                observation_config = scenario.get(
                    "observation", normalized.get("observation", {"type": "featurized", "include_agent_index": True})
                )
                obs_builder = ObservationBuilder(env, observation_config)
                roles = [0, 1] if bool(scenario.get("swap_roles", False)) else [0]
                for seed in scenario["seeds"]:
                    for student_agent_index in roles:
                        row = _run_attempt(
                            env=env,
                            obs_builder=obs_builder,
                            submission=submission,
                            scenario=scenario,
                            seed=seed,
                            student_agent_index=student_agent_index,
                            soup_reward=float(scenario.get("soup_reward", normalized["soup_reward"])),
                            renderer=renderer,
                        )
                        attempts.append(row)
                        print(
                            f"{row['group_name']} | {row['scenario_name']} | seed={seed} | "
                            f"{row['role']} | soups={row['soups']} | score={row['score']}"
                        )
    finally:
        renderer.close()

    scenario_summaries = aggregate_attempts(attempts)
    score_reports = summarize_group_scores(scenario_summaries)
    per_attempt_path = output_dir / "per_attempt.csv"
    per_scenario_path = output_dir / "per_scenario.csv"
    _write_csv(per_attempt_path, attempts, PER_ATTEMPT_FIELDS)
    _write_csv(per_scenario_path, scenario_summaries, PER_SCENARIO_FIELDS)

    return {
        "num_submissions": len(normalized["submissions"]),
        "num_enabled_scenarios": len(enabled_scenarios),
        "num_rollouts": len(attempts),
        "output_dir": str(output_dir),
        "per_attempt_csv": str(per_attempt_path),
        "per_scenario_csv": str(per_scenario_path),
        "competition_config_used": str(used_config_path),
        "score_reports": score_reports,
    }

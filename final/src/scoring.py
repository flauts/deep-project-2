"""Pure scoring and aggregation utilities for the official competition."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable


def timeout_penalty(timeout_count: int) -> int:
    """Return the capped official timeout penalty."""
    timeout_count = int(timeout_count)
    if timeout_count < 0:
        raise ValueError("timeout_count cannot be negative")
    return min(100 * timeout_count, 5000)


def compute_attempt_score(
    *,
    soups: int,
    horizon: int,
    first_soup_timestep: int | None,
    last_soup_timestep: int | None,
    student_timeouts: int,
) -> int:
    """Compute the exact official score for one rollout."""
    soups = int(soups)
    horizon = int(horizon)
    if soups < 0:
        raise ValueError("soups cannot be negative")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if soups == 0:
        return 0
    if first_soup_timestep is None or last_soup_timestep is None:
        raise ValueError("soup timesteps are required when soups > 0")

    first = int(first_soup_timestep)
    last = int(last_soup_timestep)
    if not 1 <= first <= last <= horizon:
        raise ValueError("soup timesteps must satisfy 1 <= first <= last <= horizon")

    return (
        10000 * soups
        + 10 * (horizon - last)
        + (horizon - first)
        - timeout_penalty(student_timeouts)
    )


def aggregate_attempts(attempts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average attempts per group/scenario and rank groups within scenarios."""
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    scenario_order: list[tuple[Any, Any]] = []
    seen_scenarios: set[tuple[Any, Any]] = set()

    for attempt in attempts:
        scenario_key = (attempt["scenario_id"], attempt["scenario_name"])
        if scenario_key not in seen_scenarios:
            seen_scenarios.add(scenario_key)
            scenario_order.append(scenario_key)
        grouped[(attempt["group_name"], *scenario_key)].append(attempt)

    summaries: list[dict[str, Any]] = []
    for (group_name, scenario_id, scenario_name), rows in grouped.items():
        summaries.append(
            {
                "group_name": group_name,
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "avg_score": float(fmean(float(row["score"]) for row in rows)),
                "avg_soups": float(fmean(float(row["soups"]) for row in rows)),
                "avg_student_timeouts": float(
                    fmean(float(row["student_timeouts"]) for row in rows)
                ),
                "num_rollouts": len(rows),
                "rank": 0,
            }
        )

    ordered: list[dict[str, Any]] = []
    for scenario_id, scenario_name in scenario_order:
        scenario_rows = [
            row
            for row in summaries
            if row["scenario_id"] == scenario_id and row["scenario_name"] == scenario_name
        ]
        scenario_rows.sort(key=lambda row: (-row["avg_score"], str(row["group_name"])))
        previous_score = None
        previous_rank = 0
        for index, row in enumerate(scenario_rows, start=1):
            if previous_score is None or row["avg_score"] != previous_score:
                previous_rank = index
                previous_score = row["avg_score"]
            row["rank"] = previous_rank
        ordered.extend(scenario_rows)
    return ordered


def summarize_group_scores(scenario_summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a human-facing equal-weight mean score across scenarios.

    The official score is defined per rollout and aggregated per scenario. This
    helper does not replace it: it exposes the mean of the scenario averages
    so an individual submission has a concise final report.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scenario_summaries:
        grouped[str(row["group_name"])].append(row)

    reports: list[dict[str, Any]] = []
    for group_name, rows in grouped.items():
        total_rollouts = sum(int(row["num_rollouts"]) for row in rows)
        mean_soups = sum(
            float(row["avg_soups"]) * int(row["num_rollouts"]) for row in rows
        ) / total_rollouts
        reports.append(
            {
                "group_name": group_name,
                "mean_score": float(fmean(float(row["avg_score"]) for row in rows)),
                "mean_soups": float(mean_soups),
                "num_scenarios": len(rows),
                "num_rollouts": total_rollouts,
            }
        )
    return reports

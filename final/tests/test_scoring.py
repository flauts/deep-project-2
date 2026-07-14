from src.scoring import (
    aggregate_attempts,
    compute_attempt_score,
    summarize_group_scores,
    timeout_penalty,
)


def test_score_is_zero_without_soups_even_with_timeouts():
    assert compute_attempt_score(
        soups=0,
        horizon=400,
        first_soup_timestep=None,
        last_soup_timestep=None,
        student_timeouts=99,
    ) == 0


def test_score_with_soups_uses_exact_formula():
    # 20_000 + 10 * (400 - 300) + (400 - 100) - 3 * 100
    assert compute_attempt_score(
        soups=2,
        horizon=400,
        first_soup_timestep=100,
        last_soup_timestep=300,
        student_timeouts=3,
    ) == 21000


def test_timeout_penalty_is_capped_at_5000():
    assert timeout_penalty(12) == 1200
    assert timeout_penalty(50) == 5000
    assert timeout_penalty(80) == 5000


def test_aggregation_averages_and_ranks_within_scenario():
    attempts = [
        _attempt("grupo_a", 1, 100, 1, 0),
        _attempt("grupo_a", 1, 300, 3, 2),
        _attempt("grupo_b", 1, 250, 2, 1),
        _attempt("grupo_b", 1, 350, 4, 1),
        _attempt("grupo_a", 2, 500, 5, 0),
        _attempt("grupo_b", 2, 400, 4, 0),
    ]

    rows = aggregate_attempts(attempts)
    by_key = {(row["group_name"], row["scenario_id"]): row for row in rows}

    assert by_key[("grupo_a", 1)]["avg_score"] == 200.0
    assert by_key[("grupo_a", 1)]["avg_soups"] == 2.0
    assert by_key[("grupo_a", 1)]["avg_student_timeouts"] == 1.0
    assert by_key[("grupo_a", 1)]["num_rollouts"] == 2
    assert by_key[("grupo_b", 1)]["rank"] == 1
    assert by_key[("grupo_a", 1)]["rank"] == 2
    assert by_key[("grupo_a", 2)]["rank"] == 1
    assert by_key[("grupo_b", 2)]["rank"] == 2


def test_group_summary_uses_equal_weight_per_scenario():
    report = summarize_group_scores(
        [
            {
                "group_name": "grupo_a",
                "scenario_id": 1,
                "scenario_name": "uno",
                "avg_score": 10,
                "avg_soups": 1.5,
                "num_rollouts": 2,
            },
            {
                "group_name": "grupo_a",
                "scenario_id": 2,
                "scenario_name": "dos",
                "avg_score": 30,
                "avg_soups": 2.0,
                "num_rollouts": 3,
            },
        ]
    )[0]

    assert report["mean_score"] == 20.0
    assert report["mean_soups"] == 1.8
    assert report["num_scenarios"] == 2
    assert report["num_rollouts"] == 5


def _attempt(group, scenario_id, score, soups, timeouts):
    return {
        "group_name": group,
        "scenario_id": scenario_id,
        "scenario_name": f"escenario_{scenario_id}",
        "score": score,
        "soups": soups,
        "student_timeouts": timeouts,
    }

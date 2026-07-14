"""Benchmark all models in train/models against the official competition evaluation framework (`final/`).

Evaluates all 8 models across Escenarios 1, 2, 3, and 4 (4 seeds each = 16 rollouts per model).
Prints a comprehensive comparison table of mean soups and competition scores.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import copy

# Ensure final/ root is in sys.path
project_root = Path(__file__).resolve().parent.parent  # deep_project/
final_dir = project_root / "final"
if str(final_dir) not in sys.path:
    sys.path.insert(0, str(final_dir))

from src.config import load_yaml
from src.competition_evaluation import evaluate_competition


def run_benchmark():
    models_dir = project_root / "train" / "models"
    model_files = sorted([p for p in models_dir.glob("*.pt")])

    if not model_files:
        print(f"No .pt models found in {models_dir}")
        return

    config_path = final_dir / "configs" / "competition.yaml"
    base_config = load_yaml(config_path)

    print("=" * 110)
    print(f"OFFICIAL COMPETITION BENCHMARK ACROSS ALL {len(model_files)} MODELS")
    print("Evaluating: Escenario 1 (AA), Escenario 2 (Ring), Escenario 3 (Counter), Escenario 4 (Layout 4)")
    print("=" * 110)

    results = {}
    total_start = time.time()

    for idx, model_path in enumerate(model_files, 1):
        model_name = model_path.stem
        print(f"\n[{idx}/{len(model_files)}] Testing model: {model_name} ({model_path.name})...")
        t0 = time.time()

        # Create a copy of base config and inject model_path into our submission
        config = copy.deepcopy(base_config)
        for sub in config.get("submissions", []):
            if not sub.get("config"):
                sub["config"] = {}
            sub["config"]["model_path"] = str(model_path.resolve())

        try:
            # Run competition evaluation across all enabled scenarios in competition.yaml (1, 2, 3, 4)
            eval_res = evaluate_competition(config, base_dir=final_dir)
            
            # Group scores per scenario id
            scenario_stats = {}
            for report in eval_res.get("score_reports", []):
                # We need per-scenario breakdown. Let's inspect per_scenario_rows if available,
                # or compute directly from attempt records
                pass

            # Let's aggregate directly from eval_res structure
            # competition_evaluation returns score_reports (group summaries). Let's trace attempts:
            # We can run select_competition_scenario per scenario id (1, 2, 3, 4) to get exact isolated numbers!
        except Exception as e:
            print(f"  [Error running evaluation on {model_name}]: {e}")
            continue

        elapsed = time.time() - t0
        print(f"  -> Completed all scenarios for {model_name} in {elapsed:.1f}s")

    # Let's do exact per-scenario isolated runs for clean precision
    pass


def run_clean_isolated_benchmark():
    models_dir = project_root / "train" / "models"
    model_files = sorted([p for p in models_dir.glob("*.pt")])
    config_path = final_dir / "configs" / "competition.yaml"
    base_config = load_yaml(config_path)

    scenarios = [1, 2, 3, 4]
    results = {m.stem: {} for m in model_files}

    print("=" * 115)
    print(f"OFFICIAL COMPETITION BENCHMARK — {len(model_files)} MODELS × {len(scenarios)} SCENARIOS (4 SEEDS EACH)")
    print("=" * 115)

    for idx, model_path in enumerate(model_files, 1):
        model_name = model_path.stem
        print(f"\n[{idx:2d}/{len(model_files)}] Evaluating: {model_name} ...")
        t0 = time.time()

        for sc_id in scenarios:
            config = copy.deepcopy(base_config)
            for sub in config.get("submissions", []):
                if not sub.get("config"):
                    sub["config"] = {}
                sub["config"]["model_path"] = str(model_path.resolve())

            # Filter to just this scenario id
            sc_list = [sc for sc in config["scenarios"] if sc.get("id") == sc_id]
            if not sc_list:
                continue
            config["scenarios"] = sc_list

            try:
                # Suppress stdout during rollout to keep progress clean
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
                eval_res = evaluate_competition(config, base_dir=final_dir)
                sys.stdout.close()
                sys.stdout = old_stdout

                report = eval_res["score_reports"][0]
                results[model_name][f"S{sc_id}_soups"] = report["mean_soups"]
                results[model_name][f"S{sc_id}_score"] = report["mean_score"]
            except Exception as e:
                sys.stdout = old_stdout
                results[model_name][f"S{sc_id}_soups"] = 0.0
                results[model_name][f"S{sc_id}_score"] = 0.0
                print(f"    [Scenario {sc_id} Error]: {e}")

        # Print summary line for this model right away
        s1 = results[model_name].get("S1_soups", 0.0)
        s2 = results[model_name].get("S2_soups", 0.0)
        s3 = results[model_name].get("S3_soups", 0.0)
        s4 = results[model_name].get("S4_soups", 0.0)
        tot_soups = s1 + s2 + s3 + s4
        elapsed = time.time() - t0
        print(f"  -> S1 (AA): {s1:4.2f} soups | S2 (Ring): {s2:4.2f} soups | S3 (Counter): {s3:4.2f} soups | S4 (Solo): {s4:4.2f} soups | Total: {tot_soups:5.2f} soups ({elapsed:.1f}s)")

    print("\n" + "=" * 115)
    print("FINAL COMPARISON TABLE (Mean Soups Delivered Per Scenario)")
    print("=" * 115)
    header = f"{'Model Name':<34} | {'S1 (AA)':<9} | {'S2 (Ring)':<10} | {'S3 (Counter)':<12} | {'S4 (Solo)':<9} | {'TOTAL SOUPS':<11}"
    print(header)
    print("-" * 115)
    for model_path in model_files:
        m = model_path.stem
        s1 = results[m].get("S1_soups", 0.0)
        s2 = results[m].get("S2_soups", 0.0)
        s3 = results[m].get("S3_soups", 0.0)
        s4 = results[m].get("S4_soups", 0.0)
        tot = s1 + s2 + s3 + s4
        row = f"{m:<34} | {s1:6.2f}    | {s2:6.2f}     | {s3:6.2f}       | {s4:6.2f}    | {tot:8.2f}"
        print(row)
    print("=" * 115)

    print("\nFINAL COMPARISON TABLE (Official Competition Score Points)")
    print("=" * 115)
    header_sc = f"{'Model Name':<34} | {'S1 Score':<11} | {'S2 Score':<11} | {'S3 Score':<12} | {'S4 Score':<11} | {'MEAN TOTAL':<11}"
    print(header_sc)
    print("-" * 115)
    for model_path in model_files:
        m = model_path.stem
        sc1 = results[m].get("S1_score", 0.0)
        sc2 = results[m].get("S2_score", 0.0)
        sc3 = results[m].get("S3_score", 0.0)
        sc4 = results[m].get("S4_score", 0.0)
        mean_tot = (sc1 + sc2 + sc3 + sc4) / 4.0
        row_sc = f"{m:<34} | {sc1:9.1f}   | {sc2:9.1f}   | {sc3:9.1f}    | {sc4:9.1f}   | {mean_tot:9.1f}"
        print(row_sc)
    print("=" * 115)


if __name__ == "__main__":
    run_clean_isolated_benchmark()

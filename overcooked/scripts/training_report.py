"""Generate a consolidated training report: filter stats, BC curves, PPO curves, eval scores.

Usage:
    cd overcooked
    python scripts/training_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
OVERCOOKED_DIR = SCRIPT_DIR.parent
TRAIN_DIR = OVERCOOKED_DIR.parent / "train"
DATA_DIR = TRAIN_DIR / "data"
MODELS_DIR = TRAIN_DIR / "models"


def report_filter():
    tsv_path = DATA_DIR / "recording_quality.tsv"
    if not tsv_path.exists():
        print("[Filter] recording_quality.tsv not found. Run scripts/filter_recordings.py first.")
        return

    print("=" * 80)
    print("RECORDING FILTER REPORT")
    print("=" * 80)

    by_layout = defaultdict(lambda: {"kept": 0, "dropped": 0, "tiers": defaultdict(int)})
    total_kept = 0
    total_dropped = 0
    drop_reasons = defaultdict(int)

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            layout = row["layout"]
            if row["keep"] == "True":
                by_layout[layout]["kept"] += 1
                total_kept += 1
                by_layout[layout]["tiers"][row["tier"]] += 1
            else:
                by_layout[layout]["dropped"] += 1
                total_dropped += 1
                drop_reasons[row["reason"]] += 1

    print(f"Total kept: {total_kept} | Total dropped: {total_dropped}")
    print(f"\nDrop reasons:")
    for reason, count in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        print(f"  {count:>4}  {reason}")

    print(f"\nPer-layout:")
    print(f"{'LAYOUT':<35} {'KEPT':>4} {'DROP':>4} {'GOLD':>4} {'SILV':>4} {'BRNZ':>4}")
    print("-" * 65)
    for layout in sorted(by_layout, key=lambda k: -by_layout[k]["kept"]):
        s = by_layout[layout]
        print(f"{layout:<35} {s['kept']:>4} {s['dropped']:>4} "
              f"{s['tiers']['gold']:>4} {s['tiers']['silver']:>4} {s['tiers']['bronze']:>4}")
    print()


def report_bc():
    hist_path = MODELS_DIR / "bc_agent.history.json"
    if not hist_path.exists():
        print("[BC] bc_agent.history.json not found. Run train/train_bc.py first.")
        return

    with open(hist_path) as f:
        hist = json.load(f)

    print("=" * 80)
    print("BEHAVIORAL CLONING REPORT")
    print("=" * 80)

    n = len(hist["train_loss"])
    if n == 0:
        print("No epochs trained.")
        return

    print(f"Epochs: {n}")
    print(f"Final train loss: {hist['train_loss'][-1]:.4f} | Final train acc: {hist['train_acc'][-1]:.4f}")
    print(f"Final val loss:   {hist['val_loss'][-1]:.4f} | Final val acc:   {hist['val_acc'][-1]:.4f}")

    best_idx = hist["val_loss"].index(min(hist["val_loss"]))
    print(f"Best val loss at epoch {best_idx+1}: {hist['val_loss'][best_idx]:.4f} "
          f"(acc={hist['val_acc'][best_idx]:.4f})")
    print()


def report_ppo():
    hist_path = MODELS_DIR / "ppo_agent.history.json"
    if not hist_path.exists():
        print("[PPO] ppo_agent.history.json not found. Run train/train_ppo.py first.")
        return

    with open(hist_path) as f:
        hist = json.load(f)

    print("=" * 80)
    print("PPO FINE-TUNING REPORT")
    print("=" * 80)

    n = len(hist["timesteps"])
    if n == 0:
        print("No timesteps trained.")
        return

    print(f"Total timesteps: {hist['timesteps'][-1]}")
    print(f"Final mean reward: {hist['mean_reward'][-1]:.1f}")
    print(f"Best mean reward: {max(hist['mean_reward']):.1f}")

    print(f"\n{'STEP':>8} {'REWARD':>8} {'POLICY':>8} {'VALUE':>8} {'ENTROPY':>8}")
    for i in range(0, n, max(1, n // 20)):
        print(f"{hist['timesteps'][i]:>8} {hist['mean_reward'][i]:>8.1f} "
              f"{hist['policy_loss'][i]:>8.4f} {hist['value_loss'][i]:>8.4f} "
              f"{hist['entropy'][i]:>8.4f}")
    print()


def report_eval():
    eval_dir = OVERCOOKED_DIR / "configs" / "eval"
    if not eval_dir.exists():
        print("[Eval] configs/eval/ not found. Run scripts/generate_eval_configs.py first.")
        return

    print("=" * 80)
    print("EVALUATION CONFIGS")
    print("=" * 80)
    yamls = sorted(eval_dir.glob("*.yaml"))
    print(f"Found {len(yamls)} eval configs:")
    for y in yamls:
        print(f"  python -m src.evaluate --config configs/eval/{y.name}")
    print()


def report_dataset():
    stats_path = DATA_DIR / "dataset_stats.json"
    if not stats_path.exists():
        print("[Dataset] dataset_stats.json not found. Run train/build_dataset.py first.")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    print("=" * 80)
    print("DATASET REPORT")
    print("=" * 80)
    print(f"Total timesteps: {stats['total_timesteps']}")
    print(f"Obs dim: {stats['obs_dim']}")
    print(f"Num layouts: {stats['num_layouts']}")
    print(f"\nAction distribution:")
    names = ["north", "south", "east", "west", "stay", "interact"]
    for a in range(6):
        c = stats["action_distribution"][str(a)]
        print(f"  {a} ({names[a]:>9}): {c:>6} ({100*c/stats['total_timesteps']:.1f}%)")
    print(f"\nTier distribution: {stats['tier_distribution']}")
    print(f"\nTop 10 layouts by timesteps:")
    sorted_l = sorted(stats["per_layout"].items(), key=lambda x: -x[1]["timesteps"])
    for name, s in sorted_l[:10]:
        print(f"  {name:<35} {s['timesteps']:>6} steps | avg_score={s['avg_score']:.1f}")
    print()


def main():
    report_filter()
    print()
    report_dataset()
    print()
    report_bc()
    print()
    report_ppo()
    print()
    report_eval()


if __name__ == "__main__":
    main()
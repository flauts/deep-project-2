"""Filter Overcooked recordings by quality and assign tiers for BC training.

Computes per-recording quality signals (official score, deliveries, idle %, action
entropy, recorded agent type) and applies configurable exclusion rules. Tags kept
recordings as gold/silver/bronze by per-layout official score percentile.

Outputs:
    - train/data/recording_quality.tsv  (every recording with signals + keep/drop + tier)
    - train/data/consolidated_filtered.npz (kept recordings concatenated)

Usage:
    cd overcooked
    python scripts/filter_recordings.py --config ../configs/filter.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
OVERCOOKED_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = OVERCOOKED_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
from official_score import compute_official_score


def compute_entropy(actions: np.ndarray) -> float:
    vals, counts = np.unique(actions, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum())


def get_recorded_agent(metadata: dict) -> str:
    """Determine which agent was recorded (from data_collection.record_agent_indices)."""
    rai = metadata.get("data_collection", {}).get("record_agent_indices", [])
    pols = metadata.get("policies", {})
    if not rai:
        return "unknown"
    idx = rai[0]
    if idx == 0:
        return pols.get("agent_0", {}).get("name", "unknown")
    elif idx == 1:
        return pols.get("agent_1", {}).get("name", "unknown")
    return "unknown"


def get_partner_agent(metadata: dict) -> str:
    """Determine the partner agent."""
    rai = metadata.get("data_collection", {}).get("record_agent_indices", [])
    pols = metadata.get("policies", {})
    if not rai:
        return "unknown"
    idx = rai[0]
    if idx == 0:
        return pols.get("agent_1", {}).get("name", "unknown")
    elif idx == 1:
        return pols.get("agent_0", {}).get("name", "unknown")
    return "unknown"


def extract_layout_name(npz_filename: str, metadata: dict) -> str:
    """Extract layout name from metadata first, fall back to filename prefix."""
    layout = metadata.get("layout", {})
    env = metadata.get("environment", {})
    name = layout.get("layout_name") or env.get("layout_name")
    if name:
        return str(name)
    base = os.path.basename(npz_filename)
    return base.split("_2026")[0].split("_2025")[0]


def find_recordings(root: Path) -> list[str]:
    """Find all .npz recordings, excluding the training output dirs."""
    results = []
    for p in glob.glob(str(root / "**" / "*.npz"), recursive=True):
        norm = os.path.normpath(p)
        parts = norm.split(os.sep)
        if "train" in parts and "data" in parts:
            continue
        results.append(p)
    return sorted(results)


def load_npz_safe(path: str) -> dict | None:
    try:
        return dict(np.load(path, allow_pickle=True))
    except Exception:
        return None


def load_metadata(npz_path: str) -> dict:
    meta_path = npz_path[:-4] + ".metadata.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def assign_tiers(records: list[dict], config: dict) -> None:
    """Assign gold/silver/bronze tiers per layout based on official score percentile."""
    tier_cfg = config.get("tier", {})
    gold_pct = tier_cfg.get("gold", {}).get("min_percentile", 75)
    gold_soups = tier_cfg.get("gold", {}).get("min_soups", 4)
    silver_pct = tier_cfg.get("silver", {}).get("min_percentile", 50)
    silver_soups = tier_cfg.get("silver", {}).get("min_soups", 2)
    bronze_weight = tier_cfg.get("bronze", {}).get("weight", 0.25)

    # Group kept records by layout
    by_layout = defaultdict(list)
    for r in records:
        if r["keep"]:
            by_layout[r["layout"]].append(r)

    for layout, recs in by_layout.items():
        scores = [r["official_score"] for r in recs]
        if not scores:
            continue

        # Compute percentiles
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        gold_thresh = sorted_scores[int(n * gold_pct / 100)] if n > 1 else sorted_scores[0]
        silver_thresh = sorted_scores[int(n * silver_pct / 100)] if n > 1 else sorted_scores[0]

        for r in recs:
            if r["num_soups"] >= gold_soups or r["official_score"] >= gold_thresh:
                r["tier"] = "gold"
                r["weight"] = tier_cfg.get("gold", {}).get("weight", 1.0)
            elif r["num_soups"] >= silver_soups or r["official_score"] >= silver_thresh:
                r["tier"] = "silver"
                r["weight"] = tier_cfg.get("silver", {}).get("weight", 0.6)
            else:
                r["tier"] = "bronze"
                r["weight"] = bronze_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "filter.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    recordings_root = OVERCOOKED_DIR / config.get("recordings_root", ".")
    output_tsv = PROJECT_ROOT / config.get("output_tsv", "train/data/recording_quality.tsv")
    output_npz = PROJECT_ROOT / config.get("output_npz", "train/data/consolidated_filtered.npz")

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    exclude_non_human = config.get("exclude_non_human", True)
    min_idle_zero = config.get("min_idle_excluded_if_zero_score", 70.0)
    min_entropy = config.get("min_entropy", 0.3)
    min_length = config.get("min_length", 100)
    min_recordings_for_eval = config.get("min_recordings_for_eval", 3)

    npz_files = find_recordings(recordings_root)
    print(f"Found {len(npz_files)} .npz recordings")
    print(f"Applying filters:")
    print(f"  exclude_non_human: {exclude_non_human}")
    print(f"  min_idle_if_zero_score: {min_idle_zero}%")
    print(f"  min_entropy: {min_entropy}")
    print(f"  min_length: {min_length}")
    print()

    all_records = []
    load_errors = 0

    for npz_path in npz_files:
        data = load_npz_safe(npz_path)
        if data is None:
            all_records.append({
                "file": npz_path, "layout": "?", "recorded_agent": "ERR",
                "partner": "?", "score": -1, "deliveries": 0, "idle_pct": 0,
                "length": 0, "entropy": 0, "official_score": 0, "num_soups": 0,
                "keep": False, "reason": "load_error", "tier": "none", "weight": 0,
            })
            load_errors += 1
            continue

        rewards = data.get("rewards")
        actions = data.get("actions")
        if rewards is None or actions is None:
            all_records.append({
                "file": npz_path, "layout": "?", "recorded_agent": "ERR",
                "partner": "?", "score": -1, "deliveries": 0, "idle_pct": 0,
                "length": 0, "entropy": 0, "official_score": 0, "num_soups": 0,
                "keep": False, "reason": "missing_keys", "tier": "none", "weight": 0,
            })
            load_errors += 1
            continue
        timesteps = data["timesteps"] if "timesteps" in data else np.arange(len(rewards))

        score = int(np.sum(rewards))
        deliveries = int(np.sum(rewards > 0))
        idle_pct = float(np.mean(actions == 4) * 100)
        length = int(len(actions))
        entropy = compute_entropy(actions)

        metadata = load_metadata(npz_path)
        recorded_agent = get_recorded_agent(metadata)
        partner = get_partner_agent(metadata)
        layout = extract_layout_name(npz_path, metadata)

        official = compute_official_score(rewards, timesteps=timesteps, horizon=length)

        record = {
            "file": npz_path,
            "layout": layout,
            "recorded_agent": recorded_agent,
            "partner": partner,
            "score": score,
            "deliveries": deliveries,
            "idle_pct": round(idle_pct, 1),
            "length": length,
            "entropy": round(entropy, 3),
            "official_score": official["official_score"],
            "num_soups": official["num_soups"],
            "keep": True,
            "reason": "",
            "tier": "none",
            "weight": 0.0,
        }

        # Apply exclusion rules
        if exclude_non_human and recorded_agent not in ("human_keyboard", "unknown"):
            record["keep"] = False
            record["reason"] = f"non_human ({recorded_agent})"
        elif score == 0 and idle_pct >= min_idle_zero:
            record["keep"] = False
            record["reason"] = f"disengaged (score=0, idle={idle_pct:.0f}%)"
        elif entropy < min_entropy:
            record["keep"] = False
            record["reason"] = f"low_entropy ({entropy:.3f})"
        elif length < min_length:
            record["keep"] = False
            record["reason"] = f"truncated (len={length})"
        else:
            record["reason"] = "kept"

        all_records.append(record)

    # Assign tiers
    assign_tiers(all_records, config)

    # Layout eligibility gate
    kept_by_layout = defaultdict(list)
    for r in all_records:
        if r["keep"]:
            kept_by_layout[r["layout"]].append(r)
    low_coverage = {layout for layout, recs in kept_by_layout.items() if len(recs) < min_recordings_for_eval}

    # Write TSV
    header = ["file", "layout", "recorded_agent", "partner", "score", "deliveries",
              "idle_pct", "length", "entropy", "official_score", "num_soups",
              "keep", "reason", "tier", "weight", "low_coverage"]
    with open(output_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in all_records:
            row = [
                os.path.relpath(r["file"], OVERCOOKED_DIR),
                r["layout"], r["recorded_agent"], r["partner"], str(r["score"]),
                str(r["deliveries"]), str(r["idle_pct"]), str(r["length"]),
                str(r["entropy"]), str(r["official_score"]), str(r["num_soups"]),
                str(r["keep"]), r["reason"], r["tier"], str(r["weight"]),
                "Y" if r["layout"] in low_coverage else "N",
            ]
            f.write("\t".join(row) + "\n")

    # Build consolidated filtered NPZ
    kept_files = [r for r in all_records if r["keep"]]
    obs_list, act_list, rew_list, done_list, next_obs_list = [], [], [], [], []
    tiers, weights, layouts, role_swaps, agent_indices = [], [], [], [], []
    ep_ids, ep_seeds, timesteps_list = [], [], []

    for r in kept_files:
        data = load_npz_safe(r["file"])
        if data is None:
            continue
        n = len(data["actions"])
        obs = np.asarray(data["obs"], dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        obs_list.append(obs)
        act_list.append(np.asarray(data["actions"], dtype=np.int64))
        rew_list.append(np.asarray(data["rewards"], dtype=np.float32))
        done_list.append(np.asarray(data["dones"], dtype=np.bool_))
        if "next_obs" in data:
            no = np.asarray(data["next_obs"], dtype=np.float32)
            if no.ndim == 1:
                no = no[np.newaxis, :]
            next_obs_list.append(no)
        tiers.append(np.full(n, {"gold": 0, "silver": 1, "bronze": 2}.get(r["tier"], 2), dtype=np.int64))
        weights.append(np.full(n, r["weight"], dtype=np.float32))
        layouts.append(np.array([r["layout"]] * n, dtype=object))
        rs = data["role_swaps"] if "role_swaps" in data else np.zeros(n, dtype=np.bool_)
        role_swaps.append(np.asarray(rs, dtype=np.bool_))
        ai = data["agent_indices"] if "agent_indices" in data else np.ones(n, dtype=np.int64)
        agent_indices.append(np.asarray(ai, dtype=np.int64))
        eid = data["episode_ids"] if "episode_ids" in data else np.zeros(n, dtype=np.int64)
        ep_ids.append(np.asarray(eid, dtype=np.int64))
        es = data["episode_seeds"] if "episode_seeds" in data else np.full(n, -1, dtype=np.int64)
        ep_seeds.append(np.asarray(es, dtype=np.int64))
        ts = data["timesteps"] if "timesteps" in data else np.arange(n, dtype=np.int64)
        timesteps_list.append(np.asarray(ts, dtype=np.int64))

    kept = [r for r in all_records if r["keep"]]
    dropped = [r for r in all_records if not r["keep"]]
    print(f"\n{'='*80}")
    print(f"TOTAL: {len(all_records)} | KEPT: {len(kept)} | DROPPED: {len(dropped)} | ERRORS: {load_errors}")
    print(f"{'='*80}")

    print(f"\nDrop reasons:")
    reasons = defaultdict(int)
    for r in dropped:
        reasons[r["reason"]] += 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {count:>4}  {reason}")

    print(f"\nTier distribution (kept only):")
    tier_counts = defaultdict(int)
    for r in kept:
        tier_counts[r["tier"]] += 1
    for tier in ["gold", "silver", "bronze"]:
        print(f"  {tier:>6}: {tier_counts.get(tier, 0)}")

    print(f"\nPer-layout summary (kept):")
    print(f"{'LAYOUT':<35} {'N':>3} {'GOLD':>4} {'SILV':>4} {'BRNZ':>4} {'AVG_SCORE':>9} {'LOW_COV':>7}")
    print("-" * 75)
    for layout in sorted(kept_by_layout):
        recs = kept_by_layout[layout]
        golds = sum(1 for r in recs if r["tier"] == "gold")
        silvers = sum(1 for r in recs if r["tier"] == "silver")
        bronzes = sum(1 for r in recs if r["tier"] == "bronze")
        avg_sc = np.mean([r["official_score"] for r in recs])
        lc = "Y" if layout in low_coverage else ""
        print(f"{layout:<35} {len(recs):>3} {golds:>4} {silvers:>4} {bronzes:>4} {avg_sc:>9.0f} {lc:>7}")

    print(f"\nLow-coverage layouts (excluded from per-layout eval, still trained): {len(low_coverage)}")
    for lc in sorted(low_coverage):
        print(f"  {lc}")

    if not kept:
        print("\nNo recordings kept — nothing to save.")
        print(f"Output TSV:  {output_tsv}")
        return

    save_dict = {
        "obs": np.concatenate(obs_list, axis=0),
        "actions": np.concatenate(act_list),
        "rewards": np.concatenate(rew_list),
        "dones": np.concatenate(done_list),
        "tiers": np.concatenate(tiers),
        "weights": np.concatenate(weights),
        "layouts": np.concatenate(layouts),
        "role_swaps": np.concatenate(role_swaps),
        "agent_indices": np.concatenate(agent_indices),
        "episode_ids": np.concatenate(ep_ids),
        "episode_seeds": np.concatenate(ep_seeds),
        "timesteps": np.concatenate(timesteps_list),
    }
    if next_obs_list:
        save_dict["next_obs"] = np.concatenate(next_obs_list, axis=0)
    np.savez_compressed(output_npz, **save_dict)

    print(f"\nOutput TSV:  {output_tsv}")
    print(f"Output NPZ:  {output_npz}")
    print(f"Total kept timesteps: {save_dict['actions'].shape[0]}")


if __name__ == "__main__":
    main()
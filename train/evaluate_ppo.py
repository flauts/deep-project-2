#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Overcooked-AI GNN PPO & BC Models.
Evaluates the trained PPO agent (`ppo_agent_400k_validation.pt`) across all 6 benchmark layouts
against 3 partner paradigms (`Self-Play, BC Expert, Random`) over 5 evaluation seeds (42..46).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.training.env import build_env_for_layout, load_dynamics_overrides
from train.training.models import GraphAttentionPPOPolicy

TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_DIR.parent
OVERCOOKED_DIR = PROJECT_ROOT / "overcooked"
LAYOUTS_DIR = OVERCOOKED_DIR / "layouts"


def load_policy(model_path, device, obs_dim=96, num_actions=6):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    hidden = ckpt.get("hidden", 256)
    layers = ckpt.get("layers", 3)
    
    policy = GraphAttentionPPOPolicy(obs_dim, num_actions, hidden=hidden, layers=layers).to(device)
    policy.load_state_dict(ckpt["model_state"], strict=False)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    return policy


def select_action(logits, eval_mode="stochastic", temp=1.0):
    if eval_mode == "argmax":
        return int(torch.argmax(logits, dim=-1).item())
    elif eval_mode == "temperature":
        scaled_logits = logits / max(temp, 1e-5)
        dist = torch.distributions.Categorical(logits=scaled_logits)
        return int(dist.sample().item())
    else:  # stochastic (default training rollout distribution)
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())


def run_episode(env, policy0, policy1_type, policy1_model, device, max_steps=400, seed=42, eval_mode="stochastic", temp=1.0):
    """Run a single 400-step evaluation episode and return (sparse_reward_total, shaped_reward_total, total_deliveries)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    obs = env.reset()
    total_sparse = 0.0
    total_shaped = 0.0
    total_deliveries = 0
    
    for step in range(max_steps):
        # Agent 0 (Primary PPO Agent) action
        obs0_tensor = torch.FloatTensor(obs[0]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits0, _ = policy0.forward(obs0_tensor)
            act0 = select_action(logits0, eval_mode, temp)
            
        # Agent 1 (Partner) action
        if policy1_type == "self_play":
            obs1_tensor = torch.FloatTensor(obs[1]).unsqueeze(0).to(device)
            with torch.no_grad():
                logits1, _ = policy0.forward(obs1_tensor)
                act1 = select_action(logits1, eval_mode, temp)
        elif policy1_type == "bc_expert":
            obs1_tensor = torch.FloatTensor(obs[1]).unsqueeze(0).to(device)
            with torch.no_grad():
                logits1, _ = policy1_model.forward(obs1_tensor)
                act1 = select_action(logits1, eval_mode, temp)
        elif policy1_type == "random":
            act1 = int(np.random.randint(0, 6))
        else:
            raise ValueError(f"Unknown partner type: {policy1_type}")
            
        next_obs, rewards, dones, info = env.step([act0, act1])
        
        sparse = info.get("sparse_r_by_agent", [0.0, 0.0])
        shaped = info.get("shaped_r_by_agent", [0.0, 0.0])
        
        step_sparse = float(sparse[0] + sparse[1])
        total_sparse += step_sparse
        total_shaped += float(shaped[0] + shaped[1])
        if step_sparse > 0:
            total_deliveries += 1
        
        obs = next_obs
        if dones[0]:
            break
            
    return total_sparse, total_shaped, total_deliveries


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-model", type=str, default="train/models/ppo_agent_400k_validation.pt")
    parser.add_argument("--bc-model", type=str, default="train/models/bc_agent_gnn.pt")
    parser.add_argument("--layouts", type=str, default="cramped_room,asymmetric_advantages,coordination_ring,simple_o,forced_coordination,counter_circuit")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--num-seeds", type=int, default=None, help="Automatically generate N sequential seeds starting at 42")
    parser.add_argument("--output-md", type=str, default="train/models/eval_400k_report.md")
    parser.add_argument("--output-json", type=str, default="train/models/eval_400k_results.json")
    parser.add_argument("--eval-mode", type=str, default="stochastic", choices=["stochastic", "argmax", "temperature"])
    parser.add_argument("--temp", type=float, default=1.0)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device} | Mode: {args.eval_mode} (temp={args.temp})")
    
    layout_names = [l.strip() for l in args.layouts.split(",")]
    if args.num_seeds is not None:
        seeds = list(range(42, 42 + args.num_seeds))
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    print(f"Loading PPO Policy: {args.ppo_model}")
    ppo_policy = load_policy(args.ppo_model, device)
    
    bc_policy = None
    if Path(args.bc_model).exists():
        print(f"Loading BC Expert Policy: {args.bc_model}")
        bc_policy = load_policy(args.bc_model, device)
    else:
        print(f"Warning: BC model not found at {args.bc_model}. BC expert evaluation will be skipped.")
        
    dynamics_overrides = load_dynamics_overrides()
    
    results = {}
    
    print("\n" + "="*95)
    print(f"{'LAYOUT':<24} | {'PARTNER PARADIGM':<18} | {'SPARSE REWARD (Mean±SD)':<24} | {'SOUPS DELIVERED':<16} | {'SHAPED TOTAL':<12}")
    print("="*95)
    
    for layout in layout_names:
        results[layout] = {}
        try:
            # We use shaped_reward_scale=1.0 during eval so we can track exact shaped value alongside sparse
            env = build_env_for_layout(layout, LAYOUTS_DIR, dynamics_overrides, shaped_reward_scale=1.0)
        except Exception as e:
            print(f"{layout:<24} | {'FAILED TO BUILD ENV':<18} | {str(e):<24} | {'-':<16} | -")
            continue
            
        paradigms = ["self_play", "bc_expert", "random"] if bc_policy is not None else ["self_play", "random"]
        
        for paradigm in paradigms:
            sparse_scores = []
            shaped_scores = []
            delivery_scores = []
            for seed in seeds:
                s_r, sh_r, deliv = run_episode(env, ppo_policy, paradigm, bc_policy, device, max_steps=400, seed=seed, eval_mode=args.eval_mode, temp=args.temp)
                sparse_scores.append(s_r)
                shaped_scores.append(sh_r)
                delivery_scores.append(deliv)
                
            mean_sparse = np.mean(sparse_scores)
            std_sparse = np.std(sparse_scores)
            mean_shaped = np.mean(shaped_scores)
            mean_deliveries = np.mean(delivery_scores)
            std_deliveries = np.std(delivery_scores)
            
            results[layout][paradigm] = {
                "mean_sparse": float(mean_sparse),
                "std_sparse": float(std_sparse),
                "mean_shaped": float(mean_shaped),
                "mean_deliveries": float(mean_deliveries),
                "std_deliveries": float(std_deliveries),
                "scores": [float(x) for x in sparse_scores],
                "deliveries": [int(x) for x in delivery_scores]
            }
            
            print(f"{layout:<24} | {paradigm:<18} | {mean_sparse:>7.1f} ± {std_sparse:<6.1f}          | {mean_deliveries:>4.1f} ± {std_deliveries:<4.1f} soups | {mean_shaped:>8.1f}")
            
    print("="*95 + "\n")
    
    # Save JSON results
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"JSON Results saved to: {json_path}")
    
    # Generate Markdown Report
    md_lines = [
        "# Overcooked-AI PPO Validation Scorecard (`400K Smoke Test`)",
        "",
        "**Model Evaluated:** `ppo_agent_400k_validation.pt`  ",
        f"**Evaluation Seeds:** `{', '.join(map(str, seeds))}` (400 steps per episode)  ",
        "",
        "| Layout | Partner Paradigm | Sparse Soup Reward (`Mean ± SD`) | Exact Soups Delivered (`Mean ± SD`) | Total Shaped Navigation Return |",
        "|:---|:---|:---:|:---:|:---:|"
    ]
    
    for layout in layout_names:
        if layout not in results or not results[layout]:
            continue
        for paradigm, metrics in results[layout].items():
            ms = metrics["mean_sparse"]
            ss = metrics["std_sparse"]
            md = metrics["mean_deliveries"]
            sd = metrics["std_deliveries"]
            sh = metrics["mean_shaped"]
            md_lines.append(f"| `{layout}` | **{paradigm}** | **{ms:.1f} ± {ss:.1f}** | `{md:.1f} ± {sd:.1f} soups` | `{sh:.1f}` |")
            
    md_path = Path(args.output_md)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Markdown Scorecard saved to: {md_path}")
    

if __name__ == "__main__":
    main()

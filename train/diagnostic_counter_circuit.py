#!/usr/bin/env python3
"""
Step-by-Step Qualitative Rollout Diagnostic for Counter Circuit (`counter_circuit`).
Diagnoses why `counter_circuit` achieved 0.0 ± 0.0 sparse return under self_play and random pairings,
and checks for degenerate policy loops, symmetry-breaking failures, corridor blocking, and oscillation.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.training.env import build_env_for_layout, load_dynamics_overrides
from train.training.models import GraphAttentionPPOPolicy
from src.constants import action_index_to_overcooked_action

TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_DIR.parent
OVERCOOKED_DIR = PROJECT_ROOT / "overcooked"
LAYOUTS_DIR = OVERCOOKED_DIR / "layouts"

ACTION_NAMES = {
    0: "North",
    1: "South",
    2: "East",
    3: "West",
    4: "Stay",
    5: "Interact"
}

def load_policy(model_path, device, obs_dim=96, num_actions=6):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    hidden = ckpt.get("hidden", 256)
    layers = ckpt.get("layers", 3)
    policy = GraphAttentionPPOPolicy(obs_dim, num_actions, hidden=hidden, layers=layers).to(device)
    policy.load_state_dict(ckpt["model_state"], strict=False)
    policy.eval()
    return policy


def run_diagnostic_rollout(env, policy0, policy1_type, policy1_model, device, eval_mode="argmax", max_steps=400, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs = env.reset()
    
    # Tracking counters
    p0_collisions = 0
    p1_collisions = 0
    p0_pickups = 0
    p0_drops = 0
    p1_pickups = 0
    p1_drops = 0
    soups_delivered = 0
    
    # Trajectory log snippet
    trajectory_log = []
    
    prev_p0_pos = None
    prev_p1_pos = None
    prev_p0_item = None
    prev_p1_item = None
    
    for step in range(max_steps):
        state = env.get_state()
        p0 = state.players[0]
        p1 = state.players[1]
        
        # Check item state changes
        curr_p0_item = p0.held_object.name if p0.held_object else "None"
        curr_p1_item = p1.held_object.name if p1.held_object else "None"
        
        if prev_p0_item == "None" and curr_p0_item != "None":
            p0_pickups += 1
        elif prev_p0_item != "None" and curr_p0_item == "None":
            p0_drops += 1
            
        if prev_p1_item == "None" and curr_p1_item != "None":
            p1_pickups += 1
        elif prev_p1_item != "None" and curr_p1_item == "None":
            p1_drops += 1
            
        # Action selection
        obs0_tensor = torch.FloatTensor(obs[0]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits0, _ = policy0.forward(obs0_tensor)
            if eval_mode == "argmax":
                act0 = int(torch.argmax(logits0, dim=-1).item())
            else:
                act0 = int(torch.distributions.Categorical(logits=logits0).sample().item())
                
        if policy1_type == "self_play":
            obs1_tensor = torch.FloatTensor(obs[1]).unsqueeze(0).to(device)
            with torch.no_grad():
                logits1, _ = policy0.forward(obs1_tensor)
                if eval_mode == "argmax":
                    act1 = int(torch.argmax(logits1, dim=-1).item())
                else:
                    act1 = int(torch.distributions.Categorical(logits=logits1).sample().item())
        elif policy1_type == "bc_expert":
            obs1_tensor = torch.FloatTensor(obs[1]).unsqueeze(0).to(device)
            with torch.no_grad():
                logits1, _ = policy1_model.forward(obs1_tensor)
                if eval_mode == "argmax":
                    act1 = int(torch.argmax(logits1, dim=-1).item())
                else:
                    act1 = int(torch.distributions.Categorical(logits=logits1).sample().item())
        elif policy1_type == "random":
            act1 = int(np.random.randint(0, 6))
            
        next_obs, rewards, dones, info = env.step([act0, act1])
        next_state = env.get_state()
        np0 = next_state.players[0]
        np1 = next_state.players[1]
        
        # Check collision (attempted movement 0..3 but position unchanged)
        is_p0_collision = (act0 in [0, 1, 2, 3]) and (np0.position == p0.position)
        is_p1_collision = (act1 in [0, 1, 2, 3]) and (np1.position == p1.position)
        if is_p0_collision:
            p0_collisions += 1
        if is_p1_collision:
            p1_collisions += 1
            
        sparse = info.get("sparse_r_by_agent", [0.0, 0.0])
        if sum(sparse) > 0:
            soups_delivered += int(sum(sparse) / 20)
            
        # Log first 30 steps + any delivery/item transitions
        if step < 30 or sum(sparse) > 0 or (curr_p0_item != prev_p0_item) or (curr_p1_item != prev_p1_item):
            trajectory_log.append(
                f"Step {step:>3} | P0 pos={p0.position} item={curr_p0_item:<6} act={ACTION_NAMES[act0]:<8} {'[BLOCKED]' if is_p0_collision else '':<9} | "
                f"P1 pos={p1.position} item={curr_p1_item:<6} act={ACTION_NAMES[act1]:<8} {'[BLOCKED]' if is_p1_collision else '':<9} | "
                f"Reward={sum(sparse):.0f}"
            )
            
        obs = next_obs
        prev_p0_pos = p0.position
        prev_p1_pos = p1.position
        prev_p0_item = curr_p0_item
        prev_p1_item = curr_p1_item
        
        if dones[0]:
            break
            
    summary = {
        "soups_delivered": soups_delivered,
        "p0_collisions": p0_collisions,
        "p1_collisions": p1_collisions,
        "p0_pickups": p0_pickups,
        "p0_drops": p0_drops,
        "p1_pickups": p1_pickups,
        "p1_drops": p1_drops,
        "trajectory_log": trajectory_log
    }
    return summary


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Models on {device}...")
    ppo_policy = load_policy("train/models/ppo_agent_400k_validation.pt", device)
    bc_policy = load_policy("train/models/bc_agent_gnn.pt", device)
    
    dynamics_overrides = load_dynamics_overrides()
    env = build_env_for_layout("counter_circuit", LAYOUTS_DIR, dynamics_overrides, shaped_reward_scale=1.0)
    
    print("\n" + "="*95)
    print("COUNTER CIRCUIT (`counter_circuit`) DEEP QUALITATIVE ROLLOUT DIAGNOSTIC")
    print("="*95)
    
    for mode in ["argmax", "stochastic"]:
        print(f"\n--- EVALUATION MODE: {mode.upper()} (Seed 42) ---")
        for partner in ["self_play", "bc_expert", "random"]:
            summary = run_diagnostic_rollout(env, ppo_policy, partner, bc_policy, device, eval_mode=mode, max_steps=400, seed=42)
            print(f"\n[Partner: {partner:<12}] | Soups Delivered: {summary['soups_delivered']} | P0 Collisions: {summary['p0_collisions']}/400 ({summary['p0_collisions']/4.0:.1f}%) | P1 Collisions: {summary['p1_collisions']}/400 ({summary['p1_collisions']/4.0:.1f}%)")
            print(f"                   | P0 Pickups/Drops: {summary['p0_pickups']}/{summary['p0_drops']} | P1 Pickups/Drops: {summary['p1_pickups']}/{summary['p1_drops']}")
            print("  First 15 trajectory steps & transitions:")
            for line in summary["trajectory_log"][:15]:
                print("    " + line)
            if len(summary["trajectory_log"]) > 15:
                print(f"    ... (+{len(summary['trajectory_log'])-15} more logged events)")
                
    print("\n" + "="*95)


if __name__ == "__main__":
    main()

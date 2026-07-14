"""Test all three versions of ppo_agent_master against BC baseline across curriculum and held-out layouts."""
import os
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OVERCOOKED_ROOT = PROJECT_ROOT / "overcooked"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(OVERCOOKED_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "train" / "training"))

from env import build_env_for_layout, load_dynamics_overrides
LAYOUTS_DIR = OVERCOOKED_ROOT / "layouts"
from models import get_policy
from overcooked_ai_py.mdp.actions import Action
from policies.basic_policies import GreedyFullTaskPolicy, RandomMotionPolicy
from src.constants import overcooked_action_to_index, OVERCOOKED_ACTION_TO_INDEX

def run_eval_for_model(model_path, layout_names, partner_type="greedy", n_episodes=5, device="cpu"):
    if not os.path.exists(model_path):
        return {l: 0.0 for l in layout_names}
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    obs_dim = ckpt["obs_dim"]
    num_actions = ckpt["num_actions"]
    hidden = ckpt["hidden"]
    layers = ckpt["layers"]
    arch = ckpt.get("arch", "gnn")
    topo_dim = ckpt.get("topo_dim", 0)

    policy = get_policy(obs_dim, num_actions, hidden, layers, device, arch=arch, topo_dim=topo_dim)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()

    dynamics_overrides = load_dynamics_overrides()
    results = {}

    for name in layout_names:
        try:
            env = build_env_for_layout(name, LAYOUTS_DIR, dynamics_overrides, shaped_reward_scale=0.0)
            
            # Choose partner
            partner = None
            if name == "layout_4":
                partner = RandomMotionPolicy()
            elif partner_type == "greedy":
                partner = GreedyFullTaskPolicy(ingredient="onion", avoid_teammate=True)
                partner.set_mdp(env.mdp)
            else:
                partner = None  # self-play

            sparse_scores = []
            for seed in range(42, 42 + n_episodes):
                np.random.seed(seed)
                torch.manual_seed(seed)
                total_sparse = 0.0
                for _step in range(400):
                    obs0 = torch.FloatTensor(obs[0]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits0, _ = policy.forward(obs0)
                    act0 = int(logits0.argmax(dim=-1).item())

                    if partner is not None:
                        if hasattr(partner, "agent_index"):
                            partner.agent_index = 1
                        partner_action, _ = partner.action(env.env.state)
                        act1 = overcooked_action_to_index(partner_action) if partner_action in OVERCOOKED_ACTION_TO_INDEX else 4
                    else:
                        obs1 = torch.FloatTensor(obs[1]).unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits1, _ = policy.forward(obs1)
                        act1 = int(logits1.argmax(dim=-1).item())

                    next_obs, rewards, dones, info = env.step([act0, act1])
                    obs = next_obs
                    sparse = info.get("sparse_r_by_agent", [0.0, 0.0])
                    total_sparse += float(sparse[0] + sparse[1])
                sparse_scores.append(total_sparse)
            results[name] = float(np.mean(sparse_scores)) / 20.0  # convert to soups
        except Exception as e:
            results[name] = -1.0
            print(f"Error on {name}: {e}")
            
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking across 5 evaluation episodes per map on device: {device}\n")

    layout_names = [
        "asymmetric_advantages",
        "coordination_ring",
        "simple_o",
        "cramped_room",
        "counter_circuit",
        "layout_4",
        "forced_coordination"  # Held-out competition layout for generalization check!
    ]

    models = {
        "BC Baseline": "train/models/bc_agent_gnn.pt",
        "Master v1 (200k)": "train/models/ppo_agent_master.pt",
        "Master v2 (130k)": "train/models/ppo_agent_master_v2.pt",
        "Master v3 (50k)": "train/models/ppo_agent_master_v3.pt",
    }

    all_results = {}
    for label, path in models.items():
        print(f"Evaluating {label} ({path})...")
        res = run_eval_for_model(path, layout_names, partner_type="greedy", n_episodes=5, device=device)
        all_results[label] = res

    # Print summary table
    print("\n" + "="*85)
    print(f"{'Layout (Greedy Partner except layout_4)':<25} | {'BC Baseline':<12} | {'v1 (200k)':<12} | {'v2 (130k)':<12} | {'v3 (50k)':<12}")
    print("="*85)
    for name in layout_names:
        row = f"{name:<25} | "
        for label in models.keys():
            val = all_results[label].get(name, 0.0)
            row += f"{val:<12.1f} | "
        print(row)
    print("="*85)
    
    # Calculate averages across the first 6 curriculum maps and total across all 7
    print(f"{'Curriculum Avg (6 maps)':<25} | ", end="")
    for label in models.keys():
        curr_avg = np.mean([all_results[label][l] for l in layout_names[:6]])
        print(f"{curr_avg:<12.2f} | ", end="")
    print()
    print(f"{'Total Avg (all 7 maps)':<25} | ", end="")
    for label in models.keys():
        tot_avg = np.mean([all_results[label][l] for l in layout_names])
        print(f"{tot_avg:<12.2f} | ", end="")
    print("\n" + "="*85)

if __name__ == "__main__":
    main()

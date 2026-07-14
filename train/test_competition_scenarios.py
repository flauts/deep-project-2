"""Test models against the exact official TA competition scenarios (Escenarios 1 to 4)."""
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
from models import PPOPolicy, BCPolicy, GraphAttentionPPOPolicy
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import Agent
from policies.basic_policies import GreedyFullTaskPolicy, RandomMotionPolicy
from src.constants import overcooked_action_to_index, OVERCOOKED_ACTION_TO_INDEX
from src.policy_wrappers import EpsilonActionWrapper, SafeActionWrapper

def load_policy_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    obs_dim = ckpt.get("obs_dim", 121)
    num_actions = ckpt.get("num_actions", 6)
    hidden = ckpt.get("hidden", 256)
    layers = ckpt.get("layers", 3)
    arch = ckpt.get("arch", "gnn")
    topo_dim = ckpt.get("topo_dim", 25)

    if arch == "gnn" or "topo_dim" in ckpt:
        policy = GraphAttentionPPOPolicy(obs_dim, num_actions, hidden=hidden, layers=layers, topo_dim=topo_dim).to(device)
    else:
        policy = PPOPolicy(obs_dim, num_actions, hidden=hidden, layers=layers).to(device)
        
    policy.load_state_dict(ckpt["model_state"], strict=False)
    policy.eval()
    return policy

class StickyActionWrapper(Agent):
    """Repeats previous action with probability sticky_prob."""
    def __init__(self, base_agent: Agent, sticky_prob: float = 0.10):
        self.base_agent = base_agent
        self.sticky_prob = float(sticky_prob)
        self.last_action = None
        super().__init__()

    def reset(self):
        super().reset()
        self.last_action = None
        if hasattr(self, "base_agent") and hasattr(self.base_agent, "reset"):
            self.base_agent.reset()

    def set_agent_index(self, agent_index):
        super().set_agent_index(agent_index)
        if hasattr(self.base_agent, "set_agent_index"):
            self.base_agent.set_agent_index(agent_index)

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        if hasattr(self.base_agent, "set_mdp"):
            self.base_agent.set_mdp(mdp)

    def action(self, state):
        if self.last_action is not None and np.random.random() < self.sticky_prob:
            return self.last_action, {"sticky_override": True}
        action, info = self.base_agent.action(state)
        self.last_action = action
        return action, info

def get_scenario_partner(scenario_idx, env, seed=42):
    if scenario_idx == 1:
        # Escenario 1: asymmetric_advantages + greedy_full_task
        p = GreedyFullTaskPolicy(ingredient="onion", avoid_teammate=True)
        p.set_mdp(env.mdp)
        p.set_agent_index(1)
        return p
    elif scenario_idx == 2:
        # Escenario 2: coordination_ring + greedy_full_task con sticky actions (15%)
        base = GreedyFullTaskPolicy(ingredient="onion", avoid_teammate=True)
        base.set_mdp(env.mdp)
        p = StickyActionWrapper(base, sticky_prob=0.15)
        p.set_mdp(env.mdp)
        p.set_agent_index(1)
        return p
    elif scenario_idx == 3:
        # Escenario 3: counter_circuit + greedy_full_task con sticky (10%) y random actions (10%)
        base = GreedyFullTaskPolicy(ingredient="onion", avoid_teammate=True)
        base.set_mdp(env.mdp)
        eps = EpsilonActionWrapper(base, random_action_prob=0.10, seed=seed)
        p = StickyActionWrapper(eps, sticky_prob=0.10)
        p.set_mdp(env.mdp)
        p.set_agent_index(1)
        return p
    elif scenario_idx == 4:
        # Escenario 4: layout_4 (scenario_4) + random_motion
        p = RandomMotionPolicy()
        if hasattr(p, "set_agent_index"):
            p.set_agent_index(1)
        return p
    return None

def evaluate_scenario(policy, scenario_idx, layout_name, n_episodes=10, device="cpu"):
    dynamics_overrides = load_dynamics_overrides()
    env = build_env_for_layout(layout_name, LAYOUTS_DIR, dynamics_overrides, shaped_reward_scale=0.0)
    
    sparse_scores = []
    for seed in range(100, 100 + n_episodes):
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs = env.reset()
        partner = get_scenario_partner(scenario_idx, env, seed=seed)
        if hasattr(partner, "set_agent_index"):
            partner.set_agent_index(1)
            
        total_sparse = 0.0
        for _step in range(400):
            obs0 = torch.FloatTensor(obs[0]).unsqueeze(0).to(device)
            with torch.no_grad():
                logits0, _ = policy.forward(obs0)
            act0 = int(logits0.argmax(dim=-1).item())

            partner_action, _ = partner.action(env.env.state)
            act1 = overcooked_action_to_index(partner_action) if partner_action in OVERCOOKED_ACTION_TO_INDEX else 4

            next_obs, rewards, dones, info = env.step([act0, act1])
            obs = next_obs
            sparse = info.get("sparse_r_by_agent", [0.0, 0.0])
            total_sparse += float(sparse[0] + sparse[1])
        sparse_scores.append(total_sparse)
        
    avg_sparse = float(np.mean(sparse_scores))
    avg_soups = avg_sparse / 20.0
    return avg_soups

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Official Competition Scenarios (Escenarios 1 to 4) on {device}")
    print(f"Testing 10 evaluation episodes per scenario to determine exact averages & qualification.\n")

    scenarios = [
        (1, "asymmetric_advantages", "Escenario 1 (asymmetric + pure greedy) [Target >= 1 soup]"),
        (2, "coordination_ring", "Escenario 2 (ring + sticky greedy 15%) [Target >= 2 soups avg]"),
        (3, "counter_circuit", "Escenario 3 (counter + sticky 10% & random 10%) [Target >= 2 soups avg]"),
        (4, "layout_4", "Escenario 4 (layout_4 + random_motion) [Target >= 1 soup avg]"),
    ]

    models = {
        "BC Baseline": "train/models/bc_agent_gnn.pt",
        "Master v1 (200k)": "train/models/ppo_agent_master.pt",
        "Master v2 (130k)": "train/models/ppo_agent_master_v2.pt",
        "Master v3 (50k)": "train/models/ppo_agent_master_v3.pt",
    }

    results = {label: {} for label in models.keys()}

    for label, path in models.items():
        if not os.path.exists(path):
            print(f"Skipping {label}: {path} not found")
            continue
        print(f"Loading {label} ({path})...")
        policy = load_policy_from_ckpt(path, device)

        for s_idx, l_name, desc in scenarios:
            print(f"  Running {desc}...", end="", flush=True)
            soups = evaluate_scenario(policy, s_idx, l_name, n_episodes=10, device=device)
            results[label][s_idx] = soups
            print(f" -> {soups:.2f} soups")

    print("\n" + "="*95)
    print(f"{'Scenario / Rules':<48} | {'BC Baseline':<11} | {'v1 (200k)':<10} | {'v2 (130k)':<10} | {'v3 (50k)':<10}")
    print("="*95)
    for s_idx, l_name, desc in scenarios:
        row = f"{desc:<48} | "
        for label in models.keys():
            val = results[label].get(s_idx, 0.0)
            row += f"{val:<11.2f} | " if "Baseline" in label else f"{val:<10.2f} | "
        print(row)
    print("="*95)

    # Print Qualification / Pass-Fail assessment
    print("\n--- QUALIFICATION CHECK ---")
    for s_idx, l_name, desc in scenarios:
        print(f"\n{desc}:")
        for label in models.keys():
            val = results[label].get(s_idx, 0.0)
            passed = False
            if s_idx == 1:
                passed = val >= 1.0
            elif s_idx in (2, 3):
                passed = val >= 2.0
            elif s_idx == 4:
                passed = val >= 1.0
            status = "PASS (Classified!)" if passed else "FAIL (Did not meet threshold)"
            print(f"  {label:<18}: {val:4.2f} soups -> {status}")

if __name__ == "__main__":
    main()

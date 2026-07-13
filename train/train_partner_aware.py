#!/usr/bin/env python3
"""
Partner-Aware PPO Fine-Tuning (`train_partner_aware.py`)

Fine-tunes the 1.7M PPO Production checkpoint directly against the official TA grading partners:
1. GreedyFullTaskPolicy (Pure) - 33.3% of episodes (Escenario 1 partner)
2. GreedyFullTaskPolicy (Sticky 15%) - 33.3% of episodes (Escenario 2 partner)
3. GreedyFullTaskPolicy (Sticky 10% + Random 10%) - 33.3% of episodes (Escenario 3 partner)

Features:
- No self-play (`agent vs self`) and no BC expert (`agent vs GNN`).
- Learner randomly swaps between Agent Index 0 and Agent Index 1 (50/50 per episode).
- Uses our verified clean telescoping PBRS (`gamma * phi_t - phi_t-1`) and BFS counter rerouting.
- Target layouts: asymmetric_advantages, coordination_ring, counter_circuit, simple_o, forced_coordination.
"""

import sys
import os
import argparse
import time
from pathlib import Path
import numpy as np
import torch

# Setup environment paths
PROJECT_ROOT = Path(r"C:\Users\SEBASTIAN\Documents\deep_project")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "overcooked"))
sys.path.insert(0, str(PROJECT_ROOT / "train"))
sys.path.insert(0, str(PROJECT_ROOT / "train" / "training"))

from overcooked.src.policy_wrappers import EpsilonActionWrapper, SafeActionWrapper
from overcooked.policies.basic_policies import GreedyFullTaskPolicy
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.agents.agent import Agent
from train.training.env import SelfPlayEnv
from train.training.models import GraphAttentionPPOPolicy, PPOPolicy as MLPPPOPolicy
from train.training.ppo import PPOUpdater


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


def sample_ta_partner(env, layout_name: str, seed: int = 42):
    """Return the EXACT partner used in the official TA grading scenario for this layout.

    Layout → Scenario match:
      asymmetric_advantages  → Scenario 1: pure greedy
      coordination_ring      → Scenario 2: sticky 15%
      counter_circuit        → Scenario 3: sticky 10% + random 10%
      simple_o / forced_*    → pure greedy (not graded, but keep realistic)
    """
    base_greedy = GreedyFullTaskPolicy(ingredient="onion", avoid_teammate=True, seed=seed)
    safe_greedy = SafeActionWrapper(base_greedy, max_action_time_ms=100)
    safe_greedy.set_mdp(env.mdp)

    if layout_name in ("coordination_ring",):
        # Scenario 2: sticky 15%
        sticky = StickyActionWrapper(safe_greedy, sticky_prob=0.15)
        sticky.set_mdp(env.mdp)
        return sticky, "sticky_15"
    elif layout_name in ("counter_circuit",):
        # Scenario 3: sticky 10% + random 10%
        epsilon = EpsilonActionWrapper(safe_greedy, random_action_prob=0.10, seed=seed)
        sticky_random = StickyActionWrapper(epsilon, sticky_prob=0.10)
        sticky_random.set_mdp(env.mdp)
        return sticky_random, "sticky10_random10"
    else:
        # Scenario 1 / simple_o / forced_coordination: pure greedy
        return safe_greedy, "pure"

# Shared relay counters for counter_circuit (inner wall y=2, reachable from both rings)
COUNTER_CIRCUIT_RELAY = frozenset([(2, 2), (3, 2), (4, 2), (5, 2), (6, 2)])


def collect_partner_aware_rollout(env, policy, n_steps, device, shaped_scale=1.0):
    """
    Collect n_steps of rollout where:
    - Learner is Agent `learner_idx` (either 0 or 1, flipped on reset).
    - Partner (`1 - learner_idx`) is a rule-based TA grading partner.
    - Transitions are recorded ONLY for the PPO learner (`policy`).
    """
    learner_obs = []
    learner_act = []
    learner_lp = []
    learner_rew = []
    learner_done = []
    learner_val = []
    learner_boot = []
    ep_rewards = []
    ep_soups = []

    obs = env.reset()
    
    # Assign roles: always alternate sides (50/50)
    learner_idx = int(np.random.choice([0, 1]))
    partner_idx = 1 - learner_idx
    partner_agent, _ = sample_ta_partner(env, layout_name=getattr(env, 'layout_name', ''), seed=int(np.random.randint(0, 1000000)))
    ep_sparse_total = 0.0
    ep_soup_count = 0

    def calc_phi(state):
        relay_items = sum(1 for pos in COUNTER_CIRCUIT_RELAY if state.has_object(pos))
        pot_onions = 0
        held_onions = 0
        for obj in state.objects.values():
            if obj.name == 'soup':
                pot_onions += len(obj.ingredients)
        for player in state.players:
            if player.has_object() and player.get_object().name == 'soup':
                held_onions += len(player.get_object().ingredients)
        return (5.0 * relay_items) + (10.0 * pot_onions) + (10.0 * held_onions)

    is_cc = getattr(env, 'layout_name', '') == 'counter_circuit'
    prev_phi = calc_phi(env.env.state) if is_cc else 0.0

    for step in range(n_steps):
        # 1. Get Learner action & value via PPO forward pass
        obs_tensor = torch.FloatTensor(np.array(obs[learner_idx])).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, val = policy.forward(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            lp = dist.log_prob(act)

        learner_a_int = int(act.item())

        # 2. Get Partner action via rule-based TA partner
        partner_a_action, _ = partner_agent.action(env.env.state)
        from overcooked_ai_py.mdp.actions import Action
        if partner_a_action in Action.ACTION_TO_INDEX:
            partner_a_int = Action.ACTION_TO_INDEX[partner_a_action]
        else:
            partner_a_int = 4  # stay fallback

        # Assemble joint action indices [a0, a1]
        joint_actions = [0, 0]
        joint_actions[learner_idx] = learner_a_int
        joint_actions[partner_idx] = partner_a_int

        # 3. Environment Step
        next_obs, rewards, dones, info = env.step(joint_actions)

        # Calculate exact reward for learner: sparse + c * shaped
        sparse_vec = info.get("sparse_r_by_agent", [0.0, 0.0])
        shaped_vec = info.get("shaped_r_by_agent", [0.0, 0.0])
        
        step_sparse = float(sparse_vec[learner_idx])
        
        if is_cc:
            # Disable old underlying library shaped rewards for CC to prevent compounding
            step_shaped = 0.0
        else:
            step_shaped = float(shaped_vec[learner_idx])
            
        learner_step_reward = step_sparse + shaped_scale * step_shaped

        # Potential-Based Reward Shaping (PBRS) for counter_circuit relay
        if is_cc:
            phi_t = calc_phi(env.env.state)
            
            # F_t = gamma * Phi(s_t) - Phi(s_{t-1})
            pbrs_reward = 0.99 * phi_t - prev_phi
            learner_step_reward += pbrs_reward
            
            prev_phi = phi_t

        ep_sparse_total += float(sparse_vec[0] + sparse_vec[1])
        if float(sparse_vec[0] + sparse_vec[1]) > 0:
            ep_soup_count += 1

        # Bootstrap value at horizon end if not terminal
        if dones[0] or step == n_steps - 1:
            with torch.no_grad():
                _, next_val = policy.forward(torch.FloatTensor(np.array(next_obs[learner_idx])).unsqueeze(0).to(device))
            boot_val = float(next_val.item())
        else:
            boot_val = 0.0

        learner_obs.append(obs[learner_idx])
        learner_act.append(learner_a_int)
        learner_lp.append(float(lp.item()))
        learner_rew.append(learner_step_reward)
        learner_done.append(bool(dones[0]))
        learner_val.append(float(val.item()))
        learner_boot.append(boot_val)

        if dones[0]:
            ep_rewards.append(ep_sparse_total)
            ep_soups.append(ep_soup_count)
            ep_sparse_total = 0.0
            ep_soup_count = 0
            
            obs = env.reset()
            learner_idx = int(np.random.choice([0, 1]))
            partner_idx = 1 - learner_idx
            partner_agent, _ = sample_ta_partner(env, layout_name=getattr(env, 'layout_name', ''), seed=int(np.random.randint(0, 1000000)))
            partner_agent.set_agent_index(partner_idx)
            
            # Reset PBRS tracking
            prev_phi = calc_phi(env.env.state) if is_cc else 0.0
        else:
            obs = next_obs

    # Compute GAE strictly for the learner
    gamma = 0.99
    lam = 0.95
    m = len(learner_rew)
    advantages = np.zeros(m, dtype=np.float32)
    returns = np.zeros(m, dtype=np.float32)
    gae = 0.0

    boot_arr = np.array(learner_boot, dtype=np.float32)
    val_arr = np.array(learner_val, dtype=np.float32)
    rew_arr = np.array(learner_rew, dtype=np.float32)
    done_arr = np.array(learner_done, dtype=np.bool_)

    for t in reversed(range(m)):
        if t == m - 1 or done_arr[t]:
            next_val = boot_arr[t]
            gae = 0.0
        else:
            next_val = val_arr[t + 1]
        delta = rew_arr[t] + gamma * next_val - val_arr[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + val_arr[t]

    obs_buf = np.array(learner_obs, dtype=np.float32)
    act_buf = np.array(learner_act, dtype=np.int64)
    lp_buf = np.array(learner_lp, dtype=np.float32)
    adv_buf = advantages
    ret_buf = returns
    val_buf = val_arr

    return obs_buf, act_buf, lp_buf, adv_buf, ret_buf, val_buf, ep_rewards, ep_soups


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-model", type=str, default="train/models/ppo_agent_1m7_production.pt")
    parser.add_argument("--output-model", type=str, default="train/models/ppo_agent_ta_finetuned.pt")
    parser.add_argument("--timesteps", type=int, default=250000)
    parser.add_argument("--rollout-len", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--shaped-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Partner-Aware PPO Fine-Tuning (Device: {device}) ===")
    print(f"Starting checkpoint: {args.start_model}")
    print(f"Output target: {args.output_model}")
    print(f"Timesteps: {args.timesteps:,} | Rollout length: {args.rollout_len} | LR: {args.lr}")
    print("-------------------------------------------------------------------------")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 3 graded TA scenarios: AA (25%), CR (25%), CC (50% — hardest, needs most practice)
    layouts = ["asymmetric_advantages", "coordination_ring", "counter_circuit", "counter_circuit"]
    envs = [SelfPlayEnv(layout_name=l, horizon=400) for l in layouts]
    obs_dim = envs[0].obs_dim

    # Detect architecture parameters
    ckpt_path = PROJECT_ROOT / args.start_model
    hidden = 256
    layers = 2
    arch = "mlp" # Default to MLP for bc_agent.pt compatibility
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            hidden = ckpt.get("hidden", 256) if isinstance(ckpt, dict) else 256
            layers = ckpt.get("layers", 2) if isinstance(ckpt, dict) else 2
            
            # Detect graph vs mlp
            if isinstance(ckpt, dict):
                if "arch" in ckpt:
                    arch = ckpt["arch"]
                elif any("graph_encoder" in k for k in ckpt.get("model_state", {}).keys()):
                    arch = "gnn"
        except Exception:
            pass

    # Initialize PPO policy and load checkpoint
    if arch in ["gnn", "attention", "graph"]:
        policy = GraphAttentionPPOPolicy(obs_dim=obs_dim, num_actions=6, hidden=hidden, layers=layers).to(device)
    else:
        policy = MLPPPOPolicy(obs_dim=obs_dim, num_actions=6, hidden=hidden, layers=layers).to(device)
    updater = PPOUpdater(policy, lr=args.lr, clip=args.clip, entropy_coef=args.entropy_coef,
                         ppo_epochs=args.epochs, batch_size=args.batch_size)
    
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            policy.load_state_dict(state_dict, strict=False)
            print(f"[Success] Loaded starting weights from {ckpt_path} (hidden={hidden}, layers={layers})")
        except Exception as e:
            print(f"[Warning] Could not load checkpoint ({e}). Starting from fresh weights.")
    else:
        print(f"[Warning] Checkpoint {ckpt_path} not found. Starting from fresh weights.")

    n_rollouts = args.timesteps // args.rollout_len
    start_time = time.time()

    print("\nStarting focused fine-tuning: AA(25%) | CR(25%) | CC(50%) + counter relay shaping...")
    for roll in range(1, n_rollouts + 1):
        # Sample layout right round robin / uniform
        layout_idx = (roll - 1) % len(envs)
        env = envs[layout_idx]

        obs_buf, act_buf, lp_buf, adv_buf, ret_buf, val_buf, ep_rews, ep_soups = collect_partner_aware_rollout(
            env, policy, args.rollout_len, device, shaped_scale=args.shaped_scale
        )

        # PPO Update
        metrics = updater.update(obs_buf, act_buf, lp_buf, adv_buf, ret_buf, val_buf)

        mean_sparse = np.mean(ep_rews) if ep_rews else 0.0
        mean_soups = np.mean(ep_soups) if ep_soups else 0.0
        total_steps = roll * args.rollout_len
        elapsed = time.time() - start_time
        fps = int(total_steps / elapsed) if elapsed > 0 else 0

        print(f"Step {total_steps:6d}/{args.timesteps} | Layout: {env.layout_name:22s} | "
              f"Sparse: {mean_sparse:5.1f} | Soups: {mean_soups:4.2f} | "
              f"v_loss: {metrics.get('value_loss', metrics.get('v_loss', 0.0)):.3f} | ent: {metrics.get('entropy', 0.0):.3f} | {fps} FPS")

        if roll % 25 == 0 or roll == n_rollouts:
            out_path = PROJECT_ROOT / args.output_model
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": policy.state_dict(),
                "obs_dim": obs_dim,
                "num_actions": 6,
                "arch": "gnn",
                "hidden": hidden,
                "layers": layers,
                "timesteps": total_steps
            }, out_path)
            print(f"  [Checkpoint] Saved fine-tuned model to {out_path}")

    print("\nFine-tuning completed successfully!")


if __name__ == "__main__":
    main()

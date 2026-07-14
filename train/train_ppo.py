"""PPO fine-tuner for the BC-trained agent.

Supports self-play and partner-aware (greedy) training with periodic eval + early stopping.

Usage:
    python train/train_ppo.py --bc-model train/models/bc_agent_gnn.pt \\
        --layouts counter_circuit,asymmetric_advantages,coordination_ring \\
        --partner-type greedy --timesteps 100000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import PPOPolicy, BCPolicy, GraphAttentionPPOPolicy
from ppo import PPOUpdater
from env import SelfPlayEnv, build_env_for_layout, load_dynamics_overrides
from src.constants import overcooked_action_to_index
from policies.basic_policies import GreedyFullTaskPolicy, RandomMotionPolicy

TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_DIR.parent
OVERCOOKED_DIR = PROJECT_ROOT / "overcooked"
LAYOUTS_DIR = OVERCOOKED_DIR / "layouts"


def init_from_bc(ppo_policy: PPOPolicy, bc_path: str):
    """Load BC weights into PPO policy's feature extractor + actor."""
    print(f"Loading BC model: {bc_path}")
    ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
    bc_state = ckpt["model_state"]

    ppo_state = ppo_policy.state_dict()
    loaded = 0
    for name, param in bc_state.items():
        if name in ppo_state and ppo_state[name].shape == param.shape:
            ppo_state[name] = param
            loaded += 1
    ppo_policy.load_state_dict(ppo_state)
    print(f"  Loaded {loaded} parameter tensors from BC")


def collect_rollouts(env, policy, n_steps, device, partner_policy=None,
                     frozen_self_partner=None):
    """Collect n_steps of experience.

    If partner_policy is set, agent 1 uses that (greedy) policy via state.
    If frozen_self_partner is set, agent 1 uses that frozen PPO policy via obs (FCP).
    Otherwise both agents use the live PPO policy (self-play).
    """
    has_partner = partner_policy is not None or frozen_self_partner is not None
    train_agents = 1 if has_partner else 2

    agent_obs = [[] for _ in range(train_agents)]
    agent_act = [[] for _ in range(train_agents)]
    agent_lp = [[] for _ in range(train_agents)]
    agent_rew = [[] for _ in range(train_agents)]
    agent_done = [[] for _ in range(train_agents)]
    agent_val = [[] for _ in range(train_agents)]
    agent_boot = [[] for _ in range(train_agents)]
    ep_rewards = []

    obs = env.reset()
    ep_reward = 0.0
    
    learner_idx = int(np.random.choice([0, 1])) if has_partner else 0
    partner_idx = 1 - learner_idx

    for step in range(n_steps):
        obs_tensor = torch.FloatTensor(np.array(obs)).to(device)
        with torch.no_grad():
            logits, values = policy.forward(obs_tensor)

        dist_learner = torch.distributions.Categorical(logits=logits[learner_idx:learner_idx+1])
        act_learner_t = dist_learner.sample()
        act_learner = int(act_learner_t.cpu().item())
        lp_learner = dist_learner.log_prob(act_learner_t)

        if partner_policy is not None:
            partner_policy.agent_index = partner_idx
            partner_action, _ = partner_policy.action(env.env.state)
            act_partner = overcooked_action_to_index(partner_action)
        elif frozen_self_partner is not None:
            obs_partner_frozen = torch.FloatTensor(obs[partner_idx]).unsqueeze(0).to(device)
            with torch.no_grad():
                frozen_logits, _ = frozen_self_partner.forward(obs_partner_frozen)
            frozen_dist = torch.distributions.Categorical(logits=frozen_logits)
            act_partner = int(frozen_dist.sample().item())
        else:
            dist_partner = torch.distributions.Categorical(logits=logits[partner_idx:partner_idx+1])
            act_partner_t = dist_partner.sample()
            act_partner = int(act_partner_t.cpu().item())
            lp_partner = dist_partner.log_prob(act_partner_t)

        joint_acts = [0, 0]
        joint_acts[learner_idx] = act_learner
        joint_acts[partner_idx] = act_partner

        next_obs, rewards, dones, info = env.step(joint_acts)

        if dones[0] or step == n_steps - 1:
            with torch.no_grad():
                _, next_vals = policy.forward(torch.FloatTensor(np.array(next_obs)).to(device))
            boot_raw = next_vals.cpu().numpy()
        else:
            boot_raw = np.zeros(2, dtype=np.float32)

        agent_obs[0].append(obs[learner_idx])
        agent_act[0].append(act_learner)
        agent_lp[0].append(float(lp_learner.item()))
        agent_rew[0].append(float(rewards[learner_idx]))
        agent_done[0].append(bool(dones[0]))
        agent_val[0].append(float(values[learner_idx].cpu().item()))
        agent_boot[0].append(float(boot_raw[learner_idx]))

        if not has_partner:
            agent_obs[1].append(obs[partner_idx])
            agent_act[1].append(act_partner)
            agent_lp[1].append(float(lp_partner.item()))
            agent_rew[1].append(float(rewards[partner_idx]))
            agent_done[1].append(bool(dones[1]))
            agent_val[1].append(float(values[partner_idx].cpu().item()))
            agent_boot[1].append(float(boot_raw[partner_idx]))

        ep_reward += rewards[learner_idx]

        if dones[0]:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            obs = env.reset()
            learner_idx = int(np.random.choice([0, 1])) if has_partner else 0
            partner_idx = 1 - learner_idx
            if partner_policy is not None:
                partner_policy.set_mdp(env.mdp)
        else:
            obs = next_obs

    # GAE independently per trained agent
    gamma = 0.99
    lam = 0.95
    all_obs, all_act, all_lp, all_adv, all_ret, all_val = [], [], [], [], [], []

    for i in range(train_agents):
        obs_arr = np.array(agent_obs[i], dtype=np.float32)
        act_arr = np.array(agent_act[i], dtype=np.int64)
        lp_arr = np.array(agent_lp[i], dtype=np.float32)
        rew_arr = np.array(agent_rew[i], dtype=np.float32)
        done_arr = np.array(agent_done[i], dtype=np.bool_)
        val_arr = np.array(agent_val[i], dtype=np.float32)
        boot_arr = np.array(agent_boot[i], dtype=np.float32)

        m = len(rew_arr)
        advantages = np.zeros(m, dtype=np.float32)
        returns = np.zeros(m, dtype=np.float32)
        gae = 0.0
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

        all_obs.append(obs_arr)
        all_act.append(act_arr)
        all_lp.append(lp_arr)
        all_adv.append(advantages)
        all_ret.append(returns)
        all_val.append(val_arr)

    obs_buf = np.concatenate(all_obs)
    act_buf = np.concatenate(all_act)
    log_prob_buf = np.concatenate(all_lp)
    adv_buf = np.concatenate(all_adv)
    ret_buf = np.concatenate(all_ret)
    val_buf = np.concatenate(all_val)

    return obs_buf, act_buf, log_prob_buf, adv_buf, ret_buf, val_buf, ep_rewards


def quick_eval(policy, envs, active_layout_names, partners, device, n_episodes=5):
    """Run quick eval: argmax + partner, n_episodes per layout.
    partners can be GreedyFullTaskPolicy objects (greedy mode) or None (self-play mode).
    Returns {layout: mean_sparse_reward}."""

    results = {}
    for layout, env, partner in zip(active_layout_names, envs, partners):
        sparse_scores = []
        for seed in range(42, 42 + n_episodes):
            np.random.seed(seed)
            torch.manual_seed(seed)
            obs = env.reset()
            total_sparse = 0.0
            for _step in range(400):
                obs0 = torch.FloatTensor(obs[0]).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits0, _ = policy.forward(obs0)
                act0 = int(logits0.argmax(dim=-1).item())
                # Agent 1: greedy partner or PPO self-play
                if partner is not None:
                    if hasattr(partner, "agent_index"):
                        partner.agent_index = 1
                    partner_action, _ = partner.action(env.env.state)
                    act1 = overcooked_action_to_index(partner_action)
                else:
                    obs1 = torch.FloatTensor(obs[1]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits1, _ = policy.forward(obs1)
                    act1 = int(logits1.argmax(dim=-1).item())
                next_obs, rewards, dones, info = env.step([act0, act1])
                sparse = info.get("sparse_r_by_agent", [0.0, 0.0])
                total_sparse += float(sparse[0] + sparse[1])
                obs = next_obs
                if dones[0]:
                    break
            sparse_scores.append(total_sparse)
        results[layout] = float(np.mean(sparse_scores))
    return results


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-model", type=str, default="train/models/bc_agent_gnn.pt")
    parser.add_argument("--output", type=str, default="train/models/ppo_agent.pt")
    parser.add_argument("--layouts", type=str, default="cramped_room,asymmetric_advantages,coordination_ring,simple_o,forced_coordination,counter_circuit")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--rollout-len", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--coordination-entropy-coef", type=float, default=0.05, help="Entropy coefficient used specifically on coordination bottleneck layouts")
    parser.add_argument("--kl-coef", type=float, default=0.1, help="KL divergence regularization coefficient against reference BC policy")
    parser.add_argument("--shaped-reward-scale", type=float, default=1.0)
    parser.add_argument("--layout-weights", type=str, default=None, help="Comma-separated layout sampling weights (e.g. '1,1,2,1,2,2') or 'uneven' preset")
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default="gnn", choices=["mlp", "gnn", "attention"], help="Model architecture: mlp or gnn/attention")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden/embed dimension size (default: detected from BC model or 128/256)")
    parser.add_argument("--layers", type=int, default=None, help="Number of layers/transformer blocks (default: detected from BC model or 2/3)")
    parser.add_argument("--topo-dim", type=int, default=0, help="Topology feature dimension (0=disabled, auto-detected from env/BC checkpoint if obs > 96)")
    parser.add_argument("--partner-type", type=str, default="greedy", choices=["self_play", "greedy"], help="Partner for agent 1: self_play (same PPO policy) or greedy (GreedyFullTaskPolicy)")
    parser.add_argument("--eval-interval", type=int, default=10000, help="Steps between quick evals (0 to disable)")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes per layout during quick_eval")
    parser.add_argument("--early-stop-patience", type=int, default=3, help="Stop training after N consecutive evals below BC baseline")
    parser.add_argument("--coordination-layouts", type=str, default=None, help="Comma-separated layouts to use self-play (no greedy partner). On these layouts both PPO agents explore together.")
    parser.add_argument("--solo-layouts", type=str, default=None, help="Comma-separated layouts where agent 1 is RandomMotionPolicy, forcing our agent to complete 100%% of the cooking solo.")
    parser.add_argument("--coordination-partner-model", type=str, default=None, help="Path to a BC model trained on user recordings. When set, coordination layouts use this as a frozen partner instead of FCP.")
    parser.add_argument("--bc-loss-coef", type=float, default=0.0, help="BC auxiliary loss coefficient. Adds supervised cross-entropy from consolidated dataset during PPO training (0=disabled).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dynamics_overrides = load_dynamics_overrides()
    layout_names = [l.strip() for l in args.layouts.split(",")]
    coordination_maps = [l.strip() for l in args.coordination_layouts.split(",") if l.strip()] if args.coordination_layouts else []
    solo_maps = [l.strip() for l in args.solo_layouts.split(",") if l.strip()] if args.solo_layouts else []
    if coordination_maps:
        print(f"Coordination self-play layouts ({len(coordination_maps)}): {coordination_maps}")
    if solo_maps:
        print(f"Solo cooking layouts against RandomMotionPolicy ({len(solo_maps)}): {solo_maps}")
    print(f"Training on layouts: {layout_names}")

    # Build environments
    envs = []
    active_layout_names = []
    for name in layout_names:
        try:
            scale = 0.0 if (coordination_maps and name in coordination_maps) else args.shaped_reward_scale
            env = build_env_for_layout(name, LAYOUTS_DIR, dynamics_overrides, shaped_reward_scale=scale)
            envs.append(env)
            active_layout_names.append(name)
            print(f"  {name}: obs_dim={env.obs_dim}")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    if not envs:
        print("No environments loaded!")
        return

    # Determine layout sampling probabilities
    env_probs = None
    if args.layout_weights is not None:
        if args.layout_weights.strip().lower() == "uneven":
            # Preset for the 6 canonical layouts: prioritize coordination bottlenecks (coordination_ring, forced_coordination, counter_circuit)
            preset_map = {
                "cramped_room": 1.0,
                "asymmetric_advantages": 1.0,
                "coordination_ring": 2.0,
                "simple_o": 1.0,
                "forced_coordination": 2.0,
                "counter_circuit": 2.0
            }
            raw_weights = [preset_map.get(name, 1.0) for name in active_layout_names]
        else:
            raw_weights = [float(w.strip()) for w in args.layout_weights.split(",")]
            if len(raw_weights) != len(envs):
                raise ValueError(f"Number of layout weights ({len(raw_weights)}) must match number of environments ({len(envs)})")
        
        env_probs = np.array(raw_weights, dtype=np.float64)
        env_probs /= env_probs.sum()
        prob_dict = {name: f"{p*100:.1f}%" for name, p in zip(active_layout_names, env_probs)}
        print(f"Using weighted layout sampling probabilities: {prob_dict}")

    obs_dim = envs[0].obs_dim
    num_actions = 6

    # -- Set up greedy partners (one per env, mapping to agent 1) --
    greedy_partners = []
    if args.partner_type == "greedy":
        for env in envs:
            g = GreedyFullTaskPolicy(ingredient="onion", avoid_teammate=True)
            g.set_mdp(env.mdp)
            g.agent_index = 1
            greedy_partners.append(g)
        print(f"Partner type: greedy (1 partner per layout)")
    else:
        greedy_partners = [None] * len(envs)
        print(f"Partner type: self_play")

    # Load coordination partner model if provided (for coordination layouts)
    coord_partner = None
    if args.coordination_partner_model and coordination_maps:
        coord_model_path = Path(args.coordination_partner_model)
        if coord_model_path.exists():
            print(f"Loading coordination partner from {coord_model_path}...")
            coord_ckpt = torch.load(str(coord_model_path), map_location="cpu", weights_only=False)
            coord_topo_dim = coord_ckpt.get("topo_dim", 0)
            coord_hidden = coord_ckpt.get("hidden", 128)
            coord_layers = coord_ckpt.get("layers", 2)
            coord_partner = GraphAttentionPPOPolicy(
                obs_dim, num_actions, hidden=coord_hidden, layers=coord_layers,
                topo_dim=coord_topo_dim,
            ).to(device)
            coord_partner.load_state_dict(coord_ckpt["model_state"], strict=False)
            coord_partner.eval()
            for p in coord_partner.parameters():
                p.requires_grad = False
            print(f"  Coordination partner loaded (hidden={coord_hidden}, topo_dim={coord_topo_dim})")
        else:
            print(f"  Coordination partner model not found at {coord_model_path}")

    # Determine architecture hyperparameters (CLI overrides -> BC model -> defaults)
    bc_path = Path(args.bc_model)
    bc_topo_dim = 0
    if bc_path.exists():
        try:
            ckpt = torch.load(str(bc_path), map_location="cpu", weights_only=False)
            bc_hidden = ckpt.get("hidden", 128 if args.arch in ["gnn", "attention"] else 256)
            bc_layers = ckpt.get("layers", 2 if args.arch in ["gnn", "attention"] else 3)
            bc_topo_dim = ckpt.get("topo_dim", 0)
            print(f"Detected architecture parameters from BC checkpoint: hidden={bc_hidden}, layers={bc_layers}, topo_dim={bc_topo_dim}")
        except Exception as e:
            print(f"Error reading BC checkpoint hyperparams: {e}. Using defaults.")
            bc_hidden = 128 if args.arch in ["gnn", "attention"] else 256
            bc_layers = 2 if args.arch in ["gnn", "attention"] else 3
    else:
        bc_hidden = 128 if args.arch in ["gnn", "attention"] else 256
        bc_layers = 2 if args.arch in ["gnn", "attention"] else 3

    hidden = args.hidden if args.hidden is not None else bc_hidden
    layers = args.layers if args.layers is not None else bc_layers
    topo_dim = args.topo_dim if args.topo_dim > 0 else (obs_dim - 96 if obs_dim > 96 else bc_topo_dim)
    if topo_dim > 0:
        print(f"Topology features: topo_dim={topo_dim}")

    bc_policy = None
    if args.arch in ["gnn", "attention"]:
        print(f"Instantiating Relational Graph Attention Network (GNN/Attention) PPO architecture (hidden={hidden}, layers={layers})...")
        policy = GraphAttentionPPOPolicy(obs_dim, num_actions, hidden=hidden, layers=layers, topo_dim=topo_dim).to(device)
        if bc_path.exists():
            bc_policy = GraphAttentionPPOPolicy(obs_dim, num_actions, hidden=hidden, layers=layers, topo_dim=topo_dim).to(device)
    else:
        print(f"Instantiating MLP PPO architecture (hidden={hidden}, layers={layers})...")
        policy = PPOPolicy(obs_dim, num_actions, hidden=hidden, layers=layers).to(device)
        if bc_path.exists():
            bc_policy = PPOPolicy(obs_dim, num_actions, hidden=hidden, layers=layers).to(device)

    if bc_path.exists():
        init_from_bc(policy, str(bc_path))
        if bc_policy is not None:
            init_from_bc(bc_policy, str(bc_path))
            bc_policy.eval()
            for p in bc_policy.parameters():
                p.requires_grad = False
    else:
        print(f"BC model not found at {args.bc_model}, training from scratch")

    # Load BC dataset for auxiliary supervised loss
    bc_data = None
    if args.bc_loss_coef > 0:
        bc_data_path = Path("train/data/consolidated.npz")
        if bc_data_path.exists():
            bc_data_np = np.load(str(bc_data_path))
            bc_obs = torch.FloatTensor(bc_data_np["obs"]).to(device)
            bc_actions = torch.LongTensor(bc_data_np["actions"]).to(device)
            bc_weights = torch.FloatTensor(bc_data_np["weights"]).to(device)
            bc_data = {"obs": bc_obs, "actions": bc_actions, "weights": bc_weights}
            print(f"Loaded BC data: {len(bc_actions)} timesteps for BC auxiliary loss (coef={args.bc_loss_coef})")
        else:
            print(f"BC dataset not found at {bc_data_path}")

    ppo = PPOUpdater(
        policy, lr=args.lr, clip=args.clip, entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs, batch_size=args.batch_size,
        bc_policy=bc_policy, kl_coef=args.kl_coef,
        bc_data=bc_data, bc_coef=args.bc_loss_coef,
    )

    # -- Run BC baseline eval before any PPO updates --
    bc_baseline = None
    if args.eval_interval > 0:
        print("\nRunning BC baseline eval...")
        eval_partners = []
        for name in active_layout_names:
            if solo_maps and name in solo_maps:
                eval_partners.append(RandomMotionPolicy())
            elif args.partner_type == "greedy":
                eval_partners.append(greedy_partners[active_layout_names.index(name)])
            else:
                eval_partners.append(None)
        bc_results = quick_eval(policy, envs, active_layout_names, eval_partners, device, n_episodes=args.eval_episodes)
        avg_sparse = np.mean(list(bc_results.values()))
        print(f"BC baseline: { {k: f'{v/20:.1f} soups' for k, v in bc_results.items()} }")
        bc_baseline = avg_sparse
        print(f"BC baseline avg sparse: {bc_baseline:.1f} ({bc_baseline/20:.1f} soups)")

    history = {
        "timesteps": [], "mean_reward": [], "policy_loss": [], "value_loss": [], "entropy": [],
        "layout_history": defaultdict(list), "eval_history": [],
    }
    total_steps = 0
    env_idx = 0
    start_time = time.time()
    best_eval_score = bc_baseline or 0.0
    best_state = None
    bad_eval_streak = 0

    # -- Fictitious Co-Play: checkpoint buffer for coordination-layout partners --
    # Keeps sparse checkpoints spanning near-BC -> early-PPO -> current,
    # breaking the identical-twin correlation on self-play layouts.
    checkpoint_buffer = []       # list of state_dicts, kept sparse
    partner_pool = None          # frozen GraphAttentionPPOPolicy for past self-partner
    # Seed with BC model at init so FCP has diversity from step 0
    checkpoint_buffer.append({k: v.clone() for k, v in policy.state_dict().items()})

    while total_steps < args.timesteps:
        if env_probs is not None:
            chosen_idx = int(np.random.choice(len(envs), p=env_probs))
        else:
            chosen_idx = env_idx % len(envs)
            env_idx += 1

        env = envs[chosen_idx]
        current_layout = active_layout_names[chosen_idx]
        # Use self-play on coordination layouts (greedy partner gets 0 soups there)
        is_coord = coordination_maps and current_layout in coordination_maps
        is_solo = solo_maps and current_layout in solo_maps
        if is_solo:
            # Partner is RandomMotionPolicy — agent MUST do 100% of the cooking completely solo!
            obs, acts, old_log_probs, advs, returns, vals, ep_rewards = collect_rollouts(
                env, policy, args.rollout_len, device, partner_policy=RandomMotionPolicy(),
            )
        elif is_coord:
            if coord_partner is not None and random.random() < 0.5:
                # Use coordination partner trained on user recordings 50% of the time —
                # it already knows how to coordinate (pass items, take turns)
                obs, acts, old_log_probs, advs, returns, vals, ep_rewards = collect_rollouts(
                    env, policy, args.rollout_len, device, partner_policy=None,
                    frozen_self_partner=coord_partner,
                )
            else:
                # FCP: sample a past checkpoint 50% of the remaining time; otherwise use live twin self-play
                fcp_partner = None
                if checkpoint_buffer and random.random() < 0.5:
                    past_state = random.choice(checkpoint_buffer)
                    if partner_pool is None:
                        partner_pool = GraphAttentionPPOPolicy(obs_dim, num_actions, hidden=hidden,
                                                                layers=layers, topo_dim=topo_dim).to(device)
                    partner_pool.load_state_dict(past_state)
                    partner_pool.eval()
                    fcp_partner = partner_pool
                obs, acts, old_log_probs, advs, returns, vals, ep_rewards = collect_rollouts(
                    env, policy, args.rollout_len, device, partner_policy=None,
                    frozen_self_partner=fcp_partner,
                )
        else:
            partner = greedy_partners[chosen_idx] if args.partner_type == "greedy" else None
            obs, acts, old_log_probs, advs, returns, vals, ep_rewards = collect_rollouts(
                env, policy, args.rollout_len, device, partner_policy=partner,
            )

        # Apply State-Dependent Dynamic Exploration (high on bottlenecks and solo maps, low on open maps)
        if is_coord or is_solo:
            ppo.entropy_coef = args.coordination_entropy_coef
        else:
            ppo.entropy_coef = args.entropy_coef

        stats = ppo.update(obs, acts, old_log_probs, advs, returns, vals)

        total_steps += args.rollout_len
        mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0

        history["timesteps"].append(total_steps)
        history["mean_reward"].append(mean_r)
        history["policy_loss"].append(stats["policy_loss"])
        history["value_loss"].append(stats["value_loss"])
        history["entropy"].append(stats["entropy"])
        history["layout_history"][current_layout].append({"step": total_steps, "reward": mean_r})

        elapsed = time.time() - start_time
        print(f"Steps {total_steps:>8} | layout={current_layout:<22} | reward={mean_r:>6.1f} | "
              f"policy={stats['policy_loss']:.4f} | value={stats['value_loss']:.4f} | "
              f"entropy={stats['entropy']:.4f} | {elapsed:.0f}s", flush=True)

        # Periodic evaluation + early stopping checkpoint
        if args.eval_interval > 0 and total_steps % args.eval_interval < args.rollout_len:
            print(f"\n--- Eval at step {total_steps} ---")
            # Per-layout eval partners: RandomMotionPolicy on solo maps, GreedyFullTaskPolicy elsewhere
            eval_partners = []
            for name in active_layout_names:
                if solo_maps and name in solo_maps:
                    eval_partners.append(RandomMotionPolicy())
                elif args.partner_type == "greedy":
                    eval_partners.append(greedy_partners[active_layout_names.index(name)])
                else:
                    eval_partners.append(None)
            eval_results = quick_eval(policy, envs, active_layout_names, eval_partners, device, n_episodes=args.eval_episodes)
            avg_sparse = np.mean(list(eval_results.values()))
            avg_soups = avg_sparse / 20.0
            print(f"  eval avg sparse: {avg_sparse:.1f} ({avg_soups:.1f} soups)")

            history["eval_history"].append({
                "step": total_steps, "avg_sparse": float(avg_sparse),
                "per_layout": {k: float(v) for k, v in eval_results.items()}
            })

            # FCP checkpoint buffer: keep sparse snapshots (near-BC, early-PPO, current)
            # to break identical-twin correlation on self-play layouts.
            checkpoint_buffer.append({k: v.clone() for k, v in policy.state_dict().items()})
            if len(checkpoint_buffer) > 3:
                checkpoint_buffer = [checkpoint_buffer[0], checkpoint_buffer[1], checkpoint_buffer[-1]]

            # Track best
            if avg_sparse > best_eval_score:
                best_eval_score = avg_sparse
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                bad_eval_streak = 0
                print(f"  ** new best! avg={avg_soups:.1f} soups **")
            else:
                bad_eval_streak += 1
                if bc_baseline is not None and avg_sparse < bc_baseline:
                    print(f"  below BC baseline ({bc_baseline/20:.1f} soups), streak={bad_eval_streak}/{args.early_stop_patience}")

            # Early stopping
            if args.early_stop_patience > 0 and bad_eval_streak >= args.early_stop_patience:
                print(f"\nEarly stopping triggered after {bad_eval_streak} consecutive evals below best.")
                break

        # Save checkpoint
        if total_steps % args.save_interval < args.rollout_len or total_steps >= args.timesteps:
            save_path = Path(args.output)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": policy.state_dict(),
                "obs_dim": obs_dim,
                "num_actions": num_actions,
                "hidden": hidden,
                "layers": layers,
                "arch": args.arch,
                "topo_dim": topo_dim,
                "history": history,
                "args": vars(args),
            }, save_path)
            print(f"  Saved checkpoint: {save_path}")

    # -- Final: save best model --
    if best_state is not None:
        save_path = Path(args.output).with_name(Path(args.output).stem + "_best.pt")
        torch.save({
            "model_state": best_state,
            "obs_dim": obs_dim, "num_actions": num_actions,
            "hidden": hidden, "layers": layers, "arch": args.arch,
            "topo_dim": topo_dim, "history": history, "args": vars(args),
        }, save_path)
        print(f"\nBest model saved: {save_path} (eval_score={best_eval_score/20:.1f} soups)")

    # Final save
    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": policy.state_dict(),
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "hidden": hidden,
        "layers": layers,
        "arch": args.arch,
        "topo_dim": topo_dim,
        "history": history,
        "args": vars(args),
    }, save_path)
    print(f"\nFinal model saved: {save_path}")

    hist_path = save_path.with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved: {hist_path}")


if __name__ == "__main__":
    main()
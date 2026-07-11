"""PPO self-play fine-tuner for the BC-trained agent.

Usage:
    cd overcooked
    python ..\train\train_ppo.py --bc-model train/models/bc_agent.pt --layouts cramped_room --timesteps 500000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "training"))
from models import PPOPolicy, BCPolicy
from ppo import PPOUpdater
from env import SelfPlayEnv, build_env_for_layout, load_dynamics_overrides

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


def collect_rollouts(env, policy, n_steps, device, deterministic=False):
    """Collect n_steps of self-play experience."""
    obs_buf = []
    act_buf = []
    log_prob_buf = []
    rew_buf = []
    done_buf = []
    val_buf = []
    ep_rewards = []

    obs = env.reset()
    ep_reward = 0.0
    steps_this_ep = 0

    for step in range(n_steps):
        obs_tensor = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            logits, values = policy.forward(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        actions_np = actions.cpu().numpy()
        next_obs, rewards, dones, info = env.step(actions_np)

        obs_buf.extend(obs)
        act_buf.extend(actions_np.tolist())
        log_prob_buf.extend(log_probs.cpu().numpy().tolist())
        rew_buf.extend(rewards)
        done_buf.extend(dones)
        val_buf.extend(values.cpu().numpy().tolist())

        ep_reward += rewards[0]
        steps_this_ep += 1

        if dones[0]:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            steps_this_ep = 0
            obs = env.reset()
        else:
            obs = next_obs

    obs_buf = np.array(obs_buf, dtype=np.float32)
    act_buf = np.array(act_buf, dtype=np.int64)
    log_prob_buf = np.array(log_prob_buf, dtype=np.float32)
    rew_buf = np.array(rew_buf, dtype=np.float32)
    done_buf = np.array(done_buf, dtype=np.bool_)
    val_buf = np.array(val_buf, dtype=np.float32)

    # Compute returns and advantages (GAE)
    gamma = 0.99
    lam = 0.95
    n = len(rew_buf)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1 or done_buf[t]:
            next_val = 0.0
        else:
            next_val = val_buf[t + 1]
        delta = rew_buf[t] + gamma * next_val - val_buf[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + val_buf[t]

    return obs_buf, act_buf, log_prob_buf, advantages, returns, val_buf, ep_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-model", type=str, default="train/models/bc_agent.pt")
    parser.add_argument("--output", type=str, default="train/models/ppo_agent.pt")
    parser.add_argument("--layouts", type=str, default="cramped_room,asymmetric_advantages,simple_o")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--rollout-len", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--shaped-reward-scale", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dynamics_overrides = load_dynamics_overrides()
    layout_names = args.layouts.split(",")
    print(f"Training on layouts: {layout_names}")

    # Build environments
    envs = []
    for name in layout_names:
        name = name.strip()
        try:
            env = build_env_for_layout(name, LAYOUTS_DIR, dynamics_overrides, shaped_reward_scale=args.shaped_reward_scale)
            envs.append(env)
            print(f"  {name}: obs_dim={env.obs_dim}")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    if not envs:
        print("No environments loaded!")
        return

    obs_dim = envs[0].obs_dim
    num_actions = 6

    policy = PPOPolicy(obs_dim, num_actions, hidden=256, layers=3).to(device)

    if Path(args.bc_model).exists():
        init_from_bc(policy, args.bc_model)
    else:
        print(f"BC model not found at {args.bc_model}, training from scratch")

    ppo = PPOUpdater(
        policy, lr=args.lr, clip=args.clip, entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs, batch_size=args.batch_size,
    )

    history = {"timesteps": [], "mean_reward": [], "policy_loss": [], "value_loss": [], "entropy": []}
    total_steps = 0
    env_idx = 0
    start_time = time.time()

    while total_steps < args.timesteps:
        env = envs[env_idx % len(envs)]
        env_idx += 1

        obs, acts, old_log_probs, advs, returns, vals, ep_rewards = collect_rollouts(
            env, policy, args.rollout_len, device,
        )

        stats = ppo.update(obs, acts, old_log_probs, advs, returns, vals)

        total_steps += args.rollout_len
        mean_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0

        history["timesteps"].append(total_steps)
        history["mean_reward"].append(mean_r)
        history["policy_loss"].append(stats["policy_loss"])
        history["value_loss"].append(stats["value_loss"])
        history["entropy"].append(stats["entropy"])

        elapsed = time.time() - start_time
        print(f"Steps {total_steps:>8} | reward={mean_r:>6.1f} | "
              f"policy={stats['policy_loss']:.4f} | value={stats['value_loss']:.4f} | "
              f"entropy={stats['entropy']:.4f} | {elapsed:.0f}s")

        if total_steps % args.save_interval < args.rollout_len:
            save_path = Path(args.output)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": policy.state_dict(),
                "obs_dim": obs_dim,
                "num_actions": num_actions,
                "hidden": 256,
                "layers": 3,
                "history": history,
                "args": vars(args),
            }, save_path)
            print(f"  Saved checkpoint: {save_path}")

    # Final save
    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": policy.state_dict(),
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "hidden": 256,
        "layers": 3,
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
"""PPO update core."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class PPOUpdater:
    def __init__(self, policy, lr=3e-4, clip=0.2, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5, ppo_epochs=4, batch_size=64,
                 bc_policy=None, kl_coef=0.05,
                 bc_data=None, bc_coef=0.0):
        self.policy = policy
        self.bc_policy = bc_policy
        self.kl_coef = kl_coef
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.bc_data = bc_data  # dict with "obs", "actions", "weights" tensors on device
        self.bc_coef = bc_coef
        self.bc_batch_size = min(256, len(bc_data["obs"])) if bc_data else 0

    def update(self, obs, actions, old_log_probs, advantages, returns, values):
        obs = torch.FloatTensor(obs)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        old_values = torch.FloatTensor(values)

        if obs.device != self.policy.actor.weight.device:
            device = self.policy.actor.weight.device
            obs = obs.to(device)
            actions = actions.to(device)
            old_log_probs = old_log_probs.to(device)
            advantages = advantages.to(device)
            returns = returns.to(device)
            old_values = old_values.to(device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(obs)
        idx = np.arange(n)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                b_obs = obs[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]
                b_old_values = old_values[batch_idx]

                log_probs, entropy, new_values, curr_logits = self.policy.evaluate(b_obs, b_actions)

                ratio = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(new_values, b_returns)

                kl_loss = 0.0
                if self.bc_policy is not None:
                    with torch.no_grad():
                        bc_logits, _ = self.bc_policy.forward(b_obs)
                    bc_log_probs = nn.functional.log_softmax(bc_logits, dim=-1)
                    curr_log_probs = nn.functional.log_softmax(curr_logits, dim=-1)
                    curr_probs = nn.functional.softmax(curr_logits, dim=-1)
                    # Exact reverse KL: D_KL(pi_theta || pi_BC) = sum(pi_theta * (log_pi_theta - log_pi_bc))
                    kl_loss = torch.sum(curr_probs * (curr_log_probs - bc_log_probs), dim=-1).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean() + self.kl_coef * kl_loss

                # BC auxiliary loss: supervise on recorded demonstrations
                bc_loss_val = torch.tensor(0.0, device=loss.device)
                if self.bc_data is not None and self.bc_coef > 0:
                    bc_batch_idx = np.random.choice(
                        len(self.bc_data["obs"]), self.bc_batch_size, replace=False
                    )
                    b_bc_obs = self.bc_data["obs"][bc_batch_idx]
                    b_bc_actions = self.bc_data["actions"][bc_batch_idx]
                    b_bc_logits = self.policy.forward(b_bc_obs)[0]
                    bc_per_sample = nn.functional.cross_entropy(b_bc_logits, b_bc_actions, reduction="none")
                    if "weights" in self.bc_data:
                        b_bc_weights = self.bc_data["weights"][bc_batch_idx]
                        bc_loss_val = (bc_per_sample * b_bc_weights).mean()
                    else:
                        bc_loss_val = bc_per_sample.mean()

                loss = loss + self.bc_coef * bc_loss_val

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                if isinstance(kl_loss, torch.Tensor):
                    total_kl += kl_loss.item()

        n_updates = self.ppo_epochs * (n // self.batch_size + 1)
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
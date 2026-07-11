"""PPO update core."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class PPOUpdater:
    def __init__(self, policy, lr=3e-4, clip=0.2, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5, ppo_epochs=4, batch_size=64):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

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

                log_probs, entropy, new_values = self.policy.evaluate(b_obs, b_actions)

                ratio = torch.exp(log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(new_values, b_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        n_updates = self.ppo_epochs * (n // self.batch_size + 1)
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
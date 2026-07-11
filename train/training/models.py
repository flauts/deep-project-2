"""Model definitions for BC and PPO."""

from __future__ import annotations

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """MLP actor: obs -> 6 action logits."""

    def __init__(self, obs_dim: int, num_actions: int = 6, hidden: int = 256, layers: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        modules = []
        in_dim = obs_dim
        for _ in range(layers):
            modules.append(nn.Linear(in_dim, hidden))
            modules.append(nn.ReLU())
            in_dim = hidden
        self.feature = nn.Sequential(*modules)
        self.actor = nn.Linear(in_dim, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.feature(obs))

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        logits = self.forward(obs)
        if deterministic:
            return logits.argmax(dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)


class PPOPolicy(nn.Module):
    """Actor-Critic for PPO: shared feature extractor + actor head + value head."""

    def __init__(self, obs_dim: int, num_actions: int = 6, hidden: int = 256, layers: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        modules = []
        in_dim = obs_dim
        for _ in range(layers):
            modules.append(nn.Linear(in_dim, hidden))
            modules.append(nn.ReLU())
            in_dim = hidden
        self.feature = nn.Sequential(*modules)
        self.actor = nn.Linear(in_dim, num_actions)
        self.critic = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor):
        feat = self.feature(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        feat = self.feature(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, value
        return action, None, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        feat = self.feature(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value
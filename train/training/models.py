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
            log_prob = None
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        feat = self.feature(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value, logits


class RelationalGraphAttentionExtractor(nn.Module):
    """Decomposes Overcooked featurized observations into 10 relational entity nodes

    and processes them via Multi-Head Self-Attention (Relational Graph Message Passing).
    """

    def __init__(self, obs_dim: int = 96, embed_dim: int = 128, num_heads: int = 4, layers: int = 2,
                 topo_dim: int = 0):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.topo_dim = topo_dim

        # Specific node encoders for 96-dim featurized vector
        self.proj_self = nn.Linear(14, embed_dim)
        self.proj_mate = nn.Linear(14, embed_dim)
        self.proj_onion = nn.Linear(4, embed_dim)
        self.proj_tomato = nn.Linear(4, embed_dim)
        self.proj_dish = nn.Linear(4, embed_dim)
        self.proj_soup = nn.Linear(8, embed_dim)
        self.proj_serving = nn.Linear(4, embed_dim)
        self.proj_counter = nn.Linear(4, embed_dim)
        self.proj_pot0 = nn.Linear(20, embed_dim)
        self.proj_pot1 = nn.Linear(20, embed_dim)

        # Topology feature node (layout-level static features)
        if topo_dim > 0:
            self.proj_topo = nn.Linear(topo_dim, embed_dim)

        # General fallback projection if obs_dim != 96
        num_nodes = 10 + (1 if topo_dim > 0 else 0)
        self.fallback_proj = nn.Linear(obs_dim, num_nodes * embed_dim)

        # Graph Attention / Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.graph_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out_dim = embed_dim * (4 if topo_dim > 0 else 3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        has_topo = self.topo_dim > 0 and obs.shape[-1] == 96 + self.topo_dim

        if obs.shape[-1] == 96 or has_topo:
            entity_obs = obs[:, :96]
            n_self = self.proj_self(torch.cat([entity_obs[:, 0:8], entity_obs[:, 42:46], entity_obs[:, 94:96]], dim=-1)).unsqueeze(1)
            n_mate = self.proj_mate(torch.cat([entity_obs[:, 46:54], entity_obs[:, 88:92], entity_obs[:, 92:94]], dim=-1)).unsqueeze(1)
            n_onion = self.proj_onion(torch.cat([entity_obs[:, 8:10], entity_obs[:, 54:56]], dim=-1)).unsqueeze(1)
            n_tomato = self.proj_tomato(torch.cat([entity_obs[:, 10:12], entity_obs[:, 56:58]], dim=-1)).unsqueeze(1)
            n_dish = self.proj_dish(torch.cat([entity_obs[:, 12:14], entity_obs[:, 58:60]], dim=-1)).unsqueeze(1)
            n_soup = self.proj_soup(torch.cat([entity_obs[:, 14:18], entity_obs[:, 60:64]], dim=-1)).unsqueeze(1)
            n_serv = self.proj_serving(torch.cat([entity_obs[:, 18:20], entity_obs[:, 64:66]], dim=-1)).unsqueeze(1)
            n_cntr = self.proj_counter(torch.cat([entity_obs[:, 20:22], entity_obs[:, 66:68]], dim=-1)).unsqueeze(1)
            n_pot0 = self.proj_pot0(torch.cat([entity_obs[:, 22:32], entity_obs[:, 68:78]], dim=-1)).unsqueeze(1)
            n_pot1 = self.proj_pot1(torch.cat([entity_obs[:, 32:42], entity_obs[:, 78:88]], dim=-1)).unsqueeze(1)

            nodes = torch.cat([n_self, n_mate, n_onion, n_tomato, n_dish, n_soup, n_serv, n_cntr, n_pot0, n_pot1], dim=1)
        else:
            num_nodes = 10
            nodes = self.fallback_proj(obs).view(B, num_nodes, self.embed_dim)

        graph_out = self.graph_encoder(nodes)
        self_summary = graph_out[:, 0, :]
        mate_summary = graph_out[:, 1, :]
        global_summary = graph_out.mean(dim=1)
        result = torch.cat([self_summary, mate_summary, global_summary], dim=-1)

        # Topology features bypass attention: concatenate directly with graph output
        if has_topo:
            topo_obs = obs[:, 96:96 + self.topo_dim]
            topo_emb = self.proj_topo(topo_obs)  # (B, embed_dim)
            result = torch.cat([result, topo_emb], dim=-1)
        return result


class GraphAttentionBCPolicy(nn.Module):
    """GNN / Spatial Attention actor for BC: obs -> 6 action logits."""

    def __init__(self, obs_dim: int, num_actions: int = 6, hidden: int = 128, layers: int = 2,
                 topo_dim: int = 0):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.topo_dim = topo_dim
        self.feature = RelationalGraphAttentionExtractor(obs_dim=obs_dim, embed_dim=hidden, layers=layers,
                                                          topo_dim=topo_dim)
        self.actor = nn.Linear(self.feature.out_dim, num_actions)

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


class GraphAttentionPPOPolicy(nn.Module):
    """GNN / Spatial Attention Actor-Critic for PPO."""

    def __init__(self, obs_dim: int, num_actions: int = 6, hidden: int = 128, layers: int = 2,
                 topo_dim: int = 0):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.topo_dim = topo_dim
        self.feature = RelationalGraphAttentionExtractor(obs_dim=obs_dim, embed_dim=hidden, layers=layers,
                                                          topo_dim=topo_dim)
        self.actor = nn.Linear(self.feature.out_dim, num_actions)
        self.critic = nn.Linear(self.feature.out_dim, 1)

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
            log_prob = None
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        feat = self.feature(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value, logits
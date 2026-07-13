"""Trained agent that loads a BC or PPO model and plays in the Overcooked game.

Follows the policies/template.py interface:
    __init__(self, config: dict)
    reset(self)
    act(self, obs) -> int

Action convention:
    0 = north/up, 1 = south/down, 2 = east/right,
    3 = west/left, 4 = stay, 5 = interact

Config options (in the YAML policy config):
    model_path: path to .pt file (default: ../train/models/bc_agent.pt)
    deterministic: use argmax instead of sampling (default: true)
    device: "cuda" or "cpu" (default: cuda if available, else cpu)

Usage in eval YAML:
    agent_0:
      type: python_class
      path: policies/trained_agent.py
      class_name: TrainedAgent
      config:
        model_path: ../train/models/bc_agent.pt
        deterministic: true
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


class _RelationalGraphAttentionExtractor(nn.Module if HAS_TORCH else object):
    def __init__(self, obs_dim: int = 96, embed_dim: int = 128, num_heads: int = 4, layers: int = 2,
                 topo_dim: int = 0):
        if HAS_TORCH:
            super().__init__()
            self.obs_dim = obs_dim
            self.embed_dim = embed_dim
            self.topo_dim = topo_dim
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
            if topo_dim > 0:
                self.proj_topo = nn.Linear(topo_dim, embed_dim)
            num_nodes = 10 + (1 if topo_dim > 0 else 0)
            self.fallback_proj = nn.Linear(obs_dim, num_nodes * embed_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
                dropout=0.1, activation='relu', batch_first=True
            )
            self.graph_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.out_dim = embed_dim * (4 if topo_dim > 0 else 3)
        else:
            super().__init__()

    def forward(self, obs):
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
        result = torch.cat([graph_out[:, 0, :], graph_out[:, 1, :], graph_out.mean(dim=1)], dim=-1)
        if has_topo:
            topo_obs = obs[:, 96:96 + self.topo_dim]
            topo_emb = self.proj_topo(topo_obs)
            result = torch.cat([result, topo_emb], dim=-1)
        return result


class _GNNPolicy(nn.Module if HAS_TORCH else object):
    def __init__(self, obs_dim: int, num_actions: int = 6, hidden: int = 128, layers: int = 2,
                 topo_dim: int = 0):
        if HAS_TORCH:
            super().__init__()
            self.feature = _RelationalGraphAttentionExtractor(obs_dim=obs_dim, embed_dim=hidden,
                                                               layers=layers, topo_dim=topo_dim)
            self.actor = nn.Linear(self.feature.out_dim, num_actions)
        else:
            super().__init__()

    def forward(self, obs):
        return self.actor(self.feature(obs))

    def logits(self, obs):
        return self.forward(obs)


class _MLPPolicy(nn.Module if HAS_TORCH else object):
    def __init__(self, obs_dim: int, num_actions: int = 6, hidden: int = 256, layers: int = 3):
        if HAS_TORCH:
            super().__init__()
            modules = []
            in_dim = obs_dim
            for _ in range(layers):
                modules.append(nn.Linear(in_dim, hidden))
                modules.append(nn.ReLU())
                in_dim = hidden
            self.feature = nn.Sequential(*modules)
            self.actor = nn.Linear(in_dim, num_actions)
        else:
            super().__init__()

    def forward(self, obs):
        feat = self.feature(obs)
        return self.actor(feat)

    def logits(self, obs):
        return self.forward(obs)


class TrainedAgent:
    """Trained agent for Overcooked. Loads a .pt model and returns action indices."""

    def __init__(self, config=None):
        self.config = config or {}

        env_yml_dir = Path(__file__).resolve().parent.parent
        default_model = env_yml_dir / "train" / "models" / "bc_agent.pt"
        self.model_path = str(self.config.get("model_path", default_model))
        self.deterministic = bool(self.config.get("deterministic", True))
        self.device_name = self.config.get("device", "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")

        self.model = None
        self.obs_dim = 96
        self.num_actions = 6
        self.topo_dim = 0
        self.topo_features = None

        if not HAS_TORCH:
            print("[TrainedAgent] WARNING: torch not available. Will return random actions.")
            self.rng = np.random.default_rng(self.config.get("seed", 0))
            return

        self.rng = np.random.default_rng(self.config.get("seed", 0))
        self.device = torch.device(self.device_name)

        model_path = Path(self.model_path)
        if model_path.exists():
            self._load_model(model_path)
        else:
            print(f"[TrainedAgent] WARNING: model not found at {model_path}. Will return random actions.")

        # Compute topology features if layout_name is provided and model supports them
        layout_name = self.config.get("layout_name")
        if layout_name and self.topo_dim > 0 and self.topo_features is None:
            try:
                project_root = Path(__file__).resolve().parent.parent  # overcooked/
                true_root = project_root.parent  # deep_project root
                sys.path.insert(0, str(true_root / "train" / "training"))
                from env import compute_topology_for_layout, load_dynamics_overrides
                layouts_dir = project_root / "layouts"
                dynamics_overrides = load_dynamics_overrides()
                self.topo_features = compute_topology_for_layout(layout_name, layouts_dir, dynamics_overrides)
                if len(self.topo_features) != self.topo_dim:
                    print(f"[TrainedAgent] WARNING: topo_dim mismatch (model={self.topo_dim}, computed={len(self.topo_features)}). Using zeros.")
                    self.topo_features = np.zeros(self.topo_dim, dtype=np.float32)
                else:
                    print(f"[TrainedAgent] Computed {self.topo_dim}-dim topology features for layout '{layout_name}'")
            except Exception as e:
                print(f"[TrainedAgent] WARNING: Failed to compute topology features for '{layout_name}': {e}. Using zeros.")
                self.topo_features = np.zeros(self.topo_dim, dtype=np.float32) if self.topo_dim > 0 else None

    def _load_model(self, path: Path):
        try:
            ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
            self.obs_dim = ckpt.get("obs_dim", 96)
            self.num_actions = ckpt.get("num_actions", 6)
            hidden = ckpt.get("hidden", 256)
            layers = ckpt.get("layers", 3)
            arch = ckpt.get("arch", None)
            self.topo_dim = ckpt.get("topo_dim", 0)
            if arch is None:
                if any("graph_encoder" in k for k in ckpt.get("model_state", {}).keys()):
                    arch = "gnn"
                else:
                    arch = "mlp"

            if arch in ["gnn", "attention", "graph"]:
                gnn_hidden = ckpt.get("hidden", 128)
                self.model = _GNNPolicy(self.obs_dim, self.num_actions, gnn_hidden, layers,
                                         topo_dim=self.topo_dim).to(self.device)
            else:
                self.model = _MLPPolicy(self.obs_dim, self.num_actions, hidden, layers).to(self.device)

            self.model.load_state_dict(ckpt["model_state"], strict=False)
            self.model.eval()
            topo_info = f", topo_dim={self.topo_dim}" if self.topo_dim > 0 else ""
            print(f"[TrainedAgent] Loaded {arch.upper()} model from {path} "
                  f"(obs_dim={self.obs_dim}, actions={self.num_actions}{topo_info})")
        except Exception as e:
            print(f"[TrainedAgent] ERROR loading model: {e}")
            self.model = None

    def reset(self):
        pass

    def act(self, obs):
        """Return action index 0-5 given observation.

        obs is the ObservationBuilder output: either a dict {"obs": np.array, ...}
        or a raw np.array of the featurized state.
        """
        if isinstance(obs, dict) and "obs" in obs:
            obs_array = np.asarray(obs["obs"], dtype=np.float32)
        else:
            obs_array = np.asarray(obs, dtype=np.float32)

        if obs_array.ndim > 1:
            obs_array = obs_array.flatten()

        raw_obs_dim = self.obs_dim - self.topo_dim
        if obs_array.shape[0] < raw_obs_dim:
            obs_array = np.pad(obs_array, (0, raw_obs_dim - obs_array.shape[0]))
        elif obs_array.shape[0] > raw_obs_dim:
            obs_array = obs_array[:raw_obs_dim]

        # Append topology features if available
        if self.topo_features is not None and self.topo_dim > 0:
            obs_array = np.concatenate([obs_array, self.topo_features])

        if self.model is None or not HAS_TORCH:
            return int(self.rng.integers(0, self.num_actions))

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(self.device)
            logits = self.model.logits(obs_tensor)

            if self.deterministic:
                action = int(logits.argmax(dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())

        return action


class PPODirectAgent:
    """PPO agent that uses featurize_state_mdp directly -- same obs pipeline as training.

    This bypasses StudentAgentAdapter/ObservationBuilder and produces correct results.
    Use as `type: trained_ppo` in policy YAML configs.

    Config options:
        model_path:   path to .pt checkpoint (required)
        deterministic: true = argmax, false = stochastic sampling (default: true)
        device:       cpu | cuda (default: cpu)
    """

    def __init__(self, config=None):
        self.config = config or {}

        project_root = Path(__file__).resolve().parent.parent
        self.model_path = str(self.config.get("model_path",
                             project_root / "train" / "models" / "ppo_agent_ta_finetuned.pt"))
        self.deterministic = bool(self.config.get("deterministic", True))
        device_name = self.config.get("device", "cpu")
        self.device = torch.device(device_name) if HAS_TORCH else None

        self._policy = None
        self._agent_index = 0
        self._mdp = None

        if not HAS_TORCH:
            print("[PPODirectAgent] WARNING: torch not available. Returning random actions.")
            self._rng = np.random.default_rng(0)
            return

        self._rng = np.random.default_rng(self.config.get("seed", 0))
        model_path = Path(self.model_path)
        if not model_path.exists():
            # try resolving relative to project root
            model_path = project_root / self.model_path
        if model_path.exists():
            self._load(model_path)
        else:
            print(f"[PPODirectAgent] WARNING: model not found at {model_path}. Returning random actions.")

    def _load(self, path: Path):
        import sys as _sys
        project_root = Path(__file__).resolve().parent.parent
        for p in [str(project_root), str(project_root / "train"),
                  str(project_root / "train" / "training")]:
            if p not in _sys.path:
                _sys.path.insert(0, p)
        try:
            from train.evaluate_ppo import load_policy
            self._policy = load_policy(path, self.device)
            print(f"[PPODirectAgent] Loaded model from {path}")
        except Exception as e:
            print(f"[PPODirectAgent] ERROR loading model: {e}")
            self._policy = None

    # ── StudentAgent interface (called by StudentAgentAdapter) ────────────────
    def reset(self):
        pass

    def act(self, obs):
        """Called by StudentAgentAdapter. obs comes from ObservationBuilder.
        We ignore it and use featurize_state_mdp instead (via get_action_from_state).
        This method exists only for compatibility — prefer get_action_from_state.
        """
        # Fallback: use obs as-is (may be wrong format, but keeps compat)
        return self._infer(np.asarray(obs["obs"] if isinstance(obs, dict) else obs,
                                       dtype=np.float32).flatten())

    def get_action_from_state(self, state, agent_index: int) -> int:
        """Direct state-based inference using featurize_state_mdp (correct pipeline)."""
        if self._mdp is None:
            return int(self._rng.integers(0, 6))
        obs_pair = self._mdp.featurize_state_mdp(state)
        obs = np.asarray(obs_pair[agent_index], dtype=np.float32).flatten()
        return self._infer(obs)

    def _infer(self, obs_array: np.ndarray) -> int:
        if self._policy is None or not HAS_TORCH:
            return int(self._rng.integers(0, 6))
        with torch.no_grad():
            t = torch.FloatTensor(obs_array).unsqueeze(0).to(self.device)
            logits, _ = self._policy.forward(t)
        if self.deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())
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

    def _load_model(self, path: Path):
        try:
            ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
            self.obs_dim = ckpt.get("obs_dim", 96)
            self.num_actions = ckpt.get("num_actions", 6)
            hidden = ckpt.get("hidden", 256)
            layers = ckpt.get("layers", 3)

            self.model = _MLPPolicy(self.obs_dim, self.num_actions, hidden, layers).to(self.device)
            # Use strict=False to allow loading PPO models (which have extra critic weights)
            self.model.load_state_dict(ckpt["model_state"], strict=False)
            self.model.eval()
            print(f"[TrainedAgent] Loaded model from {path} "
                  f"(obs_dim={self.obs_dim}, actions={self.num_actions})")
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

        if obs_array.shape[0] < self.obs_dim:
            obs_array = np.pad(obs_array, (0, self.obs_dim - obs_array.shape[0]))
        elif obs_array.shape[0] > self.obs_dim:
            obs_array = obs_array[:self.obs_dim]

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
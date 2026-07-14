"""Competition student policy template.

Loads our best trained PPO Graph Attention / MLP model and executes in the Overcooked competition runner.
Supports dynamic topology feature calculation even when layout_name is omitted from scenario config.
"""

from __future__ import annotations

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


class StudentAgent:
    def __init__(self, config=None):
        self.config = config or {}
        project_root = Path(__file__).resolve().parent.parent.parent  # deep_project/
        default_model = project_root / "train" / "models" / "ppo_agent_master.pt"
        if not default_model.exists():
            default_model = project_root / "train" / "models" / "bc_agent_gnn.pt"

        self.model_path = str(self.config.get("model_path", default_model))
        self.deterministic = bool(self.config.get("deterministic", True))
        self.device_name = self.config.get("device", "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")

        self.model = None
        self.obs_dim = 96
        self.num_actions = 6
        self.topo_dim = 0
        self.topo_features = None

        if not HAS_TORCH:
            print("[StudentAgent] WARNING: torch not available. Returning random actions.")
            self.rng = np.random.default_rng(self.config.get("seed", 0))
            return

        self.rng = np.random.default_rng(self.config.get("seed", 0))
        self.device = torch.device(self.device_name)

        model_path = Path(self.model_path)
        if not model_path.exists():
            # Try resolving relative to current root
            model_path = project_root / self.model_path
        if model_path.exists():
            self._load_model(model_path)
        else:
            print(f"[StudentAgent] WARNING: model not found at {model_path}. Returning random actions.")

        layout_name = self.config.get("layout_name")
        if not layout_name:
            try:
                for i in range(1, 6):
                    frame = sys._getframe(i)
                    if "obs_builder" in frame.f_locals and hasattr(frame.f_locals["obs_builder"], "env"):
                        env = frame.f_locals["obs_builder"].env
                        layout_name = getattr(env, "layout_name", None)
                        if not layout_name and hasattr(env, "layout_file") and env.layout_file:
                            layout_name = Path(env.layout_file).stem
                        if layout_name:
                            break
                    elif "scenario" in frame.f_locals and isinstance(frame.f_locals["scenario"], dict):
                        layout_name = frame.f_locals["scenario"].get("layout_name")
                        if layout_name:
                            break
            except Exception:
                pass

        if layout_name and self.topo_dim > 0:
            self._compute_topo(layout_name, project_root)

    def _compute_topo(self, layout_name: str, project_root: Path):
        try:
            for p in [str(project_root / "train" / "training"), str(project_root / "overcooked" / "src")]:
                if p not in sys.path:
                    sys.path.insert(0, p)
            from env import compute_topology_for_layout, load_dynamics_overrides
            layouts_dir = project_root / "overcooked" / "layouts"
            dynamics_overrides = load_dynamics_overrides()
            self.topo_features = compute_topology_for_layout(layout_name, layouts_dir, dynamics_overrides)
            if len(self.topo_features) != self.topo_dim:
                self.topo_features = np.zeros(self.topo_dim, dtype=np.float32)
        except Exception as e:
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
        except Exception as e:
            print(f"[StudentAgent] ERROR loading model: {e}")
            self.model = None

    def reset(self):
        pass

    def act(self, obs):
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

        # Dynamic layout/topology auto-detection if not precomputed
        if self.topo_features is None and self.topo_dim > 0:
            try:
                project_root = Path(__file__).resolve().parent.parent.parent
                frame = sys._getframe(1)
                adapter = frame.f_locals.get("self")
                if hasattr(adapter, "obs_builder") and hasattr(adapter.obs_builder, "env"):
                    env = adapter.obs_builder.env
                    layout_name = getattr(env, "layout_name", None)
                    if not layout_name and hasattr(env, "layout_file") and env.layout_file:
                        layout_name = Path(env.layout_file).stem
                    if layout_name:
                        self._compute_topo(layout_name, project_root)
            except Exception:
                pass
            if self.topo_features is None:
                self.topo_features = np.zeros(self.topo_dim, dtype=np.float32)

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


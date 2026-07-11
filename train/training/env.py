"""Vectorized Overcooked environment for PPO self-play training."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent
TRAIN_DIR = TRAINING_DIR.parent
PROJECT_ROOT = TRAIN_DIR.parent
OVERCOOKED_DIR = PROJECT_ROOT / "overcooked"

sys.path.insert(0, str(OVERCOOKED_DIR))


class SelfPlayEnv:
    """Wraps overcooked_ai_py for 2-agent self-play with featurized observations.

    Both agents use the same policy. The environment provides per-agent featurized obs.
    """

    def __init__(self, layout_name: str | None = None, layout_file: str | None = None,
                 horizon: int = 400, old_dynamics: bool = True, shaped_reward_scale: float = 1.0):
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

        self.shaped_reward_scale = shaped_reward_scale
        mdp_kwargs = {"old_dynamics": old_dynamics}

        if layout_file:
            from src.environment import load_custom_layout_dict
            layout_dict = load_custom_layout_dict(layout_file)
            grid = layout_dict.get("grid", "")
            grid_lines = [r.strip() for r in grid.split("\n") if r.strip()]
            layout_dict.setdefault("layout_name", Path(layout_file).stem)
            self.layout_name = layout_dict["layout_name"]
            self.mdp = OvercookedGridworld.from_grid(
                grid_lines,
                base_layout_params=layout_dict,
                params_to_overwrite=mdp_kwargs,
            )
        else:
            self.mdp = OvercookedGridworld.from_layout_name(layout_name, **mdp_kwargs)
            self.layout_name = layout_name

        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon, info_level=0)
        self.horizon = horizon
        self.num_agents = 2

        # Get obs dim from featurization
        self.env.reset(regen_mdp=False)
        obs_pair = self.env.featurize_state_mdp(self.env.state)
        self.obs_dim = len(np.asarray(obs_pair[0]).flatten())

    def reset(self):
        self.env.reset(regen_mdp=False)
        return self._get_obs()

    def _get_obs(self):
        obs_pair = self.env.featurize_state_mdp(self.env.state)
        return [np.asarray(obs_pair[0], dtype=np.float32).flatten(),
                np.asarray(obs_pair[1], dtype=np.float32).flatten()]

    def step(self, actions):
        """actions: list of 2 int action indices (0-5)"""
        from src.constants import action_index_to_overcooked_action

        overcooked_actions = [action_index_to_overcooked_action(int(a)) for a in actions]
        next_state, reward, done, info = self.env.step(overcooked_actions)

        obs = self._get_obs()
        shaped = info.get("shaped_r_by_agent", [0.0, 0.0])
        sparse = info.get("sparse_r_by_agent", [0.0, 0.0])
        rewards = [
            float(sparse[0] + self.shaped_reward_scale * shaped[0]),
            float(sparse[1] + self.shaped_reward_scale * shaped[1]),
        ]
        dones = [bool(done), bool(done)]

        return obs, rewards, dones, info

    def get_state(self):
        return self.env.state


def load_dynamics_overrides():
    """Load the dynamics_overrides.json file."""
    overrides_path = OVERCOOKED_DIR / "layouts" / "dynamics_overrides.json"
    if overrides_path.exists():
        with open(overrides_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_env_for_layout(layout_name: str, layouts_dir: Path, dynamics_overrides: dict, shaped_reward_scale: float = 1.0) -> SelfPlayEnv:
    """Build a SelfPlayEnv for a given layout name.

    Checks built-in layouts first, then custom layouts in layouts_dir.
    """
    # Check if it's a built-in layout
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

    built_in = False
    try:
        OvercookedGridworld.from_layout_name(layout_name)
        built_in = True
    except Exception:
        pass

    layout_file = None
    old_dynamics = True

    if built_in:
        env = SelfPlayEnv(layout_name=layout_name, horizon=400, old_dynamics=True, shaped_reward_scale=shaped_reward_scale)
    else:
        layout_file = layouts_dir / f"{layout_name}.layout"
        if not layout_file.exists():
            raise FileNotFoundError(f"Layout not found: {layout_file}")
        if layout_name in dynamics_overrides:
            old_dynamics = dynamics_overrides[layout_name].get("old_dynamics", True)
        env = SelfPlayEnv(layout_file=str(layout_file), horizon=400, old_dynamics=old_dynamics, shaped_reward_scale=shaped_reward_scale)

    return env
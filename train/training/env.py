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


def _bfs_distances(start, valid_positions):
    """BFS from start position through valid_positions. Returns {pos: distance}."""
    dist = {start: 0}
    queue = [start]
    i = 0
    while i < len(queue):
        cx, cy = queue[i]
        i += 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (cx + dx, cy + dy)
            if nxt in valid_positions and nxt not in dist:
                dist[nxt] = dist[(cx, cy)] + 1
                queue.append(nxt)
    return dist


def _dist_to_locations(start, valid_positions, targets, max_dist=999):
    """BFS distance from start to the nearest reachable target. 
    For non-walkable targets, uses distance to nearest adjacent walkable tile + 1."""
    if not targets:
        return max_dist
    bfs = _bfs_distances(start, valid_positions)
    best = max_dist
    for tx, ty in targets:
        if (tx, ty) in valid_positions:
            best = min(best, bfs.get((tx, ty), max_dist))
        else:
            adj = [(tx + dx, ty + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                   if (tx + dx, ty + dy) in valid_positions]
            if adj:
                d = min(bfs.get(a, max_dist) for a in adj) + 1
                best = min(best, d)
    return best


def compute_topology_features(mdp):
    """Compute 25 static per-layout topology features from an OvercookedGridworld MDP.
    
    Returns np.float32 array of shape (25,). Features cover:
      - Layout geometry (grid size, walkable area, narrow passages, central counter)
      - Entity counts (pots, serving, onion/dish dispensers)
      - BFS navigation distances between key locations
      - Graph-theoretic properties (dead ends, cycles, components, symmetry)
    """
    terrain = mdp.terrain_mtx
    h = len(terrain)
    w = len(terrain[0])
    valid_positions = set(mdp.get_valid_player_positions())
    all_walkable = list(valid_positions)

    # ── Layout geometry (features 0-4) ──
    grid_w = float(w)
    grid_h = float(h)
    walkable = float(len(valid_positions))
    narrow = float(sum(1 for pos in valid_positions
                       if sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                              if (pos[0] + dx, pos[1] + dy) in valid_positions) == 2))
    has_central_counter = 0.0
    for r in range(1, h - 1):
        if all(c == 'X' for c in terrain[r][1:w - 1]):
            has_central_counter = 1.0
            break
    if not has_central_counter:
        for c in range(1, w - 1):
            if all(terrain[r][c] == 'X' for r in range(1, h - 1)):
                has_central_counter = 1.0
                break

    # ── Entity counts (features 5-9) ──
    pot_locs = mdp.get_pot_locations()
    serving_locs = mdp.get_serving_locations()
    onion_locs = mdp.terrain_pos_dict.get('O', [])
    dish_locs = mdp.terrain_pos_dict.get('D', [])
    num_pots = float(len(pot_locs))
    num_serving = float(len(serving_locs))
    num_onion = float(len(onion_locs))
    num_dish = float(len(dish_locs))
    openness = walkable / max(w * h, 1)

    # ── BFS distances (features 10-18) ──
    p0_start = tuple(mdp.start_player_positions[0])
    p1_start = tuple(mdp.start_player_positions[1])

    bfs_p0_to_pot = _dist_to_locations(p0_start, valid_positions, pot_locs)
    bfs_p0_to_serv = _dist_to_locations(p0_start, valid_positions, serving_locs)
    bfs_p0_to_onion = _dist_to_locations(p0_start, valid_positions, onion_locs)
    bfs_p0_to_dish = _dist_to_locations(p0_start, valid_positions, dish_locs)
    bfs_p1_to_pot = _dist_to_locations(p1_start, valid_positions, pot_locs)
    bfs_onion_to_pot = _dist_to_locations(onion_locs[0], valid_positions, pot_locs) if onion_locs else 999
    bfs_pot_to_serv = _dist_to_locations(pot_locs[0], valid_positions, serving_locs) if pot_locs else 999
    bfs_dish_to_pot = _dist_to_locations(dish_locs[0], valid_positions, pot_locs) if dish_locs else 999
    bfs_p0_to_p1 = _bfs_distances(p0_start, valid_positions).get(p1_start, 999)

    # ── Graph properties (features 19-24) ──
    dead_ends = float(sum(1 for pos in valid_positions
                          if sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                 if (pos[0] + dx, pos[1] + dy) in valid_positions) == 1))

    # Connected components
    visited = set()
    components = 0
    for start_pos in all_walkable:
        if start_pos not in visited:
            components += 1
            queue = [start_pos]
            visited.add(start_pos)
            idx = 0
            while idx < len(queue):
                cx, cy = queue[idx]
                idx += 1
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nxt = (cx + dx, cy + dy)
                    if nxt in valid_positions and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
    n_components = float(components)

    # Cyclic: edges > vertices - components
    edges = 0
    for cx, cy in all_walkable:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (cx + dx, cy + dy) in valid_positions:
                edges += 1
    edges //= 2
    is_cyclic = 1.0 if edges > walkable - n_components else 0.0

    # Graph diameter (max BFS distance)
    max_diam = 0.0
    for start_pos in all_walkable:
        bf = _bfs_distances(start_pos, valid_positions)
        if bf:
            max_diam = max(max_diam, max(bf.values()))

    # Average BFS distance
    total_d = 0.0
    cnt_d = 0
    for start_pos in all_walkable:
        bf = _bfs_distances(start_pos, valid_positions)
        for d in bf.values():
            total_d += d
            cnt_d += 1
    avg_bfs = total_d / max(cnt_d, 1)

    # Left-right symmetry
    symmetric = 1.0
    for r in range(h):
        for c in range(w // 2):
            if terrain[r][c] != terrain[r][w - 1 - c]:
                symmetric = 0.0
                break
        if not symmetric:
            break

    return np.array([
        grid_w, grid_h, walkable, narrow, has_central_counter,
        num_pots, num_serving, num_onion, num_dish, openness,
        bfs_p0_to_pot, bfs_p0_to_serv, bfs_p0_to_onion, bfs_p0_to_dish,
        bfs_p1_to_pot, bfs_onion_to_pot, bfs_pot_to_serv, bfs_dish_to_pot, bfs_p0_to_p1,
        dead_ends, n_components, is_cyclic, max_diam, avg_bfs, symmetric,
    ], dtype=np.float32)


def compute_topology_for_layout(layout_name, layouts_dir, dynamics_overrides):
    """Compute topology features for a built-in or custom layout name."""
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    try:
        mdp = OvercookedGridworld.from_layout_name(layout_name)
    except Exception:
        # Search for custom layout file matching the layout name (may have team suffix)
        layout_file = layouts_dir / f"{layout_name}.layout"
        if not layout_file.exists():
            candidates = sorted(layouts_dir.glob(f"{layout_name}*.layout"))
            if candidates:
                layout_file = candidates[0]
        if not layout_file.exists():
            raise FileNotFoundError(f"Cannot resolve layout: {layout_name}")
        from src.environment import load_custom_layout_dict
        layout_dict = load_custom_layout_dict(str(layout_file))
        grid = layout_dict.get("grid", "")
        grid_lines = [r.strip() for r in grid.split("\n") if r.strip()]
        old_dynamics = dynamics_overrides.get(layout_name, {}).get("old_dynamics", True)
        mdp = OvercookedGridworld.from_grid(
            grid_lines, base_layout_params=layout_dict,
            params_to_overwrite={"old_dynamics": old_dynamics},
        )
    return compute_topology_features(mdp)


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
        base_obs_dim = len(np.asarray(obs_pair[0]).flatten())
        # Compute layout topology features (static per layout)
        self.topo_features = compute_topology_features(self.mdp)
        self.topo_dim = len(self.topo_features)
        self.obs_dim = base_obs_dim + self.topo_dim
        self._precompute_bfs_distances()

    def _precompute_bfs_distances(self):
        """Precompute grid-graph BFS shortest path distance from every walkable position to every target tile."""
        valid_positions = set(self.mdp.get_valid_player_positions())
        width = len(self.mdp.terrain_mtx[0])
        height = len(self.mdp.terrain_mtx)
        
        self.bfs_cache = {}
        for start_pos in valid_positions:
            dist = {start_pos: 0}
            queue = [start_pos]
            idx = 0
            while idx < len(queue):
                curr = queue[idx]
                idx += 1
                cx, cy = curr
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nxt = (cx + dx, cy + dy)
                    if nxt in valid_positions and nxt not in dist:
                        dist[nxt] = dist[curr] + 1
                        queue.append(nxt)
            
            for tx in range(width):
                for ty in range(height):
                    target = (tx, ty)
                    if target in valid_positions:
                        self.bfs_cache[(start_pos, target)] = dist.get(target, 999)
                    else:
                        adj_walkable = [(tx+dx, ty+dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)] if (tx+dx, ty+dy) in valid_positions]
                        if adj_walkable:
                            min_adj = min(dist.get(adj, 999) for adj in adj_walkable)
                            self.bfs_cache[(start_pos, target)] = min_adj + 1 if min_adj < 999 else 999
                        else:
                            self.bfs_cache[(start_pos, target)] = 999

    def get_distance(self, pos, target):
        return self.bfs_cache.get((pos, target), abs(pos[0] - target[0]) + abs(pos[1] - target[1]))

    def reset(self):
        self.env.reset(regen_mdp=False)
        return self._get_obs()

    def _get_obs(self):
        obs_pair = self.env.featurize_state_mdp(self.env.state)
        obs0 = np.asarray(obs_pair[0], dtype=np.float32).flatten()
        obs1 = np.asarray(obs_pair[1], dtype=np.float32).flatten()
        return [np.concatenate([obs0, self.topo_features]),
                np.concatenate([obs1, self.topo_features])]

    def step(self, actions):
        """actions: list of 2 int action indices (0-5)"""
        from src.constants import action_index_to_overcooked_action

        prev_state = self.env.state
        overcooked_actions = [action_index_to_overcooked_action(int(a)) for a in actions]
        next_state, reward, done, info = self.env.step(overcooked_actions)

        obs = self._get_obs()
        # Zero out upstream one-shot raw event bonuses to enforce strict telescoping PBRS
        shaped = [0.0, 0.0]
        sparse = list(info.get("sparse_r_by_agent", [0.0, 0.0]))

        # Navigation & completion reward shaping
        serving_locs = self.mdp.get_serving_locations()
        pot_locs = self.mdp.get_pot_locations()
        counter_locs = [
            (c, r)
            for r in range(len(self.mdp.terrain_mtx))
            for c in range(len(self.mdp.terrain_mtx[0]))
            if self.mdp.terrain_mtx[r][c] == "X"
        ]

        gamma = 0.99
        for i in range(2):
            prev_p = prev_state.players[i]
            next_p = next_state.players[i]

            # Literal total potential function Phi(player, state) per Ng et al. (1999) & Devlin & Kudenko (2011)
            # Status-aware pot targeting with safe hand-off routing when direct target is across an impassable barrier (BFS=999).
            def compute_phi(p, s_obj):
                phi = 0.0
                dist_out = 0

                # 1. State-Based Subgoal Potentials (halved so joint team sum across both agents equals +3.0 / +5.0)
                pot_st = self.mdp.get_pot_states(s_obj)
                for _ in pot_st.get("1_items", []):
                    phi += 1.5
                for _ in pot_st.get("2_items", []):
                    phi += 3.0
                for _ in pot_st.get("cooking", []):
                    phi += 4.5
                for _ in pot_st.get("ready", []):
                    phi += 4.5

                # 2. Navigation Potential based on held item
                if p.held_object:
                    item = p.held_object.name
                    if item == "soup":
                        phi += 2.5  # holding ready soup
                        if serving_locs:
                            min_d = min(self.get_distance(p.position, (sx, sy)) for sx, sy in serving_locs)
                            if min_d >= 999 and counter_locs:
                                other_p = s_obj.players[1] if p == s_obj.players[0] else s_obj.players[0]
                                reach_counters = [c for c in counter_locs if self.get_distance(p.position, c) < 999 and self.get_distance(other_p.position, c) < 999]
                                if not reach_counters:
                                    reach_counters = [c for c in counter_locs if self.get_distance(p.position, c) < 999]
                                min_d = min(self.get_distance(p.position, c) for c in reach_counters) if reach_counters else 0
                            elif min_d >= 999:
                                min_d = 0
                            phi -= 0.5 * min_d
                            dist_out = min_d
                    elif item in ("onion", "tomato") and pot_locs:
                        valid_pts = pot_st.get("empty", []) + pot_st.get("1_items", []) + pot_st.get("2_items", [])
                        target_pts = valid_pts if valid_pts else pot_locs
                        min_d = min(self.get_distance(p.position, (px, py)) for px, py in target_pts)
                        if min_d >= 999 and counter_locs:
                            other_p = s_obj.players[1] if p == s_obj.players[0] else s_obj.players[0]
                            reach_counters = [c for c in counter_locs if self.get_distance(p.position, c) < 999 and self.get_distance(other_p.position, c) < 999]
                            if not reach_counters:
                                reach_counters = [c for c in counter_locs if self.get_distance(p.position, c) < 999]
                            min_d = min(self.get_distance(p.position, c) for c in reach_counters) if reach_counters else 0
                        elif min_d >= 999:
                            min_d = 0
                        phi -= 0.2 * min_d
                        dist_out = min_d
                    elif item == "dish" and pot_locs:
                        valid_pts = pot_st.get("ready", [])
                        if not valid_pts:
                            valid_pts = pot_st.get("cooking", [])
                        target_pts = valid_pts if valid_pts else pot_locs
                        min_d = min(self.get_distance(p.position, (px, py)) for px, py in target_pts)
                        if min_d >= 999 and counter_locs:
                            other_p = s_obj.players[1] if p == s_obj.players[0] else s_obj.players[0]
                            reach_counters = [c for c in counter_locs if self.get_distance(p.position, c) < 999 and self.get_distance(other_p.position, c) < 999]
                            if not reach_counters:
                                reach_counters = [c for c in counter_locs if self.get_distance(p.position, c) < 999]
                            min_d = min(self.get_distance(p.position, c) for c in reach_counters) if reach_counters else 0
                        elif min_d >= 999:
                            min_d = 0
                        phi -= 0.2 * min_d
                        dist_out = min_d

                return phi, dist_out

            phi_prev, _ = compute_phi(prev_p, prev_state)
            phi_curr, dist_curr = compute_phi(next_p, next_state)

            # Apply exact potential difference unconditionally across EVERY transition
            shaped[i] += float(gamma * phi_curr - phi_prev)

            # Uniform heuristic idle penalty if staying still > 1 tile from target destination
            if int(actions[i]) == 4 and dist_curr > 1:
                shaped[i] -= 0.05

        rewards = [
            float(sparse[0] + self.shaped_reward_scale * shaped[0]),
            float(sparse[1] + self.shaped_reward_scale * shaped[1]),
        ]
        dones = [bool(done), bool(done)]
        info["sparse_r_by_agent"] = sparse
        info["shaped_r_by_agent"] = shaped

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
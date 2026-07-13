# AGENTS.md

Repo-specific guidance for agents working in this Overcooked-AI training project.
Verified against source; where README conflicts with code, code wins.

## What this is
Two-phase cooperative RL agent for Overcooked-AI: Behavioral Cloning (BC) from human
demos + greedy-generated demos, then PPO fine-tuning against greedy partners with
self-play on coordination bottleneck maps. Course/competition deliverable; graded by
official soup-throughput score, not raw reward.

## Environment (one-time)
- Conda env `overcooked_train`, Python 3.10. `pip install -r requirements.txt`.
- **`numpy<2` is mandatory** -- `overcooked_ai_py` uses `np.Inf` (breaks on numpy>=2).
- Torch must be CUDA wheels: `pip install torch==2.6.0 torchvision==0.21.0
  --index-url https://download.pytorch.org/whl/cu124`. PPO self-play is impractical on CPU.
- Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`.

## Working directory -- read before running anything
- Path resolution differs by script:
  - `overcooked/scripts/filter_recordings.py` and `train/build_dataset.py` resolve I/O
    against `PROJECT_ROOT` -> always write to `<root>/train/data/`.
  - `train/train_bc.py` and `train/train_ppo.py` use **CWD-relative** defaults
    (`train/data/consolidated.npz`, `train/models/bc_agent_gnn.pt`).
- If you `cd overcooked` (as the docstrings say) and run train_bc/train_ppo with defaults,
  they look in `overcooked/train/data/` and won't find what build_dataset wrote.
  **Run `train_bc.py`/`train_ppo.py` from the project root** (defaults then resolve to
  `<root>/train/...`), or pass explicit absolute `--data`/`--bc-model`/`--output`.
- Two `train/` dirs exist: `<root>/train/` (source: scripts, `training/`) and
  `overcooked/train/` (runtime `data/`, `models/`). Don't confuse them.

## Do not modify `overcooked/src/`
It's the competition starter code. Extend behavior via `overcooked/policies/` and YAML
configs in `overcooked/configs/`. `policies/trained_agent.py` is the integration point.

## Architecture (GNN + topology features)
- **Production model** is the Relational Graph Attention network with 25-dim static
  per-layout topology features (bypassing attention -- concatenated directly with graph
  output). obs_dim=121 (96 entity features + 25 topology features). Actions 0-5 =
  N,S,E,W,Stay,Interact. Reward +20/soup (sparse).
- **Topology features** (25 dims, computed once per layout from terrain matrix):
  - Layout geometry (5): width, height, walkable tiles, narrow passages, central counter
  - Entity counts (5): pots, serving, onion, dish, openness ratio
  - BFS distances (9): player-to-pot/serving/onion/dish, onion-to-pot, etc.
  - Graph properties (6): dead ends, components, cyclic, diameter, avg BFS, symmetry
- **Architecture details**:
  - 10 entity nodes extracted from obs[:, :96], 1 topology node from obs[:, 96:121]
  - Topology node bypasses attention: `proj_topo` projects features to `embed_dim`,
    then concatenated with graph output (`self + mate + global pool`)
  - `out_dim = embed_dim * 4` (was 3) when topo_dim > 0; actor/critic heads size accordingly
  - Checkpoints store `topo_dim` key; `trained_agent.py` auto-detects both `arch` and `topo_dim`
- **BC training REQUIRES `--arch gnn`** -- `train_bc.py` defaults to mlp, always pass it.

## Pipeline (ordered, from project root)

### 0. Generate greedy vs greedy demos (optional but recommended)
```bash
python train/generate_demos.py --episodes 10
```
Generates 10 episodes of greedy vs greedy gameplay on all 22 built-in layouts.
Records BOTH agents' perspectives (`_a0`, `_a1` suffixes) for position-independent learning.
Produces ~22 layouts ? 10 episodes ? 2 agents = 440 recordings in
`overcooked/data/user_recordings/`. The filter includes them if `exclude_non_human: false`
(currently set in `configs/filter.yaml`).

### 1. Consolidate custom layouts
```bash
python overcooked/scripts/consolidate_layouts.py
```
Gathers team `.layout` files into `overcooked/layouts/` with team suffix;
writes `layouts/dynamics_overrides.json`.

### 2. Filter recordings by quality
```bash
python overcooked/scripts/filter_recordings.py --config configs/filter.yaml
```
Top-quality filter + tier (gold/silver/bronze). Current config:
- `exclude_non_human: false` -- greedy-generated demos included
- Tier weights: gold 1.0, **silver 0.1, bronze 0.01** (aggressive downweighting)
- Outputs `<root>/train/data/recording_quality.tsv` + `consolidated_filtered.npz`.

### 3. Build training dataset
```bash
python train/build_dataset.py
```
Normalizes, pads, computes 25-dim topology features per layout, appends to obs
(96->121). Writes `<root>/train/data/consolidated.npz` + `dataset_stats.json`.

### 4. Train BC (Behavioral Cloning)
```bash
python train/train_bc.py --arch gnn --epochs 50 --batch-size 256 --lr 1e-3 \
  --scheduler plateau --weight-decay 0 --stay-weight 1.0 \
  --output train/models/bc_agent_gnn.pt
```
- `--arch gnn` is mandatory (default is mlp)
- `--scheduler plateau` (ReduceLROnPlateau, not cosine annealing) -- proven recipe
- `--topo-dim` auto-detected from obs_dim (96 -> topo_dim=25)
- Output: `train/models/bc_agent_gnn.pt`

### 5. Fine-tune with PPO
```bash
python train/train_ppo.py --bc-model train/models/bc_agent_gnn.pt \
  --layouts counter_circuit,asymmetric_advantages,coordination_ring,cramped_room,simple_o,forced_coordination \
  --coordination-layouts counter_circuit,forced_coordination,cramped_room \
  --timesteps 200000 --kl-coef 0.3 --lr 5e-6 --entropy-coef 0.05 \
  --eval-interval 20000 --early-stop-patience 5 \
  --output train/models/ppo_agent.pt
```
- **Partner-aware**: greedy partner on open maps (provides +20 reward signal),
  self-play on coordination bottleneck maps (`--coordination-layouts`) where greedy
  gets 0 soups and can't help
- KL anchoring against frozen BC policy prevents catastrophic forgetting
- Periodic eval every 20k steps with early stopping (patience=5, more tolerance
  for noisy coordination exploration)
- Higher entropy (0.05) encourages exploration on coordination layouts
- Saves `ppo_agent.pt` (checkpoints) + `ppo_agent_best.pt` (best eval score)

## Eval / watch entrypoints (exact)
- **Eval is `python -m src.evaluate`** (from `overcooked/`), NOT `python src/eval.py` --
  README's command is wrong; module imports `from src.config` so it must run as module.
  Configs: `overcooked/configs/eval/<layout>.yaml` (56 configs).
- **Eval configs MUST have `layout_name` in agent config** for topology models
  (otherwise topology features default to zeros):
  ```yaml
  agent_0:
    config:
      model_path: ../train/models/ppo_agent_best.pt
      layout_name: counter_circuit
      deterministic: true
  ```
- Use `type: python_class` with `TrainedAgent` for topology models.
  **DO NOT use `type: trained_ppo`** -- `PPODirectAgent` does not append topology features.
- Watch live (from root): `python watch_agent.py --config overcooked/configs/watch_bc_agent.yaml`.
- Benchmark + watch (from root): `python overcooked/scripts/benchmark_and_watch.py
  --model train/models/ppo_agent.pt --watch` (5 competition layouts; `--all-maps` for all 56).

## YAML policy types (`overcooked/src/policy_loader.py`)
- `builtin` -- `name:` one of `stay, random_motion, random, greedy_full_task,
  human_keyboard, greedy_human_model`. `greedy_full_task` supports `sticky_prob`,
  `random_action_prob`, `ingredient`, `avoid_teammate`.
- **`python_class`** -- loads `path:`/`class_name:` (e.g.
  `policies/trained_agent.py:TrainedAgent`). **REQUIRED for topology models** --
  `TrainedAgent` computes topology features from `layout_name` config and appends
  them to observations during inference.
- `trained_ppo` -- `PPODirectAgent`; does NOT use topology features. **Avoid for
  models trained with topo_dim > 0.** Only works with legacy 96-dim models.

## Layouts & dynamics
- Built-in layouts (~45): reference by `layout_name:` (e.g. `cramped_room`).
- Custom layouts: `overcooked/layouts/<name>_<team>.layout`, reference by `layout_file:`
  (not `layout_name:`), set `layout_name: null`.
- `old_dynamics: true` (default) for onion-soup; `false` for tomato/custom recipes/multi-order.
  `layouts/dynamics_overrides.json` tracks per-layout flags.

## Benchmark layouts
- 6 PPO training/eval maps (train_ppo default `--layouts`): `cramped_room,
  asymmetric_advantages, coordination_ring, simple_o, forced_coordination, counter_circuit`
  (5 Carroll et al. + simple_o).
- `--layout-weights uneven` over-weights coordination bottlenecks (ring/forced/counter ?2).
- `benchmark_and_watch.py` uses only 5 (excludes counter_circuit).

## Recordings & gitignore
- Team demo folders (~22) live **directly inside `overcooked/`**, gitignored via
  `overcooked/*` + exceptions. Recordings are `.npz` + `.pkl` + `.metadata.json` triples;
  not in git -- place locally before running the pipeline.
- **Filter config changes** from defaults:
  - `exclude_non_human: false` -- includes greedy-generated demos (from `generate_demos.py`)
  - Tier weights: gold 1.0, silver 0.1, bronze 0.01 (in `configs/filter.yaml`)
- `train/generate_demos.py` produces greedy vs greedy demos automatically --
  no manual recording needed. Both agents' perspectives saved (`_a0`/`_a1` suffixes).
- All `.pt`, `.npz`, `.pkl`, `outputs/` are gitignored -- checkpoints/data are local-only.
- Official score (what's optimized): `10000*num_soups + 10*(horizon-last_soup)
  + (horizon-first_soup) - penalty`. Soup count dominates.

## Gotchas
- **No tests, lint, CI, or formatter configured.** Verify changes by running eval/pipeline
  scripts directly.
- `train/train_partner_aware.py` has a **hardcoded Windows absolute `PROJECT_ROOT`** --
  won't run elsewhere without editing.
- **`PPODirectAgent` (trained_ppo) does NOT append topology features.** Only `TrainedAgent`
  (python_class) does. Using `type: trained_ppo` with a topology-trained model will silently
  cause dimension mismatch at inference. Always use `type: python_class` with
  `policies/trained_agent.py:TrainedAgent` for models with `topo_dim > 0`.
- **Eval configs MUST have `layout_name` in agent config** when using topology models.
  Without it, topology features default to zeros and the model may not work.
- **PPO's `--coordination-layouts` argument** enables self-play on bottleneck maps
  where greedy partner gets 0 soups. Without it, PPO trains against greedy on ALL maps,
  which provides no reward signal on coordination layouts.
- `build_dataset.py` oversampling is currently **disabled** (`OVERSAMPLE_FACTOR=1`,
  `CC_OVERSAMPLE=1`). Re-enable by changing these values if specific layouts need more data.
- Windows Unicode: `train_ppo.py` and `generate_demos.py` use ASCII-only characters
  to avoid `cp1252` encoding errors in PowerShell.
- Checkpoint naming across scripts (`bc_agent.pt`, `bc_agent_gnn.pt`, `ppo_agent.pt`,
  `ppo_agent_best.pt`, `ppo_agent_400k_validation.pt`). The YAML config's `model_path`
  is authoritative. Best checkpoints are `ppo_agent_best.pt` (early-stopping snapshot).
- `train/train/models/` is an accidental artifact from running train scripts with the wrong
  CWD (gitignored).
- README.md is older/Spanish and partly stale (references MLP, nonexistent `src/eval.py`).
  `TUTORIAL.md` and `methodology_evaluation_brief.md` are current; **code is the source of truth**.

# Overcooked AI Training Project - Tutorial & Reference

## What this project does

This project trains an AI agent to play Overcooked using **Behavioral Cloning (BC)** from
human demonstrations + **greedy-generated synthetic demos**, then fine-tunes it with
**Proximal Policy Optimization (PPO)** against a greedy partner with self-play on
coordination bottlenecks. The trained agent plugs directly into the existing
`overcooked/src/` game framework and can be evaluated against any layout.

---

## Project Structure

```
deep_project/
||| overcooked/                    # Competition starter code (DO NOT modify src/)
|   ||| src/                        # Game engine wrapper (runner, environment, eval, etc.)
|   ||| policies/                   # Built-in policies + our trained_agent.py
|   ||| configs/                    # YAML configs for the game (play, eval, collect demos)
|   |   ||| layouts/               # Original custom layouts (6 files)
|   ||| layouts/                    # <- NEW: canonical home for ALL custom layouts (group-suffixed)
|   |   ||| dynamics_overrides.json # Per-layout old_dynamics flags
|   ||| data/demonstrations/        # Bot-generated baseline demos (greedy_full_task)
|   ||| <team_folders>/             # ~22 team folders, each with recordings/ (538 .npz total)
|   ||| scripts/                    # Our consolidation + filter + reporting scripts
|   ||| ...
||| train/                          # <- NEW: training pipeline
|   ||| build_dataset.py            # Consolidate filtered recordings -> single dataset
|   ||| generate_demos.py           # Generate greedy vs greedy synthetic demos
|   ||| train_bc.py                 # Behavioral Cloning trainer
|   ||| train_ppo.py                # PPO fine-tuner (greedy-partner-aware)
|   ||| training/                   # PPO internals (env, models, ppo)
|   ||| data/                       # Generated datasets + stats
|   ||| models/                     # Generated .pt checkpoints
||| policies/trained_agent.py       # <- NEW: plug-in agent for the game
||| configs/filter.yaml             # <- NEW: recording quality filter thresholds
||| configs/eval/                    # <- NEW: per-layout evaluation configs
||| scripts/consolidate_layouts.py # <- NEW: gather custom layouts from team folders
||| scripts/filter_recordings.py    # <- NEW: quality-filter recordings
||| scripts/official_score.py      # <- NEW: official competition score formula
||| scripts/training_report.py     # <- NEW: consolidated stats + curves
||| requirements.txt               # Python dependencies
||| TUTORIAL.md                     # This file
||| overcooked_compiled_colab.ipynb # Madrona self-play notebook (reference only)
```

---

## The Recordings

Each team folder contains recordings as triples:
- `<name>.npz` - tensor dataset: `obs`, `actions`, `rewards`, `dones`, `next_obs`, `episode_ids`, etc.
- `<name>.pkl` - full pickle with `metadata`, `records` (per-timestep dicts), `episode_summaries`.
- `<name>.metadata.json` - full environment config, layout grid, policy config, observation type.

**Key facts:**
- 538 total `.npz` recordings across ~22 team folders.
- 515 are **human keyboard demos** (agent_1 = human, recorded at index 1).
- 13 are **bot baselines** (greedy_full_task/random/stay - excluded from training).
- Observations are **featurized** vectors (`env.featurize_state_mdp(state)[agent_index]`).
- Actions are integers 0-5: `0=N, 1=S, 2=E, 3=W, 4=stay, 5=interact`.
- Rewards: +20.0 per soup delivered (sparse).

### Official Competition Score

The TA's metric is NOT just raw reward. The official score is:

```
official_score = 10000 * num_soups
                + 10 * (horizon - last_soup_timestep)
                + (horizon - first_soup_timestep)
                - penalty
```

Where:
- `num_soups` = number of soups delivered (count of reward > 0 timesteps)
- `horizon` = episode length (usually 250)
- `first_soup_timestep` = timestep of first delivery
- `last_soup_timestep` = timestep of last delivery
- `penalty` = penalties (timeouts, etc.)

**Delivering more soups dominates** (10,000 each), so the agent must prioritize throughput.

`scripts/official_score.py` computes this from any recording or episode.

---

## The Notebook (overcooked_compiled_colab.ipynb)

This notebook trains an agent **from scratch via self-play PPO** using the Madrona
GPU-accelerated Overcooked implementation. It runs on Google Colab with a T4 GPU.

**Important:** The notebook does NOT use your recordings. Its environment, observation
space (CNN-visual), and pipeline are completely separate from `overcooked/src/`. It trains
the `"simple"` layout to ~234 score in ~2 minutes.

We keep it as reference but do NOT use it directly. Our pipeline (BC + PPO) uses your
recordings and integrates with your local `overcooked/src/` code.

---

## Environment Setup

### Prerequisites
- **Miniconda** installed (`C:\Users\<user>\miniconda3`)
- **NVIDIA GPU** (RTX 4060 or similar) with CUDA 12.4+ drivers
- Python 3.10 via conda

### One-time setup
```bat
:: Create conda env
conda create -n overcooked_train python=3.10 -y
conda activate overcooked_train

:: Install dependencies
pip install "numpy<2" pyyaml scipy
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install gym==0.26.2 pygame imageio imageio-ffmpeg
pip install overcooked_ai_py

:: Verify GPU
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

:: Verify overcooked
python -c "from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld; print(OvercookedGridworld.from_layout_name('cramped_room').layout_name)"
```

### Verify the game works
```bat
cd C:\Users\SEBASTIAN\Documents\deep_project\overcooked
python -m src.evaluate --config configs/evaluate.yaml
```
This runs the template (stay) agent vs greedy_full_task. Should print JSON with scores.

---

## Full Training Pipeline

### Step 0 (optional): Generate Greedy vs Greedy Demos
```bat
cd C:\Users\SEBASTIAN\Documents\deep_project
python train\generate_demos.py --episodes 10
```
Generates synthetic greedy vs greedy gameplay on all 22 built-in layouts. Records BOTH
agents' perspectives (`_a0`/`_a1`) for position-independent learning. Produces ~22 layouts
x 10 episodes x 2 agents = 440 recordings in `overcooked/data/user_recordings/`.

### Step 1: Consolidate Custom Layouts
```bat
cd C:\Users\SEBASTIAN\Documents\deep_project\overcooked
python scripts\consolidate_layouts.py
```
Scans all team folders for `.layout` files, copies each into `overcooked/layouts/` with a
group-name suffix (e.g., `maze_kitchen_attention_t.layout`). Validates each layout.
Writes `layouts/dynamics_overrides.json` for layouts requiring `old_dynamics: false`.

### Step 2: Filter Recordings by Quality
```bat
cd C:\Users\SEBASTIAN\Documents\deep_project\overcooked
python scripts\filter_recordings.py --config ..\configs\filter.yaml
```
Computes per-recording quality signals (score, deliveries, idle%, action entropy, recorded
agent type). Excludes idle/spam/truncated episodes. Current config includes bot baselines
(`greedy_full_task`) since `exclude_non_human: false`. Tags kept recordings as
gold/silver/bronze with aggressive tier weights (gold=1.0, silver=0.1, bronze=0.01).
Outputs `train/data/recording_quality.tsv` + `train/data/consolidated_filtered.npz`.

### Step 3: Build Training Dataset
```bat
cd C:\Users\SEBASTIAN\Documents\deep_project\overcooked
python ..\train\build_dataset.py
```
Pads observations to max shape, attaches layout/tier/role metadata. Writes
`train/data/consolidated.npz` + `train/data/dataset_stats.json`.

### Step 4: Train with Behavioral Cloning
```bat
cd C:\Users\SEBASTIAN\Documents\deep_project
python train\train_bc.py --arch gnn --epochs 50 --batch-size 256 --lr 1e-3 ^
    --scheduler plateau --weight-decay 0 --stay-weight 1.0 ^
    --output train\models\bc_agent_gnn.pt
```
Trains a GNN (Relational Graph Attention) actor with tier-weighted cross-entropy loss.
Observations are 121-dim (96 featurized + 25 topology features). The topology features
are static per-layout (BFS distances, narrow passages, graph connectivity) and bypass
attention - concatenated directly with the graph output so the actor head has direct
layout context. Saves `train/models/bc_agent_gnn.pt`.

### Step 5: Fine-tune with PPO (Partner-Aware)
```bat
cd C:\Users\SEBASTIAN\Documents\deep_project
python train\train_ppo.py --bc-model train\models\bc_agent_gnn.pt ^
    --layouts counter_circuit,asymmetric_advantages,coordination_ring,cramped_room,simple_o,forced_coordination ^
    --coordination-layouts counter_circuit,forced_coordination,cramped_room ^
    --timesteps 200000 --kl-coef 0.3 --lr 5e-6 --entropy-coef 0.05 ^
    --eval-interval 20000 --early-stop-patience 5 ^
    --output train\models\ppo_agent.pt
```
Fine-tunes the BC agent with PPO against a **greedy partner** on open maps (provides +20
sparse reward signal), switching to **self-play** on coordination bottlenecks where greedy
gets 0 soups. KL anchoring against the frozen BC policy prevents catastrophic forgetting.
Periodic eval every 20k steps with early stopping. Saves `ppo_agent_best.pt` (best eval
score) alongside `ppo_agent.pt`.

### Step 6: Evaluate the Trained Agent
```bat
cd C:\Users\SEBASTIAN\Documents\deep_project\overcooked
:: Evaluate on a specific layout (model_path must match your best checkpoint)
python -m src.evaluate --config configs\eval\counter_circuit.yaml

:: Or evaluate on all benchmark layouts
for %f in (configs\eval\cramped_room.yaml configs\eval\asymmetric_advantages.yaml configs\eval\coordination_ring.yaml configs\eval\simple_o.yaml configs\eval\forced_coordination.yaml configs\eval\counter_circuit.yaml) do python -m src.evaluate --config %f
```

### Step 7: Generate Training Report
```bat
cd C:\Users\SEBASTIAN\Documents\deep_project\overcooked
python scripts\training_report.py
```

---

## How the Trained Agent Plugs In

`policies/trained_agent.py` defines a `TrainedAgent` class with the same interface as
`policies/template.py`:
- `__init__(self, config)` - loads the `.pt` model and computes topology features
- `reset(self)` - resets any state
- `act(self, obs)` - returns action index 0-5 (appends topology features to obs)

It's loaded by the existing `src/policy_loader.py` WITHOUT changes to `src/`:
```yaml
# In any eval config (REQUIRED for topology models):
agent_0:
  type: python_class
  path: policies/trained_agent.py
  class_name: TrainedAgent
  config:
    model_path: ../train/models/ppo_agent_best.pt
    layout_name: counter_circuit    # <- REQUIRED for topology feature computation
    deterministic: true
```

Important notes:
- `layout_name` is required in agent config for models with `topo_dim > 0`. Without it,
  topology features default to zeros and the model may not work.
- Use `type: python_class` with `TrainedAgent`. **Do NOT use `type: trained_ppo`** -
  `PPODirectAgent` does NOT append topology features to observations.
- The best checkpoint is `ppo_agent_best.pt` (saved by early stopping), not `ppo_agent.pt`.

---

## Layout Notes

### Built-in layouts
Available inside `overcooked_ai_py` (no file needed):
`cramped_room`, `asymmetric_advantages`, `coordination_ring`, `counter_circuit`,
`forced_coordination`, `large_room`, `simple_o`, `simple_tomato`, `small_corridor`,
`soup_coordination`, `tutorial_0`-`tutorial_3`, and more (45 total).

### Custom layouts
Stored in `overcooked/layouts/` with group-suffixed names. The game loads them via the
`layout_file:` field in YAML configs (not `layout_name:`). Example:
```yaml
environment:
  layout_name: null
  layout_file: layouts/custom_easy_coop_flauta_s_company.layout
  old_dynamics: true
```

### old_dynamics flag
Most simple onion-soup layouts use `old_dynamics: true` (the default). Layouts with tomato
recipes, custom recipe_values/times, or multi-order start_all_orders need
`old_dynamics: false`. The `dynamics_overrides.json` file tracks this automatically.

---

## Recording Quality Filter

Not all recordings are good demonstrations. The filter (`scripts/filter_recordings.py`)
excludes:

1. **Disengaged sessions** - score == 0 AND idle >= 70% (player did nothing).
2. **Single-action spam** - action entropy < 0.3 (likely stayed or held one key).
3. **Truncated episodes** - length < 100 timesteps (player quit early).

Note: Bot baselines (`greedy_full_task`) are **NOT excluded** when `exclude_non_human: false`
(current config). This allows synthetic demos from `generate_demos.py` to flow into training.

Kept recordings are tiered by per-layout official score:
- **gold**: top 25% or >= 4 soups (weight 1.0)
- **silver**: >= median or >= 2 soups (weight 0.1)
- **bronze**: everything else kept (weight 0.01)

Silver and bronze weights are deliberately low to downweight low-quality team demos
while still providing state diversity. All thresholds are tunable in `configs/filter.yaml`.

---

## Key Files for the Deliverable

| File | Purpose |
|------|---------|
| `policies/trained_agent.py` | The trained agent (loads .pt, returns actions) |
| `train/models/bc_agent.pt` | BC-trained model weights |
| `train/models/ppo_agent.pt` | PPO-fine-tuned model weights (if trained) |
| `configs/eval/<layout>.yaml` | Per-layout eval configs |
| `scripts/official_score.py` | Official competition score calculator |
| `train/data/recording_quality.tsv` | Full filter report (which recordings kept/dropped) |

---

## Troubleshooting

**`np.Inf` error with overcooked_ai_py**: You're using numpy >= 2.0. Install `numpy<2`.
**`torch.cuda.is_available()` returns False**: Install torch with CUDA wheels
(`--index-url https://download.pytorch.org/whl/cu124`).
**Layout not found**: Ensure it's in `overcooked/layouts/` and referenced via `layout_file:`.
**`type: trained_ppo` silently ignores topology features**: If your model was trained with
`topo_dim=25` (121-dim obs), `PPODirectAgent` does NOT append topology features during
inference. Always use `type: python_class` with `policies/trained_agent.py:TrainedAgent`
and include `layout_name` in the agent config for topology models.
**Observation shape mismatch (96 vs 121)**: The dataset appends 25-dim topology features
to observations (96->121). Ensure your model architecture (`models.py`) has `topo_dim=25`
and your inference code (`trained_agent.py`) computes topology features from `layout_name`.
**Eval config missing `layout_name`**: If your eval configs don't include `layout_name`
in agent_0's config, topology features default to zeros. The model may not work correctly.
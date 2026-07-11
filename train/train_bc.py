"""Behavioral Cloning trainer.

Trains an MLP policy to imitate human demonstrations using tier-weighted
cross-entropy loss.

Usage:
    cd overcooked
    python ..\train\train_bc.py --epochs 50 --batch-size 256 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "training"))
from models import BCPolicy


def load_dataset(data_path: str):
    print(f"Loading dataset: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    obs = data["obs"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    weights = data["weights"].astype(np.float32)
    tiers = data["tiers"]
    layout_indices = data["layout_indices"].astype(np.int64)

    has_next = "next_obs" in data.files
    next_obs = data["next_obs"].astype(np.float32) if has_next else None

    print(f"  Obs shape: {obs.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Weights shape: {weights.shape}")

    return obs, actions, weights, tiers, layout_indices, next_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="train/data/consolidated.npz")
    parser.add_argument("--output", type=str, default="train/models/bc_agent.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    obs, actions, weights, tiers, layout_indices, next_obs = load_dataset(args.data)

    obs_dim = obs.shape[1]
    num_actions = 6

    # Shuffle and split
    n = len(obs)
    perm = np.random.permutation(n)
    n_val = int(n * args.val_split)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_obs = torch.FloatTensor(obs[train_idx]).to(device)
    train_actions = torch.LongTensor(actions[train_idx]).to(device)
    train_weights = torch.FloatTensor(weights[train_idx]).to(device)
    val_obs = torch.FloatTensor(obs[val_idx]).to(device)
    val_actions = torch.LongTensor(actions[val_idx]).to(device)
    val_weights = torch.FloatTensor(weights[val_idx]).to(device)

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    model = BCPolicy(obs_dim, num_actions, hidden=256, layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(reduction="none")

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    n_batches = (len(train_idx) + args.batch_size - 1) // args.batch_size

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        perm_train = np.random.permutation(len(train_idx))
        for i in range(n_batches):
            batch_idx = perm_train[i * args.batch_size:(i + 1) * args.batch_size]
            batch_obs = train_obs[batch_idx]
            batch_actions = train_actions[batch_idx]
            batch_weights = train_weights[batch_idx]

            logits = model.logits(batch_obs)
            per_sample_loss = criterion(logits, batch_actions)
            loss = (per_sample_loss * batch_weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_idx)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_actions).sum().item()
            total += len(batch_idx)

        train_loss = epoch_loss / len(train_idx)
        train_acc = correct / total

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model.logits(val_obs)
            val_per_sample = criterion(val_logits, val_actions)
            val_loss = (val_per_sample * val_weights).mean().item()
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == val_actions).float().mean().item()

            # Per-tier val accuracy
            tier_accs = {}
            for tier_name, tier_idx in [("gold", 0), ("silver", 1), ("bronze", 2)]:
                tier_mask = torch.tensor((tiers[val_idx] == tier_idx), dtype=torch.bool, device=device)
                if tier_mask.sum() > 0:
                    tier_accs[tier_name] = (val_preds[tier_mask] == val_actions[tier_mask]).float().mean().item()

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        tier_str = " | ".join(f"{k}: {v:.3f}" for k, v in tier_accs.items())
        print(f"Epoch {epoch+1:>3}/{args.epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {tier_str}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Save best model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "hidden": 256,
        "layers": 3,
        "history": history,
        "args": vars(args),
    }, args.output)
    print(f"\nSaved best model: {args.output} (val_loss={best_val_loss:.4f})")

    # Save history
    hist_path = Path(args.output).with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history: {hist_path}")

    # Final accuracy
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        all_obs = torch.FloatTensor(obs).to(device)
        all_actions = torch.LongTensor(actions).to(device)
        preds = model.act(all_obs, deterministic=True)
        final_acc = (preds == all_actions).float().mean().item()
    print(f"Final training accuracy (deterministic): {final_acc:.4f}")


if __name__ == "__main__":
    main()
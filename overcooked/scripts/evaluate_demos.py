#!/usr/bin/env python
import os
import glob
import numpy as np
import json

def analyze_demonstrations():
    search_pattern = os.path.join("data", "demonstrations", "**", "*.npz")
    npz_files = glob.glob(search_pattern, recursive=True)

    if not npz_files:
        print("No se encontraron archivos de demostración (.npz) en data/demonstrations/")
        return

    print("=" * 95)
    print(f"{'LAYOUT':<25} | {'AGENT TYPE':<15} | {'STEPS':<6} | {'SCORE':<6} | {'DELIVERIES':<10} | {'IDLE %':<8} | {'FILE NAME':<30}")
    print("=" * 95)

    stats_by_layout = {}

    for npz_path in sorted(npz_files):
        # Load npz
        try:
            data = np.load(npz_path, allow_pickle=True)
            rewards = data["rewards"]
            actions = data["actions"]
            
            # Sum of sparse rewards is the score
            score = int(np.sum(rewards))
            steps = len(rewards)
            
            # Each delivery gives a positive reward (usually 20 points, or 20 * order_bonus)
            deliveries = int(np.sum(rewards > 0))
            
            # Action 4 is 'stay' (no-op)
            idle_steps = int(np.sum(actions == 4))
            idle_percent = (idle_steps / steps) * 100 if steps > 0 else 0
            
            # Extract layout and agent type from path
            path_parts = os.path.normpath(npz_path).split(os.sep)
            # Path format: data/demonstrations/<layout>/<agent_type>/file.npz
            layout = path_parts[-3] if len(path_parts) >= 3 else "unknown"
            agent_type = path_parts[-2] if len(path_parts) >= 3 else "unknown"
            file_name = os.path.basename(npz_path)
            
            print(f"{layout:<25} | {agent_type:<15} | {steps:<6} | {score:<6} | {deliveries:<10} | {idle_percent:>6.1f}% | {file_name:<30}")
            
            if layout not in stats_by_layout:
                stats_by_layout[layout] = []
            stats_by_layout[layout].append({
                "score": score,
                "deliveries": deliveries,
                "idle_percent": idle_percent,
                "steps": steps
            })
            
        except Exception as e:
            print(f"Error cargando {npz_path}: {e}")

    print("=" * 95)
    print("\nPROMEDIOS POR MAPA:")
    print("-" * 55)
    print(f"{'LAYOUT':<25} | {'AV. SCORE':<10} | {'AV. DELIVERIES':<14} | {'AV. IDLE %':<10}")
    print("-" * 55)
    for layout, episodes in sorted(stats_by_layout.items()):
        avg_score = np.mean([e["score"] for e in episodes])
        avg_del = np.mean([e["deliveries"] for e in episodes])
        avg_idle = np.mean([e["idle_percent"] for e in episodes])
        print(f"{layout:<25} | {avg_score:<10.1f} | {avg_del:<14.1f} | {avg_idle:>8.1f}%")
    print("-" * 55)

if __name__ == "__main__":
    analyze_demonstrations()

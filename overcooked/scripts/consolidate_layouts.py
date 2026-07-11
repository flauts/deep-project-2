"""Consolidate custom .layout files from all team folders into a single canonical directory.

Scans the overcooked/ directory tree for .layout files (excluding configs/layouts/ and
the overcooked_ai_py package). Copies each into overcooked/layouts/ with a group-name
suffix: <name>_<group_slug>.layout.

Handles name collisions (e.g., custom_room appears in multiple groups with different grids)
by keeping one file per group. Validates each layout by parsing and checking terrain.

Writes layouts/dynamics_overrides.json mapping layout_name -> {old_dynamics: false} for
layouts that require non-default dynamics (tomato recipes, custom recipe_values, etc.).

Usage:
    cd overcooked
    python scripts/consolidate_layouts.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OVERCOOKED_DIR = SCRIPT_DIR.parent
LAYOUTS_OUT_DIR = OVERCOOKED_DIR / "layouts"
CONFIGS_LAYOUTS_DIR = OVERCOOKED_DIR / "configs" / "layouts"
DYNAMICS_OVERRIDES_PATH = LAYOUTS_OUT_DIR / "dynamics_overrides.json"

# Folders to skip when scanning for layouts
SKIP_DIRS = {"configs", "src", "policies", "data", "outputs", "scripts", "__pycache__",
             "overcooked_ai_py", ".git", "node_modules"}


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def grid_hash(grid_str: str) -> str:
    return hashlib.md5(grid_str.strip().encode("utf-8")).hexdigest()[:16]


def clean_grid_string(grid: str) -> list[str]:
    rows = [row.rstrip("\n") for row in grid.split("\n")]
    rows = [row.strip() for row in rows if row.strip() != ""]
    if not rows:
        return []
    return rows


def parse_layout(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8")
        layout_dict = ast.literal_eval(text)
        if not isinstance(layout_dict, dict):
            return None
        if "grid" not in layout_dict:
            return None
        return layout_dict
    except Exception:
        return None


def validate_layout(layout_dict: dict, name: str) -> tuple[bool, str]:
    grid_str = layout_dict.get("grid", "")
    rows = clean_grid_string(grid_str)
    if not rows:
        return False, "empty grid"
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        return False, f"unequal row widths: {sorted(widths)}"

    all_chars = "".join(rows)
    has_pot = "P" in all_chars
    has_serving = "S" in all_chars
    has_delivery = "D" in all_chars
    has_ingredient = "O" in all_chars or "T" in all_chars
    num_p1 = all_chars.count("1")
    num_p2 = all_chars.count("2")
    has_players = num_p1 >= 1 and num_p2 >= 1

    issues = []
    if not has_pot:
        issues.append("no pot (P)")
    if not has_ingredient:
        issues.append("no ingredient dispenser (O or T)")
    if not has_serving:
        issues.append("no serving (S)")
    if not has_delivery:
        issues.append("no delivery (D)")
    if not has_players:
        issues.append(f"players 1x{num_p1} 2x{num_p2}")

    if issues:
        return False, "; ".join(issues)
    return True, "ok"


def needs_new_dynamics(layout_dict: dict) -> bool:
    """Check if this layout requires old_dynamics=false."""
    grid_str = layout_dict.get("grid", "")
    orders = layout_dict.get("start_all_orders", [])
    bonus_orders = layout_dict.get("start_bonus_orders", [])

    # Tomato recipes
    for order in orders + bonus_orders:
        if isinstance(order, dict):
            ingredients = order.get("ingredients", [])
            if "tomato" in ingredients:
                return True

    # Custom recipe values/times
    if layout_dict.get("recipe_values") is not None:
        return True
    if layout_dict.get("recipe_times") is not None:
        return True
    if layout_dict.get("onion_value") is not None:
        return True
    if layout_dict.get("tomato_value") is not None:
        return True
    if layout_dict.get("onion_time") is not None:
        return True
    if layout_dict.get("tomato_time") is not None:
        return True

    # Multiple different recipes in start_all_orders
    if len(orders) > 1:
        # Check if they have different ingredient combos
        combos = set()
        for order in orders:
            if isinstance(order, dict):
                combos.add(tuple(sorted(order.get("ingredients", []))))
        if len(combos) > 1:
            return True

    # Non-empty bonus orders
    if bonus_orders:
        return True

    return False


def find_team_name(file_path: Path) -> str:
    """Extract the team/group folder name from a layout file path."""
    parts = file_path.relative_to(OVERCOOKED_DIR).parts
    if len(parts) < 2:
        return "unknown"
    return parts[0]


def find_layout_files() -> list[Path]:
    """Find all .layout files in the overcooked tree, excluding canonical dirs."""
    results = []
    for root, dirs, files in os.walk(OVERCOOKED_DIR):
        rel = Path(root).relative_to(OVERCOOKED_DIR)
        parts = rel.parts
        if parts and parts[0] in SKIP_DIRS:
            continue
        if "__pycache__" in parts:
            continue
        for f in files:
            if f.endswith(".layout"):
                results.append(Path(root) / f)
    return sorted(results)


def load_canonical_hashes() -> dict[str, str]:
    """Load grid hashes of layouts already in configs/layouts/."""
    hashes = {}
    if CONFIGS_LAYOUTS_DIR.exists():
        for f in CONFIGS_LAYOUTS_DIR.glob("*.layout"):
            layout = parse_layout(f)
            if layout and "grid" in layout:
                hashes[grid_hash(layout["grid"])] = f.stem
    return hashes


def main():
    LAYOUTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_layouts = find_layout_files()
    canonical_hashes = load_canonical_hashes()

    print(f"Found {len(all_layouts)} .layout files outside canonical dirs")
    print(f"Canonical (configs/layouts/) has {len(canonical_hashes)} layouts")
    print()

    copied = []
    failed = []
    skipped = []
    dynamics_overrides = {}
    seen_names = set()

    for layout_path in all_layouts:
        layout_dict = parse_layout(layout_path)
        if layout_dict is None:
            failed.append((layout_path, "parse error"))
            continue

        grid_h = grid_hash(layout_dict.get("grid", ""))
        is_canonical_copy = grid_h in canonical_hashes

        original_name = layout_path.stem
        team = find_team_name(layout_path)
        team_slug = slugify(team)

        new_name = f"{original_name}_{team_slug}"
        new_filename = f"{new_name}.layout"
        new_path = LAYOUTS_OUT_DIR / new_filename

        valid, msg = validate_layout(layout_dict, new_name)
        if not valid:
            print(f"  FAIL  {new_filename:<55} {msg}")
            failed.append((layout_path, msg))
            continue

        copy_marker = ""
        if is_canonical_copy:
            copy_marker = " (copy of canonical)"
            skipped.append((layout_path, canonical_hashes[grid_h]))
        else:
            # Only actually copy non-canonical layouts to the output dir
            new_path.write_text(layout_path.read_text(encoding="utf-8"), encoding="utf-8")
            seen_names.add(new_name)

            if needs_new_dynamics(layout_dict):
                dynamics_overrides[new_name] = {"old_dynamics": False}

            print(f"  OK    {new_filename:<55}{copy_marker}")
            copied.append(new_name)

    # Write dynamics overrides
    with open(DYNAMICS_OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(dynamics_overrides, f, indent=2, ensure_ascii=False)

    print()
    print(f"Copied: {len(copied)} layouts -> {LAYOUTS_OUT_DIR}")
    print(f"Skipped (copy of canonical configs/layouts/): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    print(f"Dynamics overrides ({len(dynamics_overrides)} layouts need old_dynamics=false):")
    for name, cfg in sorted(dynamics_overrides.items()):
        print(f"  {name} -> {cfg}")
    print(f"Written: {DYNAMICS_OVERRIDES_PATH}")

    if failed:
        print("\nFailed layouts:")
        for path, msg in failed:
            print(f"  {path} -> {msg}")


if __name__ == "__main__":
    main()
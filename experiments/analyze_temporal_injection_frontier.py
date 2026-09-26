#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


DEFAULT_ROUNDS = [50, 150, 250]
DEFAULT_BUDGETS = [0.01, 0.02, 0.05, 0.10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze temporal injection-capacity frontiers from existing BackFed probe CSVs."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/temporal_cost_retention_probe"),
    )

    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=DEFAULT_ROUNDS,
    )

    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=DEFAULT_BUDGETS,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/temporal_injection_frontier"),
    )

    return parser.parse_args()


def as_float(value: object) -> Optional[float]:
    if value is None:
        return None

    try:
        x = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x):
        return None

    return x


def read_search_csv(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: List[Dict[str, float]] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for raw in reader:
            scale = as_float(raw.get("scale"))
            attack_asr = as_float(raw.get("attack_asr"))
            clean_asr = as_float(raw.get("clean_asr"))
            excess_asr = as_float(raw.get("excess_asr"))
            attack_clean_acc = as_float(raw.get("attack_clean_acc"))
            clean_clean_acc = as_float(raw.get("clean_clean_acc"))
            clean_drop = as_float(raw.get("clean_drop"))

            if scale is None:
                continue

            if excess_asr is None and attack_asr is not None and clean_asr is not None:
                excess_asr = attack_asr - clean_asr

            if clean_drop is None and clean_clean_acc is not None and attack_clean_acc is not None:
                clean_drop = clean_clean_acc - attack_clean_acc

            if excess_asr is None or clean_drop is None:
                continue

            rows.append(
                {
                    "scale": scale,
                    "excess_asr": excess_asr,
                    "clean_drop": clean_drop,
                    "attack_asr": (
                        attack_asr if attack_asr is not None else float("nan")
                    ),
                    "clean_asr": (
                        clean_asr if clean_asr is not None else float("nan")
                    ),
                    "attack_clean_acc": (
                        attack_clean_acc if attack_clean_acc is not None else float("nan")
                    ),
                    "clean_clean_acc": (
                        clean_clean_acc if clean_clean_acc is not None else float("nan")
                    ),
                }
            )

    rows.sort(key=lambda row: row["scale"])

    if not rows:
        raise RuntimeError(f"No valid rows parsed from {path}")

    return rows


def injection_capacity(
    rows: List[Dict[str, float]],
    budget: float,
) -> Dict[str, float]:
    feasible = [
        row
        for row in rows
        if row["clean_drop"] <= budget
    ]

    if not feasible:
        return {
            "capacity": float("nan"),
            "scale": float("nan"),
            "clean_drop": float("nan"),
        }

    best = max(
        feasible,
        key=lambda row: row["excess_asr"],
    )

    return {
        "capacity": best["excess_asr"],
        "scale": best["scale"],
        "clean_drop": best["clean_drop"],
    }


def pareto_points(
    rows: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    candidates = sorted(
        rows,
        key=lambda row: (
            row["clean_drop"],
            -row["excess_asr"],
        ),
    )

    frontier: List[Dict[str, float]] = []
    best_asr = -float("inf")

    for row in candidates:
        if row["excess_asr"] > best_asr:
            frontier.append(row)
            best_asr = row["excess_asr"]

    return frontier


def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    fieldnames: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    root = (
        args.root
        if args.root.is_absolute()
        else repo_root / args.root
    )

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows: Dict[int, List[Dict[str, float]]] = {}
    frontier_rows: List[Dict[str, object]] = []
    capacity_rows: List[Dict[str, object]] = []

    for round_id in args.rounds:
        path = (
            root
            / f"round_{round_id}"
            / "injection_search.csv"
        )

        rows = read_search_csv(path)
        all_rows[round_id] = rows

        frontier = pareto_points(rows)

        for row in rows:
            frontier_rows.append(
                {
                    "round": round_id,
                    "scale": row["scale"],
                    "excess_asr": row["excess_asr"],
                    "clean_drop": row["clean_drop"],
                    "pareto": any(
                        abs(row["scale"] - p["scale"]) < 1e-12
                        for p in frontier
                    ),
                }
            )

        for budget in args.budgets:
            result = injection_capacity(
                rows,
                budget,
            )

            capacity_rows.append(
                {
                    "round": round_id,
                    "clean_budget": budget,
                    "capacity": result["capacity"],
                    "selected_scale": result["scale"],
                    "selected_clean_drop": result["clean_drop"],
                }
            )

    write_csv(
        output_dir / "frontier_points.csv",
        frontier_rows,
        [
            "round",
            "scale",
            "excess_asr",
            "clean_drop",
            "pareto",
        ],
    )

    write_csv(
        output_dir / "injection_capacity.csv",
        capacity_rows,
        [
            "round",
            "clean_budget",
            "capacity",
            "selected_scale",
            "selected_clean_drop",
        ],
    )

    with (
        output_dir
        / "injection_capacity.json"
    ).open("w") as f:
        json.dump(
            capacity_rows,
            f,
            indent=2,
        )

    plt.figure(figsize=(7, 5))

    for round_id in args.rounds:
        rows = all_rows[round_id]

        x = [
            row["clean_drop"]
            for row in rows
        ]

        y = [
            row["excess_asr"]
            for row in rows
        ]

        plt.plot(
            x,
            y,
            marker="o",
            label=f"round {round_id}",
        )

    plt.xlabel("Clean accuracy drop")
    plt.ylabel("Excess ASR")
    plt.title("Temporal Injection Frontier")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "injection_frontier.png",
        dpi=180,
    )

    plt.close()

    print("\nInjection Capacity")

    for budget in args.budgets:
        print(f"\nclean budget = {budget:.4f}")

        for round_id in args.rounds:
            row = next(
                item
                for item in capacity_rows
                if item["round"] == round_id
                and abs(item["clean_budget"] - budget) < 1e-12
            )

            print(
                f"round={round_id:4d} | "
                f"capacity={row['capacity']} | "
                f"scale={row['selected_scale']} | "
                f"clean_drop={row['selected_clean_drop']}"
            )

    print(
        f"\nSaved to: {output_dir}"
    )


if __name__ == "__main__":
    main()

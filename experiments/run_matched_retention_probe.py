#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

from temporal_cost_retention_probe import (
    launch_run,
    read_metrics,
    index_metrics,
    check_preattack_match,
    compute_retention,
)


DEFAULT_ROUNDS = [50, 150, 250]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run matched-effect retention across different FL communication rounds."
    )

    p.add_argument(
        "--attack-rounds",
        type=int,
        nargs="+",
        default=DEFAULT_ROUNDS,
    )

    p.add_argument(
        "--horizon",
        type=int,
        default=30,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    p.add_argument(
        "--gpu",
        type=str,
        default="0",
    )

    p.add_argument(
        "--training-mode",
        choices=["sequential", "parallel"],
        default="sequential",
    )

    p.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/temporal_cost_retention_probe"),
    )

    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/matched_retention_probe"),
    )

    p.add_argument(
        "--force",
        action="store_true",
    )

    return p.parse_args()


def as_float(x: object) -> Optional[float]:
    if x is None:
        return None

    try:
        y = float(str(x).strip())
    except (TypeError, ValueError):
        return None

    if not math.isfinite(y):
        return None

    return y


def read_search(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            scale = as_float(row.get("scale"))
            excess = as_float(row.get("excess_asr"))
            clean_drop = as_float(row.get("clean_drop"))

            if scale is None or excess is None or clean_drop is None:
                continue

            rows.append(
                {
                    "scale": scale,
                    "excess_asr": excess,
                    "clean_drop": clean_drop,
                }
            )

    rows.sort(
        key=lambda item: item["scale"]
    )

    return rows


def select_common_target(
    all_rows: Dict[int, List[Dict[str, float]]],
) -> float:
    maxima = []

    for round_id, rows in all_rows.items():
        feasible = [
            row["excess_asr"]
            for row in rows
            if row["clean_drop"] <= 0.10
        ]

        if not feasible:
            raise RuntimeError(
                f"No point with clean_drop <= 10% for round {round_id}"
            )

        maxima.append(
            max(feasible)
        )

    q_max = min(maxima)

    q = min(
        0.20,
        0.8 * q_max,
    )

    if q < 0.10:
        raise RuntimeError(
            f"Common matched-effect target too weak: q={q:.6f} < 0.10"
        )

    return q


def choose_scale(
    rows: List[Dict[str, float]],
    target: float,
) -> Dict[str, float]:
    candidates = [
        row
        for row in rows
        if row["clean_drop"] <= 0.10
    ]

    if not candidates:
        raise RuntimeError(
            "No candidate under 10% clean-drop budget."
        )

    return min(
        candidates,
        key=lambda row: (
            abs(row["excess_asr"] - target),
            row["clean_drop"],
        ),
    )


def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    fieldnames: List[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    source_root = (
        args.source_root
        if args.source_root.is_absolute()
        else repo_root / args.source_root
    )

    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else repo_root / args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows: Dict[int, List[Dict[str, float]]] = {}

    for t in args.attack_rounds:
        path = (
            source_root
            / f"round_{t}"
            / "injection_search.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        all_rows[t] = read_search(path)

    target = select_common_target(
        all_rows
    )

    selected: Dict[int, Dict[str, float]] = {}

    for t in args.attack_rounds:
        selected[t] = choose_scale(
            all_rows[t],
            target,
        )

    print(
        f"Common matched-effect target q={target:.6f}"
    )

    for t in args.attack_rounds:
        row = selected[t]

        print(
            f"round={t} | "
            f"scale={row['scale']} | "
            f"existing excess={row['excess_asr']:.6f} | "
            f"existing clean_drop={row['clean_drop']:.6f}"
        )

    max_round = (
        max(args.attack_rounds)
        + args.horizon
    )

    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONHASHSEED"] = str(args.seed)

    clean_csv = launch_run(
        repo_root=repo_root,
        env=env,
        run_dir=(
            output_root
            / "clean_trigger_baseline"
        ),
        seed=args.seed,
        gpu=args.gpu,
        training_mode=args.training_mode,
        num_rounds=max_round,
        attack_round=max_round + 1000,
        scale_poison=False,
        scale_factor=1.0,
        force=args.force,
    )

    clean_idx = index_metrics(
        read_metrics(clean_csv)
    )

    summary_rows: List[Dict[str, object]] = []

    for t in args.attack_rounds:
        scale = selected[t]["scale"]

        run_dir = (
            output_root
            / f"round_{t}"
            / f"scale_{str(scale).replace('.', 'p')}"
        )

        metrics_csv = launch_run(
            repo_root=repo_root,
            env=env,
            run_dir=run_dir,
            seed=args.seed,
            gpu=args.gpu,
            training_mode=args.training_mode,
            num_rounds=t + args.horizon,
            attack_round=t,
            scale_poison=True,
            scale_factor=scale,
            force=args.force,
        )

        attack_idx = index_metrics(
            read_metrics(metrics_csv)
        )

        check_preattack_match(
            clean_idx,
            attack_idx,
            t,
        )

        retention_summary, curve = compute_retention(
            clean_idx=clean_idx,
            attack_idx=attack_idx,
            attack_round=t,
            horizon=args.horizon,
        )

        curve_path = (
            output_root
            / f"round_{t}"
            / "retention_curve.csv"
        )

        write_csv(
            curve_path,
            curve,
            [
                "h",
                "round",
                "attack_asr",
                "clean_asr",
                "excess_asr",
                "attack_clean_acc",
                "clean_clean_acc",
                "clean_drop",
            ],
        )

        summary_rows.append(
            {
                "attack_round": t,
                "target_excess_asr": target,
                "selected_scale": scale,
                "existing_search_excess_asr": selected[t]["excess_asr"],
                "existing_search_clean_drop": selected[t]["clean_drop"],
                **retention_summary,
            }
        )

    write_csv(
        output_root
        / "matched_retention_summary.csv",
        summary_rows,
        [
            "attack_round",
            "target_excess_asr",
            "selected_scale",
            "existing_search_excess_asr",
            "existing_search_clean_drop",
            "immediate_excess_asr",
            "retention_auc",
            "half_life",
            "end_excess_asr",
            "peak_excess_asr",
        ],
    )

    with (
        output_root
        / "matched_retention_summary.json"
    ).open("w") as f:
        json.dump(
            summary_rows,
            f,
            indent=2,
        )

    print("\nMatched Retention Summary")

    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()

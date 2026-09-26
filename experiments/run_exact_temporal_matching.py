#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_ROUNDS = [50, 150, 250]
DEFAULT_INITIAL_SCALES = [1.0, 2.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Exact matched-effect temporal probe with a cohort-matched "
            "counterfactual branch."
        )
    )

    p.add_argument(
        "--attack-rounds",
        type=int,
        nargs="+",
        default=DEFAULT_ROUNDS,
    )

    p.add_argument(
        "--initial-scales",
        type=float,
        nargs="+",
        default=DEFAULT_INITIAL_SCALES,
    )

    p.add_argument(
        "--target",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--max-clean-drop",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--horizon",
        type=int,
        default=30,
    )

    p.add_argument(
        "--max-refine-runs",
        type=int,
        default=8,
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
        "--output-root",
        type=Path,
        default=Path("outputs/exact_temporal_matching"),
    )

    p.add_argument(
        "--force",
        action="store_true",
    )

    p.add_argument(
        "--smoke",
        action="store_true",
    )

    return p.parse_args()


def norm_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


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


def get_round(row: Dict[str, str]) -> Optional[int]:
    for key in ("round", "step"):
        if key in row:
            x = as_float(row[key])

            if x is not None:
                return int(round(x))

    return None


def find_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.rglob("*.csv")
    )

    scored: List[Tuple[int, int, Path]] = []

    for path in candidates:
        try:
            with path.open("r", newline="") as f:
                header = next(
                    csv.reader(f),
                    None,
                )
        except Exception:
            continue

        if not header:
            continue

        cols = {
            norm_col(col)
            for col in header
        }

        score = 0

        if "test_backdoor_acc" in cols:
            score += 100

        if "test_clean_acc" in cols:
            score += 50

        if "round" in cols or "step" in cols:
            score += 10

        if score:
            scored.append(
                (
                    score,
                    path.stat().st_size,
                    path,
                )
            )

    if not scored:
        raise RuntimeError(
            f"No metrics CSV found under {run_dir}"
        )

    scored.sort(
        reverse=True
    )

    return scored[0][2]


def read_metrics(
    path: Path,
) -> List[Dict[str, str]]:
    with path.open(
        "r",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        rows = []

        for raw in reader:
            rows.append(
                {
                    norm_col(k): v
                    for k, v in raw.items()
                    if k is not None
                }
            )

    return rows


def index_metrics(
    rows: List[Dict[str, str]],
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}

    for row in rows:
        r = get_round(row)

        if r is None:
            continue

        bd = as_float(
            row.get("test_backdoor_acc")
        )

        ca = as_float(
            row.get("test_clean_acc")
        )

        if bd is None:
            continue

        out[r] = {
            "backdoor_acc": bd,
            "clean_acc": (
                ca
                if ca is not None
                else float("nan")
            ),
        }

    return out


def run_cmd(
    cmd: List[str],
    repo_root: Path,
    env: Dict[str, str],
) -> None:
    print("\n" + "=" * 110)
    print(" ".join(cmd))
    print("=" * 110 + "\n")

    subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        check=True,
    )


def build_common_cmd(
    *,
    repo_root: Path,
    run_dir: Path,
    seed: int,
    gpu: str,
    training_mode: str,
    num_rounds: int,
    attack_round: int,
) -> List[str]:
    return [
        sys.executable,
        str(repo_root / "main.py"),

        "--config-name",
        "cifar10",

        "checkpoint=null",
        f"seed={seed}",
        "deterministic=true",
        f'cuda_visible_devices="{gpu}"',

        "aggregator=unweighted_fedavg",
        "dataset=CIFAR10",
        "model=ResNet18",

        f"num_rounds={num_rounds}",
        f"training_mode={training_mode}",

        "save_logging=csv",
        "progress_bar=false",
        "plot_client_selection=true",

        "no_attack=false",
        "atk_config=cifar10_multishot",

        "atk_config.data_poison_method=pattern",
        "atk_config.model_poison_method=base",

        "atk_config.poison_frequency=single-shot",
        f"atk_config.poison_start_round={attack_round}",
        f"atk_config.poison_end_round={attack_round}",
        "atk_config.poison_interval=1",

        "atk_config.adversary_selection=single",
        "atk_config.selection_scheme=all-adversary",

        "atk_config.attack_type=all2one",
        "atk_config.random_class=false",
        "atk_config.target_class=2",

        "test_every=1",

        f"hydra.run.dir={run_dir.resolve()}",
        "dir_tag=exact_temporal_matching",
    ]


def launch_variant(
    *,
    repo_root: Path,
    env: Dict[str, str],
    run_dir: Path,
    seed: int,
    gpu: str,
    training_mode: str,
    num_rounds: int,
    attack_round: int,
    mode: str,
    scale: float,
    force: bool,
    save_model: bool = False,
) -> Path:
    if run_dir.exists():
        if force:
            shutil.rmtree(run_dir)
        else:
            return find_metrics_csv(
                run_dir
            )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep each calibration probe with the run that produced it, so a later
    # scale does not overwrite the matched run's record.
    env = env.copy()
    env["BACKFED_TEMPORAL_PROBE_DIR"] = str(run_dir / "probe")

    cmd = build_common_cmd(
        repo_root=repo_root,
        run_dir=run_dir,
        seed=seed,
        gpu=gpu,
        training_mode=training_mode,
        num_rounds=num_rounds,
        attack_round=attack_round,
    )

    if mode == "attack":
        cmd += [
            "atk_config.poison_ratio=0.3125",
            "atk_config.use_atk_optimizer=true",
            "atk_config.poison_epochs=6",
            "atk_config.poison_lr=0.05",
            "atk_config.scale_poison=true",
            f"atk_config.scale_factor={scale}",
        ]

    elif mode == "counterfactual":
        cmd += [
            "atk_config.poison_ratio=0.0",
            "atk_config.use_atk_optimizer=false",
            "atk_config.poison_epochs=2",
            "atk_config.poison_lr=0.1",
            "atk_config.scale_poison=false",
            "atk_config.scale_factor=1.0",
        ]

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    if save_model:
        cmd += [
            "save_model=true",
        ]

    run_cmd(
        cmd,
        repo_root,
        env,
    )

    return find_metrics_csv(
        run_dir
    )


def metric_at(
    idx: Dict[int, Dict[str, float]],
    r: int,
) -> Dict[str, float]:
    if r not in idx:
        raise RuntimeError(
            f"Round {r} not found. "
            f"Available rounds: {sorted(idx)}"
        )

    return idx[r]


def compare_effect(
    *,
    attack_idx: Dict[int, Dict[str, float]],
    cf_idx: Dict[int, Dict[str, float]],
    attack_round: int,
) -> Dict[str, float]:
    a = metric_at(
        attack_idx,
        attack_round,
    )

    c = metric_at(
        cf_idx,
        attack_round,
    )

    return {
        "attack_asr": a["backdoor_acc"],
        "cf_asr": c["backdoor_acc"],
        "excess_asr": (
            a["backdoor_acc"]
            - c["backdoor_acc"]
        ),
        "attack_clean_acc": a["clean_acc"],
        "cf_clean_acc": c["clean_acc"],
        "clean_drop": (
            c["clean_acc"]
            - a["clean_acc"]
        ),
    }


def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    fieldnames: List[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def scale_key(
    scale: float,
) -> float:
    return round(
        float(scale),
        8,
    )


def choose_best(
    history: Dict[float, Dict[str, float]],
    target: float,
    max_clean_drop: float,
) -> Optional[Dict[str, float]]:
    feasible = [
        row
        for row in history.values()
        if row["clean_drop"] <= max_clean_drop
    ]

    if not feasible:
        return None

    return min(
        feasible,
        key=lambda row: (
            abs(
                row["excess_asr"]
                - target
            ),
            row["clean_drop"],
            row["scale"],
        ),
    )


def find_bracket(
    history: Dict[float, Dict[str, float]],
    target: float,
) -> Optional[
    Tuple[
        Dict[str, float],
        Dict[str, float],
    ]
]:
    rows = sorted(
        history.values(),
        key=lambda row: row["scale"],
    )

    candidates = []

    for i in range(
        len(rows) - 1
    ):
        left = rows[i]
        right = rows[i + 1]

        y1 = left["excess_asr"]
        y2 = right["excess_asr"]

        if (
            (y1 <= target <= y2)
            or
            (y2 <= target <= y1)
        ):
            candidates.append(
                (
                    right["scale"]
                    - left["scale"],
                    left,
                    right,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0]
    )

    return (
        candidates[0][1],
        candidates[0][2],
    )


def midpoint_scale(
    left: Dict[str, float],
    right: Dict[str, float],
) -> float:
    return (
        float(left["scale"])
        + float(right["scale"])
    ) / 2.0


def compute_retention(
    *,
    attack_idx: Dict[int, Dict[str, float]],
    cf_idx: Dict[int, Dict[str, float]],
    attack_round: int,
    horizon: int,
) -> Tuple[
    Dict[str, float],
    List[Dict[str, float]],
]:
    curve = []

    for h in range(
        0,
        horizon + 1,
    ):
        r = (
            attack_round
            + h
        )

        a = metric_at(
            attack_idx,
            r,
        )

        c = metric_at(
            cf_idx,
            r,
        )

        excess = (
            a["backdoor_acc"]
            - c["backdoor_acc"]
        )

        curve.append(
            {
                "h": h,
                "round": r,
                "attack_asr": a["backdoor_acc"],
                "cf_asr": c["backdoor_acc"],
                "excess_asr": excess,
                "attack_clean_acc": a["clean_acc"],
                "cf_clean_acc": c["clean_acc"],
                "clean_drop": (
                    c["clean_acc"]
                    - a["clean_acc"]
                ),
            }
        )

    e0 = curve[0]["excess_asr"]

    if e0 <= 0:
        raise RuntimeError(
            f"Non-positive immediate excess ASR: {e0}"
        )

    post = curve[1:]

    retention_auc = (
        sum(
            max(
                row["excess_asr"],
                0.0,
            )
            / e0
            for row in post
        )
        / len(post)
    )

    half_threshold = (
        0.5 * e0
    )

    half_life = (
        horizon + 1
    )

    for row in post:
        if (
            row["excess_asr"]
            <= half_threshold
        ):
            half_life = int(
                row["h"]
            )
            break

    return (
        {
            "immediate_excess_asr": e0,
            "retention_auc": retention_auc,
            "half_life": half_life,
            "end_excess_asr": curve[-1]["excess_asr"],
            "peak_excess_asr": max(
                row["excess_asr"]
                for row in curve
            ),
        },
        curve,
    )


def main() -> None:
    args = parse_args()

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    if not (
        repo_root
        / "main.py"
    ).exists():
        raise FileNotFoundError(
            repo_root
            / "main.py"
        )

    rounds = list(
        args.attack_rounds
    )

    if args.smoke:
        rounds = rounds[:1]
        args.horizon = min(
            args.horizon,
            5,
        )
        args.max_refine_runs = min(
            args.max_refine_runs,
            2,
        )

    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else repo_root
        / args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = (
        args.gpu
    )

    env["PYTHONHASHSEED"] = str(
        args.seed
    )

    final_rows = []

    for t in rounds:
        round_root = (
            output_root
            / f"round_{t}"
        )

        cf_long_dir = (
            round_root
            / "counterfactual"
        )

        cf_csv = launch_variant(
            repo_root=repo_root,
            env=env,
            run_dir=cf_long_dir,
            seed=args.seed,
            gpu=args.gpu,
            training_mode=args.training_mode,
            num_rounds=t + args.horizon,
            attack_round=t,
            mode="counterfactual",
            scale=1.0,
            force=args.force,
        )

        cf_idx = index_metrics(
            read_metrics(
                cf_csv
            )
        )

        history: Dict[
            float,
            Dict[str, float],
        ] = {}

        def evaluate(
            scale: float,
        ) -> Dict[str, float]:
            key = scale_key(
                scale
            )

            if key in history:
                return history[key]

            tag = (
                f"{scale:.8f}"
                .rstrip("0")
                .rstrip(".")
                .replace(".", "p")
            )

            run_dir = (
                round_root
                / "calibration"
                / f"scale_{tag}"
            )

            attack_csv = launch_variant(
                repo_root=repo_root,
                env=env,
                run_dir=run_dir,
                seed=args.seed,
                gpu=args.gpu,
                training_mode=args.training_mode,
                num_rounds=t,
                attack_round=t,
                mode="attack",
                scale=scale,
                force=args.force,
            )

            attack_idx = index_metrics(
                read_metrics(
                    attack_csv
                )
            )

            effect = compare_effect(
                attack_idx=attack_idx,
                cf_idx=cf_idx,
                attack_round=t,
            )

            row = {
                "scale": float(scale),
                **effect,
            }

            history[key] = row

            print(
                f"[MATCH t={t}] "
                f"scale={scale:.8f} "
                f"excess={row['excess_asr']:.6f} "
                f"clean_drop={row['clean_drop']:.6f}"
            )

            return row

        initial_scales = sorted(
            set(
                float(x)
                for x in args.initial_scales
            )
        )

        for scale in initial_scales:
            evaluate(
                scale
            )

            best = choose_best(
                history,
                args.target,
                args.max_clean_drop,
            )

            if (
                best is not None
                and abs(
                    best["excess_asr"]
                    - args.target
                )
                <= args.tolerance
            ):
                break

            bracket = find_bracket(
                history,
                args.target,
            )

            if bracket is not None:
                break

        refine_count = 0

        while (
            refine_count
            < args.max_refine_runs
        ):
            best = choose_best(
                history,
                args.target,
                args.max_clean_drop,
            )

            if (
                best is not None
                and abs(
                    best["excess_asr"]
                    - args.target
                )
                <= args.tolerance
            ):
                break

            bracket = find_bracket(
                history,
                args.target,
            )

            if bracket is None:
                break

            left, right = bracket

            new_scale = midpoint_scale(
                left,
                right,
            )

            if (
                scale_key(new_scale)
                in history
            ):
                break

            evaluate(
                new_scale
            )

            refine_count += 1

        calibration_rows = sorted(
            history.values(),
            key=lambda row: row["scale"],
        )

        write_csv(
            round_root
            / "calibration_search.csv",
            calibration_rows,
            [
                "scale",
                "attack_asr",
                "cf_asr",
                "excess_asr",
                "attack_clean_acc",
                "cf_clean_acc",
                "clean_drop",
            ],
        )

        best = choose_best(
            history,
            args.target,
            args.max_clean_drop,
        )

        if best is None:
            final_rows.append(
                {
                    "attack_round": t,
                    "status": "NO_FEASIBLE_SCALE",
                    "matched_scale": float("nan"),
                    "calibration_excess_asr": float("nan"),
                    "calibration_clean_drop": float("nan"),
                    "immediate_excess_asr": float("nan"),
                    "retention_auc": float("nan"),
                    "half_life": float("nan"),
                    "end_excess_asr": float("nan"),
                    "peak_excess_asr": float("nan"),
                }
            )

            continue

        if (
            abs(
                best["excess_asr"]
                - args.target
            )
            > args.tolerance
        ):
            final_rows.append(
                {
                    "attack_round": t,
                    "status": "MATCH_FAILED",
                    "matched_scale": best["scale"],
                    "calibration_excess_asr": best["excess_asr"],
                    "calibration_clean_drop": best["clean_drop"],
                    "immediate_excess_asr": float("nan"),
                    "retention_auc": float("nan"),
                    "half_life": float("nan"),
                    "end_excess_asr": float("nan"),
                    "peak_excess_asr": float("nan"),
                }
            )

            continue

        matched_scale = float(
            best["scale"]
        )

        retention_dir = (
            round_root
            / "matched_retention"
        )

        attack_long_csv = launch_variant(
            repo_root=repo_root,
            env=env,
            run_dir=retention_dir,
            seed=args.seed,
            gpu=args.gpu,
            training_mode=args.training_mode,
            num_rounds=t + args.horizon,
            attack_round=t,
            mode="attack",
            scale=matched_scale,
            force=args.force,
        )

        attack_long_idx = index_metrics(
            read_metrics(
                attack_long_csv
            )
        )

        retention_summary, curve = compute_retention(
            attack_idx=attack_long_idx,
            cf_idx=cf_idx,
            attack_round=t,
            horizon=args.horizon,
        )

        write_csv(
            round_root
            / "matched_retention_curve.csv",
            curve,
            [
                "h",
                "round",
                "attack_asr",
                "cf_asr",
                "excess_asr",
                "attack_clean_acc",
                "cf_clean_acc",
                "clean_drop",
            ],
        )

        rerun_error = abs(
            retention_summary[
                "immediate_excess_asr"
            ]
            - args.target
        )

        status = (
            "PASS"
            if rerun_error
            <= args.tolerance
            else "RERUN_MATCH_DRIFT"
        )

        final_rows.append(
            {
                "attack_round": t,
                "status": status,
                "matched_scale": matched_scale,
                "calibration_excess_asr": best["excess_asr"],
                "calibration_clean_drop": best["clean_drop"],
                **retention_summary,
            }
        )

    write_csv(
        output_root
        / "exact_matching_summary.csv",
        final_rows,
        [
            "attack_round",
            "status",
            "matched_scale",
            "calibration_excess_asr",
            "calibration_clean_drop",
            "immediate_excess_asr",
            "retention_auc",
            "half_life",
            "end_excess_asr",
            "peak_excess_asr",
        ],
    )

    with (
        output_root
        / "exact_matching_summary.json"
    ).open("w") as f:
        json.dump(
            final_rows,
            f,
            indent=2,
        )

    print("\nFINAL")
    print("=" * 110)

    for row in final_rows:
        print(row)


if __name__ == "__main__":
    main()

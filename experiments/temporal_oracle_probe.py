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


DEFAULT_ATTACK_ROUNDS = [50, 100, 150, 200, 250]
DEFAULT_HORIZON = 30
DEFAULT_SEED = 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Oracle probe for round-dependent vulnerability of FL backdoor attacks. "
            "Runs one single-shot A3FL attack at different communication rounds."
        )
    )

    parser.add_argument(
        "--attack-rounds",
        type=int,
        nargs="+",
        default=DEFAULT_ATTACK_ROUNDS,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--training-mode",
        choices=["sequential", "parallel"],
        default="sequential",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/temporal_oracle_probe"),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing run directory and rerun.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only run two early candidate rounds for a quick implementation check.",
    )

    return parser.parse_args()


def run_command(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print("\n" + "=" * 100)
    print("RUN:")
    print(" ".join(cmd))
    print("=" * 100 + "\n")

    subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=True,
    )


def normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def try_float(value: str) -> Optional[float]:
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    try:
        x = float(value)
    except ValueError:
        return None

    if not math.isfinite(x):
        return None

    return x


def find_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("*.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"No CSV log found under {run_dir}"
        )

    scored: List[Tuple[int, Path]] = []

    for path in candidates:
        try:
            with path.open("r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
        except Exception:
            continue

        if not header:
            continue

        normalized = {
            normalize_column_name(col)
            for col in header
        }

        score = 0

        if "test_backdoor_acc" in normalized:
            score += 10

        if "test_clean_acc" in normalized:
            score += 5

        if "round" in normalized or "step" in normalized:
            score += 2

        if score > 0:
            scored.append((score, path))

    if not scored:
        raise RuntimeError(
            "CSV files were found, but none contains test_backdoor_acc. "
            f"Candidates: {[str(p) for p in candidates]}"
        )

    scored.sort(
        key=lambda item: (item[0], item[1].stat().st_size),
        reverse=True,
    )

    return scored[0][1]


def read_metrics_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        rows = []

        for raw_row in reader:
            row = {
                normalize_column_name(k): v
                for k, v in raw_row.items()
                if k is not None
            }

            rows.append(row)

    return rows


def get_round(row: Dict[str, str]) -> Optional[int]:
    for key in ("round", "step"):
        if key not in row:
            continue

        value = try_float(row[key])

        if value is not None:
            return int(round(value))

    return None


def extract_attack_window(
    rows: List[Dict[str, str]],
    attack_round: int,
    horizon: int,
) -> List[Dict[str, float]]:
    selected = []

    for row in rows:
        round_number = get_round(row)

        if round_number is None:
            continue

        if round_number < attack_round:
            continue

        if round_number > attack_round + horizon:
            continue

        backdoor = try_float(
            row.get("test_backdoor_acc")
        )

        clean = try_float(
            row.get("test_clean_acc")
        )

        if backdoor is None:
            continue

        selected.append(
            {
                "round": float(round_number),
                "backdoor_acc": float(backdoor),
                "clean_acc": (
                    float(clean)
                    if clean is not None
                    else float("nan")
                ),
            }
        )

    selected.sort(
        key=lambda item: item["round"]
    )

    if not selected:
        raise RuntimeError(
            f"No test_backdoor_acc rows found in "
            f"[{attack_round}, {attack_round + horizon}]."
        )

    return selected


def compute_summary(
    attack_round: int,
    horizon: int,
    metrics_path: Path,
    window: List[Dict[str, float]],
) -> Dict[str, float | int | str]:
    asrs = [
        item["backdoor_acc"]
        for item in window
    ]

    clean_values = [
        item["clean_acc"]
        for item in window
        if math.isfinite(item["clean_acc"])
    ]

    utility = sum(asrs) / len(asrs)

    immediate_asr = asrs[0]
    end_asr = asrs[-1]
    peak_asr = max(asrs)

    mean_clean = (
        sum(clean_values) / len(clean_values)
        if clean_values
        else float("nan")
    )

    return {
        "attack_round": attack_round,
        "horizon": horizon,
        "num_eval_points": len(asrs),
        "temporal_utility": utility,
        "immediate_asr": immediate_asr,
        "peak_asr": peak_asr,
        "end_asr": end_asr,
        "mean_clean_acc": mean_clean,
        "metrics_csv": str(metrics_path),
    }


def write_window_csv(
    path: Path,
    window: List[Dict[str, float]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "backdoor_acc",
                "clean_acc",
            ],
        )

        writer.writeheader()

        for row in window:
            writer.writerow(row)


def write_summary_csv(
    path: Path,
    summaries: List[Dict],
) -> None:
    if not summaries:
        return

    fieldnames = [
        "attack_round",
        "horizon",
        "num_eval_points",
        "temporal_utility",
        "immediate_asr",
        "peak_asr",
        "end_asr",
        "mean_clean_acc",
        "metrics_csv",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for summary in summaries:
            writer.writerow(summary)


def rank_summary(
    summaries: List[Dict],
) -> List[Dict]:
    return sorted(
        summaries,
        key=lambda row: row["temporal_utility"],
        reverse=True,
    )


def main() -> None:
    args = parse_args()

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    main_py = repo_root / "main.py"

    if not main_py.exists():
        raise FileNotFoundError(
            f"BackFed main.py not found: {main_py}"
        )

    attack_rounds = list(
        args.attack_rounds
    )

    if args.smoke:
        attack_rounds = attack_rounds[:2]
        args.horizon = min(
            args.horizon,
            5,
        )

    output_root = (
        repo_root
        / args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONHASHSEED"] = str(args.seed)

    summaries: List[Dict] = []

    for attack_round in attack_rounds:
        total_rounds = (
            attack_round
            + args.horizon
        )

        run_dir = (
            output_root
            / f"attack_round_{attack_round}"
        )

        if run_dir.exists():
            if args.force:
                shutil.rmtree(run_dir)
            else:
                print(
                    f"[SKIP] Existing directory: {run_dir}"
                )

        if not run_dir.exists():
            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            command = [
                sys.executable,
                str(main_py),

                "--config-name",
                "cifar10",

                f"seed={args.seed}",
                "deterministic=true",
                "checkpoint=null",
                f'cuda_visible_devices="{args.gpu}"',

                "aggregator=unweighted_fedavg",

                "dataset=CIFAR10",
                "model=ResNet18",

                f"num_rounds={total_rounds}",

                f"training_mode={args.training_mode}",

                "save_logging=csv",
                "progress_bar=false",

                "no_attack=false",

                "atk_config=cifar10_multishot",

                "atk_config.data_poison_method=a3fl",
                "atk_config.model_poison_method=base",

                "atk_config.poison_frequency=single-shot",

                (
                    "atk_config.poison_start_round="
                    f"{attack_round}"
                ),
                (
                    "atk_config.poison_end_round="
                    f"{attack_round}"
                ),

                "atk_config.poison_interval=1",

                "atk_config.adversary_selection=single",
                "atk_config.selection_scheme=all-adversary",

                "atk_config.poison_ratio=0.3125",

                "atk_config.attack_type=all2one",
                "atk_config.random_class=false",
                "atk_config.target_class=2",

                "atk_config.use_atk_optimizer=true",
                "atk_config.poison_epochs=6",
                "atk_config.poison_lr=0.05",

                "test_every=1",

                (
                    "hydra.run.dir="
                    f"{run_dir.resolve()}"
                ),

                (
                    "dir_tag="
                    "temporal_oracle_probe"
                ),
            ]

            run_command(
                command,
                cwd=repo_root,
                env=env,
            )

        metrics_csv = find_metrics_csv(
            run_dir
        )

        rows = read_metrics_csv(
            metrics_csv
        )

        window = extract_attack_window(
            rows=rows,
            attack_round=attack_round,
            horizon=args.horizon,
        )

        write_window_csv(
            run_dir / "attack_window_metrics.csv",
            window,
        )

        summary = compute_summary(
            attack_round=attack_round,
            horizon=args.horizon,
            metrics_path=metrics_csv,
            window=window,
        )

        summaries.append(summary)

        with (
            run_dir
            / "temporal_summary.json"
        ).open("w") as f:
            json.dump(
                summary,
                f,
                indent=2,
            )

        print(
            "\n"
            f"[ROUND {attack_round}] "
            f"U_t={summary['temporal_utility']:.6f}, "
            f"Immediate={summary['immediate_asr']:.6f}, "
            f"Peak={summary['peak_asr']:.6f}, "
            f"End={summary['end_asr']:.6f}, "
            f"Clean={summary['mean_clean_acc']:.6f}"
        )

    summaries.sort(
        key=lambda row: row["attack_round"]
    )

    write_summary_csv(
        output_root
        / "temporal_oracle_summary.csv",
        summaries,
    )

    with (
        output_root
        / "temporal_oracle_summary.json"
    ).open("w") as f:
        json.dump(
            summaries,
            f,
            indent=2,
        )

    ranked = rank_summary(
        summaries
    )

    print("\n" + "=" * 100)
    print("TEMPORAL UTILITY RANKING")
    print("=" * 100)

    for rank, row in enumerate(
        ranked,
        start=1,
    ):
        print(
            f"{rank:02d}. "
            f"round={row['attack_round']:4d} | "
            f"U_t={row['temporal_utility']:.6f} | "
            f"immediate={row['immediate_asr']:.6f} | "
            f"peak={row['peak_asr']:.6f} | "
            f"end={row['end_asr']:.6f} | "
            f"clean={row['mean_clean_acc']:.6f}"
        )

    utilities = [
        row["temporal_utility"]
        for row in summaries
    ]

    if len(utilities) >= 2:
        best = max(utilities)
        worst = min(utilities)
        spread = best - worst

        print(
            "\n"
            f"Temporal utility spread: "
            f"{spread:.6f}"
        )

    print(
        "\nSaved summary to:"
    )

    print(
        output_root
        / "temporal_oracle_summary.csv"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import runpy
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import run_exact_temporal_matching as exact_matching

from run_exact_temporal_matching import (
    launch_variant as launch_exact_variant,
    read_metrics,
    index_metrics,
    compare_effect,
)


def launch_variant(**kwargs) -> Path:
    """Save the actual final global state without changing training."""
    original_run_cmd = exact_matching.run_cmd

    def run_with_snapshot(cmd, repo_root, env):
        worker_cmd = [
            cmd[0], str(Path(__file__).resolve()),
            "--geometry-worker", *cmd[1:],
        ]
        return original_run_cmd(worker_cmd, repo_root, env)

    exact_matching.run_cmd = run_with_snapshot
    try:
        return launch_exact_variant(**kwargs)
    finally:
        exact_matching.run_cmd = original_run_cmd


def run_geometry_worker(argv: List[str]) -> None:
    """Record the current global model after the final FL round."""
    main_path = Path(argv[0]).resolve()
    sys.path.insert(0, str(main_path.parent))
    from backfed.servers.base_server import BaseServer

    original_round = BaseServer.run_one_round

    def round_with_snapshot(self, round_number):
        result = original_round(self, round_number)
        final_round = self.start_round + self.config.num_rounds - 1
        if round_number == final_round:
            model_dir = Path(self.config.output_dir) / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.global_model.state_dict().items()
            }
            torch.save(state, model_dir / f"current_round_{round_number}.pth")
        return result

    BaseServer.run_one_round = round_with_snapshot
    sys.argv = [str(main_path), *argv[1:]]
    runpy.run_path(str(main_path), run_name="__main__")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Measure actual global-model displacement needed for "
            "matched temporal backdoor injection."
        )
    )

    p.add_argument(
        "--matching-summary",
        type=Path,
        default=Path(
            "outputs/exact_temporal_matching/"
            "exact_matching_summary.csv"
        ),
    )

    p.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/temporal_geometry"
        ),
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
        "--force",
        action="store_true",
    )

    return p.parse_args()


def as_float(
    value: object,
) -> Optional[float]:
    if value is None:
        return None

    try:
        x = float(
            str(value).strip()
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(x):
        return None

    return x


def read_matching_summary(
    path: Path,
) -> List[Dict[str, object]]:
    rows = []

    with path.open(
        "r",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for raw in reader:
            status = str(
                raw.get(
                    "status",
                    "",
                )
            )

            if status != "PASS":
                continue

            attack_round = int(
                float(
                    raw["attack_round"]
                )
            )

            scale = as_float(
                raw.get(
                    "matched_scale"
                )
            )

            immediate = as_float(
                raw.get(
                    "immediate_excess_asr"
                )
            )

            if (
                scale is None
                or immediate is None
            ):
                continue

            rows.append(
                {
                    "attack_round": attack_round,
                    "matched_scale": scale,
                    "immediate_excess_asr": immediate,
                }
            )

    if not rows:
        raise RuntimeError(
            f"No PASS rows in {path}"
        )

    return rows


def maybe_state_dict(
    obj: object,
) -> Optional[
    Dict[str, torch.Tensor]
]:
    if hasattr(
        obj,
        "state_dict",
    ):
        try:
            sd = obj.state_dict()

            if isinstance(
                sd,
                dict,
            ):
                return sd
        except Exception:
            pass

    if not isinstance(
        obj,
        dict,
    ):
        return None

    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "model_state",
        "global_model_state_dict",
        "global_model",
        "model",
    ]

    for key in candidate_keys:
        if key not in obj:
            continue

        value = obj[key]

        if isinstance(
            value,
            dict,
        ):
            tensor_count = sum(
                isinstance(v, torch.Tensor)
                for v in value.values()
            )

            if tensor_count >= 5:
                return value

        if hasattr(
            value,
            "state_dict",
        ):
            try:
                return value.state_dict()
            except Exception:
                pass

    tensor_count = sum(
        isinstance(v, torch.Tensor)
        for v in obj.values()
    )

    if tensor_count >= 5:
        return obj  # raw state_dict

    return None


def load_state_dict_file(
    path: Path,
) -> Optional[
    Dict[str, torch.Tensor]
]:
    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception:
        return None

    return maybe_state_dict(
        obj
    )


def find_model_file(
    run_dir: Path,
) -> Tuple[
    Path,
    Dict[str, torch.Tensor],
]:
    current_states = sorted((run_dir / "models").glob("current_round_*.pth"))
    if len(current_states) == 1:
        state = load_state_dict_file(current_states[0])
        if state is not None:
            return current_states[0], state

    suffixes = {
        ".pt",
        ".pth",
        ".ckpt",
        ".tar",
    }

    candidates = [
        path
        for path in run_dir.rglob("*")
        if (
            path.is_file()
            and (
                path.suffix in suffixes
                or path.name.endswith(
                    ".pth.tar"
                )
            )
        )
    ]

    candidates.sort(
        key=lambda path: path.stat().st_size,
        reverse=True,
    )

    for path in candidates:
        sd = load_state_dict_file(
            path
        )

        if sd is None:
            continue

        floating = sum(
            isinstance(v, torch.Tensor)
            and (
                torch.is_floating_point(v)
                or torch.is_complex(v)
            )
            for v in sd.values()
        )

        if floating >= 5:
            return path, sd

    raise RuntimeError(
        f"No model state file found under {run_dir}. "
        "Inspect BackFed save_model behavior."
    )


def common_float_keys(
    *states: Dict[
        str,
        torch.Tensor,
    ],
) -> List[str]:
    keys = set(
        states[0].keys()
    )

    for state in states[1:]:
        keys &= set(
            state.keys()
        )

    valid = []

    for key in sorted(keys):
        tensors = [
            state[key]
            for state in states
        ]

        if not all(
            isinstance(
                tensor,
                torch.Tensor,
            )
            for tensor in tensors
        ):
            continue

        if not all(
            tensor.shape
            == tensors[0].shape
            for tensor in tensors
        ):
            continue

        if not all(
            torch.is_floating_point(
                tensor
            )
            for tensor in tensors
        ):
            continue

        valid.append(
            key
        )

    if not valid:
        raise RuntimeError(
            "No common floating-point parameter keys."
        )

    return valid


def flatten_delta(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    keys: List[str],
) -> torch.Tensor:
    chunks = []

    for key in keys:
        chunks.append(
            (
                a[key]
                .detach()
                .float()
                .reshape(-1)
                -
                b[key]
                .detach()
                .float()
                .reshape(-1)
            )
        )

    return torch.cat(
        chunks
    )


def cosine(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    denom = (
        torch.linalg.vector_norm(a)
        *
        torch.linalg.vector_norm(b)
    )

    if float(denom) <= 1e-20:
        return float("nan")

    return float(
        torch.dot(
            a,
            b,
        )
        / denom
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


def main() -> None:
    args = parse_args()

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    matching_summary = (
        args.matching_summary
        if args.matching_summary.is_absolute()
        else repo_root
        / args.matching_summary
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

    rows = read_matching_summary(
        matching_summary
    )

    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = (
        args.gpu
    )

    env["PYTHONHASHSEED"] = str(
        args.seed
    )

    geometry_rows = []

    for item in rows:
        t = int(
            item["attack_round"]
        )

        scale = float(
            item["matched_scale"]
        )

        round_root = (
            output_root
            / f"round_{t}"
        )

        pre_dir = (
            round_root
            / "pre"
        )

        cf_dir = (
            round_root
            / "counterfactual_t"
        )

        atk_dir = (
            round_root
            / "attack_t"
        )

        pre_csv = launch_variant(
            repo_root=repo_root,
            env=env,
            run_dir=pre_dir,
            seed=args.seed,
            gpu=args.gpu,
            training_mode=args.training_mode,
            num_rounds=t - 1,
            attack_round=t,
            mode="counterfactual",
            scale=1.0,
            force=args.force,
            save_model=True,
        )

        cf_csv = launch_variant(
            repo_root=repo_root,
            env=env,
            run_dir=cf_dir,
            seed=args.seed,
            gpu=args.gpu,
            training_mode=args.training_mode,
            num_rounds=t,
            attack_round=t,
            mode="counterfactual",
            scale=1.0,
            force=args.force,
            save_model=True,
        )

        atk_csv = launch_variant(
            repo_root=repo_root,
            env=env,
            run_dir=atk_dir,
            seed=args.seed,
            gpu=args.gpu,
            training_mode=args.training_mode,
            num_rounds=t,
            attack_round=t,
            mode="attack",
            scale=scale,
            force=args.force,
            save_model=True,
        )

        pre_model_path, pre_state = find_model_file(
            pre_dir
        )

        cf_model_path, cf_state = find_model_file(
            cf_dir
        )

        atk_model_path, atk_state = find_model_file(
            atk_dir
        )

        keys = common_float_keys(
            pre_state,
            cf_state,
            atk_state,
        )

        benign_drift = flatten_delta(
            cf_state,
            pre_state,
            keys,
        )

        attack_disp = flatten_delta(
            atk_state,
            cf_state,
            keys,
        )

        benign_norm = float(
            torch.linalg.vector_norm(
                benign_drift
            )
        )

        attack_norm = float(
            torch.linalg.vector_norm(
                attack_disp
            )
        )

        cos = cosine(
            attack_disp,
            benign_drift,
        )

        if math.isfinite(cos):
            orthogonal_fraction = math.sqrt(
                max(
                    0.0,
                    1.0 - cos * cos,
                )
            )
        else:
            orthogonal_fraction = float("nan")

        cf_idx = index_metrics(
            read_metrics(
                cf_csv
            )
        )

        atk_idx = index_metrics(
            read_metrics(
                atk_csv
            )
        )

        effect = compare_effect(
            attack_idx=atk_idx,
            cf_idx=cf_idx,
            attack_round=t,
        )

        excess = effect[
            "excess_asr"
        ]

        injection_efficiency = (
            excess / attack_norm
            if attack_norm > 1e-20
            else float("nan")
        )

        norm_cost_per_effect = (
            attack_norm / excess
            if excess > 1e-20
            else float("nan")
        )

        relative_attack_to_benign = (
            attack_norm / benign_norm
            if benign_norm > 1e-20
            else float("nan")
        )

        geometry_rows.append(
            {
                "attack_round": t,
                "matched_scale": scale,
                "immediate_excess_asr": excess,
                "clean_drop": effect["clean_drop"],
                "attack_displacement_norm": attack_norm,
                "benign_round_drift_norm": benign_norm,
                "injection_efficiency": injection_efficiency,
                "norm_cost_per_effect": norm_cost_per_effect,
                "attack_to_benign_norm_ratio": relative_attack_to_benign,
                "cos_attack_benign": cos,
                "orthogonal_fraction": orthogonal_fraction,
                "pre_model_file": str(pre_model_path),
                "cf_model_file": str(cf_model_path),
                "attack_model_file": str(atk_model_path),
            }
        )

    write_csv(
        output_root
        / "temporal_geometry_summary.csv",
        geometry_rows,
        [
            "attack_round",
            "matched_scale",
            "immediate_excess_asr",
            "clean_drop",
            "attack_displacement_norm",
            "benign_round_drift_norm",
            "injection_efficiency",
            "norm_cost_per_effect",
            "attack_to_benign_norm_ratio",
            "cos_attack_benign",
            "orthogonal_fraction",
            "pre_model_file",
            "cf_model_file",
            "attack_model_file",
        ],
    )

    with (
        output_root
        / "temporal_geometry_summary.json"
    ).open("w") as f:
        json.dump(
            geometry_rows,
            f,
            indent=2,
        )

    print("\nTEMPORAL GEOMETRY")
    print("=" * 110)

    for row in geometry_rows:
        print(row)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--geometry-worker":
        run_geometry_worker(sys.argv[2:])
    else:
        main()

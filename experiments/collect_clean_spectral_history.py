#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import torch
from run_exact_temporal_matching import build_common_cmd


DEFAULT_TARGETS = [50, 100, 150, 175, 200, 225, 250]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", type=int, nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--max-window", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument(
        "--training-mode",
        choices=["sequential", "parallel"],
        default="sequential",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/clean_spectral_history"),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def needed_rounds(targets, max_window):
    rounds = set()
    for t in targets:
        # 构造 g_{t-K},...,g_{t-1}
        # 需要 w_{t-K-1},...,w_{t-1}
        start = t - max_window - 1
        end = t - 1
        if start < 0:
            raise ValueError("Target too early.")
        rounds.update(range(start, end + 1))
    return sorted(rounds)


def run_clean_worker(argv):
    """Save actual round-end models; BackFed's standard checkpoint stores best models."""
    main_path = Path(argv[0]).resolve()
    sys.path.insert(0, str(main_path.parent))
    from backfed.servers.base_server import BaseServer

    original_round = BaseServer.run_one_round

    def round_with_snapshot(self, round_number):
        result = original_round(self, round_number)
        if round_number in self.config.save_checkpoint_rounds:
            checkpoint_dir = Path(self.config.output_dir) / "current_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.global_model.state_dict().items()
            }
            torch.save(
                {"server_round": round_number, "model_state": state},
                checkpoint_dir / f"current_round_{round_number}.pth",
            )
        return result

    BaseServer.run_one_round = round_with_snapshot
    sys.argv = [str(main_path), *argv[1:]]
    runpy.run_path(str(main_path), run_name="__main__")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rounds = needed_rounds(args.targets, args.max_window)

    with (output_dir / "requested_rounds.json").open("w") as f:
        json.dump(
            {
                "targets": args.targets,
                "max_window": args.max_window,
                "checkpoint_rounds": rounds,
                "num_checkpoints": len(rounds),
            },
            f,
            indent=2,
        )

    round_list = "[" + ",".join(str(x) for x in rounds) + "]"

    run_dir = output_dir / "clean_run"

    # Configure the single-shot adversary for the round after training ends.
    # This consumes the same selection RNG as every matched attack run while
    # all observed history rounds remain benign.
    common = build_common_cmd(
        repo_root=repo_root,
        run_dir=run_dir,
        seed=args.seed,
        gpu=args.gpu,
        training_mode=args.training_mode,
        num_rounds=max(args.targets) - 1,
        attack_round=max(args.targets),
    )
    cmd = [
        common[0], str(Path(__file__).resolve()),
        "--clean-worker", *common[1:],
        "save_checkpoint=false",
        "save_checkpoint_rounds=" + round_list,
    ]

    print("Need checkpoints:", rounds)
    print(" ".join(cmd))

    if args.dry_run:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONHASHSEED"] = str(args.seed)

    subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        check=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--clean-worker":
        run_clean_worker(sys.argv[2:])
    else:
        main()

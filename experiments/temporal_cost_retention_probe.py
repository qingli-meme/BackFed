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
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_ROUNDS = [50, 150, 250]
DEFAULT_SCALES = [1.0, 2.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure round-wise injection cost and normalized retention in BackFed."
    )
    p.add_argument("--attack-rounds", type=int, nargs="+", default=DEFAULT_ROUNDS)
    p.add_argument("--scale-grid", type=float, nargs="+", default=DEFAULT_SCALES)
    p.add_argument("--target-excess-asr", type=float, default=0.30)
    p.add_argument("--max-clean-drop", type=float, default=0.05)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument(
        "--training-mode",
        choices=["sequential", "parallel"],
        default="sequential",
    )
    p.add_argument("--binary-refine-steps", type=int, default=2)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/temporal_cost_retention_probe"),
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def norm_col(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def as_float(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        y = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def round_from_row(row: Dict[str, str]) -> Optional[int]:
    for k in ("round", "step"):
        if k in row:
            x = as_float(row[k])
            if x is not None:
                return int(round(x))
    return None


def read_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            out.append({norm_col(k): v for k, v in row.items() if k is not None})
        return out


def find_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("*.csv"))
    scored: List[Tuple[int, int, Path]] = []
    for p in candidates:
        try:
            with p.open("r", newline="") as f:
                header = next(csv.reader(f), None)
        except Exception:
            continue
        if not header:
            continue
        cols = {norm_col(c) for c in header}
        score = 0
        if "test_backdoor_acc" in cols:
            score += 100
        if "test_clean_acc" in cols:
            score += 50
        if "round" in cols or "step" in cols:
            score += 10
        if score:
            scored.append((score, p.stat().st_size, p))
    if not scored:
        raise RuntimeError(f"No metrics CSV with test_backdoor_acc found under {run_dir}")
    scored.sort(reverse=True)
    return scored[0][2]


def index_metrics(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for row in rows:
        r = round_from_row(row)
        if r is None:
            continue
        bd = as_float(row.get("test_backdoor_acc"))
        ca = as_float(row.get("test_clean_acc"))
        if bd is None:
            continue
        out[r] = {
            "backdoor_acc": bd,
            "clean_acc": ca if ca is not None else float("nan"),
        }
    return out


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print("\n" + "=" * 110)
    print(" ".join(cmd))
    print("=" * 110)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def common_overrides(
    *,
    seed: int,
    gpu: str,
    training_mode: str,
    num_rounds: int,
    run_dir: Path,
) -> List[str]:
    return [
        "--config-name", "cifar10",
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
        "test_every=1",
        "no_attack=false",
        "atk_config=cifar10_multishot",
        "atk_config.data_poison_method=pattern",
        "atk_config.model_poison_method=base",
        "atk_config.use_atk_optimizer=true",
        "atk_config.poison_lr=0.05",
        "atk_config.poison_epochs=6",
        "atk_config.poison_ratio=0.3125",
        "atk_config.poison_mode=online",
        "atk_config.attack_type=all2one",
        "atk_config.random_class=false",
        "atk_config.target_class=2",
        "atk_config.adversary_selection=single",
        "atk_config.selection_scheme=all-adversary",
        "atk_config.poison_frequency=single-shot",
        "atk_config.poison_interval=1",
        f"hydra.run.dir={run_dir.resolve()}",
        "dir_tag=temporal_cost_retention_probe",
    ]


def launch_run(
    *,
    repo_root: Path,
    env: Dict[str, str],
    run_dir: Path,
    seed: int,
    gpu: str,
    training_mode: str,
    num_rounds: int,
    attack_round: int,
    scale_poison: bool,
    scale_factor: float,
    force: bool,
) -> Path:
    if run_dir.exists():
        if force:
            shutil.rmtree(run_dir)
        else:
            metrics = find_metrics_csv(run_dir)
            print(f"[REUSE] {run_dir}")
            return metrics

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--backfed-worker", str(repo_root / "main.py")]
    cmd += common_overrides(
        seed=seed,
        gpu=gpu,
        training_mode=training_mode,
        num_rounds=num_rounds,
        run_dir=run_dir,
    )
    cmd += [
        f"atk_config.poison_start_round={attack_round}",
        f"atk_config.poison_end_round={attack_round}",
        f"atk_config.scale_poison={'true' if scale_poison else 'false'}",
        f"atk_config.scale_factor={scale_factor}",
    ]
    run_cmd(cmd, repo_root, env)
    return find_metrics_csv(run_dir)


def run_backfed_worker(argv: List[str]) -> None:
    """Evaluate the fixed trigger every round without changing training RNG."""
    import runpy
    import torch
    sys.path.insert(0, str(Path(argv[0]).resolve().parent))
    from backfed.client_manager import ClientManager
    from backfed.servers.base_server import BaseServer

    normal_rounds = ClientManager._initialize_normal_rounds

    def matched_normal_rounds(self):
        # BackFed otherwise skips its one attack-selection RNG draw when the
        # clean counterfactual's poisoning round is beyond the training run.
        if (self.atk_config is not None and
                self.atk_config.poison_start_round > self.start_round + self.config.num_rounds):
            self._all_adversary_selection([self.atk_config.poison_start_round])
        return normal_rounds(self)

    ClientManager._initialize_normal_rounds = matched_normal_rounds
    original_evaluate = BaseServer.server_evaluate

    def evaluate_with_trigger(self, round_number=None, test_poisoned=True, model=None):
        metrics = original_evaluate(
            self, round_number=round_number, test_poisoned=False, model=model
        )
        if test_poisoned and self.poison_module is not None:
            self.poison_module.set_client_id(-1)
            with torch.random.fork_rng():
                total, loss, accuracy = self.poison_module.poison_test(
                    net=self.global_model if model is None else model,
                    test_loader=self.test_loader,
                    loss_fn=torch.nn.CrossEntropyLoss(),
                    normalization=self.normalization,
                )
            metrics.update({
                "test_backdoor_samples": total,
                "test_backdoor_loss": loss,
                "test_backdoor_acc": accuracy,
            })
        return metrics

    BaseServer.server_evaluate = evaluate_with_trigger
    sys.argv = argv
    runpy.run_path(argv[0], run_name="__main__")


def get_at_round(index: Dict[int, Dict[str, float]], r: int) -> Dict[str, float]:
    if r not in index:
        available = sorted(index)
        raise RuntimeError(
            f"Round {r} missing. Available range: "
            f"{available[0] if available else None}..{available[-1] if available else None}"
        )
    return index[r]


def check_preattack_match(
    clean_idx: Dict[int, Dict[str, float]],
    attack_idx: Dict[int, Dict[str, float]],
    attack_round: int,
    tol: float = 1e-6,
) -> None:
    r = attack_round - 1
    if r <= 0 or r not in attack_idx or r not in clean_idx:
        return
    a = attack_idx[r]["clean_acc"]
    b = clean_idx[r]["clean_acc"]
    if math.isfinite(a) and math.isfinite(b) and abs(a - b) > tol:
        raise RuntimeError(
            f"Pre-attack trajectory mismatch at round {r}: attack={a}, clean={b}, diff={abs(a-b)}"
        )


def attack_effect(
    clean_idx: Dict[int, Dict[str, float]],
    attack_idx: Dict[int, Dict[str, float]],
    attack_round: int,
) -> Dict[str, float]:
    c = get_at_round(clean_idx, attack_round)
    a = get_at_round(attack_idx, attack_round)
    return {
        "attack_asr": a["backdoor_acc"],
        "clean_asr": c["backdoor_acc"],
        "excess_asr": a["backdoor_acc"] - c["backdoor_acc"],
        "attack_clean_acc": a["clean_acc"],
        "clean_clean_acc": c["clean_acc"],
        "clean_drop": c["clean_acc"] - a["clean_acc"],
    }


def success(effect: Dict[str, float], target_excess: float, max_clean_drop: float) -> bool:
    return (
        effect["excess_asr"] >= target_excess
        and effect["clean_drop"] <= max_clean_drop
    )


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run_injection_search(
    *,
    repo_root: Path,
    env: Dict[str, str],
    output_root: Path,
    clean_idx: Dict[int, Dict[str, float]],
    attack_round: int,
    scale_grid: List[float],
    target_excess: float,
    max_clean_drop: float,
    refine_steps: int,
    seed: int,
    gpu: str,
    training_mode: str,
    force: bool,
) -> Tuple[Optional[float], List[Dict[str, float]]]:
    history: List[Dict[str, float]] = []
    low_fail: Optional[float] = None
    high_success: Optional[float] = None

    def evaluate(scale: float) -> Dict[str, float]:
        tag = str(scale).replace(".", "p")
        run_dir = output_root / f"round_{attack_round}" / "search" / f"scale_{tag}"
        metrics = launch_run(
            repo_root=repo_root,
            env=env,
            run_dir=run_dir,
            seed=seed,
            gpu=gpu,
            training_mode=training_mode,
            num_rounds=attack_round,
            attack_round=attack_round,
            scale_poison=True,
            scale_factor=scale,
            force=force,
        )
        idx = index_metrics(read_metrics(metrics))
        check_preattack_match(clean_idx, idx, attack_round)
        eff = attack_effect(clean_idx, idx, attack_round)
        row = {"scale": scale, **eff}
        history.append(row)
        print(
            f"[SEARCH t={attack_round}] scale={scale:.6g} "
            f"excess_asr={eff['excess_asr']:.4f} "
            f"clean_drop={eff['clean_drop']:.4f}"
        )
        return eff

    for scale in sorted(set(scale_grid)):
        eff = evaluate(scale)
        if success(eff, target_excess, max_clean_drop):
            high_success = scale
            break
        low_fail = scale

    if high_success is None:
        return None, history

    if low_fail is not None and refine_steps > 0:
        lo, hi = low_fail, high_success
        for _ in range(refine_steps):
            mid = (lo + hi) / 2.0
            eff = evaluate(mid)
            if success(eff, target_excess, max_clean_drop):
                hi = mid
            else:
                lo = mid
        high_success = hi

    return high_success, history


def compute_retention(
    *,
    clean_idx: Dict[int, Dict[str, float]],
    attack_idx: Dict[int, Dict[str, float]],
    attack_round: int,
    horizon: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    curve: List[Dict[str, float]] = []
    for h in range(0, horizon + 1):
        r = attack_round + h
        c = get_at_round(clean_idx, r)
        a = get_at_round(attack_idx, r)
        excess = a["backdoor_acc"] - c["backdoor_acc"]
        curve.append(
            {
                "h": h,
                "round": r,
                "attack_asr": a["backdoor_acc"],
                "clean_asr": c["backdoor_acc"],
                "excess_asr": excess,
                "attack_clean_acc": a["clean_acc"],
                "clean_clean_acc": c["clean_acc"],
                "clean_drop": c["clean_acc"] - a["clean_acc"],
            }
        )

    e0 = curve[0]["excess_asr"]
    denom = max(e0, 1e-12)
    post = curve[1:]
    normalized_auc = (
        sum(max(x["excess_asr"], 0.0) / denom for x in post) / len(post)
        if post else 0.0
    )

    half_life = horizon + 1
    threshold = 0.5 * e0
    for x in post:
        if x["excess_asr"] <= threshold:
            half_life = int(x["h"])
            break

    return (
        {
            "immediate_excess_asr": e0,
            "retention_auc": normalized_auc,
            "half_life": half_life,
            "end_excess_asr": curve[-1]["excess_asr"],
            "peak_excess_asr": max(x["excess_asr"] for x in curve),
        },
        curve,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "main.py").exists():
        raise FileNotFoundError(f"BackFed main.py not found under {repo_root}")

    attack_rounds = list(args.attack_rounds)
    scale_grid = list(args.scale_grid)
    if args.smoke:
        attack_rounds = attack_rounds[:2]
        scale_grid = scale_grid[:3]
        args.horizon = min(args.horizon, 5)
        args.binary_refine_steps = 0

    output_root = repo_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONHASHSEED"] = str(args.seed)

    max_round = max(attack_rounds) + args.horizon
    never_attack_round = max_round + 1000

    clean_dir = output_root / "clean_trigger_baseline"
    clean_csv = launch_run(
        repo_root=repo_root,
        env=env,
        run_dir=clean_dir,
        seed=args.seed,
        gpu=args.gpu,
        training_mode=args.training_mode,
        num_rounds=max_round,
        attack_round=never_attack_round,
        scale_poison=False,
        scale_factor=1.0,
        force=args.force,
    )
    clean_idx = index_metrics(read_metrics(clean_csv))

    summaries: List[Dict[str, object]] = []

    for t in attack_rounds:
        chosen_scale, search_history = run_injection_search(
            repo_root=repo_root,
            env=env,
            output_root=output_root,
            clean_idx=clean_idx,
            attack_round=t,
            scale_grid=scale_grid,
            target_excess=args.target_excess_asr,
            max_clean_drop=args.max_clean_drop,
            refine_steps=args.binary_refine_steps,
            seed=args.seed,
            gpu=args.gpu,
            training_mode=args.training_mode,
            force=args.force,
        )

        write_csv(
            output_root / f"round_{t}" / "injection_search.csv",
            search_history,
            [
                "scale",
                "attack_asr",
                "clean_asr",
                "excess_asr",
                "attack_clean_acc",
                "clean_clean_acc",
                "clean_drop",
            ],
        )

        if chosen_scale is None:
            summaries.append(
                {
                    "attack_round": t,
                    "reachable": False,
                    "injection_cost_scale": float("nan"),
                    "immediate_excess_asr": float("nan"),
                    "retention_auc": float("nan"),
                    "half_life": float("nan"),
                    "end_excess_asr": float("nan"),
                    "peak_excess_asr": float("nan"),
                    "temporal_vulnerability": float("nan"),
                }
            )
            print(f"[UNREACHABLE] round={t}")
            continue

        tag = str(chosen_scale).replace(".", "p")
        retain_dir = output_root / f"round_{t}" / "retention" / f"scale_{tag}"
        retain_csv = launch_run(
            repo_root=repo_root,
            env=env,
            run_dir=retain_dir,
            seed=args.seed,
            gpu=args.gpu,
            training_mode=args.training_mode,
            num_rounds=t + args.horizon,
            attack_round=t,
            scale_poison=True,
            scale_factor=chosen_scale,
            force=args.force,
        )
        retain_idx = index_metrics(read_metrics(retain_csv))
        check_preattack_match(clean_idx, retain_idx, t)

        retention_summary, curve = compute_retention(
            clean_idx=clean_idx,
            attack_idx=retain_idx,
            attack_round=t,
            horizon=args.horizon,
        )
        vulnerability = retention_summary["retention_auc"] / max(chosen_scale, 1e-12)

        write_csv(
            output_root / f"round_{t}" / "retention_curve.csv",
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

        summaries.append(
            {
                "attack_round": t,
                "reachable": True,
                "injection_cost_scale": chosen_scale,
                **retention_summary,
                "temporal_vulnerability": vulnerability,
            }
        )

    summary_csv = output_root / "cost_retention_summary.csv"
    write_csv(
        summary_csv,
        summaries,
        [
            "attack_round",
            "reachable",
            "injection_cost_scale",
            "immediate_excess_asr",
            "retention_auc",
            "half_life",
            "end_excess_asr",
            "peak_excess_asr",
            "temporal_vulnerability",
        ],
    )
    with (output_root / "cost_retention_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)

    print("\nFINAL SUMMARY")
    print("=" * 110)
    for row in summaries:
        print(row)
    print(f"\nSaved: {summary_csv}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--backfed-worker":
        run_backfed_worker(sys.argv[2:])
    else:
        main()

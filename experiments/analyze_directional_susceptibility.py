#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--geometry-csv",
        type=Path,
        default=Path(
            "outputs/temporal_geometry/temporal_geometry_summary.csv"
        ),
    )

    p.add_argument(
        "--probe-root",
        type=Path,
        default=Path(
            "outputs/temporal_directional_probe_raw"
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/directional_susceptibility_analysis"
        ),
    )

    return p.parse_args()


def as_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        y = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    i = 0

    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1

    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def read_geometry(path: Path) -> Dict[int, Dict[str, float]]:
    rows = {}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for raw in reader:
            r = int(float(raw["attack_round"]))
            q = as_float(raw.get("immediate_excess_asr"))
            pnorm = as_float(raw.get("attack_displacement_norm"))
            gnorm = as_float(raw.get("benign_round_drift_norm"))
            cos = as_float(raw.get("cos_attack_benign"))

            if (
                q is None
                or pnorm is None
                or gnorm is None
                or pnorm <= 0
                or gnorm <= 0
            ):
                continue

            rows[r] = {
                "attack_round": r,
                "q": q,
                "attack_displacement_norm": pnorm,
                "benign_drift_norm": gnorm,
                "injection_efficiency": q / pnorm,
                "inverse_benign_drift": 1.0 / gnorm,
                "cos_attack_benign": (
                    cos if cos is not None else float("nan")
                ),
            }

    return rows


def read_probe_files(root: Path) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = {}

    for path in sorted(root.glob("round_*_client_*.json")):
        try:
            with path.open("r") as f:
                obj = json.load(f)
        except Exception:
            continue

        r = obj.get("server_round", None)
        if r is None:
            continue

        try:
            r = int(r)
        except Exception:
            continue

        obj["_path"] = str(path)
        out.setdefault(r, []).append(obj)

    return out


def select_probe(records: List[Dict]) -> Optional[Dict]:
    valid = []

    for obj in records:
        required = [
            as_float(obj.get("probe_bds")),
            as_float(obj.get("probe_bd_gain_slope")),
            as_float(obj.get("probe_clean_abs_slope")),
            as_float(obj.get("probe_raw_update_norm")),
        ]

        if all(x is not None for x in required):
            valid.append(obj)

    if not valid:
        return None

    valid = sorted(
        valid,
        key=lambda obj: float(obj["probe_bds"]),
    )

    return valid[len(valid) // 2]


def log_safe(x: float) -> float:
    return math.log(max(float(x), 1e-12))


def fit_linear(X: np.ndarray, y: np.ndarray):
    X_aug = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return coef


def predict_linear(coef: np.ndarray, X: np.ndarray):
    X_aug = np.column_stack([np.ones(len(X)), X])
    return X_aug @ coef


def loo_mae(X: np.ndarray, y: np.ndarray) -> float:
    preds = []
    targets = []

    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False

        coef = fit_linear(X[mask], y[mask])
        pred = predict_linear(coef, X[i : i + 1])[0]

        preds.append(pred)
        targets.append(y[i])

    preds = np.asarray(preds)
    targets = np.asarray(targets)

    return float(np.mean(np.abs(preds - targets)))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    geometry_path = (
        args.geometry_csv
        if args.geometry_csv.is_absolute()
        else repo_root / args.geometry_csv
    )

    probe_root = (
        args.probe_root
        if args.probe_root.is_absolute()
        else repo_root / args.probe_root
    )

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = read_geometry(geometry_path)
    probe_files = read_probe_files(probe_root)

    rows = []

    for round_id in sorted(geometry):
        probe = select_probe(probe_files.get(round_id, []))
        if probe is None:
            continue

        g = geometry[round_id]

        row = {
            **g,
            "probe_raw_update_norm": float(probe["probe_raw_update_norm"]),
            "probe_bd_gain_slope": float(probe["probe_bd_gain_slope"]),
            "probe_clean_signed_slope": float(
                probe["probe_clean_signed_slope"]
            ),
            "probe_clean_abs_slope": float(
                probe["probe_clean_abs_slope"]
            ),
            "probe_bds": float(probe["probe_bds"]),
            "probe_bd_update_cos": float(probe["probe_bd_update_cos"]),
            "probe_clean_update_cos": float(
                probe["probe_clean_update_cos"]
            ),
            "probe_path": probe["_path"],
        }

        rows.append(row)

    if len(rows) < 6:
        raise RuntimeError(
            f"Need >= 6 matched rounds with probe records; got {len(rows)}"
        )

    write_csv(
        output_dir / "round_features.csv",
        rows,
        [
            "attack_round",
            "q",
            "attack_displacement_norm",
            "benign_drift_norm",
            "injection_efficiency",
            "inverse_benign_drift",
            "cos_attack_benign",
            "probe_raw_update_norm",
            "probe_bd_gain_slope",
            "probe_clean_signed_slope",
            "probe_clean_abs_slope",
            "probe_bds",
            "probe_bd_update_cos",
            "probe_clean_update_cos",
            "probe_path",
        ],
    )

    y = np.asarray([r["injection_efficiency"] for r in rows], dtype=float)
    inv_drift = np.asarray([r["inverse_benign_drift"] for r in rows], dtype=float)
    bds = np.asarray([r["probe_bds"] for r in rows], dtype=float)
    bd_gain = np.asarray([r["probe_bd_gain_slope"] for r in rows], dtype=float)
    clean_abs = np.asarray([r["probe_clean_abs_slope"] for r in rows], dtype=float)
    raw_update_norm = np.asarray(
        [r["probe_raw_update_norm"] for r in rows], dtype=float
    )

    metrics = {}

    candidate_features = {
        "inverse_benign_drift": inv_drift,
        "BDS": bds,
        "BD_gain_slope": bd_gain,
        "clean_abs_slope": clean_abs,
        "raw_local_update_norm": raw_update_norm,
    }

    for name, x in candidate_features.items():
        metrics[f"spearman_{name}_vs_efficiency"] = spearman(x, y)
        metrics[f"pearson_{name}_vs_efficiency"] = pearson(x, y)

    log_y = np.asarray([log_safe(v) for v in y])
    log_inv_drift = np.asarray([log_safe(v) for v in inv_drift]).reshape(-1, 1)
    log_bds = np.asarray([log_safe(v) for v in bds]).reshape(-1, 1)

    X_drift = log_inv_drift
    X_drift_bds = np.column_stack(
        [log_inv_drift[:, 0], log_bds[:, 0]]
    )

    drift_mae = loo_mae(X_drift, log_y)
    drift_bds_mae = loo_mae(X_drift_bds, log_y)

    improvement = (
        (drift_mae - drift_bds_mae)
        / max(drift_mae, 1e-12)
    )

    metrics["loo_log_mae_drift_only"] = drift_mae
    metrics["loo_log_mae_drift_plus_bds"] = drift_bds_mae
    metrics["loo_relative_improvement_with_bds"] = improvement

    coef = fit_linear(X_drift, log_y)
    drift_pred = predict_linear(coef, X_drift)
    residual = log_y - drift_pred

    metrics["spearman_BDS_vs_drift_residual"] = spearman(
        bds,
        residual,
    )

    metrics["pearson_logBDS_vs_drift_residual"] = pearson(
        log_bds[:, 0],
        residual,
    )

    with (output_dir / "analysis_summary.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nROUND FEATURES")
    print("=" * 110)

    for row in rows:
        print(
            f"round={row['attack_round']:4d} | "
            f"eff={row['injection_efficiency']:.6f} | "
            f"1/drift={row['inverse_benign_drift']:.6f} | "
            f"BDS={row['probe_bds']:.6f} | "
            f"BD_gain={row['probe_bd_gain_slope']:.6f} | "
            f"clean_abs={row['probe_clean_abs_slope']:.6f}"
        )

    print("\nANALYSIS")
    print("=" * 110)

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

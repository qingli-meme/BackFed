#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--features",
        type=Path,
        default=Path(
            "outputs/joint_spectral_conflict/"
            "joint_spectral_features.csv"
        ),
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/joint_spectral_conflict/"
            "diagnostics.json"
        ),
    )

    return p.parse_args()


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty(
        len(x),
        dtype=float,
    )

    i = 0

    while i < len(x):
        j = i

        while (
            j + 1 < len(x)
            and x[order[j + 1]]
            == x[order[i]]
        ):
            j += 1

        rank = (
            i + j
        ) / 2.0 + 1.0

        for k in range(
            i,
            j + 1,
        ):
            ranks[
                order[k]
            ] = rank

        i = j + 1

    return ranks


def pearson(x, y):
    x = np.asarray(
        x,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )

    if (
        len(x) < 2
        or np.std(x) <= 1e-12
        or np.std(y) <= 1e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def spearman(x, y):
    return pearson(
        rankdata(
            np.asarray(
                x,
                dtype=float,
            )
        ),
        rankdata(
            np.asarray(
                y,
                dtype=float,
            )
        ),
    )


def fit_linear(X, y):
    X = np.asarray(
        X,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )

    X_aug = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    coef, *_ = np.linalg.lstsq(
        X_aug,
        y,
        rcond=None,
    )

    return coef


def predict(coef, X):
    X = np.asarray(
        X,
        dtype=float,
    )

    X_aug = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    return X_aug @ coef


def loo_mae(X, y):
    X = np.asarray(
        X,
        dtype=float,
    )
    y = np.asarray(
        y,
        dtype=float,
    )

    preds = []

    for i in range(len(y)):
        mask = np.ones(
            len(y),
            dtype=bool,
        )
        mask[i] = False

        coef = fit_linear(
            X[mask],
            y[mask],
        )

        preds.append(
            predict(
                coef,
                X[i : i + 1],
            )[0]
        )

    return float(
        np.mean(
            np.abs(
                np.asarray(preds)
                - y
            )
        )
    )


def log_safe(x):
    return math.log(
        max(
            float(x),
            1e-12,
        )
    )


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    features = (
        args.features
        if args.features.is_absolute()
        else repo_root / args.features
    )

    output = (
        args.output
        if args.output.is_absolute()
        else repo_root / args.output
    )

    rows = []

    with features.open(
        "r",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for raw in reader:
            row = dict(raw)

            for key in [
                "attack_round",
                "injection_efficiency",
                "inverse_benign_drift",
                "previous_bds",
                "spectral_conflict",
                "spectral_opportunity",
            ]:
                row[key] = float(
                    row[key]
                )

            rows.append(row)

    rows = sorted(
        rows,
        key=lambda x: x["attack_round"],
    )

    eff = np.asarray(
        [
            x["injection_efficiency"]
            for x in rows
        ]
    )

    inv_drift = np.asarray(
        [
            x["inverse_benign_drift"]
            for x in rows
        ]
    )

    bds = np.asarray(
        [
            x["previous_bds"]
            for x in rows
        ]
    )

    conflict = np.asarray(
        [
            x["spectral_conflict"]
            for x in rows
        ]
    )

    opportunity = np.asarray(
        [
            x["spectral_opportunity"]
            for x in rows
        ]
    )

    log_eff = np.asarray(
        [
            log_safe(x)
            for x in eff
        ]
    )

    log_inv_drift = np.asarray(
        [
            log_safe(x)
            for x in inv_drift
        ]
    )

    log_bds = np.asarray(
        [
            log_safe(x)
            for x in bds
        ]
    )

    X_drift = (
        log_inv_drift
        .reshape(-1, 1)
    )

    X_drift_bds = (
        np.column_stack(
            [
                log_inv_drift,
                log_bds,
            ]
        )
    )

    X_full = (
        np.column_stack(
            [
                log_inv_drift,
                log_bds,
                conflict,
            ]
        )
    )

    mae_drift = loo_mae(
        X_drift,
        log_eff,
    )

    mae_drift_bds = loo_mae(
        X_drift_bds,
        log_eff,
    )

    mae_full = loo_mae(
        X_full,
        log_eff,
    )

    improvement = (
        (
            mae_drift_bds
            - mae_full
        )
        /
        max(
            mae_drift_bds,
            1e-12,
        )
    )

    coef = fit_linear(
        X_drift_bds,
        log_eff,
    )

    residual = (
        log_eff
        - predict(
            coef,
            X_drift_bds,
        )
    )

    result = {
        "spearman_inverse_drift_vs_efficiency": (
            spearman(
                inv_drift,
                eff,
            )
        ),
        "spearman_bds_vs_efficiency": (
            spearman(
                bds,
                eff,
            )
        ),
        "spearman_conflict_vs_efficiency": (
            spearman(
                conflict,
                eff,
            )
        ),
        "spearman_opportunity_vs_efficiency": (
            spearman(
                opportunity,
                eff,
            )
        ),
        "spearman_conflict_vs_drift_bds_residual": (
            spearman(
                conflict,
                residual,
            )
        ),
        "loo_log_mae_drift_only": (
            mae_drift
        ),
        "loo_log_mae_drift_plus_bds": (
            mae_drift_bds
        ),
        "loo_log_mae_drift_bds_conflict": (
            mae_full
        ),
        "relative_improvement_conflict_over_drift_bds": (
            improvement
        ),
    }

    by_round = {
        int(x["attack_round"]): x
        for x in rows
    }

    targets = [200, 225, 250]

    if all(
        x in by_round
        for x in targets
    ):
        conflicts = [
            by_round[x][
                "spectral_conflict"
            ]
            for x in targets
        ]

        efficiencies = [
            by_round[x][
                "injection_efficiency"
            ]
            for x in targets
        ]

        result[
            "reversal_conflict_increasing_200_225_250"
        ] = bool(
            conflicts[0]
            < conflicts[1]
            < conflicts[2]
        )

        result[
            "reversal_efficiency_decreasing_200_225_250"
        ] = bool(
            efficiencies[0]
            > efficiencies[1]
            > efficiencies[2]
        )

        result[
            "reversal_conflicts"
        ] = conflicts

        result[
            "reversal_efficiencies"
        ] = efficiencies

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

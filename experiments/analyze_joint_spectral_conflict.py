#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
import torch
from scipy.fft import dct


EPS = 1e-12
DEFAULT_TARGETS = [50, 100, 150, 175, 200, 225, 250]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=DEFAULT_TARGETS,
    )
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints"),
    )
    p.add_argument(
        "--spectral-state-root",
        type=Path,
        default=Path("outputs/temporal_spectral_state"),
    )
    p.add_argument(
        "--geometry-csv",
        type=Path,
        default=Path(
            "outputs/temporal_geometry/"
            "temporal_geometry_summary.csv"
        ),
    )
    p.add_argument(
        "--directional-feature-csv",
        type=Path,
        default=Path(
            "outputs/directional_susceptibility_analysis/"
            "round_features.csv"
        ),
    )
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--bands", type=int, default=8)
    p.add_argument("--wavelet", type=str, default="db2")
    p.add_argument("--wavelet-level", type=int, default=3)
    p.add_argument("--min-param-size", type=int, default=256)
    p.add_argument("--recent-fraction", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/joint_spectral_conflict"),
    )
    return p.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def as_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        y = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(y):
        return None
    return y


def maybe_state_dict(obj):
    if hasattr(obj, "state_dict"):
        try:
            sd = obj.state_dict()
            if isinstance(sd, dict):
                return sd
        except Exception:
            pass

    if not isinstance(obj, dict):
        return None

    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "model_state",
        "global_model_state_dict",
        "global_model",
        "model",
        "weights",
    ]

    for key in candidate_keys:
        value = obj.get(key, None)

        if isinstance(value, dict):
            tensor_count = sum(
                isinstance(v, torch.Tensor)
                for v in value.values()
            )
            if tensor_count >= 5:
                return value

        if hasattr(value, "state_dict"):
            try:
                return value.state_dict()
            except Exception:
                pass

    tensor_count = sum(
        isinstance(v, torch.Tensor)
        for v in obj.values()
    )
    if tensor_count >= 5:
        return obj

    return None


def extract_round_from_object(obj) -> Optional[int]:
    if not isinstance(obj, dict):
        return None

    for key in [
        "round",
        "server_round",
        "current_round",
        "global_round",
        "epoch",
    ]:
        if key not in obj:
            continue
        try:
            return int(float(obj[key]))
        except Exception:
            pass

    for key in [
        "metadata",
        "meta",
        "state",
        "trainer_state",
    ]:
        value = obj.get(key, None)
        if isinstance(value, dict):
            out = extract_round_from_object(value)
            if out is not None:
                return out

    return None


def parse_round_from_name(path: Path) -> Optional[int]:
    name = path.name.lower()

    for pattern in [
        r"round[_\-]?(\d+)",
        r"checkpoint[_\-]?(\d+)",
        r"ckpt[_\-]?(\d+)",
        r"epoch[_\-]?(\d+)",
    ]:
        m = re.search(pattern, name)
        if m:
            return int(m.group(1))

    return None


def build_checkpoint_index(root: Path) -> Dict[int, Path]:
    suffixes = {".pt", ".pth", ".ckpt", ".tar"}

    files = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and (
                path.suffix.lower() in suffixes
                or path.name.endswith(".pth.tar")
            )
        )
    ]

    index = {}

    for path in files:
        round_id = parse_round_from_name(path)

        if round_id is None:
            try:
                obj = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )
            except Exception:
                continue

            round_id = extract_round_from_object(obj)

        if round_id is None:
            continue

        if round_id in index:
            if path.stat().st_mtime <= index[round_id].stat().st_mtime:
                continue

        index[round_id] = path

    return index


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    state = maybe_state_dict(obj)

    if state is None:
        raise RuntimeError(
            f"Cannot find model state_dict in {path}"
        )

    return state


def load_spectral_state(
    root: Path,
    target: int,
) -> Dict[str, object]:
    candidates = sorted(
        root.glob(f"round_{target}_client_*.pt")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No spectral state for round {target} under {root}"
        )

    path = max(
        candidates,
        key=lambda x: x.stat().st_mtime,
    )

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    obj["_path"] = str(path)
    return obj


def dct_vec(x: torch.Tensor) -> np.ndarray:
    arr = (
        x.detach()
        .cpu()
        .float()
        .reshape(-1)
        .numpy()
    )

    return dct(
        arr,
        type=2,
        norm="ortho",
    )


def band_slices(
    n: int,
    bands: int,
) -> List[slice]:
    if n < bands:
        return [slice(0, n)]

    edges = np.linspace(
        0,
        n,
        bands + 1,
        dtype=int,
    )

    out = []

    for i in range(bands):
        a = int(edges[i])
        b = int(edges[i + 1])

        if b > a:
            out.append(slice(a, b))

    return out


def parameter_keys(
    *,
    probe,
    state,
    min_param_size,
):
    raw_update = probe["raw_update"]
    bd_grad = probe["backdoor_grad"]
    clean_grad = probe["clean_grad"]

    keys = []

    for key in raw_update:
        if (
            key not in bd_grad
            or key not in clean_grad
            or key not in state
        ):
            continue

        values = [
            raw_update[key],
            bd_grad[key],
            clean_grad[key],
            state[key],
        ]

        if not all(
            isinstance(v, torch.Tensor)
            for v in values
        ):
            continue

        if not all(
            torch.is_floating_point(v)
            for v in values
        ):
            continue

        if raw_update[key].numel() < min_param_size:
            continue

        keys.append(key)

    if not keys:
        raise RuntimeError(
            "No valid trainable parameter tensors."
        )

    return sorted(keys)


def update_between(
    newer,
    older,
    key,
):
    return (
        newer[key].detach().float().cpu()
        -
        older[key].detach().float().cpu()
    )


def safe_cosine_band(
    h: np.ndarray,
    g: np.ndarray,
) -> float:
    hn = float(np.linalg.norm(h))
    gn = float(np.linalg.norm(g))

    if hn <= EPS or gn <= EPS:
        return 0.0

    return float(
        np.dot(h, g)
        / (hn * gn)
    )


def wavelet_persistent_resistance(
    seq: np.ndarray,
    wavelet: str,
    level: int,
    recent_fraction: float,
):
    seq = np.asarray(
        seq,
        dtype=np.float64,
    )

    wave = pywt.Wavelet(wavelet)

    max_level = pywt.dwt_max_level(
        len(seq),
        wave.dec_len,
    )

    level = max(
        1,
        min(level, max_level),
    )

    coeffs = pywt.wavedec(
        seq,
        wavelet=wave,
        level=level,
        mode="periodization",
    )

    approx_energy = float(
        np.sum(
            np.square(coeffs[0])
        )
    )

    detail_energy = float(
        sum(
            np.sum(
                np.square(c)
            )
            for c in coeffs[1:]
        )
    )

    low_ratio = (
        approx_energy
        /
        (
            approx_energy
            + detail_energy
            + EPS
        )
    )

    low_coeffs = [
        coeffs[0]
    ] + [
        np.zeros_like(c)
        for c in coeffs[1:]
    ]

    low_recon = pywt.waverec(
        low_coeffs,
        wavelet=wave,
        mode="periodization",
    )[: len(seq)]

    tail = max(
        1,
        int(
            round(
                len(seq)
                * recent_fraction
            )
        ),
    )

    recent_mean = float(
        np.mean(
            low_recon[-tail:]
        )
    )

    persistent = (
        max(recent_mean, 0.0)
        * low_ratio
    )

    return {
        "persistent": persistent,
        "low_ratio": low_ratio,
        "recent_low_mean": recent_mean,
        "low_reconstruction": low_recon,
    }


def first_singular_energy_ratio(
    matrix: np.ndarray,
) -> float:
    if matrix.size == 0:
        return 0.0

    if matrix.shape[1] <= 1:
        return 1.0

    if np.linalg.norm(matrix) <= EPS:
        return 0.0

    _, s, _ = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    energy = float(
        np.sum(s * s)
    )

    if energy <= EPS:
        return 0.0

    return float(
        s[0] * s[0]
        / energy
    )


def read_geometry_csv(path: Path):
    out = {}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for raw in reader:
            r = int(float(raw["attack_round"]))

            q = as_float(
                raw.get("immediate_excess_asr")
            )
            pnorm = as_float(
                raw.get("attack_displacement_norm")
            )
            gnorm = as_float(
                raw.get("benign_round_drift_norm")
            )

            if (
                q is None
                or pnorm is None
                or gnorm is None
                or pnorm <= 0
                or gnorm <= 0
            ):
                continue

            out[r] = {
                "q": q,
                "attack_displacement_norm": pnorm,
                "benign_drift_norm": gnorm,
                "injection_efficiency": q / pnorm,
            }

    return out


def read_directional_csv(path: Path):
    if not path.exists():
        return {}

    out = {}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for raw in reader:
            r = int(float(raw["attack_round"]))
            bds = as_float(
                raw.get("probe_bds")
            )

            if bds is not None:
                out[r] = {"bds": bds}

    return out


def attack_gain_profiles(
    *,
    probe,
    keys,
    bands,
):
    raw_update = probe["raw_update"]
    bd_grad = probe["backdoor_grad"]
    clean_grad = probe["clean_grad"]

    raw_gain = {}
    cache = {}

    total_bd_dot = 0.0
    total_clean_dot = 0.0
    d_norm_sq = 0.0

    for key in keys:
        D = dct_vec(raw_update[key])
        H = dct_vec(bd_grad[key])
        C = dct_vec(clean_grad[key])

        cache[key] = {
            "D": D,
            "H": H,
            "C": C,
        }

        slices = band_slices(
            len(D),
            bands,
        )

        for band_idx, sl in enumerate(slices):
            db = D[sl]
            hb = H[sl]
            cb = C[sl]

            bd_dot = float(
                np.dot(hb, db)
            )

            clean_dot = float(
                np.dot(cb, db)
            )

            raw_gain[
                (key, band_idx)
            ] = max(
                -bd_dot,
                0.0,
            )

            total_bd_dot += bd_dot
            total_clean_dot += clean_dot
            d_norm_sq += float(
                np.dot(db, db)
            )

    gain_sum = sum(
        raw_gain.values()
    )

    if gain_sum <= EPS:
        raise RuntimeError(
            "No positive first-order backdoor gain."
        )

    attack_weight = {
        k: v / gain_sum
        for k, v in raw_gain.items()
    }

    d_norm = math.sqrt(
        max(d_norm_sq, EPS)
    )

    scalar_bd_gain = (
        max(
            -total_bd_dot,
            0.0,
        )
        / d_norm
    )

    scalar_clean_cost = (
        abs(total_clean_dot)
        / d_norm
    )

    return (
        attack_weight,
        cache,
        scalar_bd_gain,
        scalar_clean_cost,
    )


def analyze_target(
    *,
    target,
    window,
    bands,
    wavelet,
    wavelet_level,
    recent_fraction,
    checkpoint_index,
    spectral_state_root,
    min_param_size,
    gamma,
):
    probe = load_spectral_state(
        spectral_state_root,
        target,
    )

    start_state_round = (
        target
        - window
        - 1
    )

    end_state_round = (
        target
        - 1
    )

    required_rounds = list(
        range(
            start_state_round,
            end_state_round + 1,
        )
    )

    missing = [
        r
        for r in required_rounds
        if r not in checkpoint_index
    ]

    if missing:
        raise RuntimeError(
            f"Missing checkpoints for target {target}: {missing}"
        )

    states = {
        r: load_state_dict(
            checkpoint_index[r]
        )
        for r in required_rounds
    }

    keys = parameter_keys(
        probe=probe,
        state=states[end_state_round],
        min_param_size=min_param_size,
    )

    (
        attack_weight,
        cache,
        scalar_bd_gain,
        scalar_clean_cost,
    ) = attack_gain_profiles(
        probe=probe,
        keys=keys,
        bands=bands,
    )

    resistance_seq = {
        pair: []
        for pair in attack_weight
    }

    for r in range(
        target - window,
        target,
    ):
        newer = states[r]
        older = states[r - 1]

        for key in keys:
            g = update_between(
                newer,
                older,
                key,
            )

            G = dct_vec(g)
            H = cache[key]["H"]

            slices = band_slices(
                len(G),
                bands,
            )

            for band_idx, sl in enumerate(slices):
                pair = (
                    key,
                    band_idx,
                )

                if pair not in resistance_seq:
                    continue

                resistance_seq[pair].append(
                    safe_cosine_band(
                        H[sl],
                        G[sl],
                    )
                )

    pair_stats = {}

    for pair, seq in resistance_seq.items():
        pair_stats[pair] = (
            wavelet_persistent_resistance(
                np.asarray(
                    seq,
                    dtype=float,
                ),
                wavelet=wavelet,
                level=wavelet_level,
                recent_fraction=recent_fraction,
            )
        )

    band_coherence = {}

    for band_idx in range(bands):
        columns = []

        for key in keys:
            pair = (
                key,
                band_idx,
            )

            weight = attack_weight.get(
                pair,
                0.0,
            )

            stat = pair_stats.get(
                pair,
                None,
            )

            if weight <= 0 or stat is None:
                continue

            columns.append(
                np.asarray(
                    stat[
                        "low_reconstruction"
                    ],
                    dtype=float,
                )
                * math.sqrt(weight)
            )

        if not columns:
            band_coherence[band_idx] = 0.0
            continue

        matrix = np.stack(
            columns,
            axis=1,
        )

        band_coherence[band_idx] = (
            first_singular_energy_ratio(
                matrix
            )
        )

    conflict = 0.0
    conflict_no_coherence = 0.0
    band_rows = []

    for band_idx in range(bands):
        band_base = 0.0
        band_attack_mass = 0.0

        for key in keys:
            pair = (
                key,
                band_idx,
            )

            weight = attack_weight.get(
                pair,
                0.0,
            )

            stat = pair_stats.get(
                pair,
                None,
            )

            if weight <= 0 or stat is None:
                continue

            band_attack_mass += weight

            band_base += (
                weight
                * float(
                    stat["persistent"]
                )
            )

        coherence = band_coherence.get(
            band_idx,
            0.0,
        )

        band_conflict = (
            coherence
            * band_base
        )

        conflict_no_coherence += band_base
        conflict += band_conflict

        band_rows.append(
            {
                "band": band_idx,
                "attack_mass": band_attack_mass,
                "base_conflict": band_base,
                "coherence": coherence,
                "coherent_conflict": band_conflict,
            }
        )

    bds_like = (
        scalar_bd_gain
        /
        (
            scalar_clean_cost
            + 1e-8
        )
    )

    opportunity = (
        bds_like
        * math.exp(
            -gamma * conflict
        )
    )

    return {
        "attack_round": target,
        "window": window,
        "bands": bands,
        "wavelet": wavelet,
        "wavelet_level": wavelet_level,
        "num_layers": len(keys),
        "scalar_bd_gain": scalar_bd_gain,
        "scalar_clean_cost": scalar_clean_cost,
        "bds_recomputed": bds_like,
        "spectral_conflict": conflict,
        "spectral_conflict_no_coherence": conflict_no_coherence,
        "spectral_opportunity": opportunity,
        "probe_path": probe["_path"],
        "band_rows": band_rows,
    }


def write_csv(
    path,
    rows,
    fieldnames,
):
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


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    checkpoint_root = resolve(
        repo_root,
        args.checkpoint_root,
    )

    spectral_state_root = resolve(
        repo_root,
        args.spectral_state_root,
    )

    geometry_csv = resolve(
        repo_root,
        args.geometry_csv,
    )

    directional_csv = resolve(
        repo_root,
        args.directional_feature_csv,
    )

    output_dir = resolve(
        repo_root,
        args.output_dir,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_index = build_checkpoint_index(
        checkpoint_root
    )

    geometry = read_geometry_csv(
        geometry_csv
    )

    directional = read_directional_csv(
        directional_csv
    )

    rows = []
    band_rows = []

    for target in args.targets:
        if target not in geometry:
            print(
                f"[SKIP] round={target}: no geometry"
            )
            continue

        result = analyze_target(
            target=target,
            window=args.window,
            bands=args.bands,
            wavelet=args.wavelet,
            wavelet_level=args.wavelet_level,
            recent_fraction=args.recent_fraction,
            checkpoint_index=checkpoint_index,
            spectral_state_root=spectral_state_root,
            min_param_size=args.min_param_size,
            gamma=args.gamma,
        )

        row = {
            k: v
            for k, v in result.items()
            if k != "band_rows"
        }

        g = geometry[target]

        row[
            "injection_efficiency"
        ] = g[
            "injection_efficiency"
        ]

        row[
            "benign_drift_norm"
        ] = g[
            "benign_drift_norm"
        ]

        row[
            "inverse_benign_drift"
        ] = (
            1.0
            / g[
                "benign_drift_norm"
            ]
        )

        row[
            "previous_bds"
        ] = (
            directional.get(
                target,
                {},
            ).get(
                "bds",
                float("nan"),
            )
        )

        rows.append(row)

        for x in result["band_rows"]:
            band_rows.append(
                {
                    "attack_round": target,
                    **x,
                }
            )

    write_csv(
        output_dir
        / "joint_spectral_features.csv",
        rows,
        [
            "attack_round",
            "window",
            "bands",
            "wavelet",
            "wavelet_level",
            "num_layers",
            "injection_efficiency",
            "benign_drift_norm",
            "inverse_benign_drift",
            "previous_bds",
            "scalar_bd_gain",
            "scalar_clean_cost",
            "bds_recomputed",
            "spectral_conflict",
            "spectral_conflict_no_coherence",
            "spectral_opportunity",
            "probe_path",
        ],
    )

    write_csv(
        output_dir
        / "band_conflict.csv",
        band_rows,
        [
            "attack_round",
            "band",
            "attack_mass",
            "base_conflict",
            "coherence",
            "coherent_conflict",
        ],
    )

    with (
        output_dir
        / "joint_spectral_features.json"
    ).open("w") as f:
        json.dump(
            rows,
            f,
            indent=2,
        )

    print("\nJOINT SPECTRAL FEATURES")
    print("=" * 120)

    for row in rows:
        print(
            f"round={row['attack_round']:4d} | "
            f"eff={row['injection_efficiency']:.6f} | "
            f"drift={row['benign_drift_norm']:.6f} | "
            f"conflict={row['spectral_conflict']:.6f} | "
            f"bds={row['bds_recomputed']:.6f} | "
            f"opportunity={row['spectral_opportunity']:.6f}"
        )


if __name__ == "__main__":
    main()

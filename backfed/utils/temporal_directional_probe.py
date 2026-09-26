from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import numpy as np


EPS = 1e-12


def _trainable_names(model: torch.nn.Module):
    return {name for name, _ in model.named_parameters()}


def _update_norm(model, model_updates: Dict[str, torch.Tensor]) -> float:
    trainable = _trainable_names(model)
    total = 0.0

    for name, tensor in model_updates.items():
        if name not in trainable:
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        if not torch.is_floating_point(tensor):
            continue

        x = tensor.detach().float()
        total += float(torch.sum(x * x).item())

    return math.sqrt(max(total, 0.0))


def _directional_dot(model, model_updates, loss, update_norm):
    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    params = [param for _, param in named_params]

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    dot = 0.0
    grad_sq = 0.0

    for (name, _), grad in zip(named_params, grads):
        if grad is None:
            continue

        update = model_updates.get(name, None)
        if update is None:
            continue
        if not isinstance(update, torch.Tensor):
            continue
        if not torch.is_floating_point(update):
            continue

        g = grad.detach().float()
        d = update.detach().float().to(g.device)

        dot += float(torch.sum(g * d).item())
        grad_sq += float(torch.sum(g * g).item())

    normalized_dot = (
        dot / update_norm
        if update_norm > EPS
        else float("nan")
    )

    grad_norm = math.sqrt(max(grad_sq, 0.0))

    cosine = (
        dot / (update_norm * grad_norm)
        if update_norm > EPS and grad_norm > EPS
        else float("nan")
    )

    return {
        "directional_dot": normalized_dot,
        "gradient_norm": grad_norm,
        "cosine": cosine,
    }


def _prepare_clean_batch(images, labels, device, normalization):
    images = images.to(device)
    labels = labels.to(device)

    if normalization is not None:
        images = normalization(images)

    return images, labels


def _prepare_backdoor_batch(
    images,
    labels,
    device,
    normalization,
    poison_module,
):
    images = images.to(device)
    labels = labels.to(device)

    poisoned_images = poison_module.poison_inputs(images)
    poisoned_labels = poison_module.poison_labels(labels)

    poisoned_images = poisoned_images.to(device)
    poisoned_labels = poisoned_labels.to(device)

    if normalization is not None:
        poisoned_images = normalization(poisoned_images)

    return poisoned_images, poisoned_labels


def compute_directional_susceptibility(
    *,
    model: torch.nn.Module,
    global_state_dict: Dict[str, torch.Tensor],
    model_updates: Dict[str, torch.Tensor],
    probe_loader: Iterable,
    poison_module,
    criterion,
    device,
    normalization=None,
    num_batches: int = 2,
) -> Dict[str, float]:
    """
    Compute attacker-observable directional susceptibility on the
    current global model using the PRE-SCALING malicious update.
    """

    update_norm = _update_norm(model, model_updates)

    if update_norm <= EPS:
        return {
            "probe_raw_update_norm": update_norm,
            "probe_bd_gain_slope": float("nan"),
            "probe_clean_signed_slope": float("nan"),
            "probe_clean_abs_slope": float("nan"),
            "probe_bds": float("nan"),
            "probe_bd_grad_norm": float("nan"),
            "probe_clean_grad_norm": float("nan"),
            "probe_bd_update_cos": float("nan"),
            "probe_clean_update_cos": float("nan"),
            "probe_clean_loss": float("nan"),
            "probe_bd_loss": float("nan"),
            "probe_num_batches": 0.0,
        }

    saved_state = copy.deepcopy(model.state_dict())
    saved_training = model.training
    saved_python_rng = random.getstate()
    saved_numpy_rng = np.random.get_state()
    saved_torch_rng = torch.get_rng_state()
    saved_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    clean_dots = []
    bd_dots = []
    clean_grad_norms = []
    bd_grad_norms = []
    clean_cosines = []
    bd_cosines = []
    clean_losses = []
    bd_losses = []
    used_batches = 0

    try:
        model.load_state_dict(global_state_dict, strict=True)
        model.eval()
        for batch_idx, batch in enumerate(probe_loader):
            if batch_idx >= num_batches:
                break
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                continue

            images, labels = batch[0], batch[1]
            if len(labels) <= 1:
                continue

            clean_images, clean_labels = _prepare_clean_batch(
                images,
                labels,
                device,
                normalization,
            )

            model.zero_grad(set_to_none=True)
            clean_outputs = model(clean_images)
            clean_loss = criterion(clean_outputs, clean_labels)

            clean_stats = _directional_dot(
                model,
                model_updates,
                clean_loss,
                update_norm,
            )

            bd_images, bd_labels = _prepare_backdoor_batch(
                images,
                labels,
                device,
                normalization,
                poison_module,
            )

            model.zero_grad(set_to_none=True)
            bd_outputs = model(bd_images)
            bd_loss = criterion(bd_outputs, bd_labels)

            bd_stats = _directional_dot(
                model,
                model_updates,
                bd_loss,
                update_norm,
            )

            clean_dots.append(clean_stats["directional_dot"])
            bd_dots.append(bd_stats["directional_dot"])
            clean_grad_norms.append(clean_stats["gradient_norm"])
            bd_grad_norms.append(bd_stats["gradient_norm"])
            clean_cosines.append(clean_stats["cosine"])
            bd_cosines.append(bd_stats["cosine"])
            clean_losses.append(float(clean_loss.detach().item()))
            bd_losses.append(float(bd_loss.detach().item()))
            used_batches += 1

    finally:
        model.load_state_dict(saved_state, strict=True)
        model.train(saved_training)
        random.setstate(saved_python_rng)
        np.random.set_state(saved_numpy_rng)
        torch.set_rng_state(saved_torch_rng)
        if saved_cuda_rng is not None:
            torch.cuda.set_rng_state_all(saved_cuda_rng)

    def mean(values):
        valid = [x for x in values if math.isfinite(x)]
        return sum(valid) / len(valid) if valid else float("nan")

    clean_signed = mean(clean_dots)
    bd_signed = mean(bd_dots)

    bd_gain = -bd_signed if math.isfinite(bd_signed) else float("nan")
    clean_abs = abs(clean_signed) if math.isfinite(clean_signed) else float("nan")

    if math.isfinite(bd_gain) and math.isfinite(clean_abs):
        bds = max(bd_gain, 0.0) / (clean_abs + 1e-8)
    else:
        bds = float("nan")

    return {
        "probe_raw_update_norm": update_norm,
        "probe_bd_gain_slope": bd_gain,
        "probe_clean_signed_slope": clean_signed,
        "probe_clean_abs_slope": clean_abs,
        "probe_bds": bds,
        "probe_bd_grad_norm": mean(bd_grad_norms),
        "probe_clean_grad_norm": mean(clean_grad_norms),
        "probe_bd_update_cos": mean(bd_cosines),
        "probe_clean_update_cos": mean(clean_cosines),
        "probe_clean_loss": mean(clean_losses),
        "probe_bd_loss": mean(bd_losses),
        "probe_num_batches": float(used_batches),
    }


def write_probe_record(
    *,
    server_round: int,
    client_id,
    scale_factor: float,
    metrics: Dict[str, float],
    output_dir: Optional[str] = None,
) -> Path:
    if output_dir is None:
        output_dir = os.environ.get(
            "BACKFED_TEMPORAL_PROBE_DIR",
            "outputs/temporal_directional_probe_raw",
        )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    path = root / f"round_{int(server_round)}_client_{client_id}.json"

    payload = {
        "server_round": int(server_round),
        "client_id": str(client_id),
        "scale_factor": float(scale_factor),
        **{
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in metrics.items()
        },
    }

    with path.open("w") as f:
        json.dump(payload, f, indent=2)

    return path

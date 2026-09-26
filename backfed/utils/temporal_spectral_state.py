from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import numpy as np


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
    images = poison_module.poison_inputs(images)
    labels = poison_module.poison_labels(labels)
    images = images.to(device)
    labels = labels.to(device)
    if normalization is not None:
        images = normalization(images)
    return images, labels


def _trainable_names(model):
    return {
        n
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def _copy_update_cpu(model, model_updates, dtype=torch.float16):
    trainable = _trainable_names(model)
    out = {}
    for name, value in model_updates.items():
        if name not in trainable:
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if not torch.is_floating_point(value):
            continue
        out[name] = value.detach().to("cpu", dtype=dtype).clone()
    return out


def _average_gradient(
    *,
    model,
    loader: Iterable,
    criterion,
    device,
    normalization,
    poison_module,
    backdoor: bool,
    num_batches: int,
) -> Tuple[Dict[str, torch.Tensor], float, int]:
    named_params = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad
    ]

    accum = {
        n: torch.zeros_like(
            p,
            device="cpu",
            dtype=torch.float32,
        )
        for n, p in named_params
    }

    total_loss = 0.0
    used = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            continue

        images, labels = batch[0], batch[1]

        if backdoor:
            images, labels = _prepare_backdoor_batch(
                images,
                labels,
                device,
                normalization,
                poison_module,
            )
        else:
            images, labels = _prepare_clean_batch(
                images,
                labels,
                device,
                normalization,
            )

        model.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        grads = torch.autograd.grad(
            loss,
            [p for _, p in named_params],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        for (name, _), grad in zip(named_params, grads):
            if grad is None:
                continue
            accum[name] += grad.detach().to("cpu", dtype=torch.float32)

        total_loss += float(loss.detach().item())
        used += 1

    if used <= 0:
        raise RuntimeError("No valid probe batch.")

    for name in accum:
        accum[name] /= float(used)

    return accum, total_loss / float(used), used


def collect_temporal_spectral_state(
    *,
    model,
    global_state_dict,
    model_updates,
    probe_loader,
    poison_module,
    criterion,
    device,
    normalization=None,
    num_batches=4,
    storage_dtype=torch.float16,
):
    saved_state = copy.deepcopy(model.state_dict())
    saved_training = model.training
    saved_python_rng = random.getstate()
    saved_numpy_rng = np.random.get_state()
    saved_torch_rng = torch.get_rng_state()
    saved_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    raw_update = _copy_update_cpu(
        model,
        model_updates,
        dtype=storage_dtype,
    )

    try:
        model.load_state_dict(global_state_dict, strict=True)
        model.eval()

        clean_grad, clean_loss, clean_used = _average_gradient(
            model=model,
            loader=probe_loader,
            criterion=criterion,
            device=device,
            normalization=normalization,
            poison_module=poison_module,
            backdoor=False,
            num_batches=num_batches,
        )

        bd_grad, bd_loss, bd_used = _average_gradient(
            model=model,
            loader=probe_loader,
            criterion=criterion,
            device=device,
            normalization=normalization,
            poison_module=poison_module,
            backdoor=True,
            num_batches=num_batches,
        )
    finally:
        model.load_state_dict(saved_state, strict=True)
        model.train(saved_training)
        random.setstate(saved_python_rng)
        np.random.set_state(saved_numpy_rng)
        torch.set_rng_state(saved_torch_rng)
        if saved_cuda_rng is not None:
            torch.cuda.set_rng_state_all(saved_cuda_rng)

    clean_grad = {
        n: x.to(dtype=storage_dtype)
        for n, x in clean_grad.items()
    }

    bd_grad = {
        n: x.to(dtype=storage_dtype)
        for n, x in bd_grad.items()
    }

    return {
        "raw_update": raw_update,
        "backdoor_grad": bd_grad,
        "clean_grad": clean_grad,
        "backdoor_loss": float(bd_loss),
        "clean_loss": float(clean_loss),
        "num_clean_batches": int(clean_used),
        "num_backdoor_batches": int(bd_used),
    }


def save_temporal_spectral_state(
    *,
    server_round,
    client_id,
    scale_factor,
    state,
    output_dir: Optional[str] = None,
):
    if output_dir is None:
        output_dir = os.environ.get(
            "BACKFED_SPECTRAL_STATE_DIR",
            "outputs/temporal_spectral_state",
        )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    path = root / (
        f"round_{int(server_round)}_client_{client_id}.pt"
    )

    torch.save(
        {
            "server_round": int(server_round),
            "client_id": str(client_id),
            "scale_factor": float(scale_factor),
            **state,
        },
        path,
    )

    return path

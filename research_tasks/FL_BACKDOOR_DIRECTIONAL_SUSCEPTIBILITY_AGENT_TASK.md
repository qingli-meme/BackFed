# FL Backdoor Timing — Directional Susceptibility Probe
## Agent 交付任务书

---

# 0. 本轮结论

上一轮已经得到：

```text
Matched Retention: FLAT
```

因此本轮**不再研究 retention / persistence**。

同时，matched injection efficiency 随 benign global drift norm 减小而升高。已有 FL 安全文献已经讨论过：当 benign gradient / loss landscape 较平时，攻击者更容易用较小恶意扰动改变 global model。因此，如果最后只是：

```text
||global drift|| 小 → attack
```

或者：

```text
越接近收敛 → attack
```

这个 Idea 不继续。

本轮只测试一个更具体的新机制：

> 在控制 benign drift magnitude 后，当前 global model 沿“恶意后门更新方向”的函数敏感度，是否还能预测真实注入效率？

这个量定义为：

```text
Backdoor Directional Susceptibility (BDS)
```

---

# 1. BDS 定义

攻击者在 round `t` 收到 global model：

```text
w_t
```

先本地生成一个**标准 malicious local update**：

```text
d_t
```

但暂时不上传。

归一化：

\[
\hat d_t = \frac{d_t}{\|d_t\|_2}
\]

在当前 global model 上，定义 backdoor-direction gain：

\[
G_t^{bd}
=
-
\langle
\nabla_w L_{bd}(w_t),
\hat d_t
\rangle
\]

其中 `L_bd` 是 triggered inputs + target label 上的 loss。

定义 clean-direction sensitivity：

\[
G_t^{cl}
=
\left|
\langle
\nabla_w L_{clean}(w_t),
\hat d_t
\rangle
\right|
\]

然后：

\[
\boxed{
BDS_t
=
\frac{
\max(G_t^{bd},0)
}{
G_t^{cl}+\epsilon
}
}
\]

直觉：

```text
BDS 高
=
沿当前恶意方向移动一点点，
backdoor loss 很快下降，
但 clean loss 变化相对较小。
```

本轮不是直接把 BDS 当方法，而是验证：

```text
BDS 是否能解释 benign drift 解释不了的 Injection Efficiency。
```

---

# 2. 真实目标变量

继续使用上一轮已经严格定义好的：

```text
cohort-matched counterfactual
+
matched immediate effect
+
actual global parameter displacement
```

固定目标：

```text
q ≈ 20% immediate excess ASR
```

对于 round `t`：

\[
p_t
=
w_t^{attack}
-
w_t^{counterfactual}
\]

真实 Injection Efficiency：

\[
Y_t
=
\frac{q_t}{\|p_t\|_2}
\]

其中 `q_t` 用这一轮真实匹配出来的 immediate excess ASR，不要强行写成固定 0.20。

---

# 3. 本轮 round 数量

目前只有 3 个点不足以分析。

本轮固定：

```text
50
100
150
175
200
225
250
```

共 7 个 round。

其中：

```text
50 / 150 / 250
```

尽量复用已有 exact-matching 结果。

新增：

```text
100 / 175 / 200 / 225
```

不要继续加点。

---

# 4. 本轮明确不做

不要：

```text
Retention
Half-life
future simulation
A3FL
Neurotoxin
defense
trigger optimization
scheduler
Hessian
curvature
```

本轮只研究：

```text
Injection Vulnerability
```

---

# 5. 仓库

继续使用：

```text
BackFed
commit c851ce90373ef2659447ea25296c7b442e5dc5c7
```

如果仓库不存在：

```bash
git clone https://github.com/thinh-dao/BackFed.git
cd BackFed
git checkout c851ce90373ef2659447ea25296c7b442e5dc5c7
```

保留已有：

```text
experiments/run_exact_temporal_matching.py
experiments/analyze_temporal_geometry.py
```

本轮新增：

```text
backfed/utils/temporal_directional_probe.py
experiments/analyze_directional_susceptibility.py
```

并对：

```text
backfed/clients/base_malicious_client.py
```

增加一个**只记录、不改变训练结果**的 hook。

---

# 6. BDS 必须使用 scale 前 malicious update

BackFed 当前逻辑中，`base_malicious_client.py` 会先计算：

```python
model_updates = self.weight_diff_dict(
    client_state_dict=self.model.state_dict(),
    global_state_dict=train_package["global_state_dict"]
)
```

然后才做：

```python
if self.atk_config["scale_poison"]:
    self.model_replacement_inplace(
        scale_factor=self.atk_config["scale_factor"],
        model_updates=model_updates
    )
```

所以：

```text
BDS 必须计算在 model_updates 生成之后，
model_replacement_inplace 之前。
```

也就是：

```text
d_t = raw malicious local update before scaling
```

---

# 7. 新增完整代码一

创建：

```text
BackFed/backfed/utils/temporal_directional_probe.py
```

完整内容如下：

```python
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


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

    model.load_state_dict(global_state_dict, strict=True)
    model.eval()

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
```

---

# 8. 修改 base_malicious_client.py

在 import 区域增加：

```python
from backfed.utils.temporal_directional_probe import (
    compute_directional_susceptibility,
    write_probe_record,
)
```

找到现有：

```python
model_updates = self.weight_diff_dict(
    client_state_dict=self.model.state_dict(),
    global_state_dict=train_package["global_state_dict"]
)

training_metrics = {
    "train_backdoor_loss": train_loss,
    "train_backdoor_acc": train_acc,
}

if self.atk_config["scale_poison"]:
    self.model_replacement_inplace(
        scale_factor=self.atk_config["scale_factor"],
        model_updates=model_updates
    )
```

替换成：

```python
model_updates = self.weight_diff_dict(
    client_state_dict=self.model.state_dict(),
    global_state_dict=train_package["global_state_dict"]
)

training_metrics = {
    "train_backdoor_loss": train_loss,
    "train_backdoor_acc": train_acc,
}

# Logging-only temporal directional probe.
# MUST stay before model-replacement scaling.
try:
    probe_loader = (
        self.val_loader
        if self.val_loader is not None
        else self.train_loader
    )

    probe_metrics = compute_directional_susceptibility(
        model=self.model,
        global_state_dict=train_package["global_state_dict"],
        model_updates=model_updates,
        probe_loader=probe_loader,
        poison_module=self.poison_module,
        criterion=self.criterion,
        device=self.device,
        normalization=normalization,
        num_batches=2,
    )

    training_metrics.update(probe_metrics)

    write_probe_record(
        server_round=server_round,
        client_id=self.client_id,
        scale_factor=float(self.atk_config["scale_factor"]),
        metrics=probe_metrics,
    )

except Exception as probe_error:
    log(
        WARNING,
        (
            "Temporal directional probe failed "
            f"for client [{self.client_id}] "
            f"at round {server_round}: {probe_error}"
        ),
    )

if self.atk_config["scale_poison"]:
    self.model_replacement_inplace(
        scale_factor=self.atk_config["scale_factor"],
        model_updates=model_updates
    )
```

---

# 9. Smoke Test：probe 不得改变攻击结果

先跑一个短实验，例如 attack round 20。

要求生成：

```text
outputs/temporal_directional_probe_raw/
round_20_client_*.json
```

JSON 至少包含：

```text
probe_raw_update_norm
probe_bd_gain_slope
probe_clean_signed_slope
probe_clean_abs_slope
probe_bds
probe_bd_update_cos
probe_clean_update_cos
```

必须检查：

```text
probe_raw_update_norm > 0
probe_bd_gain_slope finite
probe_clean_abs_slope finite
probe_bds finite
```

然后用完全相同：

```text
round
seed
scale
```

比较：

```text
probe hook OFF
probe hook ON
```

要求：

```text
Immediate ASR difference <= 1e-6
Clean ACC difference <= 1e-6
returned model-update norm difference <= 1e-6
```

如果不满足：

```text
STOP
```

先修 probe。

---

# 10. 扩展 Exact Matching

使用已有：

```text
experiments/run_exact_temporal_matching.py
```

正式 round：

```text
50 100 150 175 200 225 250
```

本轮不研究 retention，所以：

```text
horizon = 1
```

运行：

```bash
python experiments/run_exact_temporal_matching.py \
  --attack-rounds 50 100 150 175 200 225 250 \
  --target 0.20 \
  --tolerance 0.01 \
  --max-clean-drop 0.10 \
  --horizon 1 \
  --max-refine-runs 8 \
  --seed 2026 \
  --gpu 0 \
  --training-mode sequential
```

不要直接 `--force`。

优先复用已有：

```text
50 / 150 / 250
```

只重跑缺失 round。

合格标准：

```text
至少 6 / 7 round PASS
```

且 PASS round：

```text
Immediate Excess ASR ∈ [0.19, 0.21]
```

如果 PASS <= 5：

```text
STOP
```

---

# 11. 运行 actual geometry

复用：

```text
experiments/analyze_temporal_geometry.py
```

要求它从：

```text
exact_matching_summary.csv
```

自动读取所有 PASS round，不要硬编码 50/150/250。

输出至少：

```text
attack_round
immediate_excess_asr
attack_displacement_norm
benign_round_drift_norm
injection_efficiency
attack_to_benign_norm_ratio
cos_attack_benign
```

---

# 12. BDS 在同一 round 不应依赖 model-replacement scale

由于 BDS 计算在 scale 前，理论上同一 round 不同 calibration scale 的：

```text
probe_raw_update_norm
probe_bd_gain_slope
probe_clean_abs_slope
probe_bds
```

应基本相同。

要求：

```text
relative difference <= 1e-5
```

如果不一致，说明 local malicious training 的 run-to-run determinism 有问题。

此时只保留：

```text
matched final run
```

对应的 probe 记录，并在最终报告说明。

---

# 13. 新增完整代码二

创建：

```text
BackFed/experiments/analyze_directional_susceptibility.py
```

完整内容：

```python
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
```

---

# 14. 运行分析

当满足：

```text
>= 6 个 exact-matching PASS round
+
geometry 完成
+
BDS JSON 完成
```

运行：

```bash
python experiments/analyze_directional_susceptibility.py
```

输出：

```text
outputs/directional_susceptibility_analysis/
├── round_features.csv
└── analysis_summary.json
```

---

# 15. 本轮真正的比较

不要只看：

```text
corr(BDS, Injection Efficiency)
```

因为 BDS 自己也可能只是随着训练进度变化。

必须比较三组。

## Baseline

```text
1 / ||benign global drift||
```

预测：

```text
Injection Efficiency
```

## Candidate

```text
BDS
```

预测：

```text
Injection Efficiency
```

## Combined

```text
benign drift + BDS
```

预测：

```text
Injection Efficiency
```

核心问题：

```text
BDS 是否在 benign drift 已知以后仍然增加解释能力？
```

---

# 16. GO / STOP

## STRONG GO

同时满足：

```text
Spearman(BDS, Injection Efficiency) >= 0.60
```

并且：

```text
LOO log-MAE(drift + BDS)
相对
LOO log-MAE(drift only)
降低 >= 20%
```

即：

```text
loo_relative_improvement_with_bds >= 0.20
```

并且：

```text
Spearman(BDS, drift-only residual) >= 0.50
```

满足则：

```text
STRONG GO
```

说明：

```text
“当前 round 的 targeted malicious-direction sensitivity”
提供了 global training speed 之外的独立信号。
```

下一轮才实现真正的：

```text
BDS-based attack policy
```

---

## WEAK

如果：

```text
Spearman(BDS, efficiency) >= 0.50
```

但：

```text
LOO improvement < 20%
```

或者 residual correlation 弱，则：

```text
WEAK
```

只允许再加：

```text
1 个 seed
```

不要再加特征。

---

## DRIFT-ONLY

如果：

```text
Spearman(1/||g_t||, efficiency) >= 0.85
```

并且：

```text
加入 BDS 后 LOO improvement < 10%
```

判：

```text
DRIFT-ONLY
```

说明 temporal injection phenomenon 基本就是已知的 convergence / benign-dilution effect。

这条 Idea 停止。

---

## STOP

如果：

```text
Spearman(BDS, efficiency) < 0.40
```

且加入 BDS 没有稳定改善预测：

```text
STOP
```

不要继续挖：

```text
curvature
Hessian
更多 gradient 特征
```

直接换方向。

---

# 17. Novelty 防线

如果 Strong GO，论文不能写成：

```text
“我们使用 global model feedback。”
```

因为已有工作已经用 global-model feedback 优化 trigger。

真正潜在的新点应该是：

```text
existing attacks optimize WHAT malicious update to inject;
we characterize WHEN the current global state is functionally susceptible
to a fixed malicious direction.
```

即：

```text
trigger optimization
!=
temporal vulnerability estimation
```

---

# 18. Strong GO 后的最终方法轮廓

本轮不要实现，只记录。

攻击者每次被选中：

```text
1. 从当前 global model 生成标准 candidate malicious update d_t
2. 不上传
3. 本地计算 BDS_t
4. BDS_t 高 → 消耗一次 attack budget
5. BDS_t 低 → benign-like participate / skip poisoning
```

攻击预算：

\[
\sum_t a_t \le B
\]

最终决策：

\[
a_t = \mathbf 1[BDS_t > \tau_t]
\]

核心机制是：

```text
Backdoor-direction functional susceptibility
```

不是：

```text
round index
```

也不是：

```text
global drift norm
```

---

# 19. Agent 最终汇报格式

## 19.1 Probe Fidelity

```text
probe hook 是否改变攻击输出: YES / NO

最大 ASR difference:
最大 ACC difference:
最大 update-norm difference:
```

## 19.2 Exact Matching

```text
Round    q    ||p_t||    Injection Efficiency    ||g_t||
50
100
150
175
200
225
250
```

说明：

```text
PASS round 数量
```

## 19.3 Directional Features

```text
Round    BDS    BD Gain Slope    Clean Abs Slope    Raw Local Update Norm
50
100
150
175
200
225
250
```

## 19.4 Correlation

```text
Spearman(1/||g_t||, Injection Efficiency):
Spearman(BDS, Injection Efficiency):
Spearman(BD Gain, Injection Efficiency):
Spearman(Clean Abs Slope, Injection Efficiency):
```

## 19.5 Beyond-Drift Test

```text
LOO log-MAE drift only:
LOO log-MAE drift + BDS:
Relative improvement:

Spearman(BDS, drift-only residual):
```

## 19.6 Final Verdict

只选：

```text
STRONG GO
WEAK
DRIFT-ONLY
STOP
```

最多 8 句话说明。

---

# 20. 本轮严格禁止

不要：

```text
继续测 retention
做 scheduler
换 trigger
加 A3FL
加 Neurotoxin
加 defense
换 dataset
换 model
增加 malicious clients
做 Hessian
做 curvature
堆更多 feature
```

只测试一个新机制：

\[
\boxed{
\text{Backdoor Directional Susceptibility}
}
\]

并且只有当它能解释：

```text
benign drift 解释不了的 Injection Efficiency
```

时，才继续这条研究线。

否则：

```text
放弃 when-to-inject。
```

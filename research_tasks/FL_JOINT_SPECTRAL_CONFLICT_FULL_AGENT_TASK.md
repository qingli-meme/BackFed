# 联邦后门“何时注入”的参数频率—通信时间联合谱建模
## 完整研究方案、数学定义、代码与运行说明

> 基于 BackFed `c851ce90373ef2659447ea25296c7b442e5dc5c7`。
>
> 已知：不同轮次即时注入效率不同；严格匹配后留存基本平坦；良性漂移可解释大部分差异，但无法很好解释 200→225→250 轮“漂移更小、注入效率反而下降”的反转；旧定向敏感度有预测价值但独立解释力不足。
>
> 本方案不再研究留存，不再堆单点训练动力学特征，而是验证：**候选恶意更新真正依赖的参数频带，与历史良性训练中“持续抵抗当前后门功能”的频带，在参数频率和通信时间尺度上的联合冲突，是否决定注入效率。**

---

# 1. 核心研究问题

不要写成“第几轮攻击最好”。

真正问题：

> 有限攻击预算下，当攻击者在第 \(t\) 轮被选中时，当前候选恶意更新所依赖的后门有效频带，是否正好与过去若干轮持续存在的良性抵抗频带冲突？

轮次只是索引，真正的状态变量是：

\[
\boxed{
\text{后门有效参数频谱}
\times
\text{良性功能抵抗的通信时间尺度}
}
\]

---

# 2. 为什么不用一维傅里叶比值

简单方案：

\[
s_r=\langle g_r,d_t\rangle
\]

再对最近 \(K\) 轮做傅里叶变换、计算低频/高频比，不够。

它丢失：

1. 网络层结构；
2. 候选攻击真正依赖哪些参数频带；
3. 良性更新对后门功能是帮助还是抵抗；
4. 非平稳训练中的时间定位；
5. 跨层是否形成共同持续模式；
6. 低频能量究竟是有效抵抗还是普通大幅训练。

因此完整方案采用：

\[
\boxed{
\text{逐层参数离散余弦谱}
+
\text{通信时间小波多尺度}
+
\text{后门功能梯度}
+
\text{跨层一致性}
}
\]

---

# 3. 相关工作边界

## 3.1 FreqFed

NDSS 2024 的 FreqFed 对客户端模型更新做离散余弦变换，利用低频系数差异检测恶意客户端。

```text
FreqFed: A Frequency Analysis-Based Approach for Mitigating Poisoning Attacks in Federated Learning
https://www.ndss-symposium.org/ndss-paper/freqfed-a-frequency-analysis-based-approach-for-mitigating-poisoning-attacks-in-federated-learning/
```

它解决“谁是恶意客户端”，我们解决“攻击者何时值得消耗攻击机会”。

## 3.2 FedFFT

2026 年 FedFFT 逐层对联邦尖锐度扰动做实数傅里叶分析，先统计不同频带上的客户端差异，再据此滤除主要不一致频带。

```text
FedFFT: Taming Client Drift in Federated SAM via Spectral Perturbation Filtering
https://arxiv.org/abs/2607.04170
```

启发：不能预设低频一定好/坏，应先找稳定频带结构。

## 3.3 主动频谱响应

2026 年 TIFS 的主动频谱防御先主动激发客户端模型，得到动态响应，再用离散小波分析多尺度响应。

```text
Frequency-Domain Signatures for Proactive Defense Against Model Poisoning Attacks in Federated Learning
DOI: 10.1109/TIFS.2026.3689720
```

启发：频谱对象最好有明确系统语义。因此我们分析“良性更新对当前后门功能的影响”，不是无目标参数能量。

## 3.4 Neurotoxin

```text
Neurotoxin: Durable Backdoors in Federated Learning
ICML 2022
https://arxiv.org/abs/2206.10341
```

Neurotoxin 优先攻击良性训练中变化较少的参数。若本方案退化成“找良性频谱能量低的频带”，则新颖性不足。

我们必须验证的是：

\[
\boxed{
\text{后门功能方向}
\times
\text{参数频率}
\times
\text{通信时间尺度}
\times
\text{跨层一致性}
}
\]

---

# 4. 威胁模型与因果约束

保持当前威胁模型不变。

第 \(t\) 轮攻击者只允许使用：

```text
当前收到的全局模型；
过去已经公开的全局模型；
攻击者自己的本地良性/触发数据；
上传前本地得到的候选恶意更新。
```

禁止使用本轮其他客户端尚未上传的更新。

设 \(w_r\) 为第 \(r\) 轮聚合结束后的全局模型，则第 \(t\) 轮攻击开始时收到：

\[
w_{t-1}.
\]

历史良性全局更新：

\[
g_r=w_r-w_{r-1}.
\]

仅使用：

\[
g_{t-K},\ldots,g_{t-1}.
\]

不能用当前轮 counterfactual 的 \(g_t\) 作为在线特征。

---

# 5. 候选恶意方向和功能梯度

攻击者在 \(w_{t-1}\) 上先训练但不上传，得到未缩放候选恶意更新：

\[
d_t.
\]

计算后门损失梯度：

\[
h_t=\nabla_w L_{\mathrm{后门}}(w_{t-1}),
\]

以及干净损失梯度：

\[
c_t=\nabla_w L_{\mathrm{干净}}(w_{t-1}).
\]

---

# 6. 第一维：逐层参数频率

每个可训练参数张量单独展平并做正交二型离散余弦变换：

\[
H_t^{(l)}=\mathcal D_l(h_t^{(l)}),
\]

\[
D_t^{(l)}=\mathcal D_l(d_t^{(l)}),
\]

\[
G_r^{(l)}=\mathcal D_l(g_r^{(l)}).
\]

不能跨层拼接。

主实验频带数：

```text
B = 8
```

消融：

```text
B ∈ {4, 8, 16}
```

这里的“参数频率”不是图像空间频率，论文中不得混淆。

---

# 7. 后门有效谱

一阶变化：

\[
\Delta L_{\mathrm{后门}}
\approx
\langle h_t,d_t\rangle.
\]

正交变换保持内积，因此可按层、频带分解。

第 \(l\) 层第 \(b\) 个频带的后门正向贡献：

\[
a_{t,l,b}^{raw}
=
[-\langle H_{t,l,b},D_{t,l,b}\rangle]_+.
\]

归一化：

\[
a_{t,l,b}
=
\frac{a_{t,l,b}^{raw}}
{\sum_{l',b'}a_{t,l',b'}^{raw}+\epsilon}.
\]

称为：

\[
\boxed{\text{后门有效谱}}
\]

它不是恶意更新能量谱，而是“哪些频带真正负责降低后门损失”。

---

# 8. 良性功能抵抗序列

对过去某轮良性更新 \(g_r\)，其对当前后门功能的一阶影响为：

\[
\langle h_t,g_r\rangle.
\]

逐层逐频带定义：

\[
z_{t,r,l,b}
=
\frac{\langle H_{t,l,b},G_{r,l,b}\rangle}
{\|H_{t,l,b}\|_2\|G_{r,l,b}\|_2+\epsilon}.
\]

若：

\[
z>0
\]

说明良性更新会提高当前后门损失，即抵抗后门。

若：

\[
z<0
\]

说明良性更新在该频带上反而帮助后门方向。

于是得到长度 \(K\) 的时间序列：

\[
Z_{t,l,b}
=
[z_{t,t-K,l,b},\ldots,z_{t,t-1,l,b}].
\]

---

# 9. 第二维：通信时间小波

联邦训练非平稳，因此主方案不用普通全窗傅里叶，而使用离散小波。

主配置：

```text
K = 32
小波 = db2
分解层数 J = 3
```

消融：

```text
K ∈ {8,16,32}
小波 ∈ {haar,db2}
```

对 \(Z_{t,l,b}\) 做：

\[
Z\rightarrow(A_J,D_J,\ldots,D_1).
\]

定义慢变化能量占比：

\[
\rho_{t,l,b}
=
\frac{\|A_J\|_2^2}
{\|A_J\|_2^2+\sum_j\|D_j\|_2^2+\epsilon}.
\]

只保留 \(A_J\) 重建慢变化序列：

\[
\widetilde Z_{t,l,b}^{slow}.
\]

取最后 \(K/4\) 个位置的均值：

\[
\mu_{t,l,b}^{slow}.
\]

定义持续抵抗：

\[
p_{t,l,b}
=
[\mu_{t,l,b}^{slow}]_+
\rho_{t,l,b}.
\]

---

# 10. 跨层一致性

对同一频带 \(b\)，收集所有层的慢变化序列，按后门有效谱加权：

\[
\bar M_{t,b}[:,l]
=
\sqrt{a_{t,l,b}}
\widetilde Z_{t,l,b}^{slow}.
\]

做奇异值分解：

\[
\bar M=U\Sigma V^\top.
\]

定义：

\[
\kappa_{t,b}
=
\frac{\sigma_1^2}
{\sum_i\sigma_i^2+\epsilon}.
\]

越接近 1，表示多个后门相关层共享一个主要持续抵抗模式。

---

# 11. 联合谱冲突度

频带冲突：

\[
C_{t,b}
=
\kappa_{t,b}
\sum_l a_{t,l,b}p_{t,l,b}.
\]

总冲突：

\[
\boxed{
C_t=\sum_bC_{t,b}
}
\]

解释：

> \(C_t\) 高，表示候选后门真正依赖的参数频带，恰好被最近良性训练中跨层一致、慢变化、持续抵抗后门功能的模式占据。

这才是本方案真正的“何时注入”中间量。

---

# 12. 完整机会分数

旧定向敏感度：

\[
B_t
=
\frac{[-\langle h_t,\hat d_t\rangle]_+}
{|\langle c_t,\hat d_t\rangle|+\epsilon}.
\]

它只作为候选攻击自身质量项。

完整机会分数：

\[
\boxed{
V_t=B_t\exp(-\gamma C_t)
}
\]

主实验：

```text
γ = 1
```

敏感性：

```text
γ ∈ {0.5,1,2}
```

论文真正的新机制必须来自 \(C_t\)，不是重新包装 \(B_t\)。

---

# 13. 必须解释的现有反转

已有注入效率：

```text
50   0.0344
100  0.0410
150  0.0771
175  0.1001
200  0.1234
225  0.1189
250  0.0980
```

良性漂移：

```text
50   6.354
100  5.222
150  2.393
175  2.051
200  1.705
225  1.299
250  1.324
```

核心现象：

```text
200→225→250
良性漂移没有变强，
但注入效率持续下降。
```

若完整谱机制正确，应优先看到：

```text
C_200 < C_225 < C_250
```

或至少 \(C_{200}\) 明显低于 \(C_{250}\)。

---

# 14. 新增依赖

```bash
pip install scipy PyWavelets
```

不要升级 PyTorch。

---

# 15. 文件结构

新增：

```text
backfed/utils/temporal_spectral_state.py
experiments/collect_clean_spectral_history.py
experiments/analyze_joint_spectral_conflict.py
experiments/spectral_policy_diagnostics.py
```

保留现有：

```text
experiments/run_exact_temporal_matching.py
experiments/analyze_temporal_geometry.py
backfed/utils/temporal_directional_probe.py
```

---

# 16. 新增 temporal_spectral_state.py

创建：

```text
BackFed/backfed/utils/temporal_spectral_state.py
```

```python
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch


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
```

---

# 17. base_malicious_client.py 增加零影响记录 hook

import：

```python
from backfed.utils.temporal_spectral_state import (
    collect_temporal_spectral_state,
    save_temporal_spectral_state,
)
```

必须插在：

```python
model_updates = self.weight_diff_dict(...)
```

之后、任何 model replacement scaling 之前：

```python
try:
    probe_loader = (
        self.val_loader
        if self.val_loader is not None
        else self.train_loader
    )

    spectral_state = collect_temporal_spectral_state(
        model=self.model,
        global_state_dict=train_package["global_state_dict"],
        model_updates=model_updates,
        probe_loader=probe_loader,
        poison_module=self.poison_module,
        criterion=self.criterion,
        device=self.device,
        normalization=normalization,
        num_batches=4,
    )

    save_temporal_spectral_state(
        server_round=server_round,
        client_id=self.client_id,
        scale_factor=float(self.atk_config["scale_factor"]),
        state=spectral_state,
    )

except Exception as spectral_error:
    log(
        WARNING,
        (
            "Temporal spectral state collection failed "
            f"for client [{self.client_id}] "
            f"at round {server_round}: {spectral_error}"
        ),
    )
```

必须再次做 hook 开关对照：

```text
ASR difference <= 1e-6
clean ACC difference <= 1e-6
returned update norm difference <= 1e-6
saved global model max element difference <= 1e-6
```

否则停止。

---

# 18. 采集 7 个目标轮次谱状态

目标：

```text
50 100 150 175 200 225 250
```

从：

```text
outputs/exact_temporal_matching/exact_matching_summary.csv
```

读取每个 PASS round 的 matched_scale。

只重跑最终 matched attack 到攻击轮即可，不需要 retention horizon。

结果要求：

```text
outputs/temporal_spectral_state/
round_50_client_*.pt
...
round_250_client_*.pt
```

共 7 轮。

---

# 19. 新增 collect_clean_spectral_history.py

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


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

    cmd = [
        sys.executable,
        str(repo_root / "main.py"),
        "--config-name",
        "cifar10",
        "checkpoint=null",
        f"seed={args.seed}",
        "deterministic=true",
        f"cuda_visible_devices={args.gpu}",
        "aggregator=unweighted_fedavg",
        "dataset=CIFAR10",
        "model=ResNet18",
        f"num_rounds={max(args.targets)-1}",
        f"training_mode={args.training_mode}",
        "no_attack=true",
        "save_logging=csv",
        "progress_bar=false",
        "save_checkpoint=true",
        "save_checkpoint_rounds=" + round_list,
        f"hydra.run.dir={run_dir.resolve()}",
        "dir_tag=clean_spectral_history",
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
    main()
```

运行：

```bash
python experiments/collect_clean_spectral_history.py \
  --targets 50 100 150 175 200 225 250 \
  --max-window 32 \
  --seed 2026 \
  --gpu 0 \
  --training-mode sequential \
  --dry-run
```

确认后：

```bash
python experiments/collect_clean_spectral_history.py \
  --targets 50 100 150 175 200 225 250 \
  --max-window 32 \
  --seed 2026 \
  --gpu 0 \
  --training-mode sequential
```


---

# 20. 新增 analyze_joint_spectral_conflict.py

创建：

```text
BackFed/experiments/analyze_joint_spectral_conflict.py
```

完整代码：

```python
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
```

---

# 21. 运行联合谱分析

主配置：

```bash
python experiments/analyze_joint_spectral_conflict.py \
  --targets 50 100 150 175 200 225 250 \
  --checkpoint-root checkpoints \
  --window 32 \
  --bands 8 \
  --wavelet db2 \
  --wavelet-level 3 \
  --recent-fraction 0.25 \
  --gamma 1.0
```

输出：

```text
outputs/joint_spectral_conflict/
├── joint_spectral_features.csv
├── joint_spectral_features.json
└── band_conflict.csv
```

---

# 22. 新增 spectral_policy_diagnostics.py

创建：

```text
BackFed/experiments/spectral_policy_diagnostics.py
```

```python
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
```

运行：

```bash
python experiments/spectral_policy_diagnostics.py
```

---

# 23. 主判断指标

理论方向：

\[
C_t\uparrow
\Rightarrow
\text{注入效率}\downarrow.
\]

所以首先要求：

\[
\rho(
C_t,
\text{注入效率}
)<0.
\]

但仅有相关性不够。

最重要的是：

```text
良性漂移 + 旧 BDS
```

已经作为强基线。

必须验证加入：

```text
联合谱冲突 C_t
```

以后仍然增加解释力。

---

# 24. STRONG GO

同时满足：

## A. 总体关系

\[
\rho(
C_t,
\text{注入效率}
)
\le -0.60.
\]

## B. 超越良性漂移 + 旧 BDS

加入 \(C_t\) 后留一法对数误差至少下降：

```text
15%
```

即：

```text
relative_improvement_conflict_over_drift_bds >= 0.15
```

## C. 控制已有变量后仍有剩余关系

\[
\rho(
C_t,
\text{控制良性漂移+BDS后的残差}
)
\le -0.40.
\]

## D. 解释 200/225/250

最好：

```text
C_200 < C_225 < C_250
```

最低要求：

```text
C_200 明显低于 C_250，
且谱冲突方向与效率下降一致。
```

## E. 稳定性

至少在：

```text
2 个窗口大小
×
2 个频带数
```

下保持同方向。

---

# 25. 参数稳定性实验

不要只报最好的参数。

至少跑：

```text
K=16, B=8, db2
K=32, B=8, db2
K=32, B=4, db2
K=32, B=16, db2
K=32, B=8, haar
```

示例：

```bash
python experiments/analyze_joint_spectral_conflict.py \
  --window 16 \
  --bands 8 \
  --wavelet db2 \
  --output-dir outputs/joint_spectral_conflict_k16_b8_db2

python experiments/spectral_policy_diagnostics.py \
  --features outputs/joint_spectral_conflict_k16_b8_db2/joint_spectral_features.csv \
  --output outputs/joint_spectral_conflict_k16_b8_db2/diagnostics.json
```

---

# 26. WEAK / SPECTRAL ARTIFACT / STOP

## WEAK

总体相关性明显，但加入谱冲突后无法稳定超越：

```text
良性漂移 + BDS
```

先增加 seed，不设计攻击策略。

## SPECTRAL ARTIFACT

如果：

```text
换 K/B/小波后大量符号翻转
```

或者频带划分轻微变化就导致结论完全改变：

```text
SPECTRAL ARTIFACT
```

说明参数频率缺乏稳定意义。

## STOP

如果：

```text
|Spearman(C_t,效率)| < 0.40
```

同时：

```text
加入 C_t 后 LOO 改善 < 10%
```

并且：

```text
无法解释 200/225/250 反转
```

则停止。

不要继续加：

```text
谱熵
更多小波族
Hessian
更多动力学特征
神经网络预测器
```

---

# 27. 完整方法成立后的在线攻击策略

只有 STRONG GO 后再正式实现。

第 \(t\) 轮攻击者被选中：

```text
1. 收到 w_{t-1}
2. 在本地训练候选恶意更新 d_t，但先不上传
3. 计算后门梯度 h_t 与干净梯度 c_t
4. 从过去 K 个全局模型构造历史良性更新
5. 计算后门有效谱 a_{t,l,b}
6. 计算逐频带良性功能抵抗序列
7. 做时间小波分解
8. 计算持续抵抗 p_{t,l,b}
9. 计算跨层一致性 κ_{t,b}
10. 得到联合谱冲突 C_t
11. 得到机会分数 V_t
12. 根据有限预算决定攻击或良性参与
```

---

# 28. 有限预算分配

## 28.1 Oracle 上界

若允许知道未来，仅作为上界：

\[
\text{选择最大的 }B\text{ 个 }V_t.
\]

不能当真实方法。

## 28.2 因果在线分位数策略

维护过去机会分数：

\[
\mathcal V_{<t}.
\]

设历史分位数阈值：

\[
\tau_t=Q_q(\mathcal V_{<t}).
\]

例如：

```text
q = 0.8
```

若：

\[
V_t\ge\tau_t
\]

且仍有预算，则投毒；否则以良性方式参与。

这样不需要未来信息。

正式论文阶段可进一步做预算自适应阈值，但当前不实现。

---

# 29. 完整实验矩阵

机制成立以后再做。

## 数据集

```text
CIFAR-10
CIFAR-100
TinyImageNet
```

当前阶段只 CIFAR-10。

## 模型

```text
ResNet18
VGG
```

## 聚合

```text
FedAvg
FLTrust
再选一个鲁棒聚合
```

## 基础攻击器

至少：

```text
普通本地后门 + Model Replacement
A3FL
Neurotoxin
```

目标：

```text
证明谱时机策略不依赖某一种攻击构造。
```

## 时机对照

```text
固定晚期
均匀间隔
随机
按良性漂移大小
按旧 BDS
按静态频谱能量
按联合谱冲突
oracle top-B
```

---

# 30. 最重要的消融

必须包含：

```text
仅参数频率，不做时间小波
仅时间序列，不做参数频带
去掉后门梯度，只看良性能量
去掉后门有效谱权重
去掉跨层一致性
普通傅里叶替代小波
只用低频能量比例
只用单轮余弦
完整联合谱冲突
```

如果完整方法不能稳定优于这些简化版本，论文故事不成立。

---

# 31. 数学支撑

由于正交离散余弦变换保持内积：

\[
\langle h,g\rangle
=
\langle
\mathcal D(h),
\mathcal D(g)
\rangle.
\]

因此后门损失的一阶变化：

\[
\Delta L_{\mathrm{后门}}
\approx
\langle h,g\rangle
\]

可以严格分解为各层各参数频带贡献：

\[
\Delta L_{\mathrm{后门}}
\approx
\sum_{l,b}
\langle
H_{l,b},
G_{l,b}
\rangle.
\]

所以这里的频带冲突不是纯能量启发式。

它直接来自：

\[
\boxed{
\text{后门损失一阶变化的谱分解}
}
\]

随后时间小波回答：

> 这些抵抗贡献究竟是最近持续存在的慢变化结构，还是短期振荡噪声？

这是整个方法最重要的理论连接。

---

# 32. Agent 最终汇报格式

## 32.1 数据完整性

```text
目标轮次谱状态：
7/7 或 ?

历史 checkpoint：
需要多少
找到多少

所有在线特征是否严格只使用 t-1 及以前：
YES / NO

谱状态 hook 是否零影响：
YES / NO
```

## 32.2 主结果

```text
Round    注入效率    良性漂移    旧BDS    谱冲突C_t    谱机会V_t
50
100
150
175
200
225
250
```

## 32.3 关键反转

```text
C_200:
C_225:
C_250:

C_200 < C_225 < C_250:
YES / NO
```

## 32.4 统计

```text
Spearman(1/良性漂移, 注入效率):
Spearman(旧BDS, 注入效率):
Spearman(谱冲突, 注入效率):
Spearman(谱机会, 注入效率):

LOO:
良性漂移:
良性漂移 + BDS:
良性漂移 + BDS + 谱冲突:

加入谱冲突后的相对改善:

Spearman(
    谱冲突,
    控制良性漂移+BDS后的残差
):
```

## 32.5 稳定性

至少：

```text
K=16/B=8/db2
K=32/B=8/db2
K=32/B=4/db2
K=32/B=16/db2
K=32/B=8/haar
```

不得只报最优配置。

## 32.6 最终 Verdict

只选：

```text
STRONG GO
WEAK
SPECTRAL ARTIFACT
STOP
```

最多 10 句话解释。

---

# 33. 本轮严格禁止

不要：

```text
重新研究 retention
直接做最终 scheduler
换数据集
加防御
加多种攻击器
调很多频谱超参
做 Hessian
训练预测器
拼接大量训练动力学特征
```

当前唯一问题：

\[
\boxed{
\text{联合谱冲突是否稳定解释
良性漂移和旧 BDS 无法解释的注入效率变化？}
}
\]

若不能：

```text
停止 when-to-inject 频谱线。
```

若能：

```text
下一轮再实现正式在线有限预算攻击策略。
```

---

# 34. 最终判断标准

这条 Idea 成立的标准不是：

```text
“频谱和攻击效果相关。”
```

而是：

\[
\boxed{
\text{后门有效谱与历史持续良性抵抗谱之间的冲突，
能够稳定解释单轮良性漂移与旧定向敏感度无法解释的注入效率变化。}
}
\]

只有做到这一点，才值得形成正式论文方法。

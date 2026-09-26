# FL Temporal Vulnerability — Exact Matching + Counterfactual Geometry Probe

## 0. 本轮为什么要改实验

上一轮已经出现一个值得继续追的现象：

```text
同一 clean-damage budget 下，不同 communication round 的 Injection Capacity 明显不同。
```

例如 5% clean-accuracy budget 下：

```text
round 50   : J_t ≈ 15.49%
round 150  : J_t ≈ 24.63%
round 250  : J_t ≈ 43.78%
```

但在把这个现象继续发展成方法之前，必须排除三个我们之前忽略的混杂。

### 混杂 1：scale_factor 不是跨 round 可比较的攻击成本

BackFed 中：

```text
scale_factor = λ
```

只是对当前 malicious local update 的缩放。

但不同 round 的原始 malicious update：

```text
u_t
```

本身范数可能不同。

因此：

```text
λ=5 at round 50
```

和：

```text
λ=5 at round 250
```

并不代表攻击者施加了同样大的参数扰动。

所以后续不能再把：

```text
scale_factor
```

称为 Injection Cost。

真正应比较的是：

```text
达到相同 immediate backdoor effect
需要造成多大的实际 global-model displacement。
```

---

### 混杂 2：旧 clean baseline 可能没有使用同一个 attack-round cohort

攻击分支在 attack round 会通过：

```text
adversary_selection=single
selection_scheme=all-adversary
```

保证恶意客户端出现。

但此前的 clean-trigger baseline 把 poisoning round 放在训练结束以后。

这样在真正的 attack round：

```text
attack branch
```

和：

```text
clean branch
```

可能选中了不同客户端集合。

那么：

```text
attack - clean
```

不仅包含 poisoning effect，

还可能包含：

```text
client-cohort difference
```

特别是在 non-IID FL 下，这是一个不可忽略的混杂。

因此本轮必须增加：

```text
cohort-matched counterfactual
```

它和 attack branch：

```text
使用相同 attack round
相同 adversary-selection mechanism
相同 seed
相同 client-selection protocol
```

但恶意客户端在这一轮执行 benign-like local training，而不是 poisoning。

---

### 混杂 3：之前 matched retention 失败只是 scale 网格太粗

之前：

```text
round 50  : 15.49%
round 150 : 24.63%
```

虽然共同目标约 20%，但因为只从旧 scale grid 选最近点，最终差 9.14pp。

本轮改为：

```text
adaptive scale refinement
```

真正把三个 round 校准到：

```text
20% ± 1 percentage point
```

之后才允许比较 retention。

---

# 1. 本轮真正验证的机制

本轮不实现 scheduler。

本轮只验证两个性质。

## 1.1 Functional Injection Cost

对 round t，构造 cohort-matched counterfactual：

```text
w_t^cf
```

以及 matched attack：

```text
w_t^atk
```

要求：

```text
Excess ASR ≈ q = 20%
```

定义实际攻击造成的 global-model displacement：

\[
p_t = w_t^{atk} - w_t^{cf}
\]

定义：

\[
C_t^{norm}
=
\frac{\|p_t\|_2}{q}
\]

或者等价地看：

\[
E_t^{inj}
=
\frac{q}{\|p_t\|_2}
\]

其中：

```text
C_t^norm 越小
```

表示：

```text
当前 global state 越容易被注入同样强的后门。
```

这个定义不再依赖：

```text
scale_factor
```

的物理含义。

---

## 1.2 Matched-Effect Retention

三个 round 都先校准到：

```text
Immediate Excess ASR = 20% ± 1pp
```

然后继续 benign FL：

```text
H = 30 rounds
```

对 cohort-matched counterfactual：

\[
E_t(h)
=
ASR_{t+h}^{atk}
-
ASR_{t+h}^{cf}
\]

定义：

\[
R_t
=
\frac{1}{H}
\sum_{h=1}^{H}
\frac{[E_t(h)]_+}{E_t(0)}
\]

同时记录：

```text
half-life
end excess ASR
```

---

# 2. 一个额外的 round-context 特征

本轮顺手提取一个很便宜但重要的量。

定义 cohort-matched benign round drift：

\[
g_t
=
w_t^{cf}
-
w_{t-1}
\]

这里：

```text
w_{t-1}
```

是 attack 前的 global model。

计算：

```text
Benign Drift Norm
||g_t||
```

以及攻击 displacement 与 benign drift 的：

\[
\cos(p_t,g_t)
\]

和：

\[
O_t
=
\sqrt{
1-\cos^2(p_t,g_t)
}
\]

其中：

```text
O_t
```

只是一个 one-step orthogonal fraction。

本轮不把它直接定义成最终 persistence score。

只是看：

```text
Injection Capacity / Retention
```

是否明显伴随：

```text
benign round drift magnitude
```

或者：

```text
attack–benign direction relation
```

变化。

如果后面 retention 真有信号，再升级到：

```text
K-round benign-update subspace
```

现在不要做。

---

# 3. 重要解释边界

现有工作已经知道：

```text
early-round attack
和
convergence-round attack
```

行为不同。

Model Replacement 也天然更容易在 benign aggregate update 较小时发挥作用。

Neurotoxin 也已经利用：

```text
benign training 中低更新参数
```

提高 persistence。

因此最终方法不能退化成：

```text
global update norm 小 → attack
```

也不能退化成：

```text
找 benign gradient 小的参数
```

真正值得继续的情况应该是：

```text
在严格 matched cohort、
matched immediate effect、
actual perturbation norm
控制之后，

不同 round 仍然存在明显不同的
functional injection efficiency
和/或 retention。
```

只有这样才值得后续研究真正的：

```text
Temporal Vulnerability Score
```

---

# 4. 使用仓库

继续使用：

```text
BackFed
commit: c851ce90373ef2659447ea25296c7b442e5dc5c7
```

如果仓库不存在：

```bash
git clone https://github.com/thinh-dao/BackFed.git
cd BackFed
git checkout c851ce90373ef2659447ea25296c7b442e5dc5c7
```

环境沿用：

```text
Python 3.11
PyTorch 2.6.0
RTX 4090 D
training_mode=sequential
```

---

# 5. 保留已有脚本

不要删除：

```text
experiments/temporal_cost_retention_probe.py
experiments/analyze_temporal_injection_frontier.py
experiments/run_matched_retention_probe.py
```

本轮新增两个脚本：

```text
experiments/run_exact_temporal_matching.py
experiments/analyze_temporal_geometry.py
```

BackFed 核心代码原则上不修改。

唯一允许的核心修改：

```text
如果当前代码完全无法记录每轮 selected client IDs，
允许只增加 logging，
不得改变 client-selection behavior。
```

---

# 6. Cohort-Matched Counterfactual 定义

对于 attack round t：

## Attack branch

```text
poison_start_round=t
poison_end_round=t
adversary_selection=single
selection_scheme=all-adversary

poison_ratio=0.3125
use_atk_optimizer=true
poison_epochs=6
poison_lr=0.05

scale_poison=true
scale_factor=λ
```

## Counterfactual branch

同样：

```text
poison_start_round=t
poison_end_round=t
adversary_selection=single
selection_scheme=all-adversary
```

但：

```text
poison_ratio=0
use_atk_optimizer=false
scale_poison=false
scale_factor=1
```

语义：

```text
仍然触发同一个 attack-round client-selection path，
但被控制客户端这一轮不进行 poisoning。
```

Agent 必须检查 BackFed 当前实现。

确认：

```text
poison_ratio=0
+
use_atk_optimizer=false
+
scale_poison=false
```

确实会让 malicious client 的 update 退化成 benign-like update。

如果不是：

```text
不要自行猜。
```

只做一个最小兼容：

```text
给 malicious client 增加 benign-counterfactual mode，
直接调用和普通 benign client 完全相同的 local-train path。
```

如果必须这么改：

最终报告必须明确写出修改文件和代码位置。

---

# 7. Client Selection 必须核对

Attack branch 和 Counterfactual branch 在 round t：

```text
selected client IDs 必须完全一致。
```

如果 BackFed 已经有：

```text
plot_client_selection
```

或 selection log：

直接复用。

如果没有：

只允许增加 logging：

```python
print(
    f"[CLIENT_SELECTION] round={round_idx} "
    f"clients={list(selected_client_ids)}"
)
```

不得修改 selection algorithm。

如果：

```text
attack round client IDs 不一致
```

整个实验无效。

必须先修 counterfactual。

---

# 8. Exact Matching 目标

固定：

```text
attack rounds = 50, 150, 250
q = 0.20
tolerance = 0.01
max clean drop = 0.10
horizon = 30
```

要求每个 round：

```text
0.19 <= immediate excess ASR <= 0.21
```

其中：

```text
immediate excess ASR
=
attack ASR at round t
-
cohort-matched counterfactual ASR at round t
```

---

# 9. 新增脚本一

新建：

```text
BackFed/experiments/run_exact_temporal_matching.py
```

完整代码如下。

```python
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
from typing import Dict, List, Optional, Tuple


DEFAULT_ROUNDS = [50, 150, 250]
DEFAULT_INITIAL_SCALES = [1.0, 2.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Exact matched-effect temporal probe with a cohort-matched "
            "counterfactual branch."
        )
    )

    p.add_argument(
        "--attack-rounds",
        type=int,
        nargs="+",
        default=DEFAULT_ROUNDS,
    )

    p.add_argument(
        "--initial-scales",
        type=float,
        nargs="+",
        default=DEFAULT_INITIAL_SCALES,
    )

    p.add_argument(
        "--target",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--max-clean-drop",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--horizon",
        type=int,
        default=30,
    )

    p.add_argument(
        "--max-refine-runs",
        type=int,
        default=8,
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
        "--output-root",
        type=Path,
        default=Path("outputs/exact_temporal_matching"),
    )

    p.add_argument(
        "--force",
        action="store_true",
    )

    p.add_argument(
        "--smoke",
        action="store_true",
    )

    return p.parse_args()


def norm_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def as_float(value: object) -> Optional[float]:
    if value is None:
        return None

    try:
        x = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x):
        return None

    return x


def get_round(row: Dict[str, str]) -> Optional[int]:
    for key in ("round", "step"):
        if key in row:
            x = as_float(row[key])

            if x is not None:
                return int(round(x))

    return None


def find_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.rglob("*.csv")
    )

    scored: List[Tuple[int, int, Path]] = []

    for path in candidates:
        try:
            with path.open("r", newline="") as f:
                header = next(
                    csv.reader(f),
                    None,
                )
        except Exception:
            continue

        if not header:
            continue

        cols = {
            norm_col(col)
            for col in header
        }

        score = 0

        if "test_backdoor_acc" in cols:
            score += 100

        if "test_clean_acc" in cols:
            score += 50

        if "round" in cols or "step" in cols:
            score += 10

        if score:
            scored.append(
                (
                    score,
                    path.stat().st_size,
                    path,
                )
            )

    if not scored:
        raise RuntimeError(
            f"No metrics CSV found under {run_dir}"
        )

    scored.sort(
        reverse=True
    )

    return scored[0][2]


def read_metrics(
    path: Path,
) -> List[Dict[str, str]]:
    with path.open(
        "r",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        rows = []

        for raw in reader:
            rows.append(
                {
                    norm_col(k): v
                    for k, v in raw.items()
                    if k is not None
                }
            )

    return rows


def index_metrics(
    rows: List[Dict[str, str]],
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}

    for row in rows:
        r = get_round(row)

        if r is None:
            continue

        bd = as_float(
            row.get("test_backdoor_acc")
        )

        ca = as_float(
            row.get("test_clean_acc")
        )

        if bd is None:
            continue

        out[r] = {
            "backdoor_acc": bd,
            "clean_acc": (
                ca
                if ca is not None
                else float("nan")
            ),
        }

    return out


def run_cmd(
    cmd: List[str],
    repo_root: Path,
    env: Dict[str, str],
) -> None:
    print("\n" + "=" * 110)
    print(" ".join(cmd))
    print("=" * 110 + "\n")

    subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        check=True,
    )


def build_common_cmd(
    *,
    repo_root: Path,
    run_dir: Path,
    seed: int,
    gpu: str,
    training_mode: str,
    num_rounds: int,
    attack_round: int,
) -> List[str]:
    return [
        sys.executable,
        str(repo_root / "main.py"),

        "--config-name",
        "cifar10",

        "checkpoint=null",
        f"seed={seed}",
        "deterministic=true",
        f"cuda_visible_devices={gpu}",

        "aggregator=unweighted_fedavg",
        "dataset=CIFAR10",
        "model=ResNet18",

        f"num_rounds={num_rounds}",
        f"training_mode={training_mode}",

        "save_logging=csv",
        "progress_bar=false",
        "plot_client_selection=true",

        "no_attack=false",
        "atk_config=cifar10_multishot",

        "atk_config.data_poison_method=pattern",
        "atk_config.model_poison_method=base",

        "atk_config.poison_frequency=single-shot",
        f"atk_config.poison_start_round={attack_round}",
        f"atk_config.poison_end_round={attack_round}",
        "atk_config.poison_interval=1",

        "atk_config.adversary_selection=single",
        "atk_config.selection_scheme=all-adversary",

        "atk_config.attack_type=all2one",
        "atk_config.random_class=false",
        "atk_config.target_class=2",

        "test_every=1",

        f"hydra.run.dir={run_dir.resolve()}",
        "dir_tag=exact_temporal_matching",
    ]


def launch_variant(
    *,
    repo_root: Path,
    env: Dict[str, str],
    run_dir: Path,
    seed: int,
    gpu: str,
    training_mode: str,
    num_rounds: int,
    attack_round: int,
    mode: str,
    scale: float,
    force: bool,
    save_model: bool = False,
) -> Path:
    if run_dir.exists():
        if force:
            shutil.rmtree(run_dir)
        else:
            return find_metrics_csv(
                run_dir
            )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = build_common_cmd(
        repo_root=repo_root,
        run_dir=run_dir,
        seed=seed,
        gpu=gpu,
        training_mode=training_mode,
        num_rounds=num_rounds,
        attack_round=attack_round,
    )

    if mode == "attack":
        cmd += [
            "atk_config.poison_ratio=0.3125",
            "atk_config.use_atk_optimizer=true",
            "atk_config.poison_epochs=6",
            "atk_config.poison_lr=0.05",
            "atk_config.scale_poison=true",
            f"atk_config.scale_factor={scale}",
        ]

    elif mode == "counterfactual":
        cmd += [
            "atk_config.poison_ratio=0.0",
            "atk_config.use_atk_optimizer=false",
            "atk_config.poison_epochs=2",
            "atk_config.poison_lr=0.1",
            "atk_config.scale_poison=false",
            "atk_config.scale_factor=1.0",
        ]

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    if save_model:
        cmd += [
            "save_model=true",
        ]

    run_cmd(
        cmd,
        repo_root,
        env,
    )

    return find_metrics_csv(
        run_dir
    )


def metric_at(
    idx: Dict[int, Dict[str, float]],
    r: int,
) -> Dict[str, float]:
    if r not in idx:
        raise RuntimeError(
            f"Round {r} not found. "
            f"Available rounds: {sorted(idx)}"
        )

    return idx[r]


def compare_effect(
    *,
    attack_idx: Dict[int, Dict[str, float]],
    cf_idx: Dict[int, Dict[str, float]],
    attack_round: int,
) -> Dict[str, float]:
    a = metric_at(
        attack_idx,
        attack_round,
    )

    c = metric_at(
        cf_idx,
        attack_round,
    )

    return {
        "attack_asr": a["backdoor_acc"],
        "cf_asr": c["backdoor_acc"],
        "excess_asr": (
            a["backdoor_acc"]
            - c["backdoor_acc"]
        ),
        "attack_clean_acc": a["clean_acc"],
        "cf_clean_acc": c["clean_acc"],
        "clean_drop": (
            c["clean_acc"]
            - a["clean_acc"]
        ),
    }


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


def scale_key(
    scale: float,
) -> float:
    return round(
        float(scale),
        8,
    )


def choose_best(
    history: Dict[float, Dict[str, float]],
    target: float,
    max_clean_drop: float,
) -> Optional[Dict[str, float]]:
    feasible = [
        row
        for row in history.values()
        if row["clean_drop"] <= max_clean_drop
    ]

    if not feasible:
        return None

    return min(
        feasible,
        key=lambda row: (
            abs(
                row["excess_asr"]
                - target
            ),
            row["clean_drop"],
            row["scale"],
        ),
    )


def find_bracket(
    history: Dict[float, Dict[str, float]],
    target: float,
) -> Optional[
    Tuple[
        Dict[str, float],
        Dict[str, float],
    ]
]:
    rows = sorted(
        history.values(),
        key=lambda row: row["scale"],
    )

    candidates = []

    for i in range(
        len(rows) - 1
    ):
        left = rows[i]
        right = rows[i + 1]

        y1 = left["excess_asr"]
        y2 = right["excess_asr"]

        if (
            (y1 <= target <= y2)
            or
            (y2 <= target <= y1)
        ):
            candidates.append(
                (
                    right["scale"]
                    - left["scale"],
                    left,
                    right,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0]
    )

    return (
        candidates[0][1],
        candidates[0][2],
    )


def midpoint_scale(
    left: Dict[str, float],
    right: Dict[str, float],
) -> float:
    return (
        float(left["scale"])
        + float(right["scale"])
    ) / 2.0


def compute_retention(
    *,
    attack_idx: Dict[int, Dict[str, float]],
    cf_idx: Dict[int, Dict[str, float]],
    attack_round: int,
    horizon: int,
) -> Tuple[
    Dict[str, float],
    List[Dict[str, float]],
]:
    curve = []

    for h in range(
        0,
        horizon + 1,
    ):
        r = (
            attack_round
            + h
        )

        a = metric_at(
            attack_idx,
            r,
        )

        c = metric_at(
            cf_idx,
            r,
        )

        excess = (
            a["backdoor_acc"]
            - c["backdoor_acc"]
        )

        curve.append(
            {
                "h": h,
                "round": r,
                "attack_asr": a["backdoor_acc"],
                "cf_asr": c["backdoor_acc"],
                "excess_asr": excess,
                "attack_clean_acc": a["clean_acc"],
                "cf_clean_acc": c["clean_acc"],
                "clean_drop": (
                    c["clean_acc"]
                    - a["clean_acc"]
                ),
            }
        )

    e0 = curve[0]["excess_asr"]

    if e0 <= 0:
        raise RuntimeError(
            f"Non-positive immediate excess ASR: {e0}"
        )

    post = curve[1:]

    retention_auc = (
        sum(
            max(
                row["excess_asr"],
                0.0,
            )
            / e0
            for row in post
        )
        / len(post)
    )

    half_threshold = (
        0.5 * e0
    )

    half_life = (
        horizon + 1
    )

    for row in post:
        if (
            row["excess_asr"]
            <= half_threshold
        ):
            half_life = int(
                row["h"]
            )
            break

    return (
        {
            "immediate_excess_asr": e0,
            "retention_auc": retention_auc,
            "half_life": half_life,
            "end_excess_asr": curve[-1]["excess_asr"],
            "peak_excess_asr": max(
                row["excess_asr"]
                for row in curve
            ),
        },
        curve,
    )


def main() -> None:
    args = parse_args()

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    if not (
        repo_root
        / "main.py"
    ).exists():
        raise FileNotFoundError(
            repo_root
            / "main.py"
        )

    rounds = list(
        args.attack_rounds
    )

    if args.smoke:
        rounds = rounds[:1]
        args.horizon = min(
            args.horizon,
            5,
        )
        args.max_refine_runs = min(
            args.max_refine_runs,
            2,
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

    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = (
        args.gpu
    )

    env["PYTHONHASHSEED"] = str(
        args.seed
    )

    final_rows = []

    for t in rounds:
        round_root = (
            output_root
            / f"round_{t}"
        )

        cf_long_dir = (
            round_root
            / "counterfactual"
        )

        cf_csv = launch_variant(
            repo_root=repo_root,
            env=env,
            run_dir=cf_long_dir,
            seed=args.seed,
            gpu=args.gpu,
            training_mode=args.training_mode,
            num_rounds=t + args.horizon,
            attack_round=t,
            mode="counterfactual",
            scale=1.0,
            force=args.force,
        )

        cf_idx = index_metrics(
            read_metrics(
                cf_csv
            )
        )

        history: Dict[
            float,
            Dict[str, float],
        ] = {}

        def evaluate(
            scale: float,
        ) -> Dict[str, float]:
            key = scale_key(
                scale
            )

            if key in history:
                return history[key]

            tag = (
                f"{scale:.8f}"
                .rstrip("0")
                .rstrip(".")
                .replace(".", "p")
            )

            run_dir = (
                round_root
                / "calibration"
                / f"scale_{tag}"
            )

            attack_csv = launch_variant(
                repo_root=repo_root,
                env=env,
                run_dir=run_dir,
                seed=args.seed,
                gpu=args.gpu,
                training_mode=args.training_mode,
                num_rounds=t,
                attack_round=t,
                mode="attack",
                scale=scale,
                force=args.force,
            )

            attack_idx = index_metrics(
                read_metrics(
                    attack_csv
                )
            )

            effect = compare_effect(
                attack_idx=attack_idx,
                cf_idx=cf_idx,
                attack_round=t,
            )

            row = {
                "scale": float(scale),
                **effect,
            }

            history[key] = row

            print(
                f"[MATCH t={t}] "
                f"scale={scale:.8f} "
                f"excess={row['excess_asr']:.6f} "
                f"clean_drop={row['clean_drop']:.6f}"
            )

            return row

        initial_scales = sorted(
            set(
                float(x)
                for x in args.initial_scales
            )
        )

        for scale in initial_scales:
            evaluate(
                scale
            )

            best = choose_best(
                history,
                args.target,
                args.max_clean_drop,
            )

            if (
                best is not None
                and abs(
                    best["excess_asr"]
                    - args.target
                )
                <= args.tolerance
            ):
                break

            bracket = find_bracket(
                history,
                args.target,
            )

            if bracket is not None:
                break

        refine_count = 0

        while (
            refine_count
            < args.max_refine_runs
        ):
            best = choose_best(
                history,
                args.target,
                args.max_clean_drop,
            )

            if (
                best is not None
                and abs(
                    best["excess_asr"]
                    - args.target
                )
                <= args.tolerance
            ):
                break

            bracket = find_bracket(
                history,
                args.target,
            )

            if bracket is None:
                break

            left, right = bracket

            new_scale = midpoint_scale(
                left,
                right,
            )

            if (
                scale_key(new_scale)
                in history
            ):
                break

            evaluate(
                new_scale
            )

            refine_count += 1

        calibration_rows = sorted(
            history.values(),
            key=lambda row: row["scale"],
        )

        write_csv(
            round_root
            / "calibration_search.csv",
            calibration_rows,
            [
                "scale",
                "attack_asr",
                "cf_asr",
                "excess_asr",
                "attack_clean_acc",
                "cf_clean_acc",
                "clean_drop",
            ],
        )

        best = choose_best(
            history,
            args.target,
            args.max_clean_drop,
        )

        if best is None:
            final_rows.append(
                {
                    "attack_round": t,
                    "status": "NO_FEASIBLE_SCALE",
                    "matched_scale": float("nan"),
                    "calibration_excess_asr": float("nan"),
                    "calibration_clean_drop": float("nan"),
                    "immediate_excess_asr": float("nan"),
                    "retention_auc": float("nan"),
                    "half_life": float("nan"),
                    "end_excess_asr": float("nan"),
                    "peak_excess_asr": float("nan"),
                }
            )

            continue

        if (
            abs(
                best["excess_asr"]
                - args.target
            )
            > args.tolerance
        ):
            final_rows.append(
                {
                    "attack_round": t,
                    "status": "MATCH_FAILED",
                    "matched_scale": best["scale"],
                    "calibration_excess_asr": best["excess_asr"],
                    "calibration_clean_drop": best["clean_drop"],
                    "immediate_excess_asr": float("nan"),
                    "retention_auc": float("nan"),
                    "half_life": float("nan"),
                    "end_excess_asr": float("nan"),
                    "peak_excess_asr": float("nan"),
                }
            )

            continue

        matched_scale = float(
            best["scale"]
        )

        retention_dir = (
            round_root
            / "matched_retention"
        )

        attack_long_csv = launch_variant(
            repo_root=repo_root,
            env=env,
            run_dir=retention_dir,
            seed=args.seed,
            gpu=args.gpu,
            training_mode=args.training_mode,
            num_rounds=t + args.horizon,
            attack_round=t,
            mode="attack",
            scale=matched_scale,
            force=args.force,
        )

        attack_long_idx = index_metrics(
            read_metrics(
                attack_long_csv
            )
        )

        retention_summary, curve = compute_retention(
            attack_idx=attack_long_idx,
            cf_idx=cf_idx,
            attack_round=t,
            horizon=args.horizon,
        )

        write_csv(
            round_root
            / "matched_retention_curve.csv",
            curve,
            [
                "h",
                "round",
                "attack_asr",
                "cf_asr",
                "excess_asr",
                "attack_clean_acc",
                "cf_clean_acc",
                "clean_drop",
            ],
        )

        rerun_error = abs(
            retention_summary[
                "immediate_excess_asr"
            ]
            - args.target
        )

        status = (
            "PASS"
            if rerun_error
            <= args.tolerance
            else "RERUN_MATCH_DRIFT"
        )

        final_rows.append(
            {
                "attack_round": t,
                "status": status,
                "matched_scale": matched_scale,
                "calibration_excess_asr": best["excess_asr"],
                "calibration_clean_drop": best["clean_drop"],
                **retention_summary,
            }
        )

    write_csv(
        output_root
        / "exact_matching_summary.csv",
        final_rows,
        [
            "attack_round",
            "status",
            "matched_scale",
            "calibration_excess_asr",
            "calibration_clean_drop",
            "immediate_excess_asr",
            "retention_auc",
            "half_life",
            "end_excess_asr",
            "peak_excess_asr",
        ],
    )

    with (
        output_root
        / "exact_matching_summary.json"
    ).open("w") as f:
        json.dump(
            final_rows,
            f,
            indent=2,
        )

    print("\nFINAL")
    print("=" * 110)

    for row in final_rows:
        print(row)


if __name__ == "__main__":
    main()
```

---

# 10. Exact Matching Smoke Test

先运行：

```bash
python experiments/run_exact_temporal_matching.py \
  --attack-rounds 50 \
  --target 0.20 \
  --tolerance 0.02 \
  --horizon 5 \
  --max-refine-runs 2 \
  --gpu 0 \
  --training-mode sequential \
  --smoke \
  --force
```

必须确认：

```text
1. Attack branch 和 Counterfactual branch 在 attack round 的 selected client IDs 完全相同。
2. Attack branch 只有一次 poisoning。
3. Counterfactual branch 没有 poisoned samples。
4. Counterfactual branch 仍然走同一个 attack-round selection path。
5. attack 前 trajectory 完全一致。
6. calibration_search.csv 正常生成。
```

如果：

```text
client IDs 不一致
```

停止。

不要继续。

---

# 11. 正式 Exact Matching

Smoke test 通过：

```bash
python experiments/run_exact_temporal_matching.py \
  --attack-rounds 50 150 250 \
  --target 0.20 \
  --tolerance 0.01 \
  --max-clean-drop 0.10 \
  --horizon 30 \
  --max-refine-runs 8 \
  --seed 2026 \
  --gpu 0 \
  --training-mode sequential \
  --force
```

目标：

```text
round 50:
19% <= immediate excess ASR <= 21%

round 150:
19% <= immediate excess ASR <= 21%

round 250:
19% <= immediate excess ASR <= 21%
```

如果任意 round：

```text
MATCH_FAILED
```

先停止。

不要解释 retention。

---

# 12. Matched Retention 第一判定

只有三个 round：

```text
status=PASS
```

才比较：

```text
R_50
R_150
R_250
```

### Retention Strong Signal

满足任一：

```text
max(R_t) - min(R_t) >= 0.15
```

或者：

```text
max(half_life) - min(half_life) >= 5 rounds
```

则：

```text
RETENTION GO
```

### Retention Weak

如果：

```text
0.05 <= spread < 0.15
```

则：

```text
RETENTION WEAK
```

### Retention Flat

如果：

```text
spread < 0.05
```

且 half-life 基本一致：

```text
RETENTION FLAT
```

注意：

```text
RETENTION FLAT
```

不自动杀死整个 Idea。

因为 Injection Vulnerability 仍可能成立。

---

# 13. 新增脚本二：实际参数几何

新建：

```text
BackFed/experiments/analyze_temporal_geometry.py
```

这个脚本重新运行很短的：

```text
pre-attack
counterfactual-at-t
matched-attack-at-t
```

分支，并要求：

```text
save_model=true
```

然后比较真正的参数 displacement。

完整代码如下。

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from run_exact_temporal_matching import (
    launch_variant,
    read_metrics,
    index_metrics,
    compare_effect,
)


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
    main()
```

---

# 14. 运行 Geometry Probe

只有三个 round：

```text
Exact Matching 全部 PASS
```

后运行：

```bash
python experiments/analyze_temporal_geometry.py \
  --seed 2026 \
  --gpu 0 \
  --training-mode sequential \
  --force
```

如果：

```text
save_model=true
```

没有把模型文件放在 Hydra run directory：

Agent 先查看 BackFed 当前 commit 的：

```text
save_model
```

实现。

只允许修改：

```text
analyze_temporal_geometry.py
```

里的：

```text
find_model_file()
```

或者模型保存路径适配。

不要修改训练逻辑。

---

# 15. Geometry 输出

必须得到：

```text
Round
Matched Scale
Immediate Excess ASR
Clean Drop
Actual Attack Displacement Norm
Benign Round Drift Norm
Injection Efficiency
Norm Cost per Effect
Attack/Benign Norm Ratio
cos(attack, benign)
Orthogonal Fraction
```

核心表：

```text
Round   q      ||p_t||   q/||p_t||   ||g_t||   ||p_t||/||g_t||   cos(p_t,g_t)
50
150
250
```

注意：

```text
matched scale
```

只作为工程记录。

不要再把它叫：

```text
Injection Cost
```

---

# 16. 怎么判断之前的 Injection Frontier 是真信号还是 artifact

## Case A：真正的 Functional Injection Vulnerability

三个 round 已经严格 matched：

```text
q ≈ 20%
```

如果：

```text
max(||p_t||) / min(||p_t||) >= 1.5
```

说明：

```text
为了得到同样大小的 immediate backdoor effect，
不同 global state 实际需要不同大小的参数 displacement。
```

这是强信号。

等价地：

```text
Injection Efficiency = q / ||p_t||
```

存在明显差异。

判：

```text
INJECTION GEOMETRY GO
```

---

## Case B：之前主要是 scale artifact

如果：

```text
matched scale 差异很大
```

但：

```text
max(||p_t||) / min(||p_t||) < 1.2
```

则说明：

```text
不同 round 的 base malicious update norm 不同，
导致 scale_factor 看起来差很多，
但真正所需 global displacement 基本相同。
```

此时之前的 Injection Capacity story 大幅降级。

判：

```text
SCALE ARTIFACT
```

---

## Case C：Benign-round competition 可能是主要因素

如果：

```text
Injection Efficiency
```

主要随着：

```text
Benign Round Drift Norm
```

变化，

例如：

```text
benign drift 越小
→ q/||p_t|| 越高
```

那么可能只是经典的：

```text
convergence-round / benign-dilution effect
```

不要把它包装成新机制。

判：

```text
BENIGN-DRIFT EXPLANATION
```

后续 Idea 需要再找比：

```text
||g_t||
```

更深的东西。

---

## Case D：方向关系有额外结构

如果：

```text
||g_t||
```

不能解释结果，

但：

```text
cos(p_t,g_t)
```

或者：

```text
orthogonal_fraction
```

与：

```text
clean damage
retention
```

存在明显对应，

则记录：

```text
DIRECTIONAL GEOMETRY SIGNAL
```

但本轮不要直接设计最终 score。

三个点太少，不能声称相关性。

---

# 17. 最终整条 Idea 的本轮判定

本轮最终不要只给：

```text
GO / STOP
```

而是从下面五个 verdict 里选一个。

## STRONG TEMPORAL GO

要求：

```text
1. 三个 round exact matching 全部成功；
2. actual displacement norm 明显不同；
3. matched retention 也明显不同；
4. 结果不能仅由 benign drift norm 的单调变化解释。
```

这时下一步可以正式设计：

```text
Temporal Vulnerability Score
```

---

## INJECTION-ONLY GO

要求：

```text
actual injection efficiency 明显不同，
但 matched retention 基本相同。
```

这意味着：

```text
when-to-inject 的主要价值来自“什么时候容易写进去”，
而不是“什么时候更容易留下来”。
```

此时后续方法只建模：

```text
Injection Vulnerability
```

不要硬凑 retention。

---

## RETENTION-ONLY GO

要求：

```text
actual injection cost 基本一致，
但 matched retention 明显不同。
```

此时后续重点转为：

```text
future forgetting geometry
```

---

## ARTIFACT / WEAK

如果 frontier 差异主要由：

```text
scale_factor
```

或：

```text
current benign drift magnitude
```

解释，

则：

```text
不继续做复杂 scheduler。
```

---

## STOP

如果：

```text
actual matched injection cost 基本一致
+
matched retention 基本一致
```

则：

```text
when-to-inject Idea 停止。
```

---

# 18. 如果出现 STRONG / INJECTION / RETENTION GO，下一轮才考虑的特征

这一节只记录方向。

本轮不要实现。

如果后续继续，优先考虑以下而不是“round index”。

### Feature 1：Local Counterfactual Attackability Frontier

攻击者收到：

```text
w_t
```

后，可以先本地训练一个 candidate malicious update：

```text
u_t
```

但不立即上传。

在本地虚拟构造：

\[
w_t+\lambda u_t
\]

对若干小 λ 评估：

```text
local trigger gain
local clean damage
```

形成一个很便宜的：

```text
local dose-response frontier
```

它可能直接预测当前 round 是否值得消耗攻击预算。

这个比：

```text
global update norm 小就攻击
```

更像一个真正的新机制。

---

### Feature 2：Recent Benign-Update Subspace Escape

如果 matched retention 明显不同，

下一轮再构造最近 K 个 global drift：

\[
G_t=
[g_{t-K+1},...,g_t]
\]

对 attack direction：

\[
p_t
\]

计算它在 recent benign-update subspace 中的能量：

\[
O_t^{sub}
=
\frac{
\|P_{G_t}p_t\|_2^2
}{
\|p_t\|_2^2
}
\]

以及：

\[
S_t^{escape}
=
1-O_t^{sub}
\]

如果：

```text
escape 高
→ retention 高
```

才有理由把它发展成：

```text
when-to-inject retention predictor
```

注意：

Neurotoxin 已经做过：

```text
低更新坐标
```

所以未来必须证明：

```text
cross-round subspace geometry
```

提供了 coordinate-frequency 之外的信息。

---

# 19. 最终 Agent 回报格式

只回报以下内容。

## 19.1 Counterfactual Fidelity

```text
round 50:
attack/counterfactual selected clients identical: YES/NO

round 150:
...

round 250:
...
```

说明：

```text
poison_ratio=0 + use_atk_optimizer=false
```

是否确实退化成 benign-like update。

---

## 19.2 Exact Matching

```text
Round    Scale    Immediate Excess ASR    Clean Drop    Status
50
150
250
```

必须说明：

```text
max immediate ASR spread
```

---

## 19.3 Matched Retention

```text
Round    R_t    Half-life    End Excess ASR
50
150
250
```

给：

```text
Retention verdict:
GO / WEAK / FLAT
```

---

## 19.4 Actual Geometry

```text
Round    ||p_t||    q/||p_t||    ||g_t||    ||p_t||/||g_t||    cos(p_t,g_t)
50
150
250
```

---

## 19.5 Final Verdict

只选：

```text
STRONG TEMPORAL GO
INJECTION-ONLY GO
RETENTION-ONLY GO
ARTIFACT / WEAK
STOP
```

最多 8 句话解释。

---

# 20. 本轮严格禁止

不要：

```text
实现 scheduler
加入 A3FL
加入 Neurotoxin
换 dataset
换 model
加入 defense
增加 malicious clients
扫 poison ratio
训练 predictor
设计复杂 loss
直接把 round index 当特征
用 scale_factor 继续冒充 Injection Cost
```

本轮唯一目标：

```text
把 temporal signal 从
scale artifact、
client-selection artifact、
unmatched-effect artifact
中剥离出来。
```

如果剥离后信号仍然存在，

才值得正式做方法。

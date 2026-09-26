# FL Temporal Vulnerability — Injection Capacity / Matched-Retention Probe

## 0. 本轮目标

不要实现 scheduler。

不要继续使用上一轮的：

```text
Reachable / Unreachable
```

硬阈值定义。

本轮只回答两个问题：

```text
Q1. 不同 communication round 的 stealth-constrained injection capacity 是否明显不同？

Q2. 当不同 round 的 immediate backdoor effect 被校准到相近水平后，
    它们的 retention 是否仍然明显不同？
```

如果 Q1 和 Q2 都没有明显差异：

```text
STOP
```

如果存在明显差异：

```text
GO
```

再进入下一阶段建模 Temporal Vulnerability Score。

---

# 1. 使用已有实验结果

沿用当前 BackFed 仓库：

```text
commit: c851ce90373ef2659447ea25296c7b442e5dc5c7
```

已有输出：

```text
BackFed/outputs/temporal_cost_retention_probe/
```

重点读取：

```text
round_50/injection_search.csv
round_150/injection_search.csv
round_250/injection_search.csv
```

不要重新跑 injection-search，除非文件缺失或字段损坏。

---

# 2. 废弃旧 Injection Cost 定义

不要再使用：

```text
达到 excess ASR >= 30pp
且 clean drop <= 5pp
所需最小 scale
```

这个定义会把连续 trade-off 粗暴切成：

```text
Reachable
UNREACHABLE
```

例如：

```text
round 150, scale=5:
excess ASR = 24.63pp
```

虽然没有达到 30pp，但显然并不等于“完全无法注入”。

本轮改为直接建模：

```text
攻击收益 vs clean damage
```

---

# 3. 定义 Injection Capacity

对于 communication round t 和 scale λ：

```text
A_t(λ)
=
attack_backdoor_acc
-
clean_counterfactual_backdoor_acc
```

即：

```text
excess ASR
```

定义：

```text
D_t(λ)
=
clean_counterfactual_clean_acc
-
attack_clean_acc
```

即：

```text
clean accuracy drop
```

对于给定 stealth budget δ，定义：

```text
J_t(δ)
=
max A_t(λ)
subject to D_t(λ) <= δ
```

其中：

```text
J_t(δ)
```

称为：

```text
Stealth-Constrained Injection Capacity
```

本轮计算：

```text
δ ∈ {0.01, 0.02, 0.05, 0.10}
```

对应：

```text
1pp
2pp
5pp
10pp
```

clean-accuracy budget。

---

# 4. 不做插值

第一轮只使用真实跑出来的 scale grid：

```text
1
2
5
10
15
25
40
60
```

不要对：

```text
A_t(λ)
D_t(λ)
```

做 spline、polynomial 或其他插值。

原因：

当前只想确认是否存在明显结构。

直接在已有离散点里：

```text
找满足 D_t(λ) <= δ 的点
再取其中最大的 A_t(λ)
```

即可。

如果某个 δ 下没有任何满足 clean-damage budget 的点：

```text
J_t(δ) = NaN
```

不要虚构。

---

# 5. 新增分析脚本

新建：

```text
BackFed/experiments/analyze_temporal_injection_frontier.py
```

完整代码如下。

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


DEFAULT_ROUNDS = [50, 150, 250]
DEFAULT_BUDGETS = [0.01, 0.02, 0.05, 0.10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze temporal injection-capacity frontiers from existing BackFed probe CSVs."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/temporal_cost_retention_probe"),
    )

    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=DEFAULT_ROUNDS,
    )

    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=DEFAULT_BUDGETS,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/temporal_injection_frontier"),
    )

    return parser.parse_args()


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


def read_search_csv(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: List[Dict[str, float]] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for raw in reader:
            scale = as_float(raw.get("scale"))
            attack_asr = as_float(raw.get("attack_asr"))
            clean_asr = as_float(raw.get("clean_asr"))
            excess_asr = as_float(raw.get("excess_asr"))
            attack_clean_acc = as_float(raw.get("attack_clean_acc"))
            clean_clean_acc = as_float(raw.get("clean_clean_acc"))
            clean_drop = as_float(raw.get("clean_drop"))

            if scale is None:
                continue

            if excess_asr is None and attack_asr is not None and clean_asr is not None:
                excess_asr = attack_asr - clean_asr

            if clean_drop is None and clean_clean_acc is not None and attack_clean_acc is not None:
                clean_drop = clean_clean_acc - attack_clean_acc

            if excess_asr is None or clean_drop is None:
                continue

            rows.append(
                {
                    "scale": scale,
                    "excess_asr": excess_asr,
                    "clean_drop": clean_drop,
                    "attack_asr": (
                        attack_asr if attack_asr is not None else float("nan")
                    ),
                    "clean_asr": (
                        clean_asr if clean_asr is not None else float("nan")
                    ),
                    "attack_clean_acc": (
                        attack_clean_acc if attack_clean_acc is not None else float("nan")
                    ),
                    "clean_clean_acc": (
                        clean_clean_acc if clean_clean_acc is not None else float("nan")
                    ),
                }
            )

    rows.sort(key=lambda row: row["scale"])

    if not rows:
        raise RuntimeError(f"No valid rows parsed from {path}")

    return rows


def injection_capacity(
    rows: List[Dict[str, float]],
    budget: float,
) -> Dict[str, float]:
    feasible = [
        row
        for row in rows
        if row["clean_drop"] <= budget
    ]

    if not feasible:
        return {
            "capacity": float("nan"),
            "scale": float("nan"),
            "clean_drop": float("nan"),
        }

    best = max(
        feasible,
        key=lambda row: row["excess_asr"],
    )

    return {
        "capacity": best["excess_asr"],
        "scale": best["scale"],
        "clean_drop": best["clean_drop"],
    }


def pareto_points(
    rows: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    candidates = sorted(
        rows,
        key=lambda row: (
            row["clean_drop"],
            -row["excess_asr"],
        ),
    )

    frontier: List[Dict[str, float]] = []
    best_asr = -float("inf")

    for row in candidates:
        if row["excess_asr"] > best_asr:
            frontier.append(row)
            best_asr = row["excess_asr"]

    return frontier


def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
    fieldnames: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    root = (
        args.root
        if args.root.is_absolute()
        else repo_root / args.root
    )

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows: Dict[int, List[Dict[str, float]]] = {}
    frontier_rows: List[Dict[str, object]] = []
    capacity_rows: List[Dict[str, object]] = []

    for round_id in args.rounds:
        path = (
            root
            / f"round_{round_id}"
            / "injection_search.csv"
        )

        rows = read_search_csv(path)
        all_rows[round_id] = rows

        frontier = pareto_points(rows)

        for row in rows:
            frontier_rows.append(
                {
                    "round": round_id,
                    "scale": row["scale"],
                    "excess_asr": row["excess_asr"],
                    "clean_drop": row["clean_drop"],
                    "pareto": any(
                        abs(row["scale"] - p["scale"]) < 1e-12
                        for p in frontier
                    ),
                }
            )

        for budget in args.budgets:
            result = injection_capacity(
                rows,
                budget,
            )

            capacity_rows.append(
                {
                    "round": round_id,
                    "clean_budget": budget,
                    "capacity": result["capacity"],
                    "selected_scale": result["scale"],
                    "selected_clean_drop": result["clean_drop"],
                }
            )

    write_csv(
        output_dir / "frontier_points.csv",
        frontier_rows,
        [
            "round",
            "scale",
            "excess_asr",
            "clean_drop",
            "pareto",
        ],
    )

    write_csv(
        output_dir / "injection_capacity.csv",
        capacity_rows,
        [
            "round",
            "clean_budget",
            "capacity",
            "selected_scale",
            "selected_clean_drop",
        ],
    )

    with (
        output_dir
        / "injection_capacity.json"
    ).open("w") as f:
        json.dump(
            capacity_rows,
            f,
            indent=2,
        )

    plt.figure(figsize=(7, 5))

    for round_id in args.rounds:
        rows = all_rows[round_id]

        x = [
            row["clean_drop"]
            for row in rows
        ]

        y = [
            row["excess_asr"]
            for row in rows
        ]

        plt.plot(
            x,
            y,
            marker="o",
            label=f"round {round_id}",
        )

    plt.xlabel("Clean accuracy drop")
    plt.ylabel("Excess ASR")
    plt.title("Temporal Injection Frontier")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_dir / "injection_frontier.png",
        dpi=180,
    )

    plt.close()

    print("\nInjection Capacity")

    for budget in args.budgets:
        print(f"\nclean budget = {budget:.4f}")

        for round_id in args.rounds:
            row = next(
                item
                for item in capacity_rows
                if item["round"] == round_id
                and abs(item["clean_budget"] - budget) < 1e-12
            )

            print(
                f"round={round_id:4d} | "
                f"capacity={row['capacity']} | "
                f"scale={row['selected_scale']} | "
                f"clean_drop={row['selected_clean_drop']}"
            )

    print(
        f"\nSaved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
```

---

# 6. 运行 Injection Frontier 分析

在 BackFed 根目录：

```bash
python experiments/analyze_temporal_injection_frontier.py \
  --rounds 50 150 250 \
  --budgets 0.01 0.02 0.05 0.10
```

输出：

```text
outputs/temporal_injection_frontier/
├── frontier_points.csv
├── injection_capacity.csv
├── injection_capacity.json
└── injection_frontier.png
```

---

# 7. 第一阶段判定

Agent 先只分析：

```text
Injection Capacity
```

不要立即跑 retention。

重点输出：

```text
Round    J(1%)    J(2%)    J(5%)    J(10%)
50
150
250
```

同时输出每个 budget 下：

```text
max J_t - min J_t
```

以及：

```text
max J_t / min positive J_t
```

---

# 8. Injection Frontier 的 GO / STOP

## GO

满足下面任一情况即可进入 matched-retention：

### 条件 A

在：

```text
δ = 5%
```

下：

```text
max_t J_t(δ) - min_t J_t(δ) >= 0.15
```

即至少：

```text
15 percentage points
```

### 条件 B

在至少两个 budget：

```text
δ ∈ {1%,2%,5%,10%}
```

下：

```text
max_t J_t(δ)
/
min_positive_t J_t(δ)
>= 1.5
```

### 条件 C

某个 round 的 Pareto frontier 在多个 budget 下明显支配另一个 round。

例如：

```text
round 250
```

在相同 clean-damage budget 下长期都能获得更高 excess ASR。

如果满足：

```text
GO
```

继续第 9 节。

---

## STOP

如果：

```text
J_50(δ)
J_150(δ)
J_250(δ)
```

在：

```text
δ ∈ {1%,2%,5%,10%}
```

下整体差异很小，

且：

```text
max difference < 10pp
```

同时没有明显 frontier dominance：

```text
STOP
```

不要继续 retention。

---

# 9. Matched-Effect Retention

只有 Injection Frontier 判定：

```text
GO
```

才执行。

目标：

```text
让 round 50 / 150 / 250
具有尽量接近的 immediate excess ASR，
然后比较未来 30 rounds 的 retention。
```

不要再强行指定：

```text
30pp
```

而是根据三个 round 现有 injection frontier，自动选一个共同可达目标。

---

# 10. Common Target 选择规则

先从现有 search CSV 中找到每个 round 在：

```text
clean drop <= 10%
```

约束下的最大 excess ASR：

```text
M_50
M_150
M_250
```

定义：

```text
q_max = min(M_50, M_150, M_250)
```

然后：

```text
q = min(0.20, 0.8 * q_max)
```

要求：

```text
q >= 0.10
```

如果：

```text
q < 0.10
```

则：

```text
STOP
```

说明三个 round 没有足够共同可达的 backdoor-effect 区间，无法进行公平 retention comparison。

---

# 11. 新增 Matched-Retention 脚本

新建：

```text
BackFed/experiments/run_matched_retention_probe.py
```

这个脚本允许复用上一轮：

```text
temporal_cost_retention_probe.py
```

中的函数，不修改 BackFed 核心代码。

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
from typing import Dict, List, Optional

from temporal_cost_retention_probe import (
    launch_run,
    read_metrics,
    index_metrics,
    check_preattack_match,
    compute_retention,
)


DEFAULT_ROUNDS = [50, 150, 250]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run matched-effect retention across different FL communication rounds."
    )

    p.add_argument(
        "--attack-rounds",
        type=int,
        nargs="+",
        default=DEFAULT_ROUNDS,
    )

    p.add_argument(
        "--horizon",
        type=int,
        default=30,
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
        "--source-root",
        type=Path,
        default=Path("outputs/temporal_cost_retention_probe"),
    )

    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/matched_retention_probe"),
    )

    p.add_argument(
        "--force",
        action="store_true",
    )

    return p.parse_args()


def as_float(x: object) -> Optional[float]:
    if x is None:
        return None

    try:
        y = float(str(x).strip())
    except (TypeError, ValueError):
        return None

    if not math.isfinite(y):
        return None

    return y


def read_search(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            scale = as_float(row.get("scale"))
            excess = as_float(row.get("excess_asr"))
            clean_drop = as_float(row.get("clean_drop"))

            if scale is None or excess is None or clean_drop is None:
                continue

            rows.append(
                {
                    "scale": scale,
                    "excess_asr": excess,
                    "clean_drop": clean_drop,
                }
            )

    rows.sort(
        key=lambda item: item["scale"]
    )

    return rows


def select_common_target(
    all_rows: Dict[int, List[Dict[str, float]]],
) -> float:
    maxima = []

    for round_id, rows in all_rows.items():
        feasible = [
            row["excess_asr"]
            for row in rows
            if row["clean_drop"] <= 0.10
        ]

        if not feasible:
            raise RuntimeError(
                f"No point with clean_drop <= 10% for round {round_id}"
            )

        maxima.append(
            max(feasible)
        )

    q_max = min(maxima)

    q = min(
        0.20,
        0.8 * q_max,
    )

    if q < 0.10:
        raise RuntimeError(
            f"Common matched-effect target too weak: q={q:.6f} < 0.10"
        )

    return q


def choose_scale(
    rows: List[Dict[str, float]],
    target: float,
) -> Dict[str, float]:
    candidates = [
        row
        for row in rows
        if row["clean_drop"] <= 0.10
    ]

    if not candidates:
        raise RuntimeError(
            "No candidate under 10% clean-drop budget."
        )

    return min(
        candidates,
        key=lambda row: (
            abs(row["excess_asr"] - target),
            row["clean_drop"],
        ),
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

    with path.open("w", newline="") as f:
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

    source_root = (
        args.source_root
        if args.source_root.is_absolute()
        else repo_root / args.source_root
    )

    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else repo_root / args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows: Dict[int, List[Dict[str, float]]] = {}

    for t in args.attack_rounds:
        path = (
            source_root
            / f"round_{t}"
            / "injection_search.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        all_rows[t] = read_search(path)

    target = select_common_target(
        all_rows
    )

    selected: Dict[int, Dict[str, float]] = {}

    for t in args.attack_rounds:
        selected[t] = choose_scale(
            all_rows[t],
            target,
        )

    print(
        f"Common matched-effect target q={target:.6f}"
    )

    for t in args.attack_rounds:
        row = selected[t]

        print(
            f"round={t} | "
            f"scale={row['scale']} | "
            f"existing excess={row['excess_asr']:.6f} | "
            f"existing clean_drop={row['clean_drop']:.6f}"
        )

    max_round = (
        max(args.attack_rounds)
        + args.horizon
    )

    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONHASHSEED"] = str(args.seed)

    clean_csv = launch_run(
        repo_root=repo_root,
        env=env,
        run_dir=(
            output_root
            / "clean_trigger_baseline"
        ),
        seed=args.seed,
        gpu=args.gpu,
        training_mode=args.training_mode,
        num_rounds=max_round,
        attack_round=max_round + 1000,
        scale_poison=False,
        scale_factor=1.0,
        force=args.force,
    )

    clean_idx = index_metrics(
        read_metrics(clean_csv)
    )

    summary_rows: List[Dict[str, object]] = []

    for t in args.attack_rounds:
        scale = selected[t]["scale"]

        run_dir = (
            output_root
            / f"round_{t}"
            / f"scale_{str(scale).replace('.', 'p')}"
        )

        metrics_csv = launch_run(
            repo_root=repo_root,
            env=env,
            run_dir=run_dir,
            seed=args.seed,
            gpu=args.gpu,
            training_mode=args.training_mode,
            num_rounds=t + args.horizon,
            attack_round=t,
            scale_poison=True,
            scale_factor=scale,
            force=args.force,
        )

        attack_idx = index_metrics(
            read_metrics(metrics_csv)
        )

        check_preattack_match(
            clean_idx,
            attack_idx,
            t,
        )

        retention_summary, curve = compute_retention(
            clean_idx=clean_idx,
            attack_idx=attack_idx,
            attack_round=t,
            horizon=args.horizon,
        )

        curve_path = (
            output_root
            / f"round_{t}"
            / "retention_curve.csv"
        )

        write_csv(
            curve_path,
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

        summary_rows.append(
            {
                "attack_round": t,
                "target_excess_asr": target,
                "selected_scale": scale,
                "existing_search_excess_asr": selected[t]["excess_asr"],
                "existing_search_clean_drop": selected[t]["clean_drop"],
                **retention_summary,
            }
        )

    write_csv(
        output_root
        / "matched_retention_summary.csv",
        summary_rows,
        [
            "attack_round",
            "target_excess_asr",
            "selected_scale",
            "existing_search_excess_asr",
            "existing_search_clean_drop",
            "immediate_excess_asr",
            "retention_auc",
            "half_life",
            "end_excess_asr",
            "peak_excess_asr",
        ],
    )

    with (
        output_root
        / "matched_retention_summary.json"
    ).open("w") as f:
        json.dump(
            summary_rows,
            f,
            indent=2,
        )

    print("\nMatched Retention Summary")

    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
```

---

# 12. 运行 Matched Retention

只有 Injection Frontier 判定 GO 后：

```bash
python experiments/run_matched_retention_probe.py \
  --attack-rounds 50 150 250 \
  --horizon 30 \
  --seed 2026 \
  --gpu 0 \
  --training-mode sequential \
  --force
```

---

# 13. Retention 判定

回报：

```text
Round
Immediate Excess ASR
Selected Scale
Retention AUC
Half-life
End Excess ASR
```

首先检查 immediate matching 是否成功：

```text
max immediate excess ASR
-
min immediate excess ASR
<= 0.05
```

即三组 immediate effect 最好在：

```text
5 percentage points
```

以内。

如果超过：

```text
5pp
```

不要直接解释 retention。

标记：

```text
MATCHING FAILED
```

并停止。

---

# 14. Retention GO / STOP

## GO

满足任一：

```text
max(R_t) - min(R_t) >= 0.20
```

或者：

```text
max half-life - min half-life >= 8 rounds
```

或者：

```text
某个中间 round 的 retention 明显优于 early 和 late
```

则：

```text
GO
```

---

## STOP

如果：

```text
max(R_t) - min(R_t) < 0.10
```

且：

```text
half-life 基本相同
```

则：

```text
STOP
```

---

# 15. 最终 Agent 汇报格式

只汇报下面内容。

## 15.1 Injection Frontier

```text
Round    J(1%)    J(2%)    J(5%)    J(10%)
50
150
250
```

再给：

```text
Frontier verdict:
GO / STOP
```

如果 STOP：

到此结束。

## 15.2 Matched Retention

仅 Frontier GO 时给：

```text
Common target q:

Round    Scale    Immediate Excess ASR    R_t    Half-life    End Excess ASR
50
150
250
```

## 15.3 最终判定

只选：

```text
STRONG GO
WEAK
STOP
```

解释最多 5 句话。

---

# 16. 本轮禁止

不要：

```text
设计 scheduler
设计 Temporal Vulnerability Score
加入 A3FL
加入 Neurotoxin
加入 defense
增加 malicious clients
换 trigger
换 dataset
换 model
调 poison ratio
训练 predictor
增加新的 loss
```

本轮唯一目标：

```text
先确认 communication round 是否真的具有不同的
Injection Capacity 和 matched-effect Retention。
```

如果没有：

```text
when-to-inject Idea 停止。
```

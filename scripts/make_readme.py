"""Build README.md result tables straight from the run artefacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/runs"

ORDER = [
    ("resnet18_frozen_noaug", "E1", "ResNet-18 · 冻结", "否"),
    ("resnet18_frozen_aug", "E2", "ResNet-18 · 冻结", "是"),
    ("resnet18_last_stage_noaug", "E3", "ResNet-18 · 解冻最后阶段", "否"),
    ("resnet18_last_stage_aug", "E4", "ResNet-18 · 解冻最后阶段", "是"),
    ("mobilenet_v3_small_last_stage_aug", "E5", "MobileNetV3-Small · 解冻最后阶段", "是"),
    ("resnet18_scratch_noaug", "E6", "ResNet-18 · 随机初始化（全网络）", "否"),
]


def load(run: str) -> dict:
    base = RUNS / run
    res = json.loads((base / "results.json").read_text(encoding="utf-8"))
    with open(base / "history.csv", encoding="utf-8") as fh:
        hist = list(csv.DictReader(fh))
    res["train_min"] = sum(float(r["epoch_time_s"]) for r in hist) / 60.0
    res["epoch_s"] = res["train_min"] * 60.0 / max(len(hist), 1)
    return res


def mixed_table() -> str:
    """Markdown table for the multi-label plate model (section 5.3)."""
    path = ROOT / "results/tables/mixed.json"
    if not path.exists():
        return "（尚未训练：先运行 `python scripts/train_mixed.py`）"
    r = json.loads(path.read_text(encoding="utf-8"))
    v = r.get("val_tuned", r["val_plates"])
    c = r.get("canteen_tuned", r["canteen_plates"])
    s = r["test_single"]
    rows = [
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 全局检测阈值（验证盘扫描） | {r['threshold']:.2f} |",
        "| 验证盘 · 每部分 P / R / F1（阈值在同一盘上拟合，偏乐观） | {:.1f}% / {:.1f}% / {:.1f}% |".format(
            100 * v["precision"], 100 * v["recall"], 100 * v["f1"]),
        f"| 验证盘 · 整盘完全正确 | {100 * v['exact_match']:.1f}% |",
        "| 食堂代理盘 · 同一套阈值 P / R / F1 | {:.1f}% / {:.1f}% / {:.1f}% |".format(
            100 * c["precision"], 100 * c["recall"], 100 * c["f1"]),
        f"| 单菜测试集 Top-1（同一网络） | {100 * s['top1']:.2f}% |",
        f"| 单菜照片平均预测部分数 | {s['parts_per_photo']:.2f} |",
        "| 低于 10% 的分数 | 不显示 |",
    ]
    table = r.get("class_thresholds") or {}
    if table:
        lo, hi = min(table.values()), max(table.values())
        rows.insert(3, "| 每类阈值范围（20 类各自校准） | {:.2f} – {:.2f} |".format(lo, hi))
    return "\n".join(rows)


def main() -> None:
    comp_path = ROOT / "results/tables/complexity.json"
    comp = {r["run"]: r for r in json.loads(comp_path.read_text(encoding="utf-8"))} if comp_path.exists() else {}

    rows = ["| ID | 模型 / 策略 | 颜色增强 | Test Acc | Test Macro-F1 | Proxy Acc | Cross-domain Gap | Params (M) | MACs (G) | 训练时长 (min) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    lat = ["| ID | 模型 | GPU 单张 (ms) | CPU 单张 (ms) | GPU FPS |", "| --- | --- | --- | --- | --- |"]
    for run, tag, name, aug in ORDER:
        if not (RUNS / run / "results.json").exists():
            continue
        r = load(run)
        c = comp.get(run, {})
        rows.append(
            "| {} | {} | {} | {:.2f}% | {:.2f}% | {:.2f}% | {:.1f} pt | {} | {} | {:.1f} |".format(
                tag, name, aug,
                100 * r["test"]["accuracy"], 100 * r["test"]["macro_f1"],
                100 * r["canteen"]["accuracy"],
                100 * (r["test"]["accuracy"] - r["canteen"]["accuracy"]),
                f"{c['params_m']:.2f}" if c else (f"{r['total_params'] / 1e6:.2f}" if r.get("total_params") else "-"),
                f"{c['macs_g']:.2f}" if c else "-",
                r["train_min"],
            )
        )
        if c:
            lat.append("| {} | {} | {:.2f} | {:.2f} | {:.0f} |".format(tag, name, c["gpu_latency_ms_mean"], c["cpu_latency_ms_mean"], c["gpu_fps"]))

    tpl = (ROOT / "scripts/README_template.md").read_text(encoding="utf-8")
    out = (tpl.replace("<!--RESULTS-->", "\n".join(rows))
              .replace("<!--LATENCY-->", "\n".join(lat))
              .replace("<!--MIXED-->", mixed_table()))
    (ROOT / "README.md").write_text(out, encoding="utf-8")
    print("wrote README.md")


if __name__ == "__main__":
    main()

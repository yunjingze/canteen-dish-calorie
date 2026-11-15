"""Bar chart: 20-class course model vs 116-class model on the same data.

Both pairs use the same 20-class label space, so the only thing that changes is
how much data the backbone saw (12,000 vs 70,283 training images).

Left pair  - official 20-class test split (5,000 images).
Right pair - 1,000 degraded canteen photos (rotation/blur/JPEG proxy domain).
The 116-class model is re-normalised over the 20 course classes ("restricted"),
which is the fair comparison; its raw 116-way accuracy is in wide_domain.json.

Reads results/runs/resnet18_last_stage_aug/results.json (E4) and
results/tables/wide_domain.json, writes results/figures/wide_scale.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E4 = ROOT / "results/runs/resnet18_last_stage_aug/results.json"
WIDE = ROOT / "results/tables/wide_domain.json"
OUT = ROOT / "results/figures/wide_scale.png"

for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "figure.dpi": 160,
})


def main():
    e4 = json.loads(E4.read_text())
    wide = json.loads(WIDE.read_text())
    narrow = [e4["test"]["accuracy"] * 100, e4["canteen"]["accuracy"] * 100]
    broad = [wide["test"]["top1_restricted"] * 100, wide["canteen_proxy"]["top1_restricted"] * 100]

    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    xs = [0, 1]
    w = 0.34
    b1 = ax.bar([x - w / 2 for x in xs], narrow, w, color="#2f6df6", label="20 类模型（课程）")
    b2 = ax.bar([x + w / 2 for x in xs], broad, w, color="#f59e0b", label="116 类模型（本次扩展）")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["官方 test（5,000 张）", "食堂退化照片（1,000 张）"])
    ax.set_ylabel("Top-1 准确率 (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("同一套 20 类评分：5.8 倍数据带来的提升", fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()

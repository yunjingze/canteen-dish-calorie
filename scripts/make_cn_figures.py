# -*- coding: utf-8 -*-
"""Figures for the canteen (CNFOOD-241) part of the paper.

Two artefacts, both drawn from recorded numbers rather than hand-typed ones:

* cn21_confusion.png  -- row-normalised 21-way confusion matrix of the deployed
  canteen head on data/splits/cn241_21/test (from cn241_21_confusion.json).
* cn_gate.png         -- acceptance rate of the deployed open-set gate on the six
  evaluation sets, with the Chinese-domain probe switched off and on
  (from domain_gate_pre.json).

Run: python scripts/make_cn_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"
TABLES = ROOT / "results/tables"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})

# Short axis labels; the full names live in results/tables/cn241_21_classes.json.
SHORT = {
    "hot_sour_potato": "potato silk",
    "di_san_xian": "di san xian",
    "hand_torn_cabbage": "cabbage",
    "stir_fried_greens": "greens",
    "dry_fried_beans": "green beans",
    "tomato_egg": "tomato egg",
    "steamed_egg": "steamed egg",
    "sweet_sour_ribs": "sweet-sour ribs",
    "cola_wings": "cola wings",
    "boiled_chicken": "poached chicken",
    "braised_chicken": "braised chicken",
    "spicy_chicken": "spicy chicken",
    "braised_pork": "braised pork",
    "boiled_pork": "chilli pork",
    "sweet_sour_pork": "sweet-sour pork",
    "farmhouse_pork": "farmhouse pork",
    "pepper_pork": "pepper pork",
    "mu_xi_pork": "mu xi pork",
    "baozi": "baozi",
    "fried_noodles": "fried noodles",
    "other_cn": "other (175)",
}


def fig_cn_confusion(fname: str = "cn21_confusion.png") -> None:
    res = json.loads((TABLES / "cn241_21_confusion.json").read_text("utf-8"))
    names = res["classes"]
    cm = np.asarray(res["confusion"], dtype=float)
    rate = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    labels = [SHORT.get(n, n) for n in names]
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    im = ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6.4)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.4)
    ax.set_xlabel("Predicted", fontsize=8.5)
    ax.set_ylabel("True", fontsize=8.5)
    ax.set_title(f"Canteen head on its test split (n={res['n']}, "
                 f"top-1 {res['top1']*100:.1f}%, top-5 {res['top5']*100:.1f}%)",
                 fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if rate[i, j] >= 0.01:
                ax.text(j, i, f"{rate[i, j]*100:.0f}", ha="center", va="center",
                        fontsize=4.2,
                        color="white" if rate[i, j] > 0.55 else "#243b6b")
    # the off-diagonal block that matters: "other" leaking into the 20 dishes
    ax.add_patch(plt.Rectangle((0, 20), 20, 1, fill=False, ec="#e0562a", lw=1.1))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Row-normalised rate", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)
    print("wrote", (FIG / fname).relative_to(ROOT))


def fig_gate(fname: str = "cn_gate.png") -> None:
    d = json.loads((TABLES / "domain_gate_pre.json").read_text("utf-8"))
    order = ["val", "in", "out_cn", "out_ood", "out_other", "real"]
    pretty = {
        "val": "canteen val",
        "in": "canteen test\n(20 dishes)",
        "out_cn": "other Chinese\n(15 dishes)",
        "out_ood": "non-Chinese\nfood",
        "out_other": "non-food\n/ clutter",
        "real": "real photos\n(16 dishes)",
    }
    off = [d["sets"][k]["accept_domain_off"] * 100 for k in order]
    on = [d["sets"][k]["accept_domain_on"] * 100 for k in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(x - 0.2, off, 0.38, color="#9db4d8", label="without domain probe")
    ax.bar(x + 0.2, on, 0.38, color="#2f6df6", label="deployed (domain probe on)")
    for i, (a, b) in enumerate(zip(off, on)):
        ax.text(i - 0.2, a + 1.6, f"{a:.1f}", ha="center", fontsize=6.8)
        ax.text(i + 0.2, b + 1.6, f"{b:.1f}", ha="center", fontsize=6.8)
    ax.axhline(50, color="#b9c0cc", lw=0.8, ls=":")
    ax.set_xticks(x, [pretty[k] for k in order], fontsize=7.0)
    ax.set_ylim(0, 104)
    ax.set_ylabel("Accepted (%)")
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper right")
    ax.set_title(f"Open-set gate at threshold {d['domain_threshold']:.4f}: "
                 "what the system agrees to name", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)
    print("wrote", (FIG / fname).relative_to(ROOT))


if __name__ == "__main__":
    fig_cn_confusion()
    fig_gate()

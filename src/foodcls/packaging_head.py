"""Is this a product package rather than a plate of food?

A frozen-pizza box prints a photograph of a pizza, so the 20-class head answers
"pizza, 98%" and the food gate answers "yes, food".  The numpy heuristic in
packaging.py (long straight edges, low texture) found 0 of 62 supermarket pack
shots, so the cue now comes from a linear head over the same pooled features the
food gate uses -- no second network, no extra forward pass.

The weights are fitted offline (scripts in results/_pkg_*.py) on 79 packaging
photos -- 17 catalogue negatives plus 62 archived v3 product shots -- against
3700 food photos.  Held out 5-fold over the 62: AUC 0.994, 79% recall at the
shipped operating point, 0/100 false flags on the real canteen photos.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MESSAGES = (
    "这张照片看起来是商品包装或食品标签（比如包装盒正面印着的菜），不是一个餐盘。"
    "包装照不在识别范围里：模型认出来的是包装上印的图，不是真的这道菜，所以不给出品类和热量。"
    "把包装拆开、把菜盛到盘子里再拍一张。"
)


class PackagingHead:
    """Logistic head over the whole-frame pooled feature."""

    def __init__(self, path: str | Path, threshold: float = 0.80):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.mean = np.asarray(data["mean"], dtype=np.float64)
        self.scale = np.asarray(data["scale"], dtype=np.float64)
        self.w = np.asarray(data["w"], dtype=np.float64)
        self.b = float(data["b"])
        # the operating point travels with the weights for the same reason the
        # food gate keeps its threshold in the file: a refit must not silently
        # inherit the old threshold
        self.threshold = float(data.get("threshold", threshold))
        self.view = str(data.get("view", "whole"))

    def prob(self, feats) -> float:
        f = np.asarray(feats, dtype=np.float64)
        if f.ndim == 2:
            # three framings: the whole-frame view is the one these weights were
            # fitted on (a package fills the frame; a crop can cut the label)
            f = f[1] if f.shape[0] > 1 else f[0]
        z = float(((f - self.mean) / self.scale) @ self.w + self.b)
        return float(1.0 / (1.0 + np.exp(-z)))


def message() -> str:
    return MESSAGES

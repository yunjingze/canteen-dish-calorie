"""Food / non-food gate: a linear head on the pooled features of the 20-class backbone.

The view-robust rule answers "is this one of the supported dishes?"; it says
nothing about whether the frame contains food at all, so a menu board or a cup
scores like a mediocre dish and slips through.  This head is trained on 131 real
non-food photos (menus, trays, cups, napkins, bins, packaging, hands) against
Food-101 and canteen-proxy food photos, and is scored off the same forward pass
as the classifier: no extra network, 512 weights.

Seven of the first batch of "negatives" turned out to be photographs of food
(bread rolls, salad plates, food cups, a nugget tray); they are now cross-domain
positives, which lifted the held-out AUC from 0.93 to 0.95.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FoodGate:
    """Logistic regression on the backbone's pooled 512-d feature."""

    def __init__(self, path: str | Path, threshold: float = 0.20):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.w = np.asarray(data["w"], dtype=np.float64)
        self.b = float(data["b"])
        self.mean = np.asarray(data["mean"], dtype=np.float64)
        self.scale = np.asarray(data["scale"], dtype=np.float64)
        # the threshold shipped with the weights wins, so a re-trained gate does
        # not silently keep the old operating point
        self.threshold = float(data.get("threshold", threshold))

    def prob(self, feats) -> float:
        """P(this frame contains a dish we could name)."""
        z = (np.asarray(feats, dtype=np.float64).reshape(-1) - self.mean) / self.scale
        return float(1.0 / (1.0 + np.exp(-(float(z @ self.w) + self.b))))

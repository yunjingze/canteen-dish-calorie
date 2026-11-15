"""Pick the out-of-scope threshold for the wide head.

The 20-class gate (scripts/eval_openset.py) could be calibrated against 20 *unseen*
Food-101 classes.  For the wide head every Food-101 class is in scope, so we have no
labelled "unknown dish" set.  What we can measure is the other half of the tradeoff:
how often a genuine in-scope photo gets flagged.  We therefore score a stratified
sample of the official test split with the demo's own three views and put the
threshold at the 5th percentile of the worst-view top-1 probability, i.e. accept a
~5% false-alarm rate - the same policy used for the 20-class gate.

Usage:
    python scripts/eval_wide_gate.py [--n-per-class 15]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.app import preprocess_views  # noqa: E402
from foodcls.classes_wide import WIDE_CLASS_NAMES, WIDE_NUM_CLASSES  # noqa: E402
from foodcls.models import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/runs/resnet18_food101_cn116/best.pt")
    ap.add_argument("--n-per-class", type=int, default=15)
    ap.add_argument("--out", default="results/tables/wide_gate.json")
    args = ap.parse_args()

    ckpt_path = ROOT / args.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt.get("arch", "resnet18"), len(ckpt.get("class_names") or WIDE_CLASS_NAMES))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    calib = ckpt_path.parent / "calibration.json"
    temperature = json.loads(calib.read_text())["temperature"] if calib.is_file() else 1.0

    rng = random.Random(1234)
    test_root = ROOT / "data/wide/splits/test"
    from PIL import Image

    scores, top1s = [], []
    for cls in WIDE_CLASS_NAMES:
        files = sorted((test_root / cls).glob("*.jpg"))
        for p in rng.sample(files, min(args.n_per_class, len(files))):
            img = Image.open(p).convert("RGB")
            with torch.no_grad():
                logits = model(preprocess_views(img).to(device)).float() / max(temperature, 1e-3)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            scores.append(float(min(np.max(v) for v in probs)))
            top1s.append(float(np.max(probs[0])))

    scores = np.array(scores)
    q = {f"p{int(p*1000)/10:g}": round(float(np.percentile(scores, p)), 4) for p in (1, 2.5, 5, 10, 25, 50)}
    flag = {f"{t:.2f}": round(float((scores < t).mean()) * 100, 2) for t in (0.2, 0.25, 0.3, 0.35, 0.4)}
    out = {
        "n": int(scores.size),
        "n_per_class": args.n_per_class,
        "temperature": temperature,
        "worst_view_percentiles": q,
        "flagged_percent_by_threshold": flag,
        "suggested_threshold": round(float(np.percentile(scores, 5)), 2),
        "mean_worst_view": round(float(scores.mean()), 4),
        "mean_top1": round(float(np.mean(top1s)), 4),
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"wrote {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()

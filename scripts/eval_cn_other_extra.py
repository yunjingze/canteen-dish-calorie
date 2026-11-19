# -*- coding: utf-8 -*-
"""Second pass over the same evaluation sets: per-image 20-class evidence.

eval_cn_other.py only stores the argmax over all 21 classes and the bucket
probability, which is enough to sweep a one-dimensional threshold but not enough
to ask "was the model also confident about one of the 20 dishes?".  This pass
re-runs the canteen head alone and stores, per image:

  p20      fused probability of the best of the 20 dishes
  top20    that dish
  p_other  fused probability of the "other Chinese" bucket
  rob20    worst-view gate score when the bucket is ignored
  rob20r   same score after re-normalising the 20 dishes (1 - p_other)
  rob21    the gate score used by the shipped rule (max over all 21 classes)

Writes results/tables/_extra_<set>.npz, aligned row-for-row with _other_<set>.npz.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A  # noqa: E402
from foodcls.models import build_model  # noqa: E402

CN_CKPT = Path(os.environ.get("EVAL_CN_CKPT", ROOT / "results/runs/cn241_21/best.pt"))
TAG = os.environ.get("EVAL_TAG", "")
OTHER = len(A.CN_CLASS_NAMES)
CN_LIKE = ["mapo_tofu", "kung_pao_chicken", "twice_cooked_pork", "sweet_and_sour_pork",
           "boiled_fish_sichuan", "fish_with_pickled_cabbage", "west_lake_vinegar_fish",
           "steamed_sea_bass", "peking_duck", "soup_dumplings", "dumplings", "spring_rolls",
           "fried_rice", "ramen", "braised_pork_meatballs"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(CN_CKPT, map_location="cpu", weights_only=False)
names = list(ck["class_names"])
assert len(names) == OTHER + 1 and names[-1] == A.CN_OTHER_SLUG, names
model = build_model(ck["arch"], len(names), pretrained=False)
model.load_state_dict(ck["model"])
model.to(device).eval()
T = float(json.loads((CN_CKPT.parent / "calibration.json").read_text("utf-8"))["temperature"])
print(f"model {ck['arch']} T={T:.3f} classes={len(names)}", flush=True)


@torch.no_grad()
def vectors(paths, batch=16):
    n = len(paths)
    out = {k: np.zeros(n, dtype=np.float32) for k in
           ("p20", "p_other", "rob20", "rob20r", "rob21")}
    top20 = np.zeros(n, dtype=np.int32)
    for i in range(0, n, batch):
        chunk = paths[i:i + batch]
        x = torch.cat([A.preprocess_views(Image.open(p)) for p in chunk]).to(device)
        logits = model(x).float().view(len(chunk), 6, -1)
        pv = torch.softmax(logits / max(T, 1e-3), -1)
        fused = torch.softmax(logits.mean(1) / max(T, 1e-3), -1).cpu().numpy()
        pv = pv.cpu().numpy()
        sl = slice(i, i + len(chunk))
        out["p20"][sl] = fused[:, :OTHER].max(1)
        top20[sl] = fused[:, :OTHER].argmax(1)
        out["p_other"][sl] = fused[:, OTHER]
        gate = pv[:, :3, :]
        out["rob21"][sl] = gate.max(2).min(1)
        out["rob20"][sl] = gate[:, :, :OTHER].max(2).min(1)
        denom = np.clip(1.0 - gate[:, :, OTHER], 1e-6, None)
        out["rob20r"][sl] = (gate[:, :, :OTHER] / denom[:, :, None]).max(2).min(1)
        if (i // batch) % 25 == 0:
            print(f"  {i + len(chunk)}/{n}", flush=True)
    out["top20"] = top20
    return out


def wide_test_glob(cls, k):
    return sorted((ROOT / "data/wide/splits/test" / cls).glob("*.jpg"))[:k]


sets = {}
sets["in"] = [p for c in A.CN_CLASS_NAMES
              for p in sorted((ROOT / "data/splits/cn241_full/test" / c).glob("*.jpg"))]
sets["out_cn"] = [p for c in CN_LIKE for p in wide_test_glob(c, 24)]
ood_manifest = json.loads((ROOT / "results/tables/cn241_other_heldout.json").read_text("utf-8"))
sets["out_ood"] = [ROOT / it["path"] for it in ood_manifest["items"]]
sets["out_other"] = [p for d in sorted((ROOT / "data/wide/splits/test").iterdir())
                     if d.is_dir() and d.name not in CN_LIKE for p in sorted(d.glob("*.jpg"))[:8]]
sets["real"] = sorted((ROOT / "data/splits/canteen_real").rglob("*.jpg"))
val_root = ROOT / "data/splits/cn241_21/val"
sets["val"] = [p for c in A.CN_CLASS_NAMES for p in sorted((val_root / c).glob("*.jpg"))]

for tag, paths in sets.items():
    print(f"[{tag}] n={len(paths)}", flush=True)
    t0 = time.time()
    out = vectors(paths)
    np.savez(ROOT / f"results/tables/_extra{TAG}_{tag}.npz", **out)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

print(f"wrote results/tables/_extra{TAG}_<set>.npz")

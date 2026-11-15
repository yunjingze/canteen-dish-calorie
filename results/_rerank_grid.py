"""Does the 20-class model rescue the 116-class model's top-1?

The demo's "all cuisines" tab uses the 116-class model, which scores 89.0% on
the 100 real canteen photos against 94.0% for the 20-class model, and 10 of its
11 mistakes have the right dish in its own top-5.  This script measures whether
overriding the wide prediction with the 20-class one -- only when the wide
top-1 is a Food-101 class, the 20-class answer is inside the wide top-k and the
20-class posterior is high enough -- helps on the real photos and, crucially,
does not hurt the 96 Chinese classes.

    python results/_rerank_grid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from foodcls.app import preprocess_views  # noqa: E402
from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.classes_wide import WIDE_CLASS_NAMES, WIDE_NUM_CLASSES  # noqa: E402
from foodcls.models import build_model  # noqa: E402

E7 = ROOT / "results/runs/resnet18_last_stage_aug_f101n/best.pt"
WIDE = ROOT / "results/runs/resnet18_food101_cn116/best.pt"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FOOD101 = set(CLASS_NAMES)
WIDE_IDX = {c: i for i, c in enumerate(WIDE_CLASS_NAMES)}
SUB = torch.tensor([WIDE_IDX[c] for c in CLASS_NAMES])


def load(path: Path, n_classes: int, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("arch", "resnet18"), n_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    t = 1.0
    calib = path.parent / "calibration.json"
    if calib.is_file():
        t = float(json.loads(calib.read_text())["temperature"])
    return model, t


@torch.no_grad()
def logits_of(model, paths, device, batch=16):
    out = []
    for s in range(0, len(paths), batch):
        chunk = [preprocess_views(Image.open(p).convert("RGB")) for p in paths[s:s + batch]]
        x = torch.stack(chunk).to(device)
        b, v, c, h, w = x.shape
        lg = model(x.view(b * v, c, h, w)).float().view(b, v, -1).mean(1)
        out.append(lg.cpu())
    return torch.cat(out)


def collect(root: Path, only_food101: bool = False):
    paths, labels = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name not in WIDE_IDX:
            continue
        if only_food101 and folder.name not in FOOD101:
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in EXTS:
                paths.append(p)
                labels.append(WIDE_IDX[folder.name])
    return paths, labels


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m20, t20 = load(E7, 20, device)
    m116, t116 = load(WIDE, WIDE_NUM_CLASSES, device)
    print(f"temperatures: 20-class {t20:.3f}, 116-class {t116:.3f}", flush=True)

    sets = {}
    sets["canteen_real"] = collect(ROOT / "data/splits/canteen_real")
    sets["canteen_proxy"] = collect(ROOT / "data/splits/canteen")
    sets["wide_food101"] = collect(ROOT / "data/wide/splits/test", only_food101=True)

    report = {}
    for name, (paths, labels) in sets.items():
        if not paths:
            print(f"skip {name}: no images", flush=True)
            continue
        lp = logits_of(m20, paths, device)
        lw = logits_of(m116, paths, device)
        p20 = torch.softmax(lp / t20, 1).numpy()
        p116 = torch.softmax(lw / t116, 1).numpy()
        y = np.array(labels)
        # the wide model scored against the 20-class subset, so the 20 labels map onto it
        y_wide = np.array([WIDE_IDX[WIDE_CLASS_NAMES[i]] for i in y])
        top1w = p116.argmax(1)
        pred20 = p20.argmax(1)
        conf20 = p20.max(1)
        order = np.argsort(-p116, axis=1)
        rank_of = {k: np.array([list(order[i][:k]).index(WIDE_IDX[CLASS_NAMES[pred20[i]]])
                                if WIDE_IDX[CLASS_NAMES[pred20[i]]] in list(order[i][:k]) else 99
                                for i in range(len(y))]) for k in (3, 5)}
        base = float((top1w == y_wide).mean())
        rows = {"n": len(y), "base_top1": round(base, 4), "grid": []}
        for k in (3, 5):
            for tau in (0.4, 0.5, 0.6, 0.7, 0.8):
                pred = top1w.copy()
                fire = (np.array([WIDE_CLASS_NAMES[p] in FOOD101 for p in top1w])
                        & (rank_of[k] < k) & (conf20 >= tau))
                pred[fire] = [WIDE_IDX[CLASS_NAMES[p]] for p in pred20[fire]]
                acc = float((pred == y_wide).mean())
                rows["grid"].append({"k": k, "tau": tau, "fires": int(fire.sum()),
                                     "acc": round(acc, 4), "delta": round(acc - base, 4)})
        report[name] = rows
        print(f"{name}: n={len(y)} base={base:.4f}", flush=True)
        for g in rows["grid"]:
            print(f"   k={g['k']} tau={g['tau']:.1f} fires={g['fires']:5d} "
                  f"acc={g['acc']:.4f} ({g['delta']:+.4f})", flush=True)

    out = ROOT / "results/tables/wide_rerank_grid.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()

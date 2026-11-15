"""Can the 20-class posterior rescue the 116-class tab?

The "all cuisines" tab shows the 116-class model, which reaches 71.0% on the 100
real canteen photos (against 89.0% restricted to the 20 classes, and 94.0% for
the 20-class model).  Overriding only when the wide top-1 is a Food-101 class
barely moves (results/tables/wide_rerank_grid.json): the wide model is usually
right when it names a Food-101 dish -- the errors are the other way round, it
names a Chinese dish.

So this script fuses the two posteriors instead: the 20-class posterior is
mapped into the 116-class index space (zero elsewhere) and mixed with the wide
one, and alpha is swept on the real photos and both halves of the wide test set.
The Chinese half is the guard: a fusion that gains on Food-101 dishes but loses
on Chinese ones is not shippable.

    python results/_rerank_fuse.py
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
CACHE = ROOT / "results/tables/rerank_cache"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FOOD101 = set(CLASS_NAMES)
WIDE_IDX = {c: i for i, c in enumerate(WIDE_CLASS_NAMES)}
ALPHAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


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
        out.append(model(x.view(b * v, c, h, w)).float().view(b, v, -1).mean(1).cpu())
    return torch.cat(out).numpy()


def collect(root: Path, which: str):
    paths, labels = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name not in WIDE_IDX:
            continue
        is_food = folder.name in FOOD101
        if which == "food101" and not is_food:
            continue
        if which == "chinese" and is_food:
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in EXTS:
                paths.append(p)
                labels.append(WIDE_IDX[folder.name])
    return paths, np.array(labels, dtype=np.int64)


def get_logits(name: str, root: Path, which: str, models, device):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npz"
    if f.is_file():
        d = np.load(f)
        return d["l20"], d["l116"], d["y"]
    paths, y = collect(root, which)
    l20 = logits_of(models[0][0], paths, device)
    l116 = logits_of(models[1][0], paths, device)
    np.savez_compressed(f, l20=l20, l116=l116, y=y)
    print(f"cached {name}: n={len(y)}", flush=True)
    return l20, l116, y


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load(E7, 20, device), load(WIDE, WIDE_NUM_CLASSES, device)]
    (m20, t20), (m116, t116) = models

    sets = [
        ("canteen_real", ROOT / "data/splits/canteen_real", "food101"),
        ("canteen_proxy", ROOT / "data/splits/canteen", "food101"),
        ("wide_food101", ROOT / "data/wide/splits/test", "food101"),
        ("wide_chinese", ROOT / "data/wide/splits/test", "chinese"),
    ]

    report = {}
    for name, root, which in sets:
        l20, l116, y = get_logits(name, root, which, models, device)
        p20 = torch.softmax(torch.from_numpy(l20) / t20, 1).numpy()
        p116 = torch.softmax(torch.from_numpy(l116) / t116, 1).numpy()
        p20m = np.zeros_like(p116)
        sub = [WIDE_IDX[c] for c in CLASS_NAMES]
        p20m[:, sub] = p20
        rows = []
        for a in ALPHAS:
            fused = (1 - a) * p116 + a * p20m
            acc = float((fused.argmax(1) == y).mean())
            rows.append({"alpha": a, "acc": round(acc, 4)})
        report[name] = {"n": int(len(y)), "grid": rows}
        base = rows[0]["acc"]
        print(f"{name}: n={len(y)} wide={base:.4f} " +
              " ".join(f"a{a:.2f}={r['acc']:.4f}" for a, r in zip(ALPHAS, rows)), flush=True)

    out = ROOT / "results/tables/wide_rerank_fuse.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()

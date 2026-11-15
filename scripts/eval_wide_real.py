"""Score the 116-class model on the hand-labelled canteen photographs.

wide_domain.json only covers test / canteen_clean / canteen_proxy; the "all
cuisines" tab of the demo has never been measured on the 100 real canteen
photos. This appends "canteen_real" (classes absent from the batch, e.g. sushi,
are simply not scored).

    python scripts/eval_wide_real.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.classes_wide import WIDE_CLASS_NAMES, WIDE_NUM_CLASSES  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from scripts.train_wide import tta_transforms  # noqa: E402

CKPT = ROOT / "results/runs/resnet18_food101_cn116/best.pt"
OUT = ROOT / "results/tables/wide_real.json"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@torch.no_grad()
def score_dir(model, root: Path, device, batch_size: int = 24) -> dict:
    tf = tta_transforms()
    items = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name not in CLASS_NAMES:
            continue
        label = CLASS_NAMES.index(folder.name)
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in EXTS:
                items.append((p, label))
    if not items:
        raise SystemExit(f"no labelled photographs under {root}")

    subset = [WIDE_CLASS_NAMES.index(c) for c in CLASS_NAMES]
    label_in_wide = torch.tensor([WIDE_CLASS_NAMES.index(c) for c in CLASS_NAMES], device=device)
    n = correct_full = correct_sub = top5_sub = 0
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        views = []
        for path, _ in chunk:
            img = Image.open(path).convert("RGB")
            views.append(torch.stack([t(img) for t in tf]))
        views = torch.stack(views).to(device)
        b, v, c, h, w = views.shape
        logits = model(views.view(b * v, c, h, w)).view(b, v, -1).mean(1)
        labels = torch.tensor([lab for _, lab in chunk], device=device)
        correct_full += (logits.argmax(1) == label_in_wide[labels]).sum().item()
        sub_logits = logits.index_select(1, torch.tensor(subset, device=device))
        correct_sub += (sub_logits.argmax(1) == labels).sum().item()
        top5 = sub_logits.topk(5, dim=1).indices
        top5_sub += (top5 == labels[:, None]).any(1).sum().item()
        n += b
    return {
        "n": n,
        "top1_restricted": round(correct_sub / n, 4),
        "top5_restricted": round(top5_sub / n, 4),
        "top1_full116": round(correct_full / n, 4),
        "classes_in_wide": len(subset),
    }


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt.get("arch", "resnet18"), WIDE_NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    res = score_dir(model, ROOT / "data/splits/canteen_real", device)
    print(f"canteen_real n={res['n']} restricted top1={res['top1_restricted']:.4f} "
          f"top5={res['top5_restricted']:.4f} full116={res['top1_full116']:.4f}", flush=True)

    data = json.loads(OUT.read_text()) if OUT.is_file() else {}
    data["canteen_real"] = res
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

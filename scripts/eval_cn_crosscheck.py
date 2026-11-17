"""Can the wide head veto a canteen verdict?

The 20-dish head has to refuse dishes it never learned: on 15 Chinese dishes of
the wide table it names one of its 20 classes anyway, 86 % of the time.  The
wide head does know those dishes, so this script asks whether its verdict can
veto the canteen head: for every photo it records how sure each head is and
what the wide head calls it, then sweeps the veto threshold.

Writes results/tables/cn_crosscheck.json (+ .npz with the raw scores).

Run: python scripts/eval_cn_crosscheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcls import app as A
from foodcls.classes_cn import CN_CLASS_NAMES
from foodcls.classes_wide import WIDE_CLASS_NAMES
from foodcls.models import build_model
from foodcls.utils import pick_device

ROOT = Path(__file__).resolve().parents[1]
device = pick_device()

CN_CKPT = ROOT / "results/runs/cn241_full/best.pt"
WIDE_CKPT = ROOT / "results/runs/resnet18_food101_cn116/best.pt"


def load(ckpt_path: Path, names, tag: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], len(names), pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    t = json.loads((ckpt_path.parent / "calibration.json").read_text("utf-8"))["temperature"]
    print(f"{tag}: {ckpt_path.parent.name} arch={ck['arch']} T={t:.3f}", flush=True)
    return model, float(t)


cn, cn_t = load(CN_CKPT, CN_CLASS_NAMES, "canteen")
wide, wide_t = load(WIDE_CKPT, WIDE_CLASS_NAMES, "wide")


@torch.no_grad()
def scores(paths, batch=16):
    """Six-view posteriors of both heads, plus the worst-view gate score."""
    n = len(paths)
    cn_rob = np.zeros(n, dtype=np.float32)
    cn_top = np.zeros(n, dtype=np.int32)
    cn_p = np.zeros(n, dtype=np.float32)
    w_p = np.zeros(n, dtype=np.float32)
    w_top = np.zeros(n, dtype=np.int32)
    for i in range(0, n, batch):
        chunk = paths[i:i + batch]
        x = torch.cat([A.preprocess_views(Image.open(p)) for p in chunk]).to(device)
        logits_c = cn(x).float().view(len(chunk), 6, -1)
        pc = torch.softmax(logits_c / max(cn_t, 1e-3), -1)
        fused_c = torch.softmax(logits_c.mean(1) / max(cn_t, 1e-3), -1).cpu().numpy()
        gate = pc[:, :3, :].max(-1).values.cpu().numpy()  # (b, 3) worst-view rule
        logits_w = wide(x).float().view(len(chunk), 6, -1)
        fused_w = torch.softmax(logits_w.mean(1) / max(wide_t, 1e-3), -1).cpu().numpy()
        sl = slice(i, i + len(chunk))
        cn_rob[sl] = gate.min(1)
        cn_top[sl] = fused_c.argmax(1)
        cn_p[sl] = fused_c.max(1)
        w_top[sl] = fused_w.argmax(1)
        w_p[sl] = fused_w.max(1)
        if (i // batch) % 20 == 0:
            print(f"  {i + len(chunk)}/{n}", flush=True)
    return cn_rob, cn_top, cn_p, w_top, w_p


def collect(tag, paths):
    print(f"[{tag}] n={len(paths)}", flush=True)
    out = scores(paths)
    np.savez(ROOT / f"results/tables/_cc_{tag}.npz",
             cn_rob=out[0], cn_top=out[1], cn_p=out[2], w_top=out[3], w_p=out[4])
    return out


test = ROOT / "data/splits/cn241_full/test"
in_paths = [p for cls in CN_CLASS_NAMES for p in sorted((test / cls).glob("*.jpg"))]

wide_test = ROOT / "data/wide/splits/test"
cn_like = ["mapo_tofu", "kung_pao_chicken", "twice_cooked_pork", "sweet_and_sour_pork",
           "boiled_fish_sichuan", "fish_with_pickled_cabbage", "west_lake_vinegar_fish",
           "steamed_sea_bass", "peking_duck", "soup_dumplings", "dumplings", "spring_rolls",
           "fried_rice", "ramen", "braised_pork_meatballs"]
out_cn_paths = [p for cls in cn_like for p in sorted((wide_test / cls).glob("*.jpg"))[:24]]
out_other_paths = [p for d in sorted(wide_test.iterdir())
                   if d.is_dir() and d.name not in cn_like
                   for p in sorted(d.glob("*.jpg"))[:8]]

real = ROOT / "data/splits/canteen_real"
real_paths = sorted(p for p in real.rglob("*.jpg"))

sets = {
    "in": collect("in", in_paths),
    "out_cn": collect("out_cn", out_cn_paths),
    "out_other": collect("out_other", out_other_paths),
    "real": collect("real", real_paths),
}

CN_INDEX = {c: i for i, c in enumerate(CN_CLASS_NAMES)}
cn_like_idx = np.array([WIDE_CLASS_NAMES.index(c) for c in cn_like])

TH = 0.40


def keep_rates(veto_t):
    rows = {}
    for tag, (cn_rob, cn_top, cn_p, w_top, w_p) in sets.items():
        base = cn_rob >= TH
        veto = np.isin(w_top, cn_like_idx) & (w_p >= veto_t)
        final = base & ~veto
        rows[tag] = (round(float(base.mean()), 4), round(float(final.mean()), 4),
                     round(float(veto.mean()), 4))
    return rows


table = []
for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    r = keep_rates(t)
    table.append({"veto_threshold": t,
                  "in_keep": r["in"][1], "out_cn_keep": r["out_cn"][1],
                  "out_other_keep": r["out_other"][1], "real_keep": r["real"][1],
                  "in_vetoed": r["in"][2]})
    print(f"t={t:.2f}  in kept {r['in'][1]*100:5.1f}%  "
          f"other-Chinese kept {r['out_cn'][1]*100:5.1f}%  "
          f"other dishes kept {r['out_other'][1]*100:5.1f}%  "
          f"real photos kept {r['real'][1]*100:5.1f}%", flush=True)

base = {tag: round(float((v[0] >= TH).mean()), 4) for tag, v in sets.items()}
print("no-veto baseline:", base)

# How often is the wide head even right about which dish it is?
wide_names = [WIDE_CLASS_NAMES[k] for k in sets["out_cn"][3]]
truth = [p.parent.name for p in out_cn_paths]
right = float(np.mean([a == b for a, b in zip(wide_names, truth)]))
print(f"wide head names the correct dish on the 15: {right*100:.1f}%")

res = {"baseline_keep": base, "veto_table": table,
       "wide_top1_on_15": round(right, 4),
       "cn_like_wide_classes": cn_like, "n": {k: len(v[0]) for k, v in sets.items()}}
(ROOT / "results/tables/cn_crosscheck.json").write_text(
    json.dumps(res, indent=2, ensure_ascii=False), "utf-8")

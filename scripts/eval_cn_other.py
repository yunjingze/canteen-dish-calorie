# -*- coding: utf-8 -*-
"""Does a 21st "other Chinese" class teach the canteen head to say "not mine"?

The 20-dish head force-fits anything into its 20 classes; the documented fix is
to train a 21st class on the rest of CNFOOD-241 and let the head put an unknown
dish there.  This script measures the fix on five sets:

    in       2,462 CNFOOD-241 test images of the 20 supported dishes
    out_cn   15 Chinese dishes of the wide table (different source, 360 images)
    out_ood  29 CNFOOD-241 classes never shown to the 21-class head (232 images)
    out_other non-Chinese wide dishes (808 images)
    real     100 real canteen photos

and sweeps the two knobs the head now has: the worst-view gate threshold (taken
from the in-scope validation split) and the "other" probability required to
reject.  Also re-measures the wide-head veto on the same vectors.

Run: python scripts/eval_cn_other.py
Writes results/tables/cn_other.json (+ _other_<set>.npz).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A
from foodcls.models import build_model
from foodcls.utils import pick_device

CN_CKPT = ROOT / "results/runs/cn241_21/best.pt"
WIDE_CKPT = ROOT / "results/runs/resnet18_food101_cn116/best.pt"
CLASS_TABLE = ROOT / "results/tables/cn241_21_classes.json"

device = pick_device()
table = json.loads(CLASS_TABLE.read_text(encoding="utf-8"))
CN_NAMES = [c["slug"] for c in table["classes"]]
OTHER = CN_NAMES.index("other_cn")
IN_LIST = [i for i in range(len(CN_NAMES)) if i != OTHER]

CN_LIKE = ["mapo_tofu", "kung_pao_chicken", "twice_cooked_pork", "sweet_and_sour_pork",
           "boiled_fish_sichuan", "fish_with_pickled_cabbage", "west_lake_vinegar_fish",
           "steamed_sea_bass", "peking_duck", "soup_dumplings", "dumplings", "spring_rolls",
           "fried_rice", "ramen", "braised_pork_meatballs"]
#: the two the veto leaves alone on purpose (same dish / same plate as a class)
VETO_CLASSES = ("mapo_tofu", "kung_pao_chicken", "boiled_fish_sichuan",
                "fish_with_pickled_cabbage", "west_lake_vinegar_fish", "steamed_sea_bass",
                "peking_duck", "soup_dumplings", "dumplings", "spring_rolls", "fried_rice",
                "ramen", "braised_pork_meatballs")


def load(path: Path, names):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], len(names), pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    t = json.loads((path.parent / "calibration.json").read_text("utf-8"))["temperature"]
    print(f"{path.parent.name}: arch={ck['arch']} T={t:.3f} classes={len(names)}", flush=True)
    return model, float(t)


cn, cn_t = load(CN_CKPT, CN_NAMES)
wide, wide_t = load(WIDE_CKPT, A.WIDE_CLASS_NAMES)


@torch.no_grad()
def vectors(paths, batch=16, with_wide=True):
    n = len(paths)
    rob = np.zeros(n, dtype=np.float32)
    top = np.zeros(n, dtype=np.int32)
    top_p = np.zeros(n, dtype=np.float32)
    p_other = np.zeros(n, dtype=np.float32)
    w_top = np.zeros(n, dtype=np.int32)
    w_p = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch):
        chunk = paths[i:i + batch]
        x = torch.cat([A.preprocess_views(Image.open(p)) for p in chunk]).to(device)
        lc = cn(x).float().view(len(chunk), 6, -1)
        pc = torch.softmax(lc / max(cn_t, 1e-3), -1)
        fused = torch.softmax(lc.mean(1) / max(cn_t, 1e-3), -1).cpu().numpy()
        gate = pc[:, :3, :].max(-1).values.cpu().numpy()
        if with_wide:
            lw = wide(x).float().view(len(chunk), 6, -1)
            fw = torch.softmax(lw.mean(1) / max(wide_t, 1e-3), -1).cpu().numpy()
        else:
            fw = np.zeros((len(chunk), 2), dtype=np.float32)
        sl = slice(i, i + len(chunk))
        rob[sl] = gate.min(1)
        top[sl] = fused.argmax(1)
        top_p[sl] = fused.max(1)
        p_other[sl] = fused[:, OTHER]
        w_top[sl] = fw.argmax(1)
        w_p[sl] = fw.max(1)
        if (i // batch) % 25 == 0:
            print(f"  {i + len(chunk)}/{n}", flush=True)
    return rob, top, top_p, p_other, w_top, w_p


def collect(tag, paths):
    print(f"[{tag}] n={len(paths)}", flush=True)
    out = vectors(paths)
    np.savez(ROOT / f"results/tables/_other_{tag}.npz", rob=out[0], top=out[1],
             top_p=out[2], p_other=out[3], w_top=out[4], w_p=out[5])
    return out


def wide_test_glob(cls, k):
    return sorted((ROOT / "data/wide/splits/test" / cls).glob("*.jpg"))[:k]


sets = {}
in_paths = [p for c in CN_NAMES[:OTHER]
            for p in sorted((ROOT / "data/splits/cn241_full/test" / c).glob("*.jpg"))]
sets["in"] = (in_paths, collect("in", in_paths))

out_cn = [p for c in CN_LIKE for p in wide_test_glob(c, 24)]
sets["out_cn"] = (out_cn, collect("out_cn", out_cn))

ood_manifest = json.loads((ROOT / "results/tables/cn241_other_heldout.json").read_text("utf-8"))
ood = [ROOT / it["path"] for it in ood_manifest["items"]]
sets["out_ood"] = (ood, collect("out_ood", ood))

out_other = [p for d in sorted((ROOT / "data/wide/splits/test").iterdir())
             if d.is_dir() and d.name not in CN_LIKE for p in sorted(d.glob("*.jpg"))[:8]]
sets["out_other"] = (out_other, collect("out_other", out_other))

real = sorted((ROOT / "data/splits/canteen_real").rglob("*.jpg"))
sets["real"] = (real, collect("real", real))

# ---- gate threshold from the in-scope validation split (5th percentile) ----
val_root = ROOT / "data/splits/cn241_21/val"
val_paths = [p for c in CN_NAMES[:OTHER] for p in sorted((val_root / c).glob("*.jpg"))]
print(f"[val] n={len(val_paths)}", flush=True)
vrob, vtop, _, vother, _, _ = vectors(val_paths, with_wide=False)
TH = float(round(np.quantile(vrob, 0.05), 4))
print(f"in-scope val: worst-view 5th pct {TH:.4f}, mean {vrob.mean():.3f}, "
      f"other-rate {float(np.mean(vtop == OTHER))*100:.2f}%", flush=True)

veto_idx = np.array([A.WIDE_CLASS_NAMES.index(c) for c in VETO_CLASSES])
VT = 0.50


def evaluate(tau, veto=True):
    rows = {}
    for tag, (paths, (rob, top, p, p_other, w_top, w_p)) in sets.items():
        base = rob >= TH
        other_hit = (p_other >= tau) | (top == OTHER) if tau > 0 else np.zeros_like(base)
        v = np.zeros_like(base) if not veto else (
            np.isin(w_top, veto_idx) & (w_p >= VT) & (w_p >= p + 0.02))
        final = base & ~other_hit & ~v
        rows[tag] = {
            "n": int(len(rob)),
            "gate_pass": round(float(base.mean()), 4),
            "other_rej": round(float(other_hit.mean()), 4),
            "veto_rej": round(float(v.mean()), 4),
            "accept": round(float(final.mean()), 4),
        }
    return rows


# ---- in-scope accuracy of the 21-class head ----
rob, top, p, p_other, w_top, w_p = sets["in"][1]
truth = np.array([CN_NAMES.index(Path(path).parent.name) for path in in_paths])
acc = float(np.mean(top == truth))
in_other = float(np.mean(top == OTHER))
print(f"in-scope test: top-1 {acc*100:.2f}% (20-class baseline 87.73%), "
      f"called other {in_other*100:.2f}%", flush=True)

sweep = []
for tau in [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    r = evaluate(tau)
    sweep.append({"tau_other": tau, **{k: v["accept"] for k, v in r.items()},
                  "in_other_rej": r["in"]["other_rej"], "out_cn_other_rej": r["out_cn"]["other_rej"]})
    print(f"tau={tau:.2f}  in {r['in']['accept']*100:5.1f}%  out_cn {r['out_cn']['accept']*100:5.1f}%  "
          f"out_ood {r['out_ood']['accept']*100:5.1f}%  out_other {r['out_other']['accept']*100:5.1f}%  "
          f"real {r['real']['accept']*100:5.1f}%", flush=True)

best = evaluate(0.0, veto=False)
print("no other-class, no veto baseline:", best)

res = {"cn_ckpt": str(CN_CKPT.relative_to(ROOT)), "n_classes": len(CN_NAMES),
       "threshold_from_val_5pct": TH, "in_scope_top1": round(acc, 4),
       "in_scope_called_other": round(in_other, 4),
       "val_other_rate": round(float(np.mean(vtop == OTHER)), 4),
       "sweep": sweep, "sets": {k: len(v[0]) for k, v in sets.items()},
       "baseline_20class": {"in_keep": 0.948, "out_cn_keep": 0.711,
                            "out_other_keep": 0.522, "real_keep": 0.50},
       "veto_20class": {"in_keep": 0.941, "out_cn_keep": 0.344,
                        "out_other_keep": 0.522, "real_keep": 0.43}}
(ROOT / "results/tables/cn_other.json").write_text(
    json.dumps(res, indent=2, ensure_ascii=False), "utf-8")
print("wrote results/tables/cn_other.json")

"""FastAPI web demo: photo -> Top-3 dishes + calorie range."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageOps

from . import brands
from .calorie import PORTION_LABELS, topk_with_calories
from .classes import CLASS_NAMES, KCAL_PER_100G, PORTION_G, ZH_LABELS
from .classes_cn import (
    CN_CLASS_NAMES,
    CN_KCAL_PER_100G,
    CN_NUM_CLASSES,
    CN_PORTION_G,
    CN_ZH,
)
from .classes_wide import (
    WIDE_CLASS_NAMES,
    WIDE_KCAL_PER_100G,
    WIDE_NUM_CLASSES,
    WIDE_PORTION_G,
    WIDE_ZH,
)
from .data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from .mixed import (
    DETECT_THRESHOLD,
    MAX_PARTS,
    MIN_PART_PROB,
    detect_parts,
    plate_totals,
)
from .models import build_model
from .foodgate import FoodGate
from .packaging_head import PackagingHead, message as packaging_head_message
from . import packaging
from .utils import pick_device

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "results/runs/resnet18_last_stage_aug/best.pt"
CKPT = Path(os.environ.get("FOODCLS_CKPT", DEFAULT_CKPT))
if not CKPT.is_absolute():
    CKPT = (ROOT / CKPT).resolve()

#: optional multi-label head for mixed trays; the single-dish demo works without it
DEFAULT_MIXED_CKPT = ROOT / "results/runs/resnet18_multilabel_aug/best.pt"
MIXED_CKPT = Path(os.environ.get("FOODCLS_MIXED_CKPT", DEFAULT_MIXED_CKPT))
if not MIXED_CKPT.is_absolute():
    MIXED_CKPT = (ROOT / MIXED_CKPT).resolve()
#: optional wide head (Food-101 plus external Chinese dishes) for dishes outside the canteen list
DEFAULT_WIDE_CKPT = ROOT / "results/runs/resnet18_food101_cn116/best.pt"
WIDE_CKPT = Path(os.environ.get("FOODCLS_WIDE_CKPT", DEFAULT_WIDE_CKPT))
if not WIDE_CKPT.is_absolute():
    WIDE_CKPT = (ROOT / WIDE_CKPT).resolve()
WIDE_SAMPLES = ROOT / "data/wide/splits/test"
#: optional canteen-Chinese head, 20 everyday dishes from CNFOOD-241 plus a 21st
#: "other Chinese dish" bucket (CNFOOD-241 is non-commercial; see NOTICE).  Its
#: own checkpoint, its own class table.  The 20-dish checkpoint still loads.
DEFAULT_CN_CKPT = ROOT / "results/runs/cn241_21/best.pt"
CN_CKPT = Path(os.environ.get("FOODCLS_CN_CKPT", DEFAULT_CN_CKPT))
if not CN_CKPT.is_absolute():
    CN_CKPT = (ROOT / CN_CKPT).resolve()
CN_SAMPLES = ROOT / "data/splits/cn241_full/test"
MIXED_SAMPLES = ROOT / "data/splits/mixed"
STATIC = Path(__file__).resolve().parent / "static"
_PROXY = ROOT / "data/splits/canteen"
SAMPLES = _PROXY if _PROXY.exists() else ROOT / "data/splits/test"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

#: "View-robust" confidence below this means "probably none of the 20 supported
#: dishes".  We score three crops (centre, squashed full frame, wide crop) and keep
#: the *worst* top-1 probability: an unsupported dish (a whole roast chicken, say)
#: looks like a different dish depending on the crop, while a supported dish stays
#: confident.  On supported photos (val + canteen proxy) ~5 % fall below 0.32 and
#: on 20 unseen Food-101 dishes ~25 % do, so the banner stays a soft warning.
#: scripts/eval_openset.py re-measures this.
SCOPE_THRESHOLD = 0.32

#: the wide head spreads probability over 116 classes, so the same rule is a bit
#: stricter there.  Every Food-101 and Chinese class is in scope, so instead of an
#: unseen set we put this at the 5th percentile of in-scope worst-view scores, i.e.
#: the same ~5 % false-alarm budget as the 20-class gate (eval_wide_gate.py re-measures).
WIDE_SCOPE_THRESHOLD = 0.18

#: The canteen head has the same shape of risk as the 20-class one (it names
#: something for every photo), so it gets the same worst-view rule; the value is
#: measured by scripts/eval_cn_scope.py on 794 CNFOOD-241 test images (in scope,
#: worst-view score mean 0.83) and 320 dishes outside the 20 drawn from the wide
#: test split (out of scope, mean 0.50).  0.40 is the 5th percentile of the
#: in-scope scores, the same ~5 % false-alarm budget as the two other heads: it
#: keeps 94.8 % of in-scope photos and flags 43 % of the outside dishes.
CN_SCOPE_THRESHOLD = 0.40

#: The canteen head has to refuse dishes it never learned, and it does not: on 15
#: Chinese dishes of the wide table it still names one of its 20 classes 71 % of
#: the time (scripts/eval_cn_crosscheck.py).  The wide head knows those dishes, so
#: it gets a veto: if it names one of them *and* is more confident than the
#: canteen head by CN_VETO_MARGIN, the verdict is "not one of the 20".
#:
#: Two of the fifteen are left out on purpose.  糖醋里脊 (sweet_and_sour_pork) is
#: the same dish as a canteen class, and 回锅肉 (twice_cooked_pork) is visually
#: the same plate as 农家小炒肉: vetoing either one threw away 27 % of the
#: in-scope photos to catch dishes the user would have accepted.  With them out,
#: the veto keeps 94.4 % of the 2,462 in-scope test photos (94.8 % before) and
#: rejects 56 % of the out-of-scope Chinese dishes instead of 29 %.
CN_VETO_CLASSES = (
    "mapo_tofu", "kung_pao_chicken", "boiled_fish_sichuan",
    "fish_with_pickled_cabbage", "west_lake_vinegar_fish", "steamed_sea_bass",
    "peking_duck", "soup_dumplings", "dumplings", "spring_rolls", "fried_rice",
    "ramen", "braised_pork_meatballs",
)
CN_VETO_INDEX = np.array([WIDE_CLASS_NAMES.index(c) for c in CN_VETO_CLASSES], dtype=np.int64)
CN_VETO_THRESHOLD = 0.50
CN_VETO_MARGIN = 0.02

#: The head is also trained with a 21st class, "other Chinese dish": the rest of
#: CNFOOD-241 (minus Western/dessert dishes and the classes held out for testing)
#: compressed into one bucket.  This is the structural fix for the dishes the veto
#: kept missing -- 宫保鸡丁 was 0.005 short of the veto margin, and a soup dumpling
#: is a dumpling to a head that has never seen "not on the list".  The bucket
#: gives those photos somewhere to go; CN_OTHER_THRESHOLD is how much mass it
#: needs before the verdict becomes "not one of the 20".
#:
#: Measured on 2,462 in-scope test photos, 360 Chinese dishes outside the list,
#: 232 photos of 29 never-trained Chinese classes, 808 Western test photos and
#: 100 real canteen photos (scripts/eval_cn_other.py + search_cn_other_rule.py):
#: 0.40 turns away 76 % of the outside Chinese dishes for 3 % of the in-scope
#: ones.  Money either way -- the curve is flat above 0.40 because the failures
#: left are the look-alikes (回锅肉/农家小炒肉), not low-confidence cases.
CN_OTHER_SLUG = "other_cn"
CN_OTHER_ZH = "其它中餐"
CN_OTHER_THRESHOLD = float(os.environ.get("FOODCLS_CN_OTHER", "0.40"))

#: The bucket steals probability mass from the 20 dishes, so the gate is measured
#: on the dish mass after dividing it back out: per view p_dish / (1 - p_other),
#: worst view, 5th percentile of the in-scope validation split = 0.3445
#: (scripts/search_cn_other_rule.py).  With the 20-dish checkpoint the divisor is
#: 1 and the threshold stays CN_SCOPE_THRESHOLD, so old weights behave as before.
CN_SCOPE_THRESHOLD_21 = float(os.environ.get("FOODCLS_CN_SCOPE_21", "0.3445"))

#: Optional domain probe: "is this even a canteen-style Chinese dish?"  The
#: 21-class head can only answer "which of the 21", so a pizza still comes back
#: as some stir-fry with a straight face -- the bucket catches 79 % of the real
#: western photos, and names a dish on the rest.  A logistic regression on the
#: pooled 512-d feature of the same backbone answers the other question at the
#: cost of one dot product: no second network, no extra forward pass, ~10 KB of
#: JSON (scripts/extract_domain_features.py -> scripts/fit_domain_head.py).
DOMAIN_HEAD_PATH = Path(os.environ.get("FOODCLS_DOMAIN_HEAD", "results/tables/domain_head.json"))
if not DOMAIN_HEAD_PATH.is_absolute():
    DOMAIN_HEAD_PATH = (ROOT / DOMAIN_HEAD_PATH).resolve()
#: 0 keeps the threshold stored in the head file; set it to re-tune without refitting.
DOMAIN_THRESHOLD_OVERRIDE = float(os.environ.get("FOODCLS_DOMAIN_THRESHOLD", "0"))
DOMAIN_ZH = "食堂中餐"

#: Where the 20 supported dishes sit in the 116-class table, so the two
#: posteriors can be mixed without reordering anything.
WIDE_NARROW_INDEX = np.array([WIDE_CLASS_NAMES.index(c) for c in CLASS_NAMES], dtype=np.int64)

#: Food / non-food gate (see foodcls/foodgate.py).  The view-robust rule only
#: asks "which dish?", so about half of real non-food photos still pass 0.32.
#: Combined with the gate, 116 of 131 real non-food photos are rejected while 96
#: of 100 real canteen dishes and 999/1000 proxy-domain dishes are kept
#: (results/_gate_final.py, templates in results/tables/gate_oof.json).
FOODGATE_PATH = Path(os.environ.get("FOODCLS_GATE", CKPT.parent / "foodgate.json"))
if not FOODGATE_PATH.is_absolute():
    FOODGATE_PATH = (ROOT / FOODGATE_PATH).resolve()
FOODGATE_THRESHOLD = 0.20

#: Packaging head (see foodcls/packaging_head.py).  The food gate answers "is
#: there food here?" and a photo printed on a pizza box clears it.  This head
#: answers "is this a package, not a plate?" and is consulted before the scope
#: decision, so a 98 % "pizza" on a box front no longer reaches the user.
PACKAGING_PATH = Path(os.environ.get("FOODCLS_PACK", CKPT.parent / "packaging.json"))
if not PACKAGING_PATH.is_absolute():
    PACKAGING_PATH = (ROOT / PACKAGING_PATH).resolve()
PACKAGING_THRESHOLD = 0.80

_state: dict = {}


def _load_temperature(ckpt_path: Path, key: str = "temperature") -> float:
    """Optional post-hoc calibration written next to the checkpoint.

    results/runs/<run>/calibration.json is one object, while the shared table
    results/tables/calibration.json is a list of rows keyed by run name.  Both
    shapes must work: a run without its own file used to raise AttributeError
    on the list and take the whole service down at startup.
    """
    for cand in (ckpt_path.parent / "calibration.json", ROOT / "results/tables/calibration.json"):
        try:
            if not cand.is_file():
                continue
            data = json.loads(cand.read_text(encoding="utf-8"))
            if isinstance(data, list):
                data = next(
                    (row for row in data
                     if isinstance(row, dict) and row.get("run") == ckpt_path.parent.name),
                    None,
                )
            if not isinstance(data, dict):
                continue
            t = float(data.get(key, data.get("temperature", 1.0)))
            if 0.05 <= t <= 20.0:
                print(f"{key} {t:.3f} from {cand}", flush=True)
                return t
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return 1.0


def _check_class_names(ckpt, expected, path):
    """Refuse a checkpoint whose label list does not match the code's.

    A mismatch silently maps logits to the wrong names, which is far worse than
    a loud failure at start-up.
    """
    names = ckpt.get("class_names")
    if not names:
        return
    if list(names) != list(expected):
        raise RuntimeError(
            f"{Path(path).name}: checkpoint class_names ({len(names)}) do not match the "
            f"code's class list ({len(expected)}); refusing to load"
        )


def load_model():
    device = pick_device()
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    _check_class_names(ckpt, CLASS_NAMES, CKPT)
    model = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    _state["model"] = model
    _state["device"] = device
    _state["arch"] = ckpt["arch"]
    _state["temperature"] = _load_temperature(CKPT)
    _state["temperature_fused"] = _load_temperature(CKPT, "temperature_fused")
    print(f"loaded {ckpt['arch']} from {CKPT} on {device}", flush=True)
    _state["foodgate"] = None
    if FOODGATE_PATH.is_file():
        try:
            _state["foodgate"] = FoodGate(FOODGATE_PATH, FOODGATE_THRESHOLD)
            print(f"loaded food gate (p_food >= {FOODGATE_THRESHOLD:.2f}) from {FOODGATE_PATH}",
                  flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"food gate disabled: {exc}", flush=True)
    else:
        print(f"no food gate at {FOODGATE_PATH}; relying on the view-robust rule only",
              flush=True)
    _state["packaging"] = None
    if PACKAGING_PATH.is_file():
        try:
            _state["packaging"] = PackagingHead(PACKAGING_PATH, PACKAGING_THRESHOLD)
            print(f"loaded packaging head (p_pack >= {_state['packaging'].threshold:.2f}) "
                  f"from {PACKAGING_PATH}", flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"packaging head disabled: {exc}", flush=True)
    else:
        print(f"no packaging head at {PACKAGING_PATH}; using the edge heuristic only",
              flush=True)


def load_mixed_model():
    """Load the multi-label plate model if it has been trained (optional)."""
    _state["mixed_model"] = None
    _state["mixed_threshold"] = DETECT_THRESHOLD
    _state["mixed_class_thresholds"] = {}
    if not MIXED_CKPT.is_file():
        print(f"no mixed model at {MIXED_CKPT}; mixed mode disabled", flush=True)
        return
    device = _state["device"]
    ckpt = torch.load(MIXED_CKPT, map_location=device, weights_only=False)
    _check_class_names(ckpt, CLASS_NAMES, MIXED_CKPT)
    model = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    _state["mixed_model"] = model
    _state["mixed_threshold"] = float(ckpt.get("threshold", DETECT_THRESHOLD))
    table = ckpt.get("class_thresholds") or {}
    _state["mixed_class_thresholds"] = {str(k): float(v) for k, v in table.items()}
    print(
        f"loaded mixed model {ckpt['arch']} from {MIXED_CKPT} "
        f"(threshold {_state['mixed_threshold']:.2f}, "
        f"{len(_state['mixed_class_thresholds'])} per-class cut-offs)",
        flush=True,
    )


def load_wide_model():
    """Load the optional wide head; the canteen demo works without it."""
    _state["wide_model"] = None
    _state["wide_temperature"] = 1.0
    if not WIDE_CKPT.is_file():
        print(f"no wide model at {WIDE_CKPT}; wide mode disabled", flush=True)
        return
    device = _state["device"]
    ckpt = torch.load(WIDE_CKPT, map_location=device, weights_only=False)
    _check_class_names(ckpt, WIDE_CLASS_NAMES, WIDE_CKPT)
    model = build_model(ckpt["arch"], WIDE_NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    _state["wide_model"] = model
    _state["wide_arch"] = ckpt["arch"]
    _state["wide_temperature"] = _load_temperature(WIDE_CKPT)
    _state["wide_temperature_fused"] = _load_temperature(WIDE_CKPT, "temperature_fused")
    _state["wide_fuse"] = WIDE_FUSE_ALPHA if _state.get("model") is not None else 0.0
    print(f"loaded wide {ckpt['arch']} ({WIDE_NUM_CLASSES} classes) from {WIDE_CKPT} on {device}"
          + (f", fusing {WIDE_FUSE_ALPHA:.2f} of the 20-class posterior" if _state["wide_fuse"] else ""),
          flush=True)


def load_cn_model():
    """Load the optional 20-dish canteen head (CNFOOD-241); optional like wide."""
    _state["cn_model"] = None
    _state["cn_temperature"] = 1.0
    if not CN_CKPT.is_file():
        print(f"no canteen model at {CN_CKPT}; canteen mode disabled", flush=True)
        return
    device = _state["device"]
    ckpt = torch.load(CN_CKPT, map_location=device, weights_only=False)
    names = list(ckpt.get("class_names") or CN_CLASS_NAMES)
    has_other = (len(names) == CN_NUM_CLASSES + 1
                 and names[:CN_NUM_CLASSES] == list(CN_CLASS_NAMES)
                 and names[-1] == CN_OTHER_SLUG)
    if list(names) != list(CN_CLASS_NAMES) and not has_other:
        raise RuntimeError(
            f"{CN_CKPT.name}: checkpoint class_names ({len(names)}) match neither the "
            f"{CN_NUM_CLASSES} dishes nor those plus '{CN_OTHER_SLUG}'; refusing to load"
        )
    model = build_model(ckpt["arch"], len(names), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    _state["cn_model"] = model
    _state["cn_arch"] = ckpt["arch"]
    _state["cn_other_index"] = CN_NUM_CLASSES if has_other else None
    #: the gate is re-measured on the 21-class checkpoint (see CN_SCOPE_THRESHOLD_21)
    _state["cn_scope_threshold"] = CN_SCOPE_THRESHOLD_21 if has_other else CN_SCOPE_THRESHOLD
    _state["cn_temperature"] = _load_temperature(CN_CKPT)
    print(f"loaded canteen {ckpt['arch']} ({len(names)} classes"
          + (", incl. 'other Chinese'" if has_other else "")
          + f") from {CN_CKPT} on {device}, T={_state['cn_temperature']:.3f}", flush=True)


def load_domain_head():
    """Load the optional Chinese-vs-western probe (see DOMAIN_HEAD_PATH)."""
    _state["domain_head"] = None
    if not DOMAIN_HEAD_PATH.is_file():
        print(f"no domain probe at {DOMAIN_HEAD_PATH}; the canteen gate stands alone", flush=True)
        return
    head = json.loads(DOMAIN_HEAD_PATH.read_text(encoding="utf-8"))
    w = np.asarray(head["w"], dtype=np.float64)
    _state["domain_head"] = {
        "w": w,
        "b": float(head["b"]),
        "mean": np.asarray(head["mean"], dtype=np.float64),
        "scale": np.asarray(head["scale"], dtype=np.float64),
        "threshold": DOMAIN_THRESHOLD_OVERRIDE or float(head["threshold"]),
    }
    print(f"loaded domain probe ({w.size} weights, threshold "
          f"{_state['domain_head']['threshold']:.3f}) from {DOMAIN_HEAD_PATH}", flush=True)


def _transforms(flip: bool = False):
    from torchvision import transforms

    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    mirror = [transforms.RandomHorizontalFlip(p=1.0)] if flip else []

    def view(resize_to=None, square=False):
        ops = ([transforms.Resize((IMG_SIZE, IMG_SIZE))] if square
               else [transforms.Resize(int(IMG_SIZE * resize_to / 224)), transforms.CenterCrop(IMG_SIZE)])
        return transforms.Compose(ops + mirror + norm)

    return [
        # the training-time view
        view(256),
        # whole frame, no crop: keeps the context a tight crop throws away
        view(square=True),
        # a wider crop, closer to "a plate on a table"
        view(320),
    ]


def preprocess(img: Image.Image) -> torch.Tensor:
    return _transforms()[0](img.convert("RGB")).unsqueeze(0)


#: six views = three framings + their mirrors, built once at import.  Building
#: the list lazily could race on the first requests and silently send twelve
#: views, which no longer matches the fusion temperature fitted on six.
_VIEW_TFS: list = _transforms() + _transforms(flip=True)

#: the gate is calibrated on the three unmirrored framings (scripts/eval_openset.py)
_GATE_VIEWS = 3


def preprocess_views(img: Image.Image) -> torch.Tensor:
    """Stack the six views (three framings, each mirrored) into one batch."""
    rgb = img.convert("RGB")
    return torch.stack([tf(rgb) for tf in _VIEW_TFS])


#: Decompression-bomb guard.  A 12 MP phone photo is 4032x3024; past 40 MP the
#: decode alone costs seconds and hundreds of MB, and the resize throws the
#: detail away anyway.
MAX_PIXELS = 40_000_000


class ImageTooLarge(Exception):
    """The upload decodes to more pixels than we are willing to allocate."""


def read_image(raw: bytes) -> Image.Image:
    """Decode uploaded bytes, honouring the EXIF orientation of phone photos."""
    img = Image.open(io.BytesIO(raw))
    width, height = img.size
    if width * height > MAX_PIXELS:
        raise ImageTooLarge(f"{width}x{height}")
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


@torch.no_grad()
def _view_probs(model, temperature: float, img: Image.Image,
                fused_temperature: float | None = None, want_feats: bool = False):
    """Per-view probabilities and the fused posterior.

    Views are fused in logit space and calibrated once afterwards: averaging
    probabilities that were each sharpened by their own temperature gives a
    worse posterior than fusing the logits (93.1 vs 93.4 on test, 71.2 vs 73.0
    on the proxy for E7).  The per-view probabilities keep their own temperature
    because the out-of-scope gate is calibrated on them.
    """
    x = preprocess_views(img).to(_state["device"])
    store: dict = {}
    hook = None
    if want_feats:
        # the food gate consumes the pooled penultimate feature of the first
        # (centre-crop) view, i.e. the same forward pass, no second network
        hook = model.avgpool.register_forward_hook(
            lambda _m, _i, out: store.__setitem__("feat", out))
    try:
        logits = model(x).float().cpu()
    finally:
        if hook is not None:
            hook.remove()
    views = torch.softmax(logits / max(temperature, 1e-3), dim=1).numpy()
    t_fused = fused_temperature if fused_temperature else temperature
    fused = torch.softmax(logits.mean(0) / max(t_fused, 1e-3), dim=0).numpy()
    if want_feats:
        # (6, 512) -> (3, 512): one row per framing, mirror dropped.  Row 0 is the
        # centre crop the food gate was calibrated on, row 1 the whole frame the
        # packaging head was fitted on.
        frames = store["feat"].flatten(1).float().cpu().numpy()[:_GATE_VIEWS]
        return views, fused, frames
    return views, fused


def analyse(img: Image.Image):
    return _view_probs(_state["model"], _state.get("temperature", 1.0), img,
                       _state.get("temperature_fused"))


def analyse_full(img: Image.Image):
    """analyse() plus the pooled feature the food gate needs."""
    return _view_probs(_state["model"], _state.get("temperature", 1.0), img,
                       _state.get("temperature_fused"), want_feats=True)


#: How much of the 20-class posterior to mix into the 116-class one (see
#: results/_rerank_fuse.py).  The wide head knows which cuisine, the narrow head
#: is far better on the 20 dishes it was trained on; mixing them lifts the wide
#: tab on Food-101 dishes at a measurable cost on Chinese ones, which is what
#: set this value.  On the 100 real canteen photos 0.20 fixes 8 dishes and
#: breaks none (71.0 -> 79.0, p = 0.008); on 20,830 Chinese test images it
#: breaks 212 and fixes none (82.9 -> 81.9); on 5,000 Food-101 test images it
#: fixes 205 and breaks 3 (83.3 -> 87.4).  0 disables the fusion and restores
#: the plain wide head.
WIDE_FUSE_ALPHA = float(os.environ.get("FOODCLS_WIDE_FUSE", "0.2"))


def analyse_wide(img: Image.Image):
    model = _state.get("wide_model")
    if model is None:
        raise RuntimeError("wide model is not loaded")
    views, fused = _view_probs(model, _state.get("wide_temperature", 1.0), img,
                               _state.get("wide_temperature_fused"))
    alpha = _state.get("wide_fuse", 0.0)
    narrow = _state.get("model")
    if alpha > 0 and narrow is not None:
        # The narrow posterior is mapped into the wide index space and left
        # there: its mass is not renormalised, so alpha also controls how much
        # total probability the 20 known dishes may claim.
        _, p_narrow = _view_probs(narrow, _state.get("temperature", 1.0), img,
                                  _state.get("temperature_fused"))
        full = np.zeros_like(fused)
        full[WIDE_NARROW_INDEX] = p_narrow
        fused = (1.0 - alpha) * fused + alpha * full
    return views, fused


def analyse_cn(img: Image.Image, want_feats: bool = False):
    """Six-view posterior of the canteen head, plus the pooled feature."""
    model = _state.get("cn_model")
    if model is None:
        raise RuntimeError("canteen model is not loaded")
    return _view_probs(model, _state.get("cn_temperature", 1.0), img, want_feats=want_feats)


def _domain_prob(frames: np.ndarray | None) -> Optional[float]:
    """P(this photo is a canteen-style Chinese dish) from the pooled features."""
    head = _state.get("domain_head")
    if head is None or frames is None:
        return None
    x = np.asarray(frames, dtype=np.float64).mean(axis=0)
    z = float(((x - head["mean"]) / head["scale"]) @ head["w"] + head["b"])
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))


@torch.no_grad()
def cn_veto(img: Image.Image) -> Optional[dict]:
    """The wide head's objection to a canteen verdict, if it has one.

    The wide posterior is used *as trained* here: analyse_wide() mixes the
    canteen head into it, which would make the two heads agree by construction
    and hide exactly the disagreement this veto looks for.
    """
    wide = _state.get("wide_model")
    if wide is None:
        return None
    views, fused = _view_probs(wide, _state.get("wide_temperature", 1.0), img,
                               _state.get("wide_temperature_fused"))
    top = int(np.argmax(fused))
    return {
        "class": WIDE_CLASS_NAMES[top],
        "zh": WIDE_ZH[WIDE_CLASS_NAMES[top]],
        "prob": float(fused[top]),
        "views": views,
    }


@torch.no_grad()
def infer(img: Image.Image) -> np.ndarray:
    """Fused posterior: the six views averaged in logit space, then calibrated."""
    return analyse(img)[1]


def scope_of(views: np.ndarray, probs: np.ndarray, wide: bool = False,
             food_prob: float | None = None,
             packaging_prob: float | None = None, cn: bool = False,
             cn_veto_result: dict | None = None,
             cn_other_prob: float | None = None,
             cn_other_top: bool = False,
             cn_domain_prob: float | None = None) -> dict:
    """Is this photo one of the supported dishes?

    The reported confidence is the fused posterior; the decision still uses the
    worst single view, because one frame that looks like another dish is exactly
    the failure this flag exists to catch.
    """
    if wide:
        names, zh, threshold = WIDE_CLASS_NAMES, WIDE_ZH, WIDE_SCOPE_THRESHOLD
        label = f"{WIDE_NUM_CLASSES} 类"
        examples = f"火锅、麻辣烫、家常炒菜等中式吃法都不在这 {WIDE_NUM_CLASSES} 类里"
    elif cn:
        names, zh = CN_CLASS_NAMES, CN_ZH
        # With the bucket the gate must ignore it: an in-scope dish can put 40 %
        # on "some other dish" and still be the right one, and the old threshold
        # would then throw the photo away for the wrong reason.
        threshold = (CN_SCOPE_THRESHOLD_21 if cn_other_prob is not None
                     else CN_SCOPE_THRESHOLD)
        label = f"{CN_NUM_CLASSES} 道"
        examples = "牛肉面、麻辣烫、米线、手抓饼、清蒸鱼这些食堂主食和小吃不在这 20 道里"
    else:
        names, zh, threshold, label = CLASS_NAMES, ZH_LABELS, SCOPE_THRESHOLD, "20 类"
        examples = "整只烤鸡/烤鸭、火锅、家常炒菜等都不在支持的 20 类里"
    top1 = float(np.max(probs))
    # Mirrored views sharpen the fused posterior but lower the worst-view score
    # on every photo, so the gate keeps the three original framings only; that
    # is the calibration the 0.32 / 0.18 thresholds were measured on.
    gate_views = views[:_GATE_VIEWS] if len(views) > _GATE_VIEWS else views
    if cn_other_prob is not None and views.shape[1] > CN_NUM_CLASSES:
        # Divide the bucket back out of every view before asking "which of the 20
        # dishes is this?": the dish mass is what the 20-dish gate was calibrated
        # on, and the bucket answers the other question separately.
        dish = views[:, :CN_NUM_CLASSES].astype(np.float64)
        bucket = views[:, CN_NUM_CLASSES].astype(np.float64)
        renorm = dish / np.clip(1.0 - bucket, 1e-6, None)[:, None]
        robust = float(min(np.max(v) for v in renorm[:_GATE_VIEWS]))
    else:
        robust = float(min(np.max(v) for v in gate_views))
    # Two questions, two answers: the gate asks "is there food here at all?",
    # the robust score asks "is it one of the supported dishes?".  A menu board
    # clears neither, but it does clear the second one often enough to matter.
    gate = _state.get("foodgate")
    food_ok = True
    if gate is not None and food_prob is not None and not wide and not cn:
        food_ok = bool(food_prob >= gate.threshold)
    # A package prints a picture of a dish: the 20-class head is confident and
    # the food gate agrees, so neither of them can catch it.  This is a veto --
    # but only over photos the gate called food.  When the gate already said
    # "no food here", its own message is the accurate one (a napkin is not a
    # product package), and 43 of the 46 flagged negatives were exactly that.
    pack_head = _state.get("packaging")
    package = False
    if pack_head is not None and packaging_prob is not None and not wide and not cn and food_ok:
        package = bool(packaging_prob >= pack_head.threshold)
    # The wide head's veto runs last: it can only take a verdict away, never add
    # one, and it only speaks about dishes the canteen head cannot represent.
    vetoed = False
    if cn and cn_veto_result is not None:
        vetoed = bool(cn_veto_result["class"] in CN_VETO_CLASSES
                      and cn_veto_result["prob"] >= CN_VETO_THRESHOLD
                      and cn_veto_result["prob"] >= top1 + CN_VETO_MARGIN)
    # The 21st class is not a dish: it is the head saying "some other Chinese
    # dish", which is a refusal, not a name.  It is measured on the fused
    # posterior because a single view can be fooled by a garnish (see README).
    other_hit = False
    if cn and cn_other_prob is not None:
        other_hit = bool(cn_other_prob >= CN_OTHER_THRESHOLD or cn_other_top)
    # The bucket is still a Chinese dish; a pizza is not, and the head has no
    # class for "wrong cuisine".  The probe asks that question first, so the
    # message says "not canteen food" instead of "some other Chinese dish".
    dom_th = (_state.get("domain_head") or {}).get("threshold", 0.5)
    domain_hit = bool(cn and cn_domain_prob is not None and cn_domain_prob < dom_th)
    in_scope = (bool(robust >= threshold) and food_ok and not package
                and not vetoed and not other_hit and not domain_hit)
    # Name the nearest *dish* even when the bucket won: "closest: 其它中餐" tells
    # the user nothing about what the photo looks like.
    top_idx = int(np.argmax(probs))
    if cn and cn_other_prob is not None and top_idx >= CN_NUM_CLASSES:
        top_idx = int(np.argmax(probs[:CN_NUM_CLASSES]))
    closest = zh[names[top_idx]]
    scope = {
        "in_scope": in_scope,
        "top1_prob": round(top1, 4),
        "robust_prob": round(robust, 4),
        "threshold": threshold,
        "closest": closest,
        "label": label,
    }
    if cn and cn_other_prob is not None:
        scope["other"] = {
            "zh": CN_OTHER_ZH,
            "prob": round(float(cn_other_prob), 4),
            "threshold": CN_OTHER_THRESHOLD,
            "fired": other_hit,
        }
    if cn and cn_domain_prob is not None:
        scope["domain"] = {
            "zh": DOMAIN_ZH,
            "prob": round(float(cn_domain_prob), 4),
            "threshold": dom_th,
            "fired": domain_hit,
        }
    if cn and cn_veto_result is not None:
        scope["veto"] = {
            "class": cn_veto_result["class"],
            "zh": cn_veto_result["zh"],
            "prob": round(cn_veto_result["prob"], 4),
            "threshold": CN_VETO_THRESHOLD,
            "fired": vetoed,
        }
    if food_prob is not None and not wide and not cn:
        scope["food_prob"] = round(float(food_prob), 4)
        scope["food_threshold"] = gate.threshold if gate is not None else None
    if packaging_prob is not None and not wide and not cn:
        scope["packaging"] = package
        scope["packaging_prob"] = round(float(packaging_prob), 4)
        scope["packaging_threshold"] = pack_head.threshold if pack_head is not None else None
    if not in_scope and package:
        scope["not_food"] = True
        scope["message"] = packaging_head_message()
    elif not in_scope and not food_ok:
        scope["not_food"] = True
        chance = ("基本为零" if food_prob < 0.02
                  else f"约 {max(1, round(food_prob * 100))}%")
        scope["message"] = (
            f"这张照片里没找到菜（是菜的可能性{chance}）：空盘子、饮料杯、包装袋、"
            "菜单牌、人手这些都不在识别范围里。把镜头对准餐盘里的菜再拍一张，"
            "或者切到「混合餐盘」逐个标出盘里的部分。"
        )
    elif not in_scope and domain_hit:
        scope["message"] = (
            f"这张看起来不是食堂窗口的家常菜：域判别认为它是「{DOMAIN_ZH}」的把握只有 "
            f"{cn_domain_prob * 100:.0f}%（低于 {dom_th * 100:.0f}%），"
            "更像西餐、甜点或饮料。食堂模式只认识 20 道中餐——"
            "换成「单菜识别」或「116 类」标签再试。"
        )
    elif not in_scope and other_hit:
        # The bucket caught what the wide-head veto kept missing (宫保鸡丁 was
        # 0.005 short of the margin, and 汤包/饺子 fell into 包子).
        scope["message"] = (
            f"这张看起来不在食堂模式的 20 道清单里：模型把 {cn_other_prob * 100:.0f}% 的把握"
            f"放在「{CN_OTHER_ZH}」这一类上，也就是说它更像清单之外的菜"
            f"（最接近的「{closest}」只是强行归类的候选）。"
            "想吃别的菜可以切到「116 类」标签试试。"
        )
    elif not in_scope and vetoed:
        # A dish the canteen head would have force-fitted into its 20 classes:
        # name what the photo actually looks like instead of the nearest match.
        scope["message"] = (
            f"这张更像「{cn_veto_result['zh']}」——它不在食堂模式的 20 道清单里"
            f"（最接近的「{closest}」是强行归类的候选，把握只有 {robust * 100:.0f}%）。"
            "切到「116 类」标签可以认出这道菜。"
        )
    elif not in_scope:
        # The head always names *something*; say out loud when that something is
        # only the nearest of an unsupported dish.
        scope["message"] = (
            f"这张照片看起来不是清单里的菜：模型最像「{closest}」，但换个取景就认成别的，"
            f"最差的一次把握只有 {robust * 100:.0f}%。{examples}，"
            "下面只是最接近的候选，不是识别结果。"
        )
    return scope


def payload(probs: np.ndarray, source: str, portion: str = "normal", grams=None, views=None,
            wide: bool = False, img: Image.Image | None = None, food_prob: float | None = None,
            packaging_prob: float | None = None, cn: bool = False,
            cn_frames: np.ndarray | None = None):
    portion = str(portion or "normal").strip().lower()
    cn_other_prob = None
    cn_other_top = False
    if cn:
        # Split off the "other Chinese" bucket: it has no name, no calories and
        # no portion, so it must never reach topk_with_calories().
        if len(probs) > CN_NUM_CLASSES:
            cn_other_prob = float(probs[CN_NUM_CLASSES])
            cn_other_top = bool(int(np.argmax(probs)) == CN_NUM_CLASSES)
            probs = probs[:CN_NUM_CLASSES]
        items = topk_with_calories(probs, k=3, portion=portion, grams=grams,
                                   class_names=CN_CLASS_NAMES, zh_table=CN_ZH,
                                   kcal_table=CN_KCAL_PER_100G, portion_table=CN_PORTION_G)
    elif wide:
        items = topk_with_calories(probs, k=3, portion=portion, grams=grams,
                                   class_names=WIDE_CLASS_NAMES, zh_table=WIDE_ZH,
                                   kcal_table=WIDE_KCAL_PER_100G, portion_table=WIDE_PORTION_G)
    else:
        items = topk_with_calories(probs, k=3, portion=portion, grams=grams)
    weight_source = items[0]["weight_source"] if items else "portion"
    views = views if views is not None else probs[None, :]
    # The canteen tab is the only one that consults a second head: after the
    # worst-view rule said "one of the 20", the wide head may still object.
    veto = cn_veto(img) if (cn and img is not None) else None
    cn_domain_prob = _domain_prob(cn_frames) if cn else None
    scope = scope_of(views, probs, wide=wide, food_prob=food_prob,
                    packaging_prob=packaging_prob, cn=cn, cn_veto_result=veto,
                    cn_other_prob=cn_other_prob, cn_other_top=cn_other_top,
                    cn_domain_prob=cn_domain_prob)
    in_scope = scope["in_scope"]
    if img is not None and not in_scope and not scope.get("packaging"):
        # No head weights for this run: the edge heuristic is the only cue left.
        pack = packaging.signals(img)
        scope["packaging"] = bool(pack["detected"])
        if pack["detected"]:
            scope["packaging_score"] = pack["score"]
            scope["message"] = packaging.message()
    if not in_scope and scope.get("packaging"):
        # A package is not a plate: say so instead of naming the nearest dish.
        items = []
        weight_source = "portion"
    if in_scope:
        if weight_source == "user":
            note = "热量 = 每 100 g 热量先验 × 你输入的重量（±15% 反映做法差异，低置信度时进一步放宽），仅供参考。"
        else:
            note = "热量区间按常见份量估算（±15%，低置信度时放宽）；填写实际重量可得到更贴合这一份的结果。"
    elif items:
        note = "未识别为支持的菜品：模型给出的只是最接近的候选，类别与热量仅供演示。"
    else:
        note = "包装照不在训练范围里：这不是识别失败，而是这个演示本来就不认识包装食品。"
    for item in items:
        item["brand_refs"] = brands.refs_for_class(item["class"])
    return {
        "source": source,
        "scope": scope,
        "portion": PORTION_LABELS.get(portion, "常规"),
        "grams": items[0]["portion_g"] if items else None,
        "weight_source": weight_source,
        "temperature": _state.get(
            "wide_temperature_fused" if wide else "cn_temperature" if cn else "temperature_fused",
            _state.get("wide_temperature" if wide else "cn_temperature" if cn else "temperature",
                       1.0),
        ),
        "mode": "cn" if cn else "wide" if wide else "single",
        "top3": items,
        "note": note,
    }


@torch.no_grad()
def infer_mixed(img: Image.Image) -> np.ndarray:
    """Per-class sigmoid scores: 1.0 means "this dish is on the tray"."""
    model, device = _state.get("mixed_model"), _state["device"]
    if model is None:
        raise RuntimeError("mixed model is not loaded")
    x = preprocess(img).to(device)
    # same flip-TTA as training/eval: a small side dish is often clearer in one
    # of the two orientations, so keep the max score per class
    batch = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
    probs = torch.sigmoid(model(batch).float()).max(dim=0).values
    return probs.cpu().numpy()


def mixed_payload(probs: np.ndarray, source: str, portion: str = "normal", truth=None, scope=None):
    portion = str(portion or "normal").strip().lower()
    threshold = float(_state.get("mixed_threshold", DETECT_THRESHOLD))
    class_thresholds = _state.get("mixed_class_thresholds") or {}
    parts = detect_parts(
        probs,
        threshold=threshold,
        max_parts=MAX_PARTS,
        class_thresholds=class_thresholds,
        min_prob=MIN_PART_PROB,
    )
    for part in parts:
        part["portion"] = portion
        part["grams"] = None
        part["brand_refs"] = brands.refs_for_class(part["class"])
    plate = plate_totals(parts, portion=portion)
    note = "多标签模型判断盘子里有哪几样菜；每一部分可以单独选份量或填克数，总热量是各部分区间之和。"
    if not parts:
        note = "没有哪一样菜的得分过线：可能是一道菜为主，也可能是清单外的菜。换个角度再拍一张，或切到「单菜识别」。"
    return {
        "mode": "mixed",
        "source": source,
        "threshold": threshold,
        "min_prob": MIN_PART_PROB,
        "class_thresholds": class_thresholds,
        "scope": scope,
        "parts": parts,
        "plate": plate,
        "truth": truth,
        "note": note,
    }


def mixed_truth(name: str):
    """True labels of a baked demo plate, when labels.json is present."""
    for item in _mixed_labels():
        if item["name"] == name:
            return item
    return None


def _mixed_labels() -> List[dict]:
    cache = _state.get("mixed_labels")
    if cache is None:
        path = MIXED_SAMPLES / "labels.json"
        try:
            cache = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        except (OSError, ValueError):
            cache = []
        _state["mixed_labels"] = cache
    return cache


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    load_mixed_model()
    load_wide_model()
    load_cn_model()
    load_domain_head()
    _state["brand_items"] = int(brands.menu().get("count", 0))
    yield


app = FastAPI(title="Campus Canteen Dish Recognition", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

#: Optional access token.  Unset (the default) keeps the old behaviour — anyone
#: who can reach the port may use the demo, which is what a same-Wi-Fi demo
#: wants.  Set FOODCLS_TOKEN=... before opening a public tunnel: the first visit
#: with ?token=... drops a one-week cookie and every later request is checked.
AUTH_TOKEN = os.environ.get("FOODCLS_TOKEN", "").strip()


@app.middleware("http")
async def _require_token(request, call_next):
    if not AUTH_TOKEN:
        return await call_next(request)
    given = (request.headers.get("x-food-token")
             or request.query_params.get("token")
             or request.cookies.get("food_token"))
    if given != AUTH_TOKEN:
        message = "需要访问口令：在网址后面加上 ?token=…"
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": message}, status_code=401)
        return Response(f"<h1>需要访问口令</h1><p>{message}</p>", status_code=401,
                        media_type="text/html; charset=utf-8")
    response = await call_next(request)
    if request.query_params.get("token") and not request.cookies.get("food_token"):
        response.set_cookie("food_token", AUTH_TOKEN, httponly=True, samesite="lax",
                            max_age=7 * 24 * 3600)
    return response


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/manifest.webmanifest", include_in_schema=False)
def web_manifest():
    """PWA manifest: lets the page be added to a phone home screen."""
    return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js", include_in_schema=False)
def service_worker():
    return FileResponse(
        STATIC / "sw.js",
        media_type="text/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-store"},
    )


@lru_cache(maxsize=32)
def _qr_png(url: str) -> bytes:
    import segno

    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="png", scale=8, border=2, dark="#1b2430", light="#ffffff")
    return buf.getvalue()


@app.get("/api/qr.png", include_in_schema=False)
def qr_png(url: str):
    """QR for the desktop page to hand the demo over to a phone."""
    if not url.startswith(("http://", "https://")) or len(url) > 400:
        return JSONResponse({"error": "url must be an http(s) address"}, status_code=400)
    try:
        return Response(_qr_png(url), media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
    except ImportError:
        return JSONResponse({"error": "请先安装 segno：pip install segno"}, status_code=503)


@app.get("/api/brands")
def brand_menu():
    """Official chain-menu nutrition, for the brand reference panel."""
    return brands.menu()


@app.get("/api/classes")
def classes():
    """The 20 dishes the model can name (used by the "what is supported" panel)."""
    return {
        "count": len(CLASS_NAMES),
        "threshold": SCOPE_THRESHOLD,
        "gate": {
            "method": "worst top-1 probability over three crops",
            "threshold": SCOPE_THRESHOLD,
            "false_alarm": 0.05,
        },
        "classes": [
            {
                "class": c,
                "zh": ZH_LABELS[c],
                "kcal_per_100g": KCAL_PER_100G[c],
                "portion_g": PORTION_G[c],
            }
            for c in CLASS_NAMES
        ],
    }


@app.get("/api/samples")
def samples():
    if not SAMPLES.exists():
        return {"samples": []}
    out: List[dict] = []
    for cls_dir in sorted(p for p in SAMPLES.iterdir() if p.is_dir()):
        files = sorted(cls_dir.glob("*.jpg"))[:2]
        for f in files:
            out.append({"name": f.name, "class": cls_dir.name, "zh": ZH_LABELS.get(cls_dir.name, "")})
    return {"samples": out}


def _resolve_in(root: Path, name: str) -> Optional[Path]:
    """Find name as root/<class>/<name> (split layout) or root/<name> (flat)."""
    if not root.exists():
        return None
    flat = (root / name).resolve()
    if flat.parent == root.resolve() and flat.is_file():
        return flat
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        candidate = (sub / name).resolve()
        if candidate.parent == sub.resolve() and candidate.is_file():
            return candidate
    return None


def _find_sample(name: str) -> Optional[Path]:
    """Resolve a gallery file name inside any of the three galleries.

    The name arrives straight from a query string, so ".." or a separator would
    otherwise let a caller read any file the server can read; keep the resolved
    parent inside the split directory.

    Feedback is collected in all three modes, so a mixed-plate or 116-class
    sample name has to resolve too, not just the 20-class gallery.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    #: the 116-class gallery prefixes the class, and its files live in a
    #: per-class directory, so "apple_pie__test_0.jpg" is class + file name
    if "__" in name:
        cls, fname = name.split("__", 1)
        if cls and fname and "/" not in cls and "\\" not in cls:
            candidate = (WIDE_SAMPLES / cls / fname).resolve()
            if candidate.parent == (WIDE_SAMPLES / cls).resolve() and candidate.is_file():
                return candidate
    for root in (SAMPLES, WIDE_SAMPLES, MIXED_SAMPLES):
        hit = _resolve_in(root, name)
        if hit is not None:
            return hit
    return None


@app.get("/api/sample_image")
def sample_image(name: str):
    path = _find_sample(name)
    if path is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/predict_sample")
def predict_sample(name: str, portion: str = "normal", grams: Optional[str] = None):
    path = _find_sample(name)
    if path is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(path, "rb") as fh:
        img = read_image(fh.read())
    views, fused, frames = analyse_full(img)
    gate = _state.get("foodgate")
    food_prob = gate.prob(frames[0]) if gate is not None else None
    pack = _state.get("packaging")
    packaging_prob = pack.prob(frames) if pack is not None else None
    return payload(fused, source=name, portion=portion, grams=grams, views=views, img=img,
                   food_prob=food_prob, packaging_prob=packaging_prob)


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    portion: str = "normal",
    grams: Optional[str] = Form(None),
):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "图片过大，请上传 15 MB 以内的照片"}, status_code=413)
    try:
        img = read_image(raw)
    except ImageTooLarge:
        return JSONResponse({"error": "图片分辨率过高（超过 4000 万像素），请先缩小再上传"}, status_code=413)
    except Exception:
        return JSONResponse({"error": "无法解析这张图片，请换一张 JPG/PNG 照片"}, status_code=400)
    views, fused, frames = await run_in_threadpool(analyse_full, img)
    gate = _state.get("foodgate")
    food_prob = gate.prob(frames[0]) if gate is not None else None
    pack = _state.get("packaging")
    packaging_prob = pack.prob(frames) if pack is not None else None
    return payload(fused, source=file.filename or "upload", portion=portion, grams=grams,
                   views=views, img=img, food_prob=food_prob, packaging_prob=packaging_prob)


# --------------------------------------------------------------------------- #
# Feedback: "the answer is wrong, it was actually ..." kept as a labelled set
# --------------------------------------------------------------------------- #
#: Corrections are the cheapest labelled data in the domain we are worst at: the
#: photo is a real canteen tray and the user names the dish.  One CSV row plus the
#: image per submission; nothing is trained on it automatically.
FEEDBACK_DIR = ROOT / "data/splits/feedback"
FEEDBACK_CSV = FEEDBACK_DIR / "labels.csv"
FEEDBACK_SPECIAL = {"not_a_dish": "照片里没有菜", "not_sure": "我也说不准"}
#: A correction can name any class any of the three models knows, not just the 20.
FEEDBACK_CLASSES = set(CLASS_NAMES) | set(WIDE_CLASS_NAMES) | set(FEEDBACK_SPECIAL)
FEEDBACK_COLUMNS = ["timestamp", "file", "predicted", "corrected", "removed", "note", "source", "mode"]


def parse_correction(raw: str, allow_empty: bool = False) -> List[str]:
    """Split "a;b;c" into class names, rejecting anything no model knows.

    A tray names several dishes at once, so the field is a semicolon list; the
    single-class case is just a one-element list.  The strike-out column may be
    empty (most corrections only add dishes), hence the allow_empty flag.
    """
    parts = [p.strip() for p in (raw or "").split(";") if p.strip()]
    if not parts:
        if allow_empty:
            return []
        raise ValueError("请先选择正确的类别")
    if len(parts) > 6:
        raise ValueError("最多标记 6 样菜")
    unknown = [p for p in parts if p not in FEEDBACK_CLASSES]
    if unknown:
        raise ValueError("未知的类别：" + "、".join(unknown[:3]))
    return parts


@app.get("/api/feedback")
def feedback_stats():
    """How many corrections have been collected so far."""
    if not FEEDBACK_CSV.is_file():
        return {"n": 0, "path": "data/splits/feedback"}
    by_mode: Dict[str, int] = {}
    strikes = 0
    with FEEDBACK_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("mode") or "single").strip() or "single"
            by_mode[key] = by_mode.get(key, 0) + 1
            if (row.get("removed") or "").strip():
                strikes += 1
    return {
        "n": sum(by_mode.values()),
        "by_mode": by_mode,
        "with_removal": strikes,
        "path": "data/splits/feedback",
    }


@app.post("/api/feedback")
async def feedback(
    file: Optional[UploadFile] = File(None),
    sample: str = Form(""),
    predicted: str = Form(""),
    corrected: str = Form(""),
    removed: str = Form(""),
    note: str = Form(""),
    mode: str = Form("single"),
):
    """Store one correction: the photo, what we said, and what it really was.

    On a tray the user can also strike out a dish we guessed that is not there;
    the removed column holds those names, and a submission may consist of
    strike-outs alone when every guess was wrong.
    """
    try:
        parts = parse_correction(corrected, allow_empty=True)
        dropped = parse_correction(removed, allow_empty=True)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not parts and not dropped:
        return JSONResponse(
            {"error": "请先选出正确的菜，或者点一下预测里其实没有的菜把它划掉"},
            status_code=400,
        )
    corrected = ";".join(parts)
    removed = ";".join(dropped)
    mode = mode if mode in ("single", "wide", "mixed") else "single"

    upload = file is not None and bool(file.filename)
    raw = b""
    if upload:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "图片过大，请上传 15 MB 以内的照片"}, status_code=413)
    elif sample:
        path = _find_sample(sample)
        if path is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        raw = path.read_bytes()
    if not raw:
        return JSONResponse({"error": "缺少图片：请重新选择一张照片"}, status_code=400)

    try:
        img = read_image(raw)
    except ImageTooLarge:
        return JSONResponse({"error": "图片分辨率过高（超过 4000 万像素），请先缩小再上传"}, status_code=413)
    except Exception:
        return JSONResponse({"error": "无法解析这张图片，请换一张 JPG/PNG 照片"}, status_code=400)

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    if FEEDBACK_CSV.is_file():
        with FEEDBACK_CSV.open("r", encoding="utf-8", newline="") as fh:
            head = fh.readline().strip()
        if [c.strip() for c in head.split(",")] != FEEDBACK_COLUMNS:
            #: an older layout: keep it beside the new file rather than appending
            #: rows with a different number of columns
            legacy = FEEDBACK_DIR / f"labels_legacy_{datetime.now():%Y%m%d-%H%M%S}.csv"
            FEEDBACK_CSV.replace(legacy)
    digest = hashlib.sha1(raw).hexdigest()[:10]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #: the corrected class in the name keeps the folder browsable without the CSV
    if parts:
        slug = parts[0] if len(parts) == 1 else f"{parts[0]}+{len(parts) - 1}"
    else:
        slug = f"minus_{dropped[0]}" if len(dropped) == 1 else f"minus_{dropped[0]}+{len(dropped) - 1}"
    name = f"{stamp}_{slug}_{digest}.jpg"
    img.convert("RGB").save(FEEDBACK_DIR / name, "JPEG", quality=90)

    fresh = not FEEDBACK_CSV.is_file()
    with FEEDBACK_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if fresh:
            writer.writerow(FEEDBACK_COLUMNS)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            name,
            predicted.strip()[:80],
            corrected,
            removed,
            note.strip().replace("\n", " ")[:200],
            "upload" if upload else "sample",
            mode,
        ])
    return {"ok": True, "saved": name, "corrected": corrected, "removed": removed, "mode": mode}


# --------------------------------------------------------------------------- #
# Wide head: Food-101 classes plus external Chinese dishes
# --------------------------------------------------------------------------- #
def _wide_sample_name(cls: str, fname: str) -> str:
    return f"{cls}__{fname}"


def _find_wide_sample(name: str) -> Optional[Path]:
    if "__" not in name or not WIDE_SAMPLES.exists():
        return None
    cls, fname = name.split("__", 1)
    if cls not in WIDE_CLASS_NAMES or "/" in fname or "\\" in fname:
        return None
    path = WIDE_SAMPLES / cls / fname
    return path if path.is_file() else None


@app.get("/api/wide/classes")
def wide_classes():
    """All 116 names, for the "what is supported" panel of the wide mode."""
    return {
        "count": WIDE_NUM_CLASSES,
        "threshold": WIDE_SCOPE_THRESHOLD,
        "enabled": _state.get("wide_model") is not None,
        "classes": [
            {"class": c, "zh": WIDE_ZH[c], "kcal_per_100g": WIDE_KCAL_PER_100G[c],
             "portion_g": WIDE_PORTION_G[c]}
            for c in WIDE_CLASS_NAMES
        ],
    }


@app.get("/api/wide/samples")
def wide_samples():
    if not WIDE_SAMPLES.exists():
        return {"samples": [], "enabled": _state.get("wide_model") is not None}
    out: List[dict] = []
    for idx, cls in enumerate(WIDE_CLASS_NAMES):
        if idx % 2:                     # every other class keeps the gallery short
            continue
        for f_ in sorted((WIDE_SAMPLES / cls).glob("*.jpg"))[:1]:
            out.append({"name": _wide_sample_name(cls, f_.name), "class": cls, "zh": WIDE_ZH[cls]})
    return {"samples": out, "enabled": _state.get("wide_model") is not None}


@app.get("/api/wide/sample_image")
def wide_sample_image(name: str):
    path = _find_wide_sample(name)
    if path is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/wide/predict_sample")
def wide_predict_sample(name: str, portion: str = "normal", grams: Optional[str] = None):
    if _state.get("wide_model") is None:
        return JSONResponse({"error": f"{WIDE_NUM_CLASSES} 类模型尚未训练完成"}, status_code=503)
    path = _find_wide_sample(name)
    if path is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(path, "rb") as fh:
        img = read_image(fh.read())
    views, fused = analyse_wide(img)
    return payload(fused, source=name, portion=portion, grams=grams, views=views, wide=True, img=img)


@app.post("/api/wide/predict")
async def wide_predict(file: UploadFile = File(...), portion: str = "normal", grams: Optional[str] = Form(None)):
    if _state.get("wide_model") is None:
        return JSONResponse({"error": f"{WIDE_NUM_CLASSES} 类模型尚未训练完成"}, status_code=503)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "图片过大，请上传 15 MB 以内的照片"}, status_code=413)
    try:
        img = read_image(raw)
    except ImageTooLarge:
        return JSONResponse({"error": "图片分辨率过高（超过 4000 万像素），请先缩小再上传"}, status_code=413)
    except Exception:
        return JSONResponse({"error": "无法解析这张图片，请换一张 JPG/PNG 照片"}, status_code=400)
    views, fused = await run_in_threadpool(analyse_wide, img)
    return payload(fused, source=file.filename or "upload", portion=portion, grams=grams,
                   views=views, wide=True, img=img)


# --------------------------------------------------------------------------- #
# Canteen head: 20 everyday Chinese dishes (CNFOOD-241 subset)
# --------------------------------------------------------------------------- #
def _sample_name(cls: str, fname: str) -> str:
    return f"{cls}__{fname}"


def _find_cn_sample(name: str) -> Optional[Path]:
    """Look up a gallery image by "class__file.jpg", refusing any path escape."""
    if "__" not in name or not CN_SAMPLES.exists():
        return None
    cls, fname = name.split("__", 1)
    if cls not in CN_CLASS_NAMES or "/" in fname or "\\" in fname:
        return None
    path = CN_SAMPLES / cls / fname
    return path if path.is_file() else None


@app.get("/api/cn/classes")
def cn_classes():
    return {
        "count": CN_NUM_CLASSES,
        "threshold": (_state.get("cn_scope_threshold") or CN_SCOPE_THRESHOLD),
        "other": {
            "enabled": _state.get("cn_other_index") is not None,
            "zh": CN_OTHER_ZH,
            "threshold": CN_OTHER_THRESHOLD,
        },
        "enabled": _state.get("cn_model") is not None,
        "classes": [
            {"class": c, "zh": CN_ZH[c], "kcal_per_100g": CN_KCAL_PER_100G[c],
             "portion_g": CN_PORTION_G[c]}
            for c in CN_CLASS_NAMES
        ],
    }


@app.get("/api/cn/samples")
def cn_samples():
    """One test photo per dish, so the mode can be tried without a camera."""
    if not CN_SAMPLES.exists():
        return {"samples": [], "enabled": _state.get("cn_model") is not None}
    out: List[dict] = []
    for cls in CN_CLASS_NAMES:
        for f_ in sorted((CN_SAMPLES / cls).glob("*.jpg"))[:1]:
            out.append({"name": _sample_name(cls, f_.name), "class": cls, "zh": CN_ZH[cls]})
    return {"samples": out, "enabled": _state.get("cn_model") is not None}


@app.get("/api/cn/sample_image")
def cn_sample_image(name: str):
    path = _find_cn_sample(name)
    if path is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/cn/predict_sample")
def cn_predict_sample(name: str, portion: str = "normal", grams: Optional[str] = None):
    if _state.get("cn_model") is None:
        return JSONResponse({"error": f"{CN_NUM_CLASSES} 道食堂菜模型尚未训练完成"}, status_code=503)
    path = _find_cn_sample(name)
    if path is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(path, "rb") as fh:
        img = read_image(fh.read())
    views, fused, frames = analyse_cn(img, want_feats=True)
    return payload(fused, source=name, portion=portion, grams=grams, views=views, cn=True,
                   img=img, cn_frames=frames)


@app.post("/api/cn/predict")
async def cn_predict(file: UploadFile = File(...), portion: str = "normal",
                     grams: Optional[str] = Form(None)):
    if _state.get("cn_model") is None:
        return JSONResponse({"error": f"{CN_NUM_CLASSES} 道食堂菜模型尚未训练完成"}, status_code=503)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "图片过大，请上传 15 MB 以内的照片"}, status_code=413)
    try:
        img = read_image(raw)
    except ImageTooLarge:
        return JSONResponse({"error": "图片分辨率过高（超过 4000 万像素），请先缩小再上传"}, status_code=413)
    except Exception:
        return JSONResponse({"error": "无法解析这张图片，请换一张 JPG/PNG 照片"}, status_code=400)
    views, fused, frames = await run_in_threadpool(analyse_cn, img, True)
    return payload(fused, source=file.filename or "upload", portion=portion, grams=grams,
                   views=views, cn=True, img=img, cn_frames=frames)


def _packaging_scope(img: Image.Image) -> dict:
    """Scope of the single-dish head, plus the packaging cue for the tray view."""
    views, fused, frames = analyse_full(img)
    pack = _state.get("packaging")
    scope = scope_of(views, fused, packaging_prob=pack.prob(frames) if pack is not None else None)
    if not scope["in_scope"] and not scope.get("packaging") and packaging.signals(img)["detected"]:
        scope["packaging"] = True
        scope["message"] = packaging.message()
    return scope


@app.get("/api/mixed/samples")
def mixed_samples():
    items = [
        {"name": it["name"], "classes": it.get("classes", []), "zh": it.get("zh", [])}
        for it in _mixed_labels()
    ]
    return {"samples": items, "enabled": _state.get("mixed_model") is not None}


@app.get("/api/mixed/sample_image")
def mixed_sample_image(name: str):
    path = (MIXED_SAMPLES / name).resolve()
    if path.parent != MIXED_SAMPLES.resolve() or not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/mixed/predict_sample")
def mixed_predict_sample(name: str, portion: str = "normal"):
    path = (MIXED_SAMPLES / name).resolve()
    if path.parent != MIXED_SAMPLES.resolve() or not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    if _state.get("mixed_model") is None:
        return JSONResponse({"error": "混合模型尚未训练完成"}, status_code=503)
    with open(path, "rb") as fh:
        img = read_image(fh.read())
    return mixed_payload(infer_mixed(img), source=name, portion=portion,
                         truth=mixed_truth(name), scope=_packaging_scope(img))


@app.post("/api/mixed/predict")
async def mixed_predict(file: UploadFile = File(...), portion: str = "normal"):
    if _state.get("mixed_model") is None:
        return JSONResponse({"error": "混合模型尚未训练完成"}, status_code=503)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "图片过大，请上传 15 MB 以内的照片"}, status_code=413)
    try:
        img = read_image(raw)
    except ImageTooLarge:
        return JSONResponse({"error": "图片分辨率过高（超过 4000 万像素），请先缩小再上传"}, status_code=413)
    except Exception:
        return JSONResponse({"error": "无法解析这张图片，请换一张 JPG/PNG 照片"}, status_code=400)
    parts = await run_in_threadpool(infer_mixed, img)
    return mixed_payload(parts, source=file.filename or "upload", portion=portion,
                         scope=_packaging_scope(img))


@app.post("/api/mixed/total")
async def mixed_total(body: dict):
    """Recompute the per-part and total kcal after the user edits weights."""
    default_portion = str(body.get("portion") or "normal")
    parts = body.get("parts")
    if not isinstance(parts, list):
        return JSONResponse({"error": "parts 必须是数组"}, status_code=400)
    rows = []
    for item in parts:
        if not isinstance(item, dict):
            continue
        name = str(item.get("class", ""))
        if name not in CLASS_NAMES:
            continue
        try:
            prob = float(item.get("prob", 1.0))
        except (TypeError, ValueError):
            prob = 1.0
        rows.append(
            {
                "class": name,
                "prob": prob,
                "portion": item.get("portion") or default_portion,
                "grams": item.get("grams"),
            }
        )
    if not rows:
        return JSONResponse({"error": "没有可计算的菜品"}, status_code=400)
    return plate_totals(rows, portion=default_portion)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "arch": _state.get("arch"),
        "checkpoint": CKPT.name,
        "temperature": _state.get("temperature", 1.0),
        "temperature_fused": _state.get("temperature_fused", _state.get("temperature", 1.0)),
        "views": len(_VIEW_TFS) or 6,
        "mixed_enabled": _state.get("mixed_model") is not None,
        "mixed_threshold": _state.get("mixed_threshold", DETECT_THRESHOLD),
        "wide_enabled": _state.get("wide_model") is not None,
        "wide_arch": _state.get("wide_arch"),
        "wide_classes": WIDE_NUM_CLASSES,
        "wide_temperature": _state.get("wide_temperature", 1.0),
        "wide_fuse": _state.get("wide_fuse", 0.0),
        "cn_enabled": _state.get("cn_model") is not None,
        "cn_arch": _state.get("cn_arch"),
        "cn_classes": CN_NUM_CLASSES,
        "cn_temperature": _state.get("cn_temperature", 1.0),
        "cn_threshold": (_state.get("cn_scope_threshold") or CN_SCOPE_THRESHOLD),
        "cn_other": {
            "enabled": _state.get("cn_other_index") is not None,
            "zh": CN_OTHER_ZH,
            "threshold": CN_OTHER_THRESHOLD,
            "gate_threshold": CN_SCOPE_THRESHOLD_21,
        },
        "domain": {
            "enabled": _state.get("domain_head") is not None,
            "zh": DOMAIN_ZH,
            "threshold": (_state.get("domain_head") or {}).get("threshold"),
        },
        "cn_veto": {
            "enabled": _state.get("wide_model") is not None,
            "threshold": CN_VETO_THRESHOLD,
            "margin": CN_VETO_MARGIN,
            "classes": len(CN_VETO_CLASSES),
        },
        "brand_items": _state.get("brand_items", 0),
        "food_gate": {
            "enabled": _state.get("foodgate") is not None,
            "threshold": FOODGATE_THRESHOLD,
            "path": FOODGATE_PATH.name,
        },
        "packaging_head": {
            "enabled": _state.get("packaging") is not None,
            "threshold": (_state["packaging"].threshold
                          if _state.get("packaging") is not None else PACKAGING_THRESHOLD),
            "path": PACKAGING_PATH.name,
        },
    }


def _lan_urls(port: int) -> list[str]:
    """Every non-loopback IPv4 address that a phone on the same Wi-Fi can reach."""
    import socket

    urls = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith(("127.", "169.254.")) and f"http://{ip}:{port}" not in urls:
                urls.append(f"http://{ip}:{port}")
    except OSError:
        pass
    return urls


if __name__ == "__main__":
    host = os.environ.get("FOODCLS_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print("=" * 60, flush=True)
    print(f"  Food demo  ->  http://127.0.0.1:{port}", flush=True)
    for url in _lan_urls(port):
        print(f"  Same Wi-Fi ->  {url}", flush=True)
    print("  Stop with Ctrl+C", flush=True)
    print("=" * 60, flush=True)
    uvicorn.run(
        "foodcls.app:app",
        host=host,
        port=port,
        app_dir=str(Path(__file__).resolve().parents[1]),
    )

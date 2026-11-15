"""FastAPI web demo: photo -> Top-3 dishes + calorie range."""
from __future__ import annotations

import io
import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

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
    print(f"loaded wide {ckpt['arch']} ({WIDE_NUM_CLASSES} classes) from {WIDE_CKPT} on {device}",
          flush=True)


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


def analyse_wide(img: Image.Image):
    model = _state.get("wide_model")
    if model is None:
        raise RuntimeError("wide model is not loaded")
    return _view_probs(model, _state.get("wide_temperature", 1.0), img,
                       _state.get("wide_temperature_fused"))


@torch.no_grad()
def infer(img: Image.Image) -> np.ndarray:
    """Fused posterior: the six views averaged in logit space, then calibrated."""
    return analyse(img)[1]


def scope_of(views: np.ndarray, probs: np.ndarray, wide: bool = False,
             food_prob: float | None = None,
             packaging_prob: float | None = None) -> dict:
    """Is this photo one of the supported dishes?

    The reported confidence is the fused posterior; the decision still uses the
    worst single view, because one frame that looks like another dish is exactly
    the failure this flag exists to catch.
    """
    if wide:
        names, zh, threshold = WIDE_CLASS_NAMES, WIDE_ZH, WIDE_SCOPE_THRESHOLD
        label = f"{WIDE_NUM_CLASSES} 类"
        examples = f"火锅、麻辣烫、家常炒菜等中式吃法都不在这 {WIDE_NUM_CLASSES} 类里"
    else:
        names, zh, threshold, label = CLASS_NAMES, ZH_LABELS, SCOPE_THRESHOLD, "20 类"
        examples = "整只烤鸡/烤鸭、火锅、家常炒菜等都不在支持的 20 类里"
    top1 = float(np.max(probs))
    # Mirrored views sharpen the fused posterior but lower the worst-view score
    # on every photo, so the gate keeps the three original framings only; that
    # is the calibration the 0.32 / 0.18 thresholds were measured on.
    gate_views = views[:_GATE_VIEWS] if len(views) > _GATE_VIEWS else views
    robust = float(min(np.max(v) for v in gate_views))
    # Two questions, two answers: the gate asks "is there food here at all?",
    # the robust score asks "is it one of the supported dishes?".  A menu board
    # clears neither, but it does clear the second one often enough to matter.
    gate = _state.get("foodgate")
    food_ok = True
    if gate is not None and food_prob is not None and not wide:
        food_ok = bool(food_prob >= gate.threshold)
    # A package prints a picture of a dish: the 20-class head is confident and
    # the food gate agrees, so neither of them can catch it.  This is a veto --
    # but only over photos the gate called food.  When the gate already said
    # "no food here", its own message is the accurate one (a napkin is not a
    # product package), and 43 of the 46 flagged negatives were exactly that.
    pack_head = _state.get("packaging")
    package = False
    if pack_head is not None and packaging_prob is not None and not wide and food_ok:
        package = bool(packaging_prob >= pack_head.threshold)
    in_scope = bool(robust >= threshold) and food_ok and not package
    closest = zh[names[int(np.argmax(probs))]]
    scope = {
        "in_scope": in_scope,
        "top1_prob": round(top1, 4),
        "robust_prob": round(robust, 4),
        "threshold": threshold,
        "closest": closest,
        "label": label,
    }
    if food_prob is not None and not wide:
        scope["food_prob"] = round(float(food_prob), 4)
        scope["food_threshold"] = gate.threshold if gate is not None else None
    if packaging_prob is not None and not wide:
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
            packaging_prob: float | None = None):
    portion = str(portion or "normal").strip().lower()
    if wide:
        items = topk_with_calories(probs, k=3, portion=portion, grams=grams,
                                   class_names=WIDE_CLASS_NAMES, zh_table=WIDE_ZH,
                                   kcal_table=WIDE_KCAL_PER_100G, portion_table=WIDE_PORTION_G)
    else:
        items = topk_with_calories(probs, k=3, portion=portion, grams=grams)
    weight_source = items[0]["weight_source"] if items else "portion"
    views = views if views is not None else probs[None, :]
    scope = scope_of(views, probs, wide=wide, food_prob=food_prob,
                    packaging_prob=packaging_prob)
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
            "wide_temperature_fused" if wide else "temperature_fused",
            _state.get("wide_temperature" if wide else "temperature", 1.0),
        ),
        "mode": "wide" if wide else "single",
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


def _find_sample(name: str) -> Optional[Path]:
    """Resolve a gallery file name inside SAMPLES only.

    The name arrives straight from a query string, so ".." or a separator would
    otherwise let a caller read any file the server can read; keep the resolved
    parent inside the split directory.
    """
    if not SAMPLES.exists() or not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    for cls_dir in sorted(p for p in SAMPLES.iterdir() if p.is_dir()):
        candidate = (cls_dir / name).resolve()
        if candidate.parent == cls_dir.resolve() and candidate.is_file():
            return candidate
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

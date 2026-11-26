"""FastAPI web demo: photo -> Top-3 dishes + calorie range."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import time
from datetime import datetime
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
    CN_MENU_KCAL_PER_100G,
    CN_MENU_NAMES,
    CN_MENU_PORTION_G,
    CN_MENU_ZH,
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
from .plate_classes import (
    PLATE_CLASS_NAMES,
    PLATE_KCAL_PER_100G,
    PLATE_PORTION_G,
    PLATE_ZH,
)
from .foodgate import FoodGate
from .packaging_head import PackagingHead, message as packaging_head_message
from . import packaging
from .utils import pick_device

ROOT = Path(__file__).resolve().parents[2]
#: The deployed 20-class backbone.  It is the Food-101N clean-label run, not the
#: plain Food-101 one, because the food gate and the packaging head were fitted
#: on its pooled features and lose most of their power on another backbone
#: (measured end to end: 2/123 non-food photos accepted here, 43/123 with the
#: same two gates on the Food-101 run).  Pointing FOODCLS_CKPT at a different
#: run without also pointing FOODCLS_GATE / FOODCLS_PACK at matching files
#: silently turns both heads off -- /api/health shows enabled:false when it
#: happens.
DEFAULT_CKPT = ROOT / "results/runs/resnet18_last_stage_aug_f101n/best.pt"
CKPT = Path(os.environ.get("FOODCLS_CKPT", DEFAULT_CKPT))
if not CKPT.is_absolute():
    CKPT = (ROOT / CKPT).resolve()

#: optional multi-label head for mixed trays; the single-dish demo works without it
DEFAULT_MIXED_CKPT = ROOT / "results/runs/resnet18_multilabel_plate40/best.pt"
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
#: This is the second-generation head: a 241-class CNFOOD-241 backbone pretrained
#: on 153,746 images and then finetuned on the 21-way split (README 5.7.3); test
#: top-1 79.83 % against 75.40 % for the f101n-initialised one.
DEFAULT_CN_CKPT = ROOT / "results/runs/cn241_22_pre/best.pt"
CN_CKPT = Path(os.environ.get("FOODCLS_CN_CKPT", DEFAULT_CN_CKPT))
if not CN_CKPT.is_absolute():
    CN_CKPT = (ROOT / CN_CKPT).resolve()
CN_SAMPLES = ROOT / "data/splits/cn241_full/test"
MIXED_SAMPLES = ROOT / "data/splits/mixed"

#: The single-dish tab serves a five-run ensemble, not one checkpoint.  All five
#: are the same architecture and label space (20 Food-101 classes); their logits
#: are averaged and calibrated once with the temperature in
#: results/tables/ensemble_calibration.json.  Measured on the 5,000-image test
#: split: 94.40 % top-1 against 93.40 % for the deployed run alone, ECE 1.52 %
#: after calibration, and 18.7 ms for five forward passes on the laptop GPU --
#: nothing next to the image decode.  The gates (food / packaging / cuisine) and
#: the wide-head fusion still run on the deployed run's features, because they
#: were fitted on that backbone.  FOODCLS_ENSEMBLE=0 restores the single model.
ENSEMBLE_RUNS = (
    "resnet18_last_stage_aug_f101n",
    "resnet18_last_stage_aug",
    "resnet18_last_stage_aug_f101n_nodedup",
    "resnet18_last_stage_aug_f101n_e24",
    "resnet18_last_stage_aug_seed7",
)
ENSEMBLE_TABLE = ROOT / "results/tables/ensemble_calibration.json"
ENSEMBLE = os.environ.get("FOODCLS_ENSEMBLE", "1").strip().lower() not in {
    "0", "off", "false", "no"}

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
#: Rice ships as a dish of its own only in the 22-class canteen checkpoint; see
#: src/foodcls/classes_cn.py for why it is kept out of CLASS_SPEC_CN.
CN_RICE_SLUG = "rice"
CN_OTHER_THRESHOLD = float(os.environ.get("FOODCLS_CN_OTHER", "0.40"))

#: The bucket steals probability mass from the 20 dishes, so the gate is measured
#: on the dish mass after dividing it back out: per view p_dish / (1 - p_other),
#: worst view, 5th percentile of the in-scope validation split = 0.4088 with the
#: 241-class-pretrained backbone (0.3445 with the f101n-initialised one;
#: scripts/search_cn_other_rule.py re-measures both).  With the 20-dish
#: checkpoint the divisor is 1 and the threshold stays CN_SCOPE_THRESHOLD, so old
#: weights behave as before.
CN_SCOPE_THRESHOLD_21 = float(os.environ.get("FOODCLS_CN_SCOPE_21", "0.4088"))

#: One gate for every photo leaves ~1.5 pt on the table.  The two questions are
#: not independent: when the domain probe is certain the photo is canteen food,
#: a dish score just under the gate is still worth a name instead of a refusal.
#: "0.90:0.36" loosens the gate to 0.36 above 0.90 domain probability; "0" turns
#: it off.  Tuned on data/splits/cn241_full/val and reported on test by
#: scripts/eval_domain_gate.py: in-scope +1.46 pt for +0.87 pt of the 29 held-out
#: Chinese classes, with the other three out-of-scope columns flat.
CN_SCOPE_TRUST = os.environ.get("FOODCLS_CN_TRUST", "0.90:0.36")
_TRUST_AT, _TRUST_GATE = 1.01, 9.99
if CN_SCOPE_TRUST.strip() not in ("", "0"):
    _TRUST_AT, _TRUST_GATE = (float(x) for x in CN_SCOPE_TRUST.split(":"))

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
#: where the gate pair belonging to the deployed backbone lives; quoted in the
#: start-up warning so a missing file is a one-line fix, not a silent loss
GATE_HOME = "results/runs/resnet18_last_stage_aug_f101n"

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


def _load_ensemble_temperature() -> float:
    """One temperature for the averaged logits (results/tables/ensemble_calibration.json).

    It is lower than any member's own temperature because averaging shrinks the
    logit scale: the ensemble is calibrated once, after the mean, never per member.
    """
    try:
        data = json.loads(ENSEMBLE_TABLE.read_text(encoding="utf-8"))
        t = float(data["temperature"])
        if 0.05 <= t <= 20.0:
            return t
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return 1.0


def _load_member(path: Path, device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    _check_class_names(ckpt, CLASS_NAMES, path)
    model = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


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
    _state["ensemble"] = None
    _state["ensemble_runs"] = []
    if ENSEMBLE:
        # The deployed run is always member #0 (it is the one whose features the
        # gates use), so a missing sibling can shrink the ensemble to one member
        # without changing which forward pass produces the pooled feature.
        members = [model]
        runs = [CKPT.parent.name]
        for run in ENSEMBLE_RUNS:
            path = ROOT / "results/runs" / run / "best.pt"
            if path == CKPT or not path.is_file():
                continue
            try:
                members.append(_load_member(path, device))
                runs.append(run)
            except (OSError, RuntimeError, KeyError) as exc:
                print(f"ensemble member {run} skipped: {exc}", flush=True)
        if len(members) > 1:
            _state["ensemble"] = members
            _state["ensemble_runs"] = runs
            _state["temperature"] = _load_ensemble_temperature()
            _state["temperature_fused"] = _state["temperature"]
            print(f"ensemble: {len(members)} members ({', '.join(runs)}) "
                  f"at temperature {_state['temperature']:.2f}", flush=True)
    _state["foodgate"] = None
    if FOODGATE_PATH.is_file():
        try:
            _state["foodgate"] = FoodGate(FOODGATE_PATH, FOODGATE_THRESHOLD)
            print(f"loaded food gate (p_food >= {FOODGATE_THRESHOLD:.2f}) from {FOODGATE_PATH}",
                  flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"food gate disabled: {exc}", flush=True)
    else:
        print(f"no food gate at {FOODGATE_PATH}; relying on the view-robust rule "
              f"only (this backbone's gate pair lives in {GATE_HOME}/)", flush=True)
    _state["packaging"] = None
    if PACKAGING_PATH.is_file():
        try:
            _state["packaging"] = PackagingHead(PACKAGING_PATH, PACKAGING_THRESHOLD)
            print(f"loaded packaging head (p_pack >= {_state['packaging'].threshold:.2f}) "
                  f"from {PACKAGING_PATH}", flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"packaging head disabled: {exc}", flush=True)
    else:
        print(f"no packaging head at {PACKAGING_PATH}; using the edge heuristic "
              f"only (this backbone's gate pair lives in {GATE_HOME}/)", flush=True)


def _mixed_tables(ckpt):
    """Class order and lookup tables for the tray head (20 or 40 classes)."""
    names = list(ckpt.get("class_names") or CLASS_NAMES)
    if names == list(CLASS_NAMES):
        return names, ZH_LABELS, KCAL_PER_100G, PORTION_G
    if names == list(PLATE_CLASS_NAMES):
        return names, PLATE_ZH, PLATE_KCAL_PER_100G, PLATE_PORTION_G
    raise RuntimeError(
        f"mixed checkpoint {MIXED_CKPT.name} has {len(names)} classes, which match "
        f"neither the {len(CLASS_NAMES)} western dishes nor those plus the canteen "
        "menu; refusing to load"
    )


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
    names, zh, kcal, grams_table = _mixed_tables(ckpt)
    model = build_model(ckpt["arch"], len(names), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    _state["mixed_model"] = model
    _state["mixed_names"] = names
    _state["mixed_zh"] = zh
    _state["mixed_kcal"] = kcal
    _state["mixed_portion"] = grams_table
    _state["mixed_threshold"] = float(ckpt.get("threshold", DETECT_THRESHOLD))
    table = ckpt.get("class_thresholds") or {}
    _state["mixed_class_thresholds"] = {str(k): float(v) for k, v in table.items()}
    cuisines = (
        f"{len(CLASS_NAMES)} western + {len(names) - len(CLASS_NAMES)} canteen"
        if len(names) > len(CLASS_NAMES) else "western only"
    )
    print(
        f"loaded mixed model {ckpt['arch']} from {MIXED_CKPT} "
        f"({len(names)} classes, {cuisines}; threshold {_state['mixed_threshold']:.2f}, "
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
    # Three class lists are accepted; the difference is only whether the bucket
    # (and rice) have their own output units:
    #   20  the dishes alone (the pre-bucket checkpoint);
    #   21  the dishes plus the 'other Chinese' bucket;
    #   22  the dishes plus rice plus the bucket, i.e. the 22-class head where
    #       rice is a named dish instead of being poured into the bucket.
    # Everything downstream reads the *dishes* count from _state, never
    # CN_NUM_CLASSES, so a 22-class checkpoint names rice like any other dish.
    has_rice = (len(names) == CN_NUM_CLASSES + 2
                and names[:CN_NUM_CLASSES] == list(CN_CLASS_NAMES)
                and names[-2] == CN_RICE_SLUG
                and names[-1] == CN_OTHER_SLUG)
    has_other = has_rice or (len(names) == CN_NUM_CLASSES + 1
                             and names[:CN_NUM_CLASSES] == list(CN_CLASS_NAMES)
                             and names[-1] == CN_OTHER_SLUG)
    if list(names) != list(CN_CLASS_NAMES) and not has_other:
        raise RuntimeError(
            f"{CN_CKPT.name}: checkpoint class_names ({len(names)}) match neither the "
            f"{CN_NUM_CLASSES} dishes, nor those plus '{CN_OTHER_SLUG}', nor those plus "
            f"'{CN_RICE_SLUG}' and '{CN_OTHER_SLUG}'; refusing to load"
        )
    model = build_model(ckpt["arch"], len(names), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    _state["cn_model"] = model
    _state["cn_arch"] = ckpt["arch"]
    #: the dish table follows the checkpoint: rice is a dish only in the 22-class
    #: head, and the calorie/portion tables have to grow with it.
    if has_rice:
        _state["cn_names"] = CN_MENU_NAMES
        _state["cn_zh"] = CN_MENU_ZH
        _state["cn_kcal"] = CN_MENU_KCAL_PER_100G
        _state["cn_portion"] = CN_MENU_PORTION_G
    else:
        _state["cn_names"] = list(CN_CLASS_NAMES)
        _state["cn_zh"] = CN_ZH
        _state["cn_kcal"] = CN_KCAL_PER_100G
        _state["cn_portion"] = CN_PORTION_G
    _state["cn_dish_count"] = len(_state["cn_names"])
    _state["cn_other_index"] = len(names) - 1 if has_other else None
    #: the gate is re-measured on the 21-class checkpoint (see CN_SCOPE_THRESHOLD_21)
    _state["cn_scope_threshold"] = CN_SCOPE_THRESHOLD_21 if has_other else CN_SCOPE_THRESHOLD
    _state["cn_temperature"] = _load_temperature(CN_CKPT)
    print(f"loaded canteen {ckpt['arch']} ({len(names)} classes"
          + (", incl. rice" if has_rice else "")
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


#: iPhones upload HEIC by default and Pillow cannot open it on its own.  The
#: plugin registers a decoder for the HEIF family the first time it is imported;
#: without it every HEIC upload would come back as "unreadable image".
def _register_heif() -> bool:
    try:
        import pillow_heif
    except ImportError:
        return False
    pillow_heif.register_heif_opener()
    return True


HEIF_OK = _register_heif()
#: The 400 message names only the formats this build can actually decode.
UNREADABLE_IMAGE = ("无法解析这张图片，请换一张 JPG/PNG"
                   + ("/HEIC 照片" if HEIF_OK else " 照片"))


def read_image(raw: bytes) -> Image.Image:
    """Decode uploaded bytes, honouring the EXIF orientation of phone photos."""
    img = Image.open(io.BytesIO(raw))
    width, height = img.size
    if width * height > MAX_PIXELS:
        raise ImageTooLarge(f"{width}x{height}")
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _narrow_models():
    """The single-dish head as a list: the ensemble when loaded, else one model.

    Member #0 is always the deployed Food-101N run, the one the gate heads were
    fitted on, so the pooled feature below never changes owner.
    """
    members = _state.get("ensemble")
    if members:
        return members
    model = _state.get("model")
    return [model] if model is not None else []


@torch.no_grad()
def _view_probs(model, temperature: float, img: Image.Image,
                fused_temperature: float | None = None, want_feats: bool = False):
    """Per-view probabilities and the fused posterior.

    Views are fused in logit space and calibrated once afterwards: averaging
    probabilities that were each sharpened by their own temperature gives a
    worse posterior than fusing the logits (93.1 vs 93.4 on test, 71.2 vs 73.0
    on the proxy for E7).  The per-view probabilities keep their own temperature
    because the out-of-scope gate is calibrated on them.

    A model may be one network or a list of them (the single-dish ensemble).
    Members are averaged in logit space *before* any temperature is applied --
    calibrating each member and then averaging is a different, worse estimator.
    """
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    x = preprocess_views(img).to(_state["device"])
    store: dict = {}
    hook = None
    if want_feats:
        # the food gate consumes the pooled penultimate feature of the first
        # (centre-crop) view, i.e. the same forward pass, no second network.
        # Only member #0 is hooked: it is the backbone every gate was fitted on.
        hook = models[0].avgpool.register_forward_hook(
            lambda _m, _i, out: store.__setitem__("feat", out))
    try:
        if len(models) == 1:
            logits = models[0](x).float().cpu()
        else:
            logits = torch.stack([m(x).float().cpu() for m in models]).mean(0)
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
    return _view_probs(_narrow_models(), _state.get("temperature", 1.0), img,
                       _state.get("temperature_fused"))


def analyse_full(img: Image.Image):
    """analyse() plus the pooled feature the food gate needs."""
    return _view_probs(_narrow_models(), _state.get("temperature", 1.0), img,
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


def analyse_wide(img: Image.Image, want_feats: bool = False):
    model = _state.get("wide_model")
    if model is None:
        raise RuntimeError("wide model is not loaded")
    views, fused = _view_probs(model, _state.get("wide_temperature", 1.0), img,
                               _state.get("wide_temperature_fused"))
    alpha = _state.get("wide_fuse", 0.0)
    # the narrow side is the ensemble too: same posterior the single-dish tab shows
    narrow = _narrow_models()
    frames = None
    if narrow and (alpha > 0 or want_feats):
        # One narrow forward pass serves both jobs: its posterior is mapped into
        # the wide index space for the fusion, and its pooled feature is what
        # the food gate was fitted on.  Asking for the feature costs nothing.
        _, p_narrow, frames = _view_probs(narrow, _state.get("temperature", 1.0), img,
                                          _state.get("temperature_fused"), want_feats=True)
        if alpha > 0:
            # The narrow mass is not renormalised, so alpha also controls how
            # much total probability the 20 known dishes may claim.
            full = np.zeros_like(fused)
            full[WIDE_NARROW_INDEX] = p_narrow
            fused = (1.0 - alpha) * fused + alpha * full
    if want_feats:
        return views, fused, frames
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


def _food_prob(frames: np.ndarray | None) -> Optional[float]:
    """P(there is a dish in the frame), or None when no gate is loaded.

    The gate lives on the narrow backbone's pooled feature; every tab that runs
    that backbone can ask it, so the wide and tray tabs get the same
    "is there food here at all?" answer the single-dish tab already had.
    """
    gate = _state.get("foodgate")
    if gate is None or frames is None:
        return None
    return gate.prob(frames[0])


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


#: One table for the three tabs: names, the calibrated gate, and the one line
#: that says what the tab does *not* cover.  Keeping them side by side is what
#: makes "which tab does this photo belong to?" a lookup instead of three copies
#: of the same branch (see suggest_tab() below).
_SCOPE_SPECS = {
    "single": {
        "names": CLASS_NAMES, "zh": ZH_LABELS, "threshold": SCOPE_THRESHOLD,
        "label": "20 类", "tab": "单菜识别",
        "examples": "整只烤鸡/烤鸭、火锅、家常炒菜等都不在支持的 20 类里",
    },
    "wide": {
        "names": WIDE_CLASS_NAMES, "zh": WIDE_ZH, "threshold": WIDE_SCOPE_THRESHOLD,
        "label": f"{WIDE_NUM_CLASSES} 类", "tab": "116 类",
        "examples": f"火锅、麻辣烫、家常炒菜等中式吃法都不在这 {WIDE_NUM_CLASSES} 类里",
    },
    "cn": {
        "names": CN_CLASS_NAMES, "zh": CN_ZH, "threshold": CN_SCOPE_THRESHOLD,
        "label": f"{CN_NUM_CLASSES} 道", "tab": "食堂菜",
        "examples": "牛肉面、麻辣烫、米线、手抓饼、清蒸鱼这些食堂主食和小吃不在这 20 道里",
    },
}


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
    mode = "cn" if cn else "wide" if wide else "single"
    spec = _SCOPE_SPECS[mode]
    names, zh, label, examples = spec["names"], spec["zh"], spec["label"], spec["examples"]
    threshold = spec["threshold"]
    #: The canteen tab's dish list comes from the loaded checkpoint, not from the
    #: module constants: the 20-dish head, the 21-class one (plus the 'other
    #: Chinese' bucket) and the 22-class one (plus rice) all share this code.
    menu_size = CN_NUM_CLASSES
    if cn:
        names = _state.get("cn_names") or list(CN_CLASS_NAMES)
        zh = _state.get("cn_zh") or CN_ZH
        menu_size = int(_state.get("cn_dish_count") or CN_NUM_CLASSES)
        label = f"{menu_size} 道"
        examples = (f"牛肉面、麻辣烫、米线、手抓饼、清蒸鱼这些食堂主食和小吃不在这 "
                    f"{menu_size} 道里")
    if cn and cn_other_prob is not None:
        # With the bucket the gate must ignore it: an in-scope dish can put 40 %
        # on "some other dish" and still be the right one, and the old threshold
        # would then throw the photo away for the wrong reason.
        threshold = CN_SCOPE_THRESHOLD_21
    top1 = float(np.max(probs))
    # Mirrored views sharpen the fused posterior but lower the worst-view score
    # on every photo, so the gate keeps the three original framings only; that
    # is the calibration the 0.32 / 0.18 thresholds were measured on.
    gate_views = views[:_GATE_VIEWS] if len(views) > _GATE_VIEWS else views
    #: The bucket is always the checkpoint's last unit.  With no model loaded --
    #: the tests call this directly -- the last column of `views` is the bucket.
    other_index = _state.get("cn_other_index") if cn else None
    if cn and other_index is None and cn_other_prob is not None and views.shape[1] > CN_NUM_CLASSES:
        other_index = int(views.shape[1]) - 1
    if cn_other_prob is not None and other_index is not None and views.shape[1] > other_index:
        # Divide the bucket back out of every view before asking "which of the 20
        # dishes is this?": the dish mass is what the 20-dish gate was calibrated
        # on, and the bucket answers the other question separately.
        dish = views[:, :other_index].astype(np.float64)
        bucket = views[:, other_index].astype(np.float64)
        renorm = dish / np.clip(1.0 - bucket, 1e-6, None)[:, None]
        robust = float(min(np.max(v) for v in renorm[:_GATE_VIEWS]))
    else:
        robust = float(min(np.max(v) for v in gate_views))
    # A trusted probe buys a lower bar, not a free pass: the bucket, the veto and
    # "no food here" still apply, and the relaxed number is what the payload
    # reports back so the UI can explain why a 0.37 photo was accepted.
    if cn and cn_domain_prob is not None and cn_domain_prob >= _TRUST_AT:
        threshold = min(threshold, _TRUST_GATE)
    # Two questions, two answers: the gate asks "is there food here at all?",
    # the robust score asks "is it one of the supported dishes?".  A menu board
    # clears neither, but it does clear the second one often enough to matter.
    gate = _state.get("foodgate")
    food_ok = True
    if gate is not None and food_prob is not None and not cn:
        # Not "not wide": the gate rides on the narrow backbone, which the wide
        # tab already runs for the fusion, so a menu board is as much
        # "no dish here" for 116 classes as it is for 20.
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
    if cn and cn_other_prob is not None and top_idx >= len(names):
        top_idx = int(np.argmax(probs[:len(names)]))
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
    if food_prob is not None and not cn:
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
            f"更像西餐、甜点或饮料。食堂模式只认识 {menu_size} 道中餐——"
            "换成「单菜识别」或「116 类」标签再试。"
        )
    elif not in_scope and other_hit:
        # The bucket caught what the wide-head veto kept missing (宫保鸡丁 was
        # 0.005 short of the margin, and 汤包/饺子 fell into 包子).
        scope["message"] = (
            f"这张看起来不在食堂模式的 {menu_size} 道清单里：模型把 {cn_other_prob * 100:.0f}% 的把握"
            f"放在「{CN_OTHER_ZH}」这一类上，也就是说它更像清单之外的菜"
            f"（最接近的「{closest}」只是强行归类的候选）。"
            "想吃别的菜可以切到「116 类」标签试试。"
        )
    elif not in_scope and vetoed:
        # A dish the canteen head would have force-fitted into its 20 classes:
        # name what the photo actually looks like instead of the nearest match.
        scope["message"] = (
            f"这张更像「{cn_veto_result['zh']}」——它不在食堂模式的 {menu_size} 道清单里"
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


#: Where a refused photo is worth trying next, most likely first.  The tray tab
#: is deliberately absent: it is a correction tool, not a classifier.
_TAB_ORDER = {"cn": ("single", "wide"), "single": ("cn",), "wide": ("cn",)}


def _tab_scope(img: Image.Image, mode: str) -> Optional[dict]:
    """The verdict the *other* tab would show for this photo.

    It runs that tab's own decision path -- its head, its gate, its bucket, its
    veto -- rather than asking the current head for a second opinion.  That
    costs one extra forward pass per candidate, but only on photos this tab
    refused, and the button it produces is a promise the other tab can keep.
    """
    if mode == "single":
        views, fused, frames = analyse_full(img)
        pack = _state.get("packaging")
        return scope_of(views, fused, food_prob=_food_prob(frames),
                        packaging_prob=pack.prob(frames) if pack is not None else None)
    if mode == "wide":
        if _state.get("wide_model") is None:
            return None
        views, fused, frames = analyse_wide(img, want_feats=True)
        return scope_of(views, fused, wide=True, food_prob=_food_prob(frames))
    if _state.get("cn_model") is None:
        return None
    views, fused, frames = analyse_cn(img, want_feats=True)
    other_index = _state.get("cn_other_index")
    other = float(fused[other_index]) if other_index is not None and fused.size > other_index else None
    other_top = bool(int(np.argmax(fused)) == other_index) if other is not None else False
    dish = fused[:other_index] if other is not None else fused
    return scope_of(views, dish, cn=True, cn_veto_result=cn_veto(img),
                    cn_other_prob=other, cn_other_top=other_top,
                    cn_domain_prob=_domain_prob(frames))


def _refusal_reason(scope: dict) -> str:
    """Why this tab said no, in the order scope_of checks the cues."""
    if (scope.get("domain") or {}).get("fired"):
        return "domain"
    if (scope.get("other") or {}).get("fired"):
        return "other"
    if (scope.get("veto") or {}).get("fired"):
        return "veto"
    return "weak"


def suggest_tab(img: Image.Image | None, mode: str, scope: dict) -> Optional[dict]:
    """Which tab would recognise this photo, when this one would not.

    A refusal is not the end of the story: a pizza is out of scope for the 20
    canteen dishes but in scope for 单菜识别, and a 红烧肉 photo is the reverse.
    The answer is the tab whose *own* gate accepted the photo, so the UI can
    offer one button that will not refuse it a second time.
    """
    if img is None or scope.get("in_scope"):
        return None
    # "No dish in the frame" and "this is a package" are not tab problems: the
    # advice is about the photo or the tray tab, and the message already says it.
    if scope.get("not_food") or scope.get("packaging"):
        return None
    # A canteen refusal on the veto/weak cues means the photo *is* canteen food
    # that this tab simply could not name -- the probe was quiet and the bucket
    # was empty.  Sending it to the Western tabs there is how a bowl of 木须肉
    # comes back as a hamburger, so no button is offered for those.
    if mode == "cn" and _refusal_reason(scope) in ("veto", "weak"):
        return None
    for alt in _TAB_ORDER.get(mode, ()):
        other = _tab_scope(img, alt)
        if other is not None and other["in_scope"]:
            spec = _SCOPE_SPECS[alt]
            return {"mode": alt, "label": spec["label"], "tab": spec["tab"],
                    "reason": _refusal_reason(scope)}
    return None


def payload(probs: np.ndarray, source: str, portion: str = "normal", grams=None, views=None,
            wide: bool = False, img: Image.Image | None = None, food_prob: float | None = None,
            packaging_prob: float | None = None, cn: bool = False,
            cn_frames: np.ndarray | None = None):
    portion = str(portion or "normal").strip().lower()
    cn_other_prob = None
    cn_other_top = False
    if cn:
        # Split off the "other Chinese" bucket: it has no name, no calories and
        # no portion, so it must never reach topk_with_calories().  Its index is
        # the checkpoint's last unit once a model is loaded (21 units with the
        # bucket alone, 22 with rice in front of it); with nothing loaded -- the
        # tests call this function directly -- fall back to the historical 21.
        other_index = _state.get("cn_other_index")
        if other_index is None and len(probs) > CN_NUM_CLASSES:
            other_index = len(probs) - 1
        if other_index is not None and len(probs) > other_index:
            cn_other_prob = float(probs[other_index])
            cn_other_top = bool(int(np.argmax(probs)) == other_index)
            probs = probs[:other_index]
        items = topk_with_calories(probs, k=3, portion=portion, grams=grams,
                                   class_names=_state.get("cn_names") or CN_CLASS_NAMES,
                                   zh_table=_state.get("cn_zh") or CN_ZH,
                                   kcal_table=_state.get("cn_kcal") or CN_KCAL_PER_100G,
                                   portion_table=_state.get("cn_portion") or CN_PORTION_G)
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
    # Which tab would recognise it instead?  Only asked when this tab refused,
    # and only after the packaging cue is settled, so a package photo keeps its
    # "拆开装盘" message rather than being sent to another classifier.
    scope["suggest"] = suggest_tab(img, "cn" if cn else "wide" if wide else "single", scope)
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
    names = _state.get("mixed_names") or list(CLASS_NAMES)
    zh = _state.get("mixed_zh") or ZH_LABELS
    kcal = _state.get("mixed_kcal") or KCAL_PER_100G
    grams_table = _state.get("mixed_portion") or PORTION_G
    parts = detect_parts(
        probs,
        threshold=threshold,
        max_parts=MAX_PARTS,
        class_thresholds=class_thresholds,
        min_prob=MIN_PART_PROB,
        class_names=names,
        zh_table=zh,
        kcal_table=kcal,
        portion_table=grams_table,
    )
    for part in parts:
        part["portion"] = portion
        part["grams"] = None
        part["cuisine"] = "canteen" if part["class"] in CN_CLASS_NAMES else "western"
        part["brand_refs"] = brands.refs_for_class(part["class"])
    plate = plate_totals(parts, portion=portion, zh_table=zh, kcal_table=kcal,
                         portion_table=grams_table)
    note = "多标签模型判断盘子里有哪几样菜；每一部分可以单独选份量或填克数，总热量是各部分区间之和。"
    mixed_cuisines = {p["cuisine"] for p in parts}
    if len(mixed_cuisines) > 1:
        note = ("盘子里同时有西餐和食堂菜，热量按各自的分量区间相加；"
                "中西混搭时两份主食容易都算成整份，可以手动把其中一份改成「小份」。")
    elif parts and parts[0]["cuisine"] == "canteen":
        note = "这是一份食堂托盘：模型按 20 道菜 + 20 道西餐的 40 类清单找菜，每部分的克数是按食堂常见份量估的。"
    if not parts:
        note = "没有哪一样菜的得分过线：可能是一道菜为主，也可能是清单外的菜。换个角度再拍一张，或切到「单菜识别」。"
    if scope and scope.get("not_food"):
        # Naming parts on a napkin or a menu board is the exact false positive
        # the gate exists to stop, so the tray reports nothing here.
        parts = []
        plate = plate_totals([], portion=portion, zh_table=zh, kcal_table=kcal,
                             portion_table=grams_table)
        note = ("这张照片里没找到菜：门控认为画面是菜单牌、饮料杯、包装袋或空盘一类的场景，"
                "所以不列出盘里的部分。把镜头对准托盘里的菜再拍一张。")
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


#: Cross-origin use.  The mini-program (miniprogram/) calls the API from a
#: different origin, and so does a page opened over a tunnel; both were blocked
#: by the browser before this.  The default is *any* origin because the demo has
#: no cookies to steal (the token, if set, is read from a query string or header).
#: FOODCLS_ORIGINS=a,b narrows it.
_ORIGINS = [o.strip() for o in os.environ.get("FOODCLS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

LOG = logging.getLogger("foodcls.api")

#: Per-path request counters for /api/health: a demo that is being recorded has
#: no terminal in shot, so the payload is the only place a stall or a burst of
#: 500s can be seen.  Counts since process start, plus the slowest call.
_metrics: dict = {}


@app.middleware("http")
async def _count_requests(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000.0
    path = request.url.path
    if path.startswith("/api/"):
        row = _metrics.setdefault(path, {"n": 0, "errors": 0, "ms_total": 0.0,
                                    "ms_max": 0.0, "last_status": 200})
        row["n"] += 1
        row["ms_total"] += elapsed
        row["ms_max"] = max(row["ms_max"], elapsed)
        row["last_status"] = response.status_code
        if response.status_code >= 400:
            row["errors"] += 1
        level = logging.WARNING if (response.status_code >= 500 or elapsed > 3000) else logging.INFO
        LOG.log(level, "%s %s -> %d in %.0f ms", request.method, path,
                response.status_code, elapsed)
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

    Feedback is collected in all four modes, so a mixed-plate, 116-class or
    canteen sample name has to resolve too, not just the 20-class gallery.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    #: the 116-class and canteen galleries prefix the class, and their files live
    #: in a per-class directory, so "apple_pie__test_0.jpg" is class + file name
    if "__" in name:
        cls, fname = name.split("__", 1)
        if cls and fname and "/" not in cls and "\\" not in cls:
            for root in (WIDE_SAMPLES, CN_SAMPLES):
                candidate = (root / cls / fname).resolve()
                if candidate.parent == (root / cls).resolve() and candidate.is_file():
                    return candidate
    for root in (SAMPLES, WIDE_SAMPLES, MIXED_SAMPLES, CN_SAMPLES):
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
    food_prob = _food_prob(frames)
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
        return JSONResponse({"error": UNREADABLE_IMAGE}, status_code=400)
    views, fused, frames = await run_in_threadpool(analyse_full, img)
    food_prob = _food_prob(frames)
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
#: A correction can name any class any of the four heads knows, not just the 20:
#: the canteen menu has its own slugs (tomato_egg, di_san_xian, ...), and without
#: them every correction made in the canteen tab failed validation.
FEEDBACK_CLASSES = (
    set(CLASS_NAMES) | set(WIDE_CLASS_NAMES) | set(CN_CLASS_NAMES) | set(CN_MENU_NAMES)
    | set(FEEDBACK_SPECIAL)
)
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
    mode = mode if mode in ("single", "wide", "mixed", "cn") else "single"

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
        return JSONResponse({"error": UNREADABLE_IMAGE}, status_code=400)

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
    views, fused, frames = analyse_wide(img, want_feats=True)
    return payload(fused, source=name, portion=portion, grams=grams, views=views, wide=True,
                   img=img, food_prob=_food_prob(frames))


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
        return JSONResponse({"error": UNREADABLE_IMAGE}, status_code=400)
    views, fused, frames = await run_in_threadpool(analyse_wide, img, True)
    return payload(fused, source=file.filename or "upload", portion=portion, grams=grams,
                   views=views, wide=True, img=img, food_prob=_food_prob(frames))


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
        #: `count` stays the 20 dishes the module ships with: it is the front end's
        #: fallback, and the endpoint must answer the same before and after a
        #: checkpoint loads.  `dishes` below is what the loaded head actually has.
        "count": CN_NUM_CLASSES,
        "dishes": int(_state.get("cn_dish_count") or CN_NUM_CLASSES),
        "threshold": (_state.get("cn_scope_threshold") or CN_SCOPE_THRESHOLD),
        "other": {
            "enabled": _state.get("cn_other_index") is not None,
            "zh": CN_OTHER_ZH,
            "threshold": CN_OTHER_THRESHOLD,
        },
        "enabled": _state.get("cn_model") is not None,
        "classes": [
            {"class": c, "zh": _state.get("cn_zh", CN_ZH)[c],
             "kcal_per_100g": _state.get("cn_kcal", CN_KCAL_PER_100G)[c],
             "portion_g": _state.get("cn_portion", CN_PORTION_G)[c]}
            for c in (_state.get("cn_names") or CN_CLASS_NAMES)
        ],
    }


@app.get("/api/cn/samples")
def cn_samples():
    """One test photo per dish, so the mode can be tried without a camera."""
    if not CN_SAMPLES.exists():
        return {"samples": [], "enabled": _state.get("cn_model") is not None}
    out: List[dict] = []
    zh = _state.get("cn_zh") or CN_ZH
    #: Follows the loaded head, but a dish with no sample folder is simply absent
    #: (rice ships without one: the gallery predates the 22-class checkpoint).
    for cls in (_state.get("cn_names") or CN_CLASS_NAMES):
        for f_ in sorted((CN_SAMPLES / cls).glob("*.jpg"))[:1]:
            out.append({"name": _sample_name(cls, f_.name), "class": cls, "zh": zh[cls]})
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
        return JSONResponse({"error": f"{_state.get('cn_dish_count') or CN_NUM_CLASSES} 道食堂菜模型"
                             "尚未训练完成"}, status_code=503)
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
        return JSONResponse({"error": f"{_state.get('cn_dish_count') or CN_NUM_CLASSES} 道食堂菜模型"
                             "尚未训练完成"}, status_code=503)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "图片过大，请上传 15 MB 以内的照片"}, status_code=413)
    try:
        img = read_image(raw)
    except ImageTooLarge:
        return JSONResponse({"error": "图片分辨率过高（超过 4000 万像素），请先缩小再上传"}, status_code=413)
    except Exception:
        return JSONResponse({"error": UNREADABLE_IMAGE}, status_code=400)
    views, fused, frames = await run_in_threadpool(analyse_cn, img, True)
    return payload(fused, source=file.filename or "upload", portion=portion, grams=grams,
                   views=views, cn=True, img=img, cn_frames=frames)


def _packaging_scope(img: Image.Image) -> dict:
    """Scope of the single-dish head, plus the packaging cue for the tray view."""
    views, fused, frames = analyse_full(img)
    pack = _state.get("packaging")
    scope = scope_of(views, fused, food_prob=_food_prob(frames),
                     packaging_prob=pack.prob(frames) if pack is not None else None)
    if not scope["in_scope"] and not scope.get("packaging") and packaging.signals(img)["detected"]:
        scope["packaging"] = True
        scope["message"] = packaging.message()
    return scope


@app.get("/api/mixed/samples")
def mixed_samples():
    items = [
        {
            "name": it["name"],
            "classes": it.get("classes", []),
            "zh": it.get("zh", []),
            "cuisine": it.get("cuisine", ""),
        }
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
        return JSONResponse({"error": UNREADABLE_IMAGE}, status_code=400)
    parts = await run_in_threadpool(infer_mixed, img)
    return mixed_payload(parts, source=file.filename or "upload", portion=portion,
                         scope=_packaging_scope(img))


@app.post("/api/mixed/total")
async def mixed_total(body: dict):
    """Recompute the per-part and total kcal after the user edits weights."""
    default_portion = str(body.get("portion") or "normal")
    names = _state.get("mixed_names") or list(CLASS_NAMES)
    zh = _state.get("mixed_zh") or ZH_LABELS
    kcal = _state.get("mixed_kcal") or KCAL_PER_100G
    grams_table = _state.get("mixed_portion") or PORTION_G
    parts = body.get("parts")
    if not isinstance(parts, list):
        return JSONResponse({"error": "parts 必须是数组"}, status_code=400)
    rows = []
    for item in parts:
        if not isinstance(item, dict):
            continue
        name = str(item.get("class", ""))
        if name not in names:
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
    return plate_totals(rows, portion=default_portion, zh_table=zh, kcal_table=kcal,
                        portion_table=grams_table)


@app.get("/api/health")
def health():
    mixed_names = list(_state.get("mixed_names") or CLASS_NAMES)
    return {
        "ok": True,
        "arch": _state.get("arch"),
        "checkpoint": CKPT.name,
        "temperature": _state.get("temperature", 1.0),
        "temperature_fused": _state.get("temperature_fused", _state.get("temperature", 1.0)),
        "views": len(_VIEW_TFS) or 6,
        "heif": HEIF_OK,
        "ensemble": {
            "enabled": bool(_state.get("ensemble")),
            "members": len(_state.get("ensemble") or []),
            "runs": list(_state.get("ensemble_runs") or []),
            "temperature": _state.get("temperature", 1.0),
            "gates_on_member": (_state.get("ensemble_runs") or [CKPT.parent.name])[0],
        },
        "mixed_enabled": _state.get("mixed_model") is not None,
        "mixed_threshold": _state.get("mixed_threshold", DETECT_THRESHOLD),
        "mixed_classes": len(mixed_names),
        "mixed_canteen_classes": max(0, len(mixed_names) - len(CLASS_NAMES)),
        "wide_enabled": _state.get("wide_model") is not None,
        "wide_arch": _state.get("wide_arch"),
        "wide_classes": WIDE_NUM_CLASSES,
        "wide_temperature": _state.get("wide_temperature", 1.0),
        "wide_fuse": _state.get("wide_fuse", 0.0),
        "cn_enabled": _state.get("cn_model") is not None,
        "cn_arch": _state.get("cn_arch"),
        "cn_classes": int(_state.get("cn_dish_count") or CN_NUM_CLASSES),
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
        "requests": {k: {"n": v["n"], "errors": v["errors"],
                            "ms_avg": round(v["ms_total"] / max(1, v["n"]), 1),
                            "ms_max": round(v["ms_max"], 1),
                            "last_status": v["last_status"]}
                    for k, v in sorted(_metrics.items())},
        "checkpoint": CKPT.parent.name,
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

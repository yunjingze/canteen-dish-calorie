"""Mixed-plate support: synthetic composites, multi-label inference, totals.

A single-dish classifier cannot answer "this tray holds fries *and* wings": the
softmax forces one label.  So we train a small multi-label model on *composites*
built from Food-101 photographs (two or three dishes share one plate) and read
the per-class sigmoid scores as "which parts are on the tray".  The same module
builds the training plates, the deterministic evaluation plates, the demo
samples and the per-part / total calorie arithmetic.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

from .calorie import class_calorie_range, portion_scale
from .classes import CLASS_NAMES, KCAL_PER_100G, PORTION_G, ZH_LABELS
from .plate_classes import PLATE_KCAL_PER_100G, PLATE_PORTION_G, PLATE_ZH
from .plate_prior import default_names, pick_addition

PLATE_SIZE = 256
DETECT_THRESHOLD = 0.5
MAX_PARTS = 3
#: below this score a part is never shown, whatever the per-class threshold is
MIN_PART_PROB = 0.10

#: share of plates with 1 / 2 / 3 dishes during training
SINGLE_MIX = (0.35, 0.45, 0.20)


# --------------------------------------------------------------------------- #
# Composite construction
# --------------------------------------------------------------------------- #
def _cover(img: Image.Image, size: int) -> Image.Image:
    """Resize to cover a size x size square, then centre-crop (no letterboxing)."""
    w, h = img.size
    scale = max(size / w, size / h)
    img = img.resize(
        (max(size, int(round(w * scale))), max(size, int(round(h * scale)))),
        Image.BICUBIC,
    )
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _random_cover(img: Image.Image, size: int, rng: np.random.Generator) -> Image.Image:
    """Cover-crop with a random offset, so the same photo never lands identically."""
    zoom = float(rng.uniform(1.0, 1.18))
    big = _cover(img, int(round(size * zoom)))
    if big.size[0] == size and big.size[1] == size:
        return big
    left = int(rng.integers(0, big.size[0] - size + 1))
    top = int(rng.integers(0, big.size[1] - size + 1))
    return big.crop((left, top, left + size, top + size))


def _feather(mask: np.ndarray, radius: float) -> np.ndarray:
    """Blur a binary mask so a pasted part does not look cut out with scissors."""
    m = Image.fromarray((mask * 255).astype(np.uint8))
    m = m.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(m, dtype=np.float32) / 255.0


def _split_masks(n: int, rng: np.random.Generator, size: int) -> List[np.ndarray]:
    """Divide the canvas into n regions: one oblique cut, or three wedges."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = cx = (size - 1) / 2.0
    if n == 2:
        theta = float(rng.uniform(0, np.pi))
        frac = float(rng.uniform(0.35, 0.65))          # both parts stay visible
        nx, ny = np.cos(theta), np.sin(theta)
        proj = (xx - cx) * nx + (yy - cy) * ny
        span = size * 0.5 * (abs(nx) + abs(ny))
        cut = (frac - 0.5) * 2.0 * span
        return [(proj >= cut).astype(np.float32), (proj < cut).astype(np.float32)]

    ang = np.arctan2(yy - cy, xx - cx)
    for _ in range(50):
        gaps = rng.uniform(70.0, 150.0, size=2)
        if gaps.sum() < 290.0:
            break
    else:                                              # pragma: no cover
        gaps = np.array([100.0, 100.0])
    base = float(rng.uniform(-np.pi, np.pi))
    edges = np.array([base, base + gaps[0], base + gaps[0] + gaps[1]])
    edges = (edges + 2 * np.pi) % (2 * np.pi)
    edges.sort()
    masks = []
    for i in range(3):
        lo, hi = edges[i], edges[(i + 1) % 3]
        if i == 2:                                     # wraps through 0
            m = ((ang >= lo) | (ang < hi)).astype(np.float32)
        else:
            m = ((ang >= lo) & (ang < hi)).astype(np.float32)
        masks.append(m)
    return masks


def _blob_mask(size: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Elliptical region: a side dish sits in one corner of the tray, not half of it."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return ((((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0).astype(np.float32)


def _compose_blobs(
    images: Sequence[Image.Image],
    rng: np.random.Generator,
    size: int,
) -> Image.Image:
    """One dish fills the frame; the others are smaller blobs pasted on top.

    Half-and-half cuts teach "both parts occupy half the picture", which is not
    what a real tray looks like: a heap of fries with a few wings at the top
    edge.  These blobs cover roughly 6-35% of the canvas, so the model also
    learns to keep a small side dish in the label set.
    """
    canvas = np.asarray(_random_cover(images[0], size, rng), dtype=np.float32)
    for img in images[1:]:
        arr = np.asarray(_random_cover(img, size, rng), dtype=np.float32)
        rx = float(rng.uniform(0.14, 0.34)) * size
        ry = float(rng.uniform(0.14, 0.34)) * size
        cx = float(rng.uniform(0.22, 0.78)) * size
        cy = float(rng.uniform(0.20, 0.80)) * size
        m = _feather(_blob_mask(size, cx, cy, rx, ry), size * 0.05)[..., None]
        canvas = canvas * (1.0 - m) + arr * m
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def _blend(images: Sequence[Image.Image], masks: Sequence[np.ndarray]) -> Image.Image:
    acc = np.zeros((images[0].size[1], images[0].size[0], 3), dtype=np.float32)
    wsum = np.zeros(acc.shape, dtype=np.float32)
    for img, m in zip(images, masks):
        arr = np.asarray(img, dtype=np.float32)
        m3 = m[..., None]
        acc += arr * m3
        wsum += m3
    return Image.fromarray(np.clip(acc / np.maximum(wsum, 1e-4), 0, 255).astype(np.uint8))


def build_composite(
    images: Sequence[Image.Image],
    rng: np.random.Generator,
    size: int = PLATE_SIZE,
) -> Image.Image:
    """Paste two or three dishes onto one plate-like canvas.

    Two layouts are mixed 45/55 during training: an oblique cut (both parts
    large, like rice next to a stir fry) and one full-frame dish with smaller
    blobs on top (like fries with a few wings).  The blob layout is what fixes
    the "small side dish is dropped" failure mode; the split layout stays so
    the model keeps working on evenly divided trays.
    """
    if len(images) == 1:
        return _random_cover(images[0].convert("RGB"), size, rng)
    if rng.random() < 0.45:
        masks = [_feather(m, size * 0.055) for m in _split_masks(len(images), rng, size)]
        prepared = [_random_cover(img.convert("RGB"), size, rng) for img in images]
        return _blend(prepared, masks)
    return _compose_blobs([img.convert("RGB") for img in images], rng, size)


def pair_indices(rng: np.random.Generator, labels: Sequence[int], n: int,
                 num_classes: int | None = None) -> List[int]:
    """Pick n distinct classes at random from the class list."""
    n_classes = len(CLASS_NAMES) if num_classes is None else int(num_classes)
    picks: List[int] = []
    tries = 0
    while len(picks) < n and tries < 200:
        lab = int(rng.integers(n_classes))
        tries += 1
        if lab not in picks:
            picks.append(lab)
    return picks


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
try:                                            # torch is optional for the CLI tools
    from torch.utils.data import Dataset

    _BASE = Dataset
except Exception:                               # pragma: no cover
    _BASE = object


class MixedPlateDataset(_BASE):
    """On-the-fly plates: 1-3 dishes share a 256 px canvas (multi-hot target)."""

    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        length: int,
        transform=None,
        seed: int = 0,
        sizes: Sequence[int] = (1, 2, 3),
        probs: Sequence[float] = SINGLE_MIX,
        num_classes: int | None = None,
        prior: bool = False,
    ):
        self.samples = list(samples)
        self.num_classes = len(CLASS_NAMES) if num_classes is None else int(num_classes)
        self.length = length
        self.transform = transform
        self.seed = seed
        self.sizes = list(sizes)
        self.probs = list(probs)
        self.prior = bool(prior)
        # label -> row indices, so a prior-guided draw can find a photo fast
        self._by_label: Dict[int, List[int]] = {}
        for i, (_, lab) in enumerate(self.samples):
            self._by_label.setdefault(int(lab), []).append(i)

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _load(path: str) -> Image.Image:
        with open(path, "rb") as fh:
            return Image.open(fh).convert("RGB")

    def _prior_row(self, rng: np.random.Generator, names: Sequence[str],
                   labels: Sequence[int]) -> int | None:
        """One photo of a dish the pairing prior wants next to ``labels``."""
        from .plate_prior import weights_for_many

        anchors = [names[l] for l in labels if 0 <= l < len(names)]
        w = weights_for_many(names, anchors, exclude=labels)
        for i, name in enumerate(names):
            if i not in self._by_label:       # class absent from this pool
                w[i] = 0.0
        total = float(w.sum())
        if total <= 0.0:
            return None
        cls = int(rng.choice(len(names), p=w / total))
        rows = self._by_label[cls]
        return int(rows[int(rng.integers(len(rows)))])

    def __getitem__(self, index: int):
        import torch

        rng = np.random.default_rng((self.seed, index))
        n = int(rng.choice(self.sizes, p=self.probs))
        path, label = self.samples[index % len(self.samples)]
        labels = [int(label)]
        parts = [self._load(path)]
        names = default_names(self.num_classes) if self.prior else []
        for _ in range(n - 1):
            if self.prior:
                j = self._prior_row(rng, names, labels)
                if j is not None:
                    other_path, other_label = self.samples[j]
                    labels.append(int(other_label))
                    parts.append(self._load(other_path))
                    continue
            for _try in range(30):
                j = int(rng.integers(len(self.samples)))
                other_path, other_label = self.samples[j]
                if int(other_label) not in labels:
                    labels.append(int(other_label))
                    parts.append(self._load(other_path))
                    break
        img = parts[0] if len(parts) == 1 else build_composite(parts, rng)
        if self.transform is not None:
            img = self.transform(img)
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[labels] = 1.0
        return img, target


def build_eval_set(
    samples: Sequence[Tuple[str, int]],
    count: int,
    seed: int = 1234,
    sizes: Sequence[int] = (2, 3),
    probs: Sequence[float] = (0.6, 0.4),
    size: int = PLATE_SIZE,
    num_classes: int | None = None,
    label_map: Sequence[int] | None = None,
    prior: bool = False,
    prior_names: Sequence[str] | None = None,
) -> List[Tuple[Image.Image, List[int]]]:
    """Deterministic composites, rebuilt identically on every evaluation run.

    ``label_map`` translates the local labels of ``samples`` (0..k-1) into the
    model's output units: the canteen pool is a 20-class dataset but occupies
    units 20..39 of the 40-class tray head.
    """
    out: List[Tuple[Image.Image, List[int]]] = []
    n_samples = len(samples)
    names = list(prior_names) if prior_names is not None else default_names(num_classes)
    by_label: Dict[int, List[int]] = {}
    for i, (_, lab) in enumerate(samples):
        by_label.setdefault(int(lab), []).append(i)
    for i in range(count):
        rng = np.random.default_rng((seed, i))
        n = int(rng.choice(sizes, p=probs))
        if prior:
            picks = []
            for _ in range(n):
                if not picks:
                    picks = [int(rng.integers(len(names)))]
                    continue
                picks += [pick_addition(rng, names, [names[p] for p in picks], exclude=picks)]
        else:
            picks = pair_indices(rng, [], n, num_classes=num_classes)
        paths = []
        for lab in picks:
            # deterministic member of the class: index walks the file list
            cands = [s for s in samples if s[1] == lab] or [samples[0]]
            paths.append(cands[i % len(cands)][0])
        labels = picks if label_map is None else [int(label_map[l]) for l in picks]
        parts = [MixedPlateDataset._load(p) for p in paths]
        out.append((build_composite(parts, rng, size=size), labels))
    return out


# --------------------------------------------------------------------------- #
# Reading the multi-label scores
# --------------------------------------------------------------------------- #
def detect_parts(
    probs,
    threshold: float = DETECT_THRESHOLD,
    max_parts: int = MAX_PARTS,
    class_thresholds: Dict[str, float] | None = None,
    min_prob: float = MIN_PART_PROB,
    class_names: Sequence[str] | None = None,
    zh_table: Dict[str, str] | None = None,
    kcal_table: Dict[str, float] | None = None,
    portion_table: Dict[str, float] | None = None,
) -> List[Dict]:
    """Turn per-class sigmoid scores into the parts on the plate.

    ``class_thresholds`` lets each dish keep its own cut-off (a side dish that
    rarely fills the frame scores lower on average than a main).  Anything below
    ``min_prob`` is dropped even if its per-class threshold is lower, and a
    plate with no confident part returns an empty list instead of a forced
    top-1: showing "steak 4%" as the answer is worse than saying nothing.

    ``class_names`` and the three lookup tables let one call serve both the
    20-class western checkpoints and the 40-class western+canteen one; they
    default to the western tables so old callers keep working unchanged.
    """
    class_names = CLASS_NAMES if class_names is None else list(class_names)
    zh_table = ZH_LABELS if zh_table is None else zh_table
    kcal_table = KCAL_PER_100G if kcal_table is None else kcal_table
    portion_table = PORTION_G if portion_table is None else portion_table
    probs = np.asarray(probs, dtype=float).ravel()
    if probs.size != len(class_names):
        raise ValueError(f"got {probs.size} scores for {len(class_names)} classes")
    order = np.argsort(-probs)
    picked = []
    for i in order:
        i = int(i)
        cut = threshold
        if class_thresholds:
            cut = float(class_thresholds.get(class_names[i], threshold))
        if probs[i] >= max(cut, min_prob):
            picked.append(i)
        if len(picked) >= max_parts:
            break
    return [
        {
            "class": class_names[i],
            "zh": zh_table.get(class_names[i], class_names[i]),
            "prob": float(probs[i]),
            "kcal_per_100g": kcal_table.get(class_names[i]),
            "base_portion_g": portion_table.get(class_names[i]),
        }
        for i in picked
    ]


# --------------------------------------------------------------------------- #
# Calorie arithmetic for a plate
# --------------------------------------------------------------------------- #
def plate_totals(
    parts: Sequence[Dict],
    portion: str = "normal",
    zh_table: Dict[str, str] | None = None,
    kcal_table: Dict[str, float] | None = None,
    portion_table: Dict[str, float] | None = None,
) -> Dict:
    """Per-part intervals plus their sum.

    Each part uses its own weight (grams) or portion multiplier; the total is the
    sum of the part intervals, which keeps the report honest: the range still
    covers recipe variability, it is not a single point estimate added up.
    """
    zh_table = ZH_LABELS if zh_table is None else zh_table
    kcal_table = KCAL_PER_100G if kcal_table is None else kcal_table
    portion_table = PORTION_G if portion_table is None else portion_table
    rows = []
    for part in parts:
        name = str(part["class"])
        if name not in kcal_table:
            continue
        conf = float(part.get("prob", 1.0))
        scale = portion_scale(part.get("portion", portion))
        rng = class_calorie_range(name, conf, scale, grams=part.get("grams"),
                                  kcal_table=kcal_table, portion_table=portion_table)
        rows.append(
            {
                "class": name,
                "zh": zh_table[name],
                "prob": conf,
                "grams": rng["portion_g"],
                "weight_source": rng["weight_source"],
                "kcal_per_100g": rng["kcal_per_100g"],
                "kcal_point": rng["point"],
                "kcal_low": rng["low"],
                "kcal_high": rng["high"],
                "spread": rng["spread"],
            }
        )
    total = {
        "kcal_point": int(sum(r["kcal_point"] for r in rows)),
        "kcal_low": int(sum(r["kcal_low"] for r in rows)),
        "kcal_high": int(sum(r["kcal_high"] for r in rows)),
        "grams": int(sum(r["grams"] for r in rows)),
    }
    return {"parts": rows, "total": total}


def format_plate(plate: Dict) -> str:
    parts = " + ".join(f"{r['zh']} {r['kcal_low']}-{r['kcal_high']}" for r in plate["parts"])
    t = plate["total"]
    return f"{parts} = {t['kcal_low']}-{t['kcal_high']} kcal ({t['grams']} g)"

"""Dataset preparation and transforms.

Data layout produced by `prepare_data.py`:

    data/splits/train/<food101_class>/*.jpg    600 img/class (official train split)
    data/splits/val/<food101_class>/*.jpg      100 img/class (held out of official train)
    data/splits/canteen_clean/<class>/*.jpg     50 img/class (held out of official train)
    data/splits/canteen/<class>/*.jpg           degraded copies of canteen_clean
    data/splits/test/<food101_class>/*.jpg     250 img/class (official test split)

The canteen-proxy images never participate in training or model selection.
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
try:  # torch is optional for the data-preparation-only path
    from torch.utils.data import Dataset
    _BASE = Dataset
except Exception:  # pragma: no cover
    _BASE = object


class SplitDataset(_BASE):
    """ImageFolder-like dataset with an explicit (ordered) class list."""

    def __init__(self, root: str | Path, class_names: Sequence[str], transform=None):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        for idx, name in enumerate(self.class_names):
            folder = self.root / name
            if not folder.is_dir():
                raise FileNotFoundError(f"missing class folder: {folder}")
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    self.samples.append((str(p), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with open(path, "rb") as fh:
            img = Image.open(fh).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(train: bool, color_aug: bool):
    from torchvision import transforms

    if train:
        ops = [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
        ]
        if color_aug:
            ops.append(
                transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.06)
            )
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        return transforms.Compose(ops)
    return transforms.Compose(
        [
            transforms.Resize(int(IMG_SIZE * 256 / 224)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# Food-101 preparation
# --------------------------------------------------------------------------- #
def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_splits(
    data_root: str | Path,
    class_names: Sequence[str],
    seed: int = 42,
    n_train: int = 600,
    n_val: int = 100,
    n_canteen: int = 50,
) -> None:
    """Materialise the five splits from the official Food-101 images.

    The official train/test folders must already exist under data/raw/food-101
    (created by scripts/extract_food101_hf.py); a missing folder is an error, not
    a prompt to download, because torchvision drops both splits into one folder
    and the train pool would then contain test photos.
    """
    data_root = Path(data_root)
    raw = data_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    img_root = raw / "food-101" / "images"
    test_root = raw / "food-101" / "images_test"
    missing = [c for c in class_names if not (img_root / c).is_dir()]
    if missing:
        # Downloading here would put the official test images into the same
        # folder as train, and the shuffle below would then leak test photos
        # into train/val.  Refuse instead of quietly building a leaky split.
        raise RuntimeError(
            f"{len(missing)} class folders missing under {img_root} "
            f"(first: {missing[0]}). Run scripts/extract_food101_hf.py first; "
            "the official train/test folders must exist before splits are built."
        )
    if not all((test_root / c).is_dir() for c in class_names):
        raise RuntimeError(
            f"official test split missing under {test_root}; "
            "run scripts/extract_food101_hf.py so images_test/ exists."
        )

    split_root = data_root / "splits"
    rng = random.Random(seed)

    for split, expected in (("train", n_train + n_val + n_canteen), ("test", 250)):
        for name in class_names:
            files = sorted((img_root / name).glob("*.jpg"))
            if len(files) < expected:
                raise RuntimeError(f"{name}: found {len(files)} images, need {expected}")
            if split == "test":
                chosen = sorted((test_root / name).glob("*.jpg"))[:250]
                if len(chosen) < 250:
                    raise RuntimeError(f"{name}: test images missing under {test_root}")
                for src in chosen:
                    _link_or_copy(src, split_root / "test" / name / src.name)
            else:
                rng.shuffle(files)
                for src in files[:n_canteen]:
                    _link_or_copy(src, split_root / "canteen_clean" / name / src.name)
                for src in files[n_canteen:n_canteen + n_val]:
                    _link_or_copy(src, split_root / "val" / name / src.name)
                for src in files[n_canteen + n_val:n_canteen + n_val + n_train]:
                    _link_or_copy(src, split_root / "train" / name / src.name)

    make_canteen_proxy(split_root / "canteen_clean", split_root / "canteen", seed=seed)


# --------------------------------------------------------------------------- #
# Synthetic "canteen capture" domain shift
# --------------------------------------------------------------------------- #
def _apply_color_cast(img: Image.Image, rng: random.Random) -> Image.Image:
    """Warm tungsten / green fluorescent / cool daylight casts seen indoors."""
    cast = rng.choice(["warm", "warm", "fluorescent", "cool", "neutral"])
    gains = {
        "warm": (1.12, 1.00, 0.86),
        "fluorescent": (0.93, 1.06, 0.96),
        "cool": (0.94, 1.00, 1.10),
        "neutral": (1.0, 1.0, 1.0),
    }[cast]
    arr = np.asarray(img, dtype=np.float32)
    arr *= np.array(gains, dtype=np.float32)[None, None, :]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _perspective(img: Image.Image, rng: random.Random, strength: float = 0.035) -> Image.Image:
    w, h = img.size
    d = strength * min(w, h)
    quad = [
        rng.uniform(0, d), rng.uniform(0, d),
        w - rng.uniform(0, d), rng.uniform(0, d),
        w - rng.uniform(0, d), h - rng.uniform(0, d),
        rng.uniform(0, d), h - rng.uniform(0, d),
    ]
    return img.transform((w, h), Image.QUAD, quad, Image.BICUBIC)


def _vignette(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    r = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)
    mask = np.clip(1.0 - rng.uniform(0.25, 0.7) * r ** 2, 0.35, 1.0)[..., None]
    return Image.fromarray(np.clip(arr * mask * 255.0, 0, 255).astype(np.uint8))


def _motion_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    size = rng.choice([3, 5])  # Pillow's Kernel filter only supports 3x3 and 5x5
    kernel = np.zeros((size, size), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[size // 2, :] = 1.0
    else:
        kernel[:, size // 2] = 1.0
    kernel /= kernel.sum()
    from PIL import ImageFilter as IF

    return img.filter(IF.Kernel((size, size), kernel.ravel().tolist(), scale=1.0, offset=0))


def degrade_canteen_image(src: Path, dst: Path, seed: int) -> None:
    """Simulate a hand-held phone photo taken in a canteen."""
    rng = random.Random(seed)
    with open(src, "rb") as fh:
        img = Image.open(fh).convert("RGB")

    # short side ~ 512 px, as from a phone camera
    scale = 512.0 / min(img.size)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)

    # framing jitter: rotate, crop, perspective
    img = img.rotate(rng.uniform(-7, 7), resample=Image.BICUBIC, expand=False)
    w, h = img.size
    cw = int(w * rng.uniform(0.72, 0.92))
    ch = int(h * rng.uniform(0.72, 0.92))
    left = rng.randint(0, max(0, w - cw))
    top = rng.randint(0, max(0, h - ch))
    img = img.crop((left, top, left + cw, top + ch))
    img = _perspective(img, rng)
    img = img.resize((320, 320), Image.BICUBIC)

    # photometry
    img = _apply_color_cast(img, rng)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.72, 1.25))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.78, 1.20))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.70, 1.30))
    if rng.random() < 0.55:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.4)))
    if rng.random() < 0.35:
        img = _motion_blur(img, rng)
    img = _vignette(img, rng)

    arr = np.asarray(img, dtype=np.float32)
    arr += np.random.default_rng(seed).normal(0, rng.uniform(1.5, 6.0), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=rng.randint(35, 65))


def make_canteen_proxy(src_root: str | Path, dst_root: str | Path, seed: int = 42) -> None:
    src_root, dst_root = Path(src_root), Path(dst_root)
    if not src_root.is_dir():
        # the wide 101-class splits have no canteen proxy
        print(f"[canteen-proxy] no source images under {src_root}; skipped")
        return
    count = 0
    for cls_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for i, src in enumerate(sorted(cls_dir.iterdir())):
            dst = dst_root / cls_dir.name / f"{src.stem}_ccp.jpg"
            if dst.exists():
                continue
            degrade_canteen_image(src, dst, seed=seed + count + i * 7919)
            count += 1
    print(f"[canteen-proxy] generated {count} images under {dst_root}")

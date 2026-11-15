"""Command-line inference: image -> Top-3 dishes + calorie ranges."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .calorie import format_range, topk_with_calories
from .classes import CLASS_NAMES
from .data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from .models import build_model
from .utils import pick_device


def load(ckpt_path):
    device = pick_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), device


def infer(model, device, path, portion="normal", grams=None, tta=False):
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 256 / 224)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    with open(path, "rb") as fh:
        img = ImageOps.exif_transpose(Image.open(fh)).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        if tta:
            # original + horizontal flip averaged in probability space; one extra
            # forward, worth ~+0.8 (in-domain) to +1.0 (proxy) points, Section V-I
            x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
            probs = torch.softmax(model(x).float(), dim=1).mean(0).cpu().numpy()
        else:
            probs = torch.softmax(model(x).float(), dim=1)[0].cpu().numpy()
    return topk_with_calories(probs, 3, portion=portion, grams=grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parents[2] / "results/runs/resnet18_last_stage_aug/best.pt"))
    ap.add_argument("--portion", default="normal", choices=["small", "normal", "large"])
    ap.add_argument("--grams", type=float, default=None,
                    help="actual serving weight in grams; overrides --portion")
    ap.add_argument("--tta", action="store_true",
                    help="average the softmax over the image and its horizontal flip")
    args = ap.parse_args()
    model, device = load(args.ckpt)
    top = infer(model, device, args.image, args.portion, grams=args.grams, tta=args.tta)
    print(json.dumps(top, indent=2, ensure_ascii=False))
    for i, t in enumerate(top, 1):
        print(f"Top{i}: {t['zh']} ({t['class']})  p={t['prob']*100:.1f}%  {format_range(t)}  ({t['portion_g']} g)")


if __name__ == "__main__":
    sys.exit(main())

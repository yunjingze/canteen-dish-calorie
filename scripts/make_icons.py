"""Render the PWA icons (app icon / maskable icon) with Pillow.

    python scripts/make_icons.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/foodcls/static"
S = 4  # supersampling factor


def gradient(size: int) -> Image.Image:
    """Diagonal blue -> violet gradient, same colours as the page header."""
    base = Image.new("RGB", (size, size))
    px = base.load()
    c1 = (47, 109, 246)
    c2 = (122, 90, 248)
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * size - 2)
            px[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))
    return base


def draw_bowl(img: Image.Image, scale: float = 1.0) -> None:
    """White bowl with a rice mound and steam, centred on the icon.

    Coordinates are written for a 512 px canvas and scaled to the actual
    (supersampled) image size, so the art keeps its proportions.
    """
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    k = w / 512 * scale
    cx = w / 2
    rim = h / 2 + 30 * k
    # steam (behind the rice)
    for dx in (-58, 0, 58):
        d.arc([cx + dx * k - 22 * k, rim - 262 * k, cx + dx * k + 22 * k, rim - 186 * k],
              200, 340, fill=(255, 255, 255, 185), width=max(1, int(9 * k)))
    # rice mound sitting on the rim
    d.ellipse([cx - 100 * k, rim - 158 * k, cx + 100 * k, rim + 4 * k],
              fill=(255, 255, 255, 236))
    # bowl body: lower half of a wide ellipse whose flat edge is exactly the rim
    d.pieslice([cx - 138 * k, rim - 120 * k, cx + 138 * k, rim + 120 * k],
               0, 180, fill=(255, 255, 255, 242))
    d.rounded_rectangle([cx - 142 * k, rim - 12 * k, cx + 142 * k, rim + 8 * k],
                        radius=10 * k, fill=(255, 255, 255, 250))


def render(px: int, bg_scale: float, bowl_scale: float, rounded: bool) -> Image.Image:
    big = gradient(px * S).convert("RGBA")
    draw_bowl(big, bowl_scale)
    if rounded:
        mask = Image.new("L", (px * S, px * S), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, px * S - 1, px * S - 1], radius=int(px * S * 0.22), fill=255)
        big.putalpha(mask)
    return big.resize((px, px), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render(192, 1.0, 0.78, True).save(OUT / "icon-192.png")
    render(512, 1.0, 0.78, True).save(OUT / "icon-512.png")
    # maskable icons need the art inside the inner 80% safe zone and a full-bleed square
    render(512, 1.0, 0.62, False).save(OUT / "icon-maskable-512.png")
    print("wrote", OUT / "icon-192.png", OUT / "icon-512.png", OUT / "icon-maskable-512.png")


if __name__ == "__main__":
    main()

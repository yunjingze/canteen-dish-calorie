# -*- coding: utf-8 -*-
"""The ingest tool decides what counts as an honest outside test image, so its
judgements get pinned down here rather than eyeballed once."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ingest_web as W


def photo(w, h, seed):
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 40, (max(2, h // 4), max(2, w // 4), 3))
    img = Image.fromarray(np.clip(base, 0, 255).astype("uint8")).resize((w, h), Image.BICUBIC)
    d = ImageDraw.Draw(img)
    for _ in range(30):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(6, max(7, w // 12)))
        d.ellipse([x, y, x + r, y + r], fill=tuple(int(c) for c in rng.integers(30, 230, 3)))
    return img


def grid(w, h, seed):
    g = Image.new("RGB", (w, h), (255, 255, 255))
    hw, hh = w // 2, h // 2
    for k, (x, y) in enumerate([(0, 0), (hw + 4, 0), (0, hh + 4), (hw + 4, hh + 4)]):
        g.paste(photo(hw - 8, hh - 8, seed + k), (x, y))
    return g


def test_dhash_survives_recompression_and_a_halving():
    """dHash is the near-duplicate test, so it has to be stable under the two
    things a re-download does: re-encode and shrink. A noisy synthetic image is
    a harsher case than a photo, which is the point."""
    a = photo(900, 700, 1).filter(ImageFilter.GaussianBlur(2))
    b = a.resize((450, 350), Image.LANCZOS)
    same = bin(W.dhash(a) ^ W.dhash(b)).count("1")
    far = bin(W.dhash(a) ^ W.dhash(photo(900, 700, 99).filter(ImageFilter.GaussianBlur(2)))).count("1")
    assert same <= 4
    assert far > 4


def test_blur_score_separates_texture_from_flat():
    sharp = photo(800, 600, 2)
    from PIL import ImageFilter
    soft = sharp.filter(ImageFilter.GaussianBlur(12))
    assert W.blur_score(W._gray(sharp)) > W.blur_score(W._gray(soft))


def test_collage_cues_fire_on_a_grid_and_not_on_a_photo():
    bands, sr, sc = W.collage_cues(grid(1200, 1600, 5))
    assert bands >= 1
    bands2, sr2, sc2 = W.collage_cues(photo(1200, 900, 6))
    assert bands2 == 0 and not (sr2 and sc2)


def test_frame_frac_fires_on_a_letterboxed_card():
    card = Image.new("RGB", (1000, 800), (255, 255, 255))
    card.paste(photo(600, 420, 7), (200, 190))
    assert W.frame_frac(card) >= 0.75
    assert W.frame_frac(photo(1000, 800, 8)) < 0.75


def test_nearest_reports_hamming_distance_to_the_closest_known_image():
    known = 0b1011_0000_1111_0000_1010_0101_1100_0011_0000_1111_1010_0101_1100_0011_1011_0000
    bits = np.zeros((3, 64), dtype=bool)
    bits[1] = np.unpackbits(np.frombuffer(known.to_bytes(8, "big"), dtype=np.uint8))
    assert W.nearest(bits, known) == 0
    assert W.nearest(bits, known ^ 0b111) == 3
    assert W.nearest(np.zeros((0, 64), dtype=bool), 7) == 64, "empty fingerprint = nothing can be close"


def _inbox(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    (inbox / "baozi").mkdir(parents=True)
    (inbox / "not_a_dish").mkdir(parents=True)
    photo(1200, 900, 11).save(inbox / "baozi/good.jpg", quality=92)
    photo(1200, 900, 11).save(inbox / "baozi/copy.jpg", quality=92)
    photo(300, 200, 12).save(inbox / "baozi/tiny.jpg", quality=85)
    photo(1600, 400, 13).save(inbox / "baozi/wide.jpg", quality=88)
    photo(900, 900, 14).save(inbox / "not_a_dish/mystery.jpg", quality=88)
    photo(1000, 750, 15).save(inbox / "loose.jpg", quality=88)
    (inbox / "baozi/sources.csv").write_text("file,url\ngood.jpg,https://example.invalid/good\n", "utf-8")
    return inbox


def test_scan_rejects_the_obvious_and_keeps_the_rest(tmp_path):
    inbox = _inbox(tmp_path, None)
    out = tmp_path / "out"
    rc = W.main(["scan", "--inbox", str(inbox), "--out", str(out), "--fresh"])
    assert rc == 0
    rows = {r["file"]: r for r in csv.DictReader((out / "review.csv").open(encoding="utf-8-sig"))}
    pair = [rows["baozi/good.jpg"]["verdict"], rows["baozi/copy.jpg"]["verdict"]]
    assert sorted(v.split(":")[0] for v in pair) == ["keep", "reject"], "one of two identical files survives"
    assert any(v.startswith("reject:dup") for v in pair)
    assert rows["baozi/tiny.jpg"]["verdict"] == "reject:small"
    assert rows["baozi/wide.jpg"]["verdict"] == "reject:aspect"
    assert rows["not_a_dish/mystery.jpg"]["verdict"] == "reject:unknown_class"
    assert rows["loose.jpg"]["verdict"] == "reject:no_class"
    assert rows["baozi/good.jpg"]["url"] == "https://example.invalid/good"
    assert list((out / "review/sheets").glob("*.png"))


def test_apply_writes_clean_jpegs_and_parks_rejections(tmp_path):
    inbox = _inbox(tmp_path, None)
    out = tmp_path / "out"
    W.main(["scan", "--inbox", str(inbox), "--out", str(out), "--fresh"])
    review = out / "review.csv"
    rows = list(csv.DictReader(review.open(encoding="utf-8-sig")))
    for r in rows:
        if r["file"] in ("baozi/good.jpg", "baozi/copy.jpg"):
            r["verdict"] = "keep"
        elif not r["verdict"].startswith("keep"):
            pass
    with review.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    assert W.main(["apply", "--inbox", str(inbox), "--out", str(out), "--target", "1"]) == 0
    kept = out / "accepted/baozi/good.jpg"
    assert kept.exists()
    assert (out / "accepted/baozi/copy.jpg").exists()
    with Image.open(kept) as im:
        assert im.format == "JPEG"
        assert len(im.getexif()) == 0
    assert (out / "rejected/baozi/tiny.jpg").exists()
    assert not (inbox / "baozi/tiny.jpg").exists()
    assert (inbox / "baozi/good.jpg").exists(), "kept originals stay put"
    man = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8-sig")))
    assert sorted(m["file"] for m in man) == ["baozi/copy.jpg", "baozi/good.jpg"]
    assert all(m["label"] == "baozi" for m in man)


def test_scan_says_so_when_there_is_nothing_to_do(tmp_path):
    empty = tmp_path / "inbox"
    empty.mkdir()
    assert W.main(["scan", "--inbox", str(empty), "--out", str(tmp_path / "out")]) == 2

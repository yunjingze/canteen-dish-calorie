# -*- coding: utf-8 -*-
"""Turn a hand-collected folder of web photos into an honest outside test set.

Collecting is the manual part: drop pictures into
    data/external/web20/inbox/<slug>/
one folder per dish. Everything after that is the part a human cannot do by
eye at 600 images, so this tool does it:

  scan    hash + measure every file, throw out exact and near duplicates of
          anything the 21-way head already saw (train AND val AND test),
          reject undersized / corrupt / degenerate aspect files, flag
          collages, screenshot frames and blur, then write review.csv and a
          numbered contact sheet per dish so you can look at them.
  apply   take the review.csv verdicts, strip EXIF, re-encode to JPEG and
          write accepted/<slug>/ plus manifest.csv.
  status  how many images each dish has so far against --target.

The test set is frozen once you report a number. Adding to it afterwards is
just tuning on the test set, which is exactly what an outside set is for
avoiding.

Usage
    python scripts/ingest_web.py scan
    python scripts/ingest_web.py apply
    python scripts/ingest_web.py status
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = ROOT / "results/tables/_fingerprint_cn241_21.json"
SPLIT_DIR = ROOT / "data/splits/cn241_21/train"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_INBOX = ROOT / "data/external/web20/inbox"
DEFAULT_OUT = ROOT / "data/external/web20"


# --------------------------------------------------------------------------- #
# per-image measurements
# --------------------------------------------------------------------------- #
def dhash(im):
    """64-bit difference hash; same recipe as results/_hash_cn241.py."""
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _gray(im, width=320):
    w, h = im.size
    nh = max(1, round(h * width / w))
    return np.asarray(im.convert("L").resize((width, nh), Image.BILINEAR), dtype=np.float32)


def blur_score(g):
    """Variance of the 4-neighbour Laplacian. Textured photo high, soft photo low."""
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    k = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(k.var())


def _flat_runs(a, axis, tol=6.0, frac=0.97):
    """Runs of rows (axis=1) or columns (axis=0) that are near-uniform."""
    med = np.median(a, axis=axis, keepdims=True)
    flat = (np.abs(a - med) <= tol).mean(axis=axis) >= frac
    runs, start = [], None
    for i, f in enumerate(flat):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flat)))
    return runs


def collage_cues(im):
    """Cheap evidence that the file is a multi-panel grid rather than one photo.

    Two independent cues, both of which need a human glance to confirm, so they
    are reported as flags and never auto-rejected:
      * narrow near-uniform bands inside the frame -- the gutter between panels
      * full-width or full-height sharp edges -- the seam of a grid
    """
    g = _gray(im, 480)
    H, W = g.shape
    bands = 0
    for runs, n in ((_flat_runs(g, 1), H), (_flat_runs(g.T, 1), W)):
        for a, b in runs:
            thick = b - a
            if not (0.03 * n <= a and b <= 0.97 * n):
                continue
            if thick > max(3, 0.02 * n):
                continue
            if float(g[:, a:b].std()) > 6.0:
                continue
            bands += 1
    seam_row = seam_col = 0
    dy = np.abs(np.diff(g, axis=0))
    dx = np.abs(np.diff(g, axis=1))
    if dy.size:
        seam_row = int((dy > 25).mean(axis=1).max() >= 0.9)
    if dx.size:
        seam_col = int((dx > 25).mean(axis=0).max() >= 0.9)
    return bands, seam_row, seam_col


def frame_frac(im, pad=0.03):
    """Fraction of the outer frame that is a flat, near-white or near-black band.

    High values mean a screenshot, a menu card or a photo letterboxed onto a
    background -- all of which change the colour statistics the model sees.
    """
    a = _gray(im, 320)
    H, W = a.shape
    k = max(1, int(round(pad * min(H, W))))
    strips = [a[:k], a[-k:], a[:, :k], a[:, -k:]]
    hit = 0
    for s in strips:
        if float(s.std()) < 4.0 and (float(s.mean()) > 238 or float(s.mean()) < 18):
            hit += 1
    return hit / 4.0


def measure(p):
    b = p.read_bytes()
    rec = {"md5": hashlib.md5(b).hexdigest(), "kb": round(len(b) / 1024, 1)}
    with Image.open(p) as im:
        rec["w"], rec["h"] = im.size
        rec["mode"] = im.mode
        rec["fmt"] = im.format
        rec["frames"] = getattr(im, "n_frames", 1)
        rec["exif"] = bool(im.getexif())
        rec["dh"] = dhash(im)
        rec["blur"] = round(blur_score(_gray(im)), 1)
        rec["bands"], rec["seam_row"], rec["seam_col"] = collage_cues(im)
        rec["frame"] = round(frame_frac(im), 2)
    return rec


# --------------------------------------------------------------------------- #
# fingerprint of everything the model has already seen
# --------------------------------------------------------------------------- #
def load_fingerprint():
    fp = json.loads(FINGERPRINT.read_text("utf-8"))
    items = fp["items"]
    md5 = {v[0]: k for k, v in items.items()}
    hashes = [v[1] for v in items.values() if v[1] is not None]
    bits = np.zeros((len(hashes), 64), dtype=bool)
    for i, h in enumerate(hashes):
        bits[i] = np.unpackbits(np.frombuffer(int(h).to_bytes(8, "big"), dtype=np.uint8))
    return md5, bits


def nearest(bits, h):
    if not len(bits):
        return 64
    q = np.unpackbits(np.frombuffer(int(h).to_bytes(8, "big"), dtype=np.uint8))
    return int((bits != q).sum(axis=1).min())


def load_dishes():
    """The 20 scored dishes. other_cn is the catch-all bucket, not a dish."""
    return sorted(x.name for x in SPLIT_DIR.iterdir() if x.is_dir() and x.name != "other_cn")


def load_labels():
    """Every folder name an image may legitimately sit in.

    other_cn is allowed on purpose: hard negatives (yu-xiang pork next to
    pepper pork) are worth collecting even though they are not a scored dish.
    """
    return sorted(x.name for x in SPLIT_DIR.iterdir() if x.is_dir())

# --------------------------------------------------------------------------- #
# contact sheet: the point is that a human looks at 96 images per page
# --------------------------------------------------------------------------- #
def _rel(p):
    try:
        return p.resolve().relative_to(ROOT)
    except ValueError:
        return p


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_sheet(rows, path, title, cols=8, tile=230):
    label_h = 30
    pad = 8
    nrow = (len(rows) + cols - 1) // cols
    head = 40 if title else 0
    W = cols * (tile + pad) + pad
    H = head + nrow * (tile + label_h + pad) + pad
    sheet = Image.new("RGB", (W, H), (26, 26, 29))
    d = ImageDraw.Draw(sheet)
    f_idx = _font(26)
    f_lab = _font(17)
    if title:
        d.text((pad + 2, 8), title, fill=(235, 235, 240), font=_font(24))
    for k, r in enumerate(rows):
        cx = pad + (k % cols) * (tile + pad)
        cy = head + pad + (k // cols) * (tile + label_h + pad)
        verdict = r["verdict"]
        if verdict.startswith("reject"):
            edge = (220, 70, 70)
        elif r["flags"]:
            edge = (225, 165, 40)
        else:
            edge = (55, 55, 60)
        d.rectangle([cx, cy, cx + tile, cy + tile + label_h],
                    fill=(38, 38, 42), outline=edge, width=2)
        try:
            with Image.open(r["path"]) as im:
                thumb = ImageOps.contain(im.convert("RGB"), (tile - 6, tile - 6), Image.LANCZOS)
            sheet.paste(thumb, (cx + (tile - thumb.width) // 2, cy + (tile - thumb.height) // 2))
        except Exception:
            d.text((cx + 10, cy + 10), "UNREADABLE", fill=(255, 120, 120), font=f_lab)
        d.text((cx + 6, cy + 4), str(r["idx"]), fill=(255, 255, 120), font=f_idx)
        if verdict.startswith("reject"):
            d.text((cx + 6, cy + 30), "REJECT", fill=(255, 90, 90), font=f_lab)
        line = r.get("tags", "")
        if line:
            d.text((cx + 5, cy + tile + 3), line[:34], fill=(210, 210, 215), font=f_lab)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
COLS = ["idx", "verdict", "flags", "label", "file", "w", "h", "kb", "blur",
        "bands", "frame", "nearest", "note", "url"]


def _read_sources(folder):
    """Optional <label>/sources.csv with file,url[,license] next to the photos."""
    f = folder / "sources.csv"
    out = {}
    if not f.exists():
        return out
    with f.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("file") or "").strip()
            if name:
                out[name] = (row.get("url") or "").strip()
    return out


def scan(args):
    inbox, out = Path(args.inbox).resolve(), Path(args.out).resolve()
    labels = set(load_labels())
    files = [p for p in sorted(inbox.rglob("*"))
             if p.is_file() and p.suffix.lower() in IMG_EXT]
    if not files:
        print("no images under", inbox)
        print("drop them in as", inbox / "<slug>" / "*.jpg")
        return 2
    print(len(files), "files under", inbox, flush=True)
    seen_md5, seen_dh = {}, []
    md5_cn, bits = load_fingerprint()
    rows = []
    for i, p in enumerate(files):
        rel = p.relative_to(inbox)
        label = rel.parts[0] if len(rel.parts) > 1 else ""
        reasons, flags = [], []
        rec = {"idx": len(rows), "path": str(p), "label": label, "file": rel.as_posix()}
        try:
            m = measure(p)
        except Exception as e:
            m = None
            reasons.append("unreadable")
            rec["note"] = type(e).__name__
        if m:
            rec.update(w=m["w"], h=m["h"], kb=m["kb"], blur=m["blur"],
                       bands=m["bands"], frame=m["frame"])
            if not label:
                reasons.append("no_class")
            elif label not in labels:
                reasons.append("unknown_class")
            if m["md5"] in md5_cn:
                reasons.append("dup_cn241")
                rec["note"] = md5_cn[m["md5"]]
            elif m["md5"] in seen_md5:
                j = seen_md5[m["md5"]]
                reasons.append("dup_batch:%d" % j)
            else:
                hit = None
                for j, h in seen_dh:
                    dd = bin(h ^ m["dh"]).count("1")
                    if dd <= args.hd:
                        hit = (j, dd)
                        break
                if hit:
                    reasons.append("near_dup_batch:%d(h=%d)" % hit)
                d_cn = nearest(bits, m["dh"])
                rec["nearest"] = d_cn
                if d_cn <= args.hd:
                    reasons.append("near_dup_cn241(h=%d)" % d_cn)
            if min(m["w"], m["h"]) < args.min_side:
                reasons.append("small")
            ar = m["w"] / max(1, m["h"])
            if ar > 2.6 or ar < 0.38:
                reasons.append("aspect")
            if m["frames"] > 1:
                reasons.append("animated")
            if m["bands"] or (m["seam_row"] and m["seam_col"]):
                flags.append("collage")
            elif m["seam_row"] or m["seam_col"]:
                flags.append("seam")
            if m["frame"] >= 0.75:
                flags.append("frame")
            if m["blur"] < 40:
                flags.append("blurry")
            if m["kb"] < 15:
                flags.append("tiny_file")
            seen_md5.setdefault(m["md5"], rec["idx"])
            seen_dh.append((rec["idx"], m["dh"]))
        rec["flags"] = ";".join(flags)
        rec["verdict"] = ("reject:" + reasons[0]) if reasons else "keep"
        if len(reasons) > 1:
            extra = ";".join(reasons[1:])
            rec["note"] = ((rec.get("note") or "") + " | " + extra).strip(" |")
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print("  %d/%d" % (i + 1, len(files)), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    src_map = {}
    for lab in sorted({r["label"] for r in rows if r["label"]}):
        src_map.update(_read_sources(inbox / lab))
    for r in rows:
        r["url"] = src_map.get(Path(r["file"]).name, "")

    sheet_dir = out / "review/sheets"
    if sheet_dir.exists() and args.fresh:
        shutil.rmtree(sheet_dir)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"] or "_unlabeled", []).append(r)
    per_page = 96
    n_sheets = 0
    for lab, group in sorted(by_label.items()):
        for k in range(0, len(group), per_page):
            chunk = group[k:k + per_page]
            page = k // per_page + 1
            for r in chunk:
                bits_txt = "%dx%d" % (r.get("w", 0), r.get("h", 0))
                r["tags"] = bits_txt + (("  " + r["flags"]) if r["flags"] else "")
            title = "%s  page %d  (%d of %d)" % (lab, page, len(chunk), len(group))
            draw_sheet(chunk, sheet_dir / ("%s_%02d.png" % (lab, page)), title, tile=args.tile)
            n_sheets += 1

    review = out / "review.csv"
    with review.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    keep = [r for r in rows if r["verdict"] == "keep"]
    why = {}
    for r in rows:
        if r["verdict"].startswith("reject:"):
            k = r["verdict"].split(":")[1].split("_")[0]
            why[k] = why.get(k, 0) + 1
    print("")
    print("%d keep, %d auto-reject" % (len(keep), len(rows) - len(keep)), flush=True)
    for k in sorted(why, key=lambda x: -why[x]):
        print("  %-12s %d" % (k, why[k]))
    flagged = sum(1 for r in rows if r["flags"])
    print("%d flagged for a human look, %d contact sheets" % (flagged, n_sheets), flush=True)
    print("")
    print(" ", _rel(review))
    print(" ", _rel(sheet_dir), "\\*.png")
    print("")
    print("mark reject in the verdict column, then:")
    print("  python scripts/ingest_web.py apply")
    return 0

# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
MANIFEST = ["file", "label", "w", "h", "kb", "md5", "verdict", "url", "note"]


def _to_jpeg(src, dst, max_side=1600, quality=95):
    """Re-encode without EXIF so the pixels, not the phone, are what we test."""
    with Image.open(src) as im:
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im = ImageOps.contain(im, (max_side, max_side), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=quality, optimize=True, progressive=True)
    return dst


def apply(args):
    out = Path(args.out).resolve()
    args.inbox = str(Path(args.inbox).resolve())
    review = out / "review.csv"
    if not review.exists():
        print("no review.csv; run scan first")
        return 2
    with review.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("review.csv is empty")
        return 2
    labels = set(load_labels())
    kept, moved, skipped = [], [], 0
    for r in rows:
        verdict = (r.get("verdict") or "").strip().lower()
        if not verdict.startswith("k"):
            continue
        src = None
        for base in (Path(args.inbox), out / "inbox"):
            cand = base / r["file"]
            if cand.exists():
                src = cand
                break
        if src is None:
            print("missing", r["file"])
            skipped += 1
            continue
        label = (r.get("label") or "").strip()
        if label not in labels:
            label = "_unlabeled"
        r["label"] = label
        stem = Path(r["file"]).stem
        dst = out / "accepted" / label / (stem + ".jpg")
        if dst.exists():
            dst = out / "accepted" / label / ("%s_%s.jpg" % (stem, r["idx"]))
        if args.dry_run:
            kept.append(r)
            continue
        _to_jpeg(src, dst, max_side=args.max_side)
        with Image.open(dst) as im:
            r["w"], r["h"] = im.size
        r["kb"] = round(dst.stat().st_size / 1024, 1)
        kept.append(r)

    # rejections are moved, not deleted: the reason you threw one out is worth
    # keeping, otherwise the same kind comes back next batch.
    if not args.dry_run:
        for r in rows:
            verdict = (r.get("verdict") or "").strip().lower()
            if verdict.startswith("k"):
                continue
            for base in (Path(args.inbox), out / "inbox"):
                src = base / r["file"]
                if src.exists():
                    break
            else:
                continue
            label = (r.get("label") or "").strip() or "_unlabeled"
            r["label"] = label
            dst = out / "rejected" / label / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst = dst.with_name("%s_%s%s" % (src.stem, r["idx"], src.suffix))
            shutil.move(str(src), str(dst))
            moved.append(r)

    if not args.dry_run:
        man = out / "manifest.csv"
        with man.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST, extrasaction="ignore")
            w.writeheader()
            for r in sorted(kept, key=lambda x: (x["label"], x["file"])):
                w.writerow(r)

    per = {}
    for r in kept:
        per[r["label"]] = per.get(r["label"], 0) + 1
    print("")
    print("%s %d into accepted/, %d to rejected/, %d missing"
          % ("would write" if args.dry_run else "wrote", len(kept), len(moved), skipped))
    for lab in sorted(per):
        print("  %-22s %d" % (lab, per[lab]))
    short = [d for d in load_dishes() if per.get(d, 0) < args.target]
    if short:
        print("")
        print("still under %d per dish: %s" % (args.target, ", ".join(short)))
    return 0


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def status(args):
    out = Path(args.out).resolve()
    acc = out / "accepted"
    dishes = load_dishes()
    total, short = 0, []
    for d in dishes:
        n = len(list((acc / d).glob("*.jpg"))) if (acc / d).exists() else 0
        total += n
        bar = "#" * min(30, n * 30 // max(1, args.target))
        flag = "" if n >= args.target else "  (need %d)" % (args.target - n)
        print("  %-22s %4d %-30s%s" % (d, n, bar, flag))
        if n < args.target:
            short.append(d)
    print("")
    print("  total %d / target %d  (%d dishes still short)"
          % (total, args.target * len(dishes), len(short)))
    return 0


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--inbox", default=str(DEFAULT_INBOX))
    common.add_argument("--out", default=str(DEFAULT_OUT))
    common.add_argument("--target", type=int, default=40,
                        help="how many images per dish you are aiming for (status/apply)")
    common.add_argument("--min-side", type=int, default=400)
    common.add_argument("--hd", type=int, default=4, help="dHash Hamming radius for near-duplicates")
    common.add_argument("--tile", type=int, default=230, help="contact sheet tile size")
    common.add_argument("--max-side", type=int, default=1600, help="downscale on apply, 0 = keep")
    common.add_argument("--fresh", action="store_true", help="delete old contact sheets first")
    common.add_argument("--dry-run", action="store_true")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("scan", scan), ("apply", apply), ("status", status)):
        sp = sub.add_parser(name, parents=[common],
                            help={"scan": "hash, screen and draw contact sheets",
                                  "apply": "write accepted/ from the review verdicts",
                                  "status": "per-dish counts against --target"}[name])
        sp.set_defaults(func=fn)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())



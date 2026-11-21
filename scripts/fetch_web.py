"""Collect candidate web images for the canteen external test set.

Bing Images is used only as a *search index*: we take the URLs it returns,
download the bytes from the original host, and record where each file came
from in sources.csv next to it.  Nothing here judges whether an image is
usable -- that is scripts/ingest_web.py (dedup, collage/frame/blur cues,
contact sheets) plus a human looking at those sheets.

Usage::

    python scripts/fetch_web.py --per-dish 45            # all 20 dishes
    python scripts/fetch_web.py --only braised_pork --per-dish 4 --dry-run

Three rules this script enforces so the numbers stay honest later:

* the query is exactly the dish name and nothing else.  A second variant such
  as "{zh} 食堂 快餐" was tried and thrown away: Bing answers it with stock art
  and news photos of canteens and business meetings, which then have to be
  deleted by hand;
* a dish is labelled by the query, never by what the picture looks like --
  screening decides only "is this a real photo", never "can I recognise the
  dish";
* every file keeps its source URL, so the report can say where the images came
  from and the package never has to ship them.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes_cn import CN_CLASS_NAMES, CN_ZH  # noqa: E402

INBOX = ROOT / "data" / "external" / "web20" / "inbox"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_BYTES = 12 * 1024 * 1024
MIN_BYTES = 12 * 1024

#: One query per dish: the dish name, verbatim.
QUERY_TEMPLATE = "{zh}"


def parse_bing(html_text: str) -> list[str]:
    """Pull the original image URLs out of a Bing Images result page."""
    out = []
    seen = set()
    for raw in re.findall(r"murl&quot;:&quot;(.*?)&quot;", html_text):
        url = html.unescape(raw)
        if url.startswith("//"):
            url = "https:" + url
        if not url.lower().startswith(("http://", "https://")):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def search(query: str, first: int = 1, timeout: int = 25) -> list[str]:
    """One page of Bing Images results for a query.

    The plain /images/search page only ever returns the first 34 tiles, even
    when "first" is set; the async endpoint the page itself calls for infinite
    scroll does honour it, and that is what this uses.
    """
    qs = urllib.parse.urlencode(
        {
            "q": query,
            "first": first,
            "count": 35,
            "mkt": "zh-CN",
            "qft": "+filterui:imagesize-large",
        }
    )
    req = urllib.request.Request(
        "https://cn.bing.com/images/async?" + qs,
        headers={
            "User-Agent": UA,
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "https://cn.bing.com/images/search",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        page = resp.read().decode("utf-8", "replace")
    return parse_bing(page)


def search_many(query: str, pages: int = 3, timeout: int = 25) -> list[str]:
    """Several result pages, de-duplicated and in rank order."""
    out: list[str] = []
    seen = set()
    for page in range(pages):
        try:
            urls = search(query, first=1 + page * 35, timeout=timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"  ! page {page + 1} failed: {exc}", flush=True)
            continue
        fresh = [u for u in urls if u not in seen]
        if not fresh:
            break
        seen.update(fresh)
        out.extend(fresh)
        time.sleep(0.4)
    return out


def download(url: str, timeout: int = 25) -> bytes | None:
    """Fetch the bytes of one image, or None if the host refuses."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "image/avif,image/webp,image/jpeg,image/png,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "https://cn.bing.com/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if ctype and not ctype.startswith("image/"):
                return None
            data = resp.read(MAX_BYTES + 1)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None
    if len(data) > MAX_BYTES or len(data) < MIN_BYTES:
        return None
    return data


def looks_like_image(data: bytes) -> bool:
    """Cheap magic-number check before paying for a full decode."""
    return data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n" or (
        data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    )


def read_sources(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    rows = []
    text = path.read_bytes().decode("utf-8-sig", "replace")
    for line in text.splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            rows.append((parts[0], parts[1]))
    return rows


def write_sources(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = ["file,url"] + [f"{name},{url}" for name, url in rows]
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def existing_hashes(folder: Path) -> set[str]:
    out = set()
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            out.add(hashlib.md5(p.read_bytes()).hexdigest())
    return out


def next_index(folder: Path) -> int:
    used = [0]
    if folder.is_dir():
        for p in folder.iterdir():
            if p.stem.isdigit():
                used.append(int(p.stem))
    return max(used) + 1


def _grab(url: str) -> tuple[str, bytes] | None:
    data = download(url)
    if data is None or not looks_like_image(data):
        return None
    return url, data


def fetch_dish(slug: str, per_dish: int, pages: int, workers: int, delay: float, dry: bool) -> int:
    """Download up to per_dish new images for one dish.  Returns how many."""
    zh = CN_ZH[slug]
    folder = INBOX / slug
    folder.mkdir(parents=True, exist_ok=True)
    sources_path = folder / "sources.csv"
    sources = read_sources(sources_path)
    known = {url for _, url in sources}
    hashes = existing_hashes(folder)

    urls = [u for u in search_many(QUERY_TEMPLATE.format(zh=zh), pages) if u not in known]
    print(f"  {zh!r} -> {len(urls)} urls", flush=True)
    if dry:
        for u in urls[:per_dish]:
            print(f"    would fetch {u[:110]}")
        return min(per_dish, len(urls))

    start = next_index(folder)
    got = 0
    # Downloads dominate the runtime (0.5-3 s each), so they run in parallel and
    # the results are replayed in order to keep the numbering deterministic.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_grab, u) for u in urls]
        for fut in futures:
            if got >= per_dish:
                break
            res = fut.result()
            if res is None:
                continue
            url, data = res
            digest = hashlib.md5(data).hexdigest()
            if digest in hashes:
                continue
            ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
            if ext not in IMG_EXT:
                ext = ".jpg"
            name = f"{start + got:04d}{ext}"
            (folder / name).write_bytes(data)
            hashes.add(digest)
            known.add(url)
            sources.append((name, url))
            got += 1
            if delay:
                time.sleep(delay)
    write_sources(sources_path, sources)
    return got


def main(argv: list[str] | None = None) -> int:
    global INBOX
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma separated slugs (default: all 20)")
    ap.add_argument("--per-dish", type=int, default=45)
    ap.add_argument("--pages", type=int, default=3, help="result pages per dish (35 each)")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between saves")
    ap.add_argument("--inbox", type=Path, default=INBOX)
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    args = ap.parse_args(argv)

    INBOX = args.inbox.resolve()

    slugs = [s.strip() for s in args.only.split(",") if s.strip()] or list(CN_CLASS_NAMES)
    unknown = [s for s in slugs if s not in CN_ZH]
    if unknown:
        print(f"unknown slug(s): {', '.join(unknown)}")
        return 2

    total = 0
    t0 = time.time()
    for i, slug in enumerate(slugs, 1):
        print(f"[{i}/{len(slugs)}] {slug} ({CN_ZH[slug]})", flush=True)
        got = fetch_dish(slug, args.per_dish, args.pages, args.workers, args.delay, args.dry_run)
        total += got
        print(f"  -> {got} new ({time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"\ntotal {total} new image(s) under {INBOX} in {time.time() - t0:.0f}s")
    if total:
        print("next: python scripts/ingest_web.py scan --fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay the demo shots against a running server and check they still say what the script claims.

The storyboard (docs/演示视频分镜.md) quotes exact numbers -- "0.95 酸辣土豆丝",
"域头只有 23%", "汉堡 + 薯条 + 牛排".  Those numbers are only true for one set of
weights.  Re-training, re-calibrating or changing a threshold silently breaks
them, and the failure mode is ugly: you discover it while recording.

So the numbers live in results/tables/demo_shots.json, and this script diffs the
live server against that file.  Run it once before every recording.

    python scripts/check_demo_shots.py            # check, exit 1 on drift
    python scripts/check_demo_shots.py --update   # accept the current output

Only the fields the video actually shows are compared (verdict, top-1 name and
probability, threshold, plate total), not the whole payload.
"""
import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "data/external/demo_shots"
BASELINE = ROOT / "results/tables/demo_shots.json"
BASE_URL = "http://127.0.0.1:8000"

#: tab -> (endpoint, json key holding the ranked dishes)
TABS = {
    "cn": ("/api/cn/predict", "top3"),
    "single": ("/api/predict", "top3"),
    "wide": ("/api/wide/predict", "top3"),
    "mixed": ("/api/mixed/predict", "parts"),
}


def snapshot(client: httpx.Client, base: str, path: Path) -> dict:
    """What the page will show for this photo, one entry per tab."""
    out = {}
    for tab, (endpoint, key) in TABS.items():
        with open(path, "rb") as fh:
            js = client.post(base + endpoint, files={"file": (path.name, fh, "image/jpeg")}).json()
        scope = js.get("scope") or {}
        items = js.get(key) or []
        head = items[0] if items else {}
        total = (js.get("plate") or {}).get("total") or {}
        out[tab] = {
            "in_scope": scope.get("in_scope"),
            "threshold": scope.get("threshold"),
            "domain": round((scope.get("domain") or {}).get("prob") or 0.0, 4) or None,
            "suggest": (scope.get("suggest") or {}).get("tab"),
            "top": head.get("zh"),
            "prob": round(head.get("prob", 0.0), 3),
            "plate_kcal": total.get("kcal_high") if key == "parts" else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="overwrite the baseline with current output")
    ap.add_argument("--url", default=BASE_URL)
    args = ap.parse_args()

    try:
        client = httpx.Client(timeout=120)
        client.get(args.url + "/api/health").raise_for_status()
    except Exception as exc:  # noqa: BLE001 - any failure here means "start the server"
        print(f"cannot reach {args.url}: {exc}")
        print("start it with: powershell -File scripts/start_demo.ps1 -Port 8000")
        return 2

    now = {p.name: snapshot(client, args.url, p) for p in sorted(SHOTS.glob("*.jpg"))}
    if args.update or not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(now, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote baseline for {len(now)} shots -> {BASELINE.relative_to(ROOT)}")
        return 0

    want = json.loads(BASELINE.read_text(encoding="utf-8"))
    drift = 0
    for name in sorted(now):
        for tab in TABS:
            a, b = want.get(name, {}).get(tab, {}), now[name][tab]
            diff = [k for k in b if a.get(k) != b[k]]
            if diff:
                drift += 1
                print(f"DRIFT {name} [{tab}]: " + ", ".join(f"{k} {a.get(k)!r} -> {b[k]!r}" for k in diff))
    if drift:
        print(f"\n{drift} field(s) drifted. The storyboard quotes the baseline numbers --")
        print("either re-record with the new numbers, or re-run with --update if the change is intended.")
        return 1
    print(f"all {len(now)} shots match {BASELINE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

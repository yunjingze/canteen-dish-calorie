
"""End-to-end sweep: does the packaging head catch packs without losing plates?"""
from __future__ import annotations
import csv, hashlib, json, sys, time, uuid, urllib.request
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\项目作业\food_project")
API = "http://127.0.0.1:8000/api/predict"


def post(path: Path):
    boundary = uuid.uuid4().hex
    body = b""
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"portion\"\r\n\r\nnormal\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"{path.name}\"\r\nContent-Type: image/jpeg\r\n\r\n").encode()
    body += path.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def unique_discarded():
    seen, out = set(), []
    for p in sorted((ROOT/"results/_discarded_v3").rglob("*.jpg")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in seen: continue
        seen.add(h); out.append(p)
    return out


def neg_packaging():
    neg = ROOT/"data/splits/negatives"
    cats = {}
    with open(neg/"sources.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            cats[Path(row["file"]).name] = row["category"]
    return [neg/n for n, c in sorted(cats.items()) if c == "packaging"] + [neg/"005.jpg"]


def clean_sample(n=300):
    ps = sorted((ROOT/"data/splits/canteen_clean").rglob("*.jpg"))
    idx = [round(i*(len(ps)-1)/(n-1)) for i in range(n)]
    return [ps[i] for i in idx]


groups = {
    "pack_v3": unique_discarded(),
    "pack_old": neg_packaging(),
    "real": sorted((ROOT/"data/splits/canteen_real").rglob("*.jpg")),
    "negatives": sorted((ROOT/"data/splits/negatives").glob("*.jpg")),
    "clean": clean_sample(),
}
out = {}
for name, paths in groups.items():
    rows, t0 = [], time.time()
    for i, p in enumerate(paths):
        try:
            j = post(p); s = j["scope"]
            rows.append({"path": str(p.relative_to(ROOT)), "in_scope": s["in_scope"],
                         "not_food": s.get("not_food", False), "packaging": s.get("packaging", False),
                         "p_pack": s.get("packaging_prob"), "p_food": s.get("food_prob"),
                         "robust": s.get("robust_prob"), "top1": s.get("closest"),
                         "top1_prob": s.get("top1_prob")})
        except Exception as exc:
            rows.append({"path": str(p), "error": str(exc)[:140]})
        if i and i % 100 == 0:
            print(f"  {name} {i}/{len(paths)} {time.time()-t0:.0f}s", flush=True)
    out[name] = rows
    ok = [r for r in rows if "error" not in r]
    print(f"{name}: n={len(ok)}/{len(rows)} in_scope={sum(r['in_scope'] for r in ok)} "
          f"packflag={sum(r['packaging'] for r in ok)} "
          f"not_food={sum(r['not_food'] for r in ok)}  {time.time()-t0:.0f}s", flush=True)

(ROOT/"results/tables/pkg_sweep.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved results/tables/pkg_sweep.json")

"""Run the 62 unique v3 images through the live API: they are supermarket packaging photos, so the gate and the packaging detector should reject them."""
import hashlib, json, urllib.request
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"E:\项目作业\food_project")
V3 = ROOT/"data/splits/canteen_real_v3"
B = "----v3sweep"

def post(p):
    body = (f"--{B}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{p.name}\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + p.read_bytes() + \
           (f"\r\n--{B}\r\nContent-Disposition: form-data; name=\"portion\"\r\n\r\nnormal\r\n--{B}--\r\n").encode()
    req = urllib.request.Request("http://127.0.0.1:8000/api/predict", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={B}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())

by_sha = defaultdict(list)
for f in sorted(V3.rglob("*.jpg")):
    by_sha[hashlib.sha256(f.read_bytes()).hexdigest()].append(f)

rows = []
for i, (k, grp) in enumerate(sorted(by_sha.items()), 1):
    p = grp[0]
    j = post(p)
    sc = j.get("scope", {})
    rows.append({"file": p.relative_to(V3).as_posix(),
                 "labels": sorted({x.parent.name for x in grp}),
                 "in_scope": bool(sc.get("in_scope")),
                 "packaging": bool(sc.get("packaging")),
                 "not_food": bool(sc.get("not_food")),
                 "p_food": sc.get("food_prob"), "robust": sc.get("robust_prob"),
                 "top1": (j.get("top3") or [{}])[0].get("class")})
    if i % 15 == 0: print(i, "/", len(by_sha), flush=True)

json.dump(rows, open(ROOT/"results/tables/v3_api.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n = len(rows)
print(f"unique images {n}")
print(f"in_scope   {sum(r['in_scope'] for r in rows)}/{n}")
print(f"packaging  {sum(r['packaging'] for r in rows)}/{n}")
print(f"not_food   {sum(r['not_food'] for r in rows)}/{n}")
print(f"p_food<0.2 {sum((r['p_food'] or 0) < 0.2 for r in rows)}/{n}")
print("accepted anyway:")
for r in rows:
    if r["in_scope"]:
        print(f"   {r['file']:44} p_food={r['p_food']:.2f} robust={r['robust']:.2f} top1={r['top1']} labels={r['labels']}")

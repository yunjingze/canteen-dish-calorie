"""End-to-end check: every negative and every real dish photo through the live HTTP API."""
import json, time, urllib.request
from pathlib import Path
ROOT = Path(r"E:\项目作业\food_project")
B = "----sweep7f3a"

def post(p):
    body = (f"--{B}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{p.name}\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + p.read_bytes() + \
           (f"\r\n--{B}\r\nContent-Disposition: form-data; name=\"portion\"\r\n\r\nnormal\r\n--{B}--\r\n").encode()
    req = urllib.request.Request("http://127.0.0.1:8000/api/predict", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={B}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())

neg = sorted((ROOT/"data/splits/negatives").glob("*.jpg"))
real = sorted((ROOT/"data/splits/canteen_real").rglob("*.jpg"))
rows = []
t0 = time.time()
for kind, paths in (("neg", neg), ("real", real)):
    for i, p in enumerate(paths, 1):
        try:
            j = post(p)
            sc = j.get("scope", {})
            rows.append({"kind": kind, "file": str(p.relative_to(ROOT)), "in_scope": bool(sc.get("in_scope")),
                         "food_prob": sc.get("food_prob"), "robust": sc.get("robust_prob"),
                         "top1": (j.get("top3") or [{}])[0].get("class"),
                         "top1_prob": (j.get("top3") or [{}])[0].get("prob")})
        except Exception as exc:
            rows.append({"kind": kind, "file": str(p.relative_to(ROOT)), "error": repr(exc)[:200]})
        if i % 25 == 0:
            print(f"{kind} {i}/{len(paths)} elapsed {time.time()-t0:.0f}s", flush=True)

json.dump(rows, open(ROOT/"results/tables/api_sweep.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
neg_ok = sum(r.get("in_scope") for r in rows if r["kind"] == "neg")
real_ok = sum(r.get("in_scope") for r in rows if r["kind"] == "real")
print(f"DONE negatives accepted {neg_ok}/{len(neg)}  real kept {real_ok}/{len(real)}  ({time.time()-t0:.0f}s)")

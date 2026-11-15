import json, urllib.request
from pathlib import Path
ROOT = Path(r"E:\项目作业\food_project")
imgs = ["canteen_real/hamburger/009.jpg","canteen_real/pizza/001.jpg","canteen_real/ramen/001.jpg",
        "canteen_real/fried_rice/002.jpg","canteen_real/dumplings/003.jpg","negatives/006.jpg","negatives/021.jpg"]

def post(p):
    b = "----probe7f3a"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{p.name}\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + p.read_bytes() + \
           (f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"portion\"\r\n\r\nnormal\r\n--{b}--\r\n").encode()
    req = urllib.request.Request("http://127.0.0.1:8000/api/predict", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())

for i, rel in enumerate(imgs):
    p = ROOT/"data/splits"/rel
    if not p.is_file():
        print(rel, "MISSING"); continue
    j = post(p)
    if i == 0:
        print("KEYS:", sorted(j.keys()))
        print("SCOPE:", json.dumps(j.get("scope"), ensure_ascii=False)[:400])
    sc = j.get("scope", {})
    top = j.get("top") or j.get("top3") or j.get("predictions") or []
    t1 = (top[0].get("class") or top[0].get("name"), top[0].get("prob")) if top else None
    print(f"{rel:34} in_scope={sc.get('in_scope')} p_food={sc.get('food_prob')} robust={sc.get('robust_prob')} top1={t1}")

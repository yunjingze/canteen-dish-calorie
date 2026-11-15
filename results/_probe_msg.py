import json, urllib.request
from pathlib import Path
ROOT = Path(r"E:\项目作业\food_project")
B = "----x1"
def post(p):
    body = (f"--{B}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{p.name}\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + p.read_bytes() + \
           (f"\r\n--{B}\r\nContent-Disposition: form-data; name=\"portion\"\r\n\r\nnormal\r\n--{B}--\r\n").encode()
    req = urllib.request.Request("http://127.0.0.1:8000/api/predict", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={B}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())

for rel in ("negatives/013.jpg", "negatives/006.jpg"):
    j = post(ROOT/"data/splits"/rel)
    print(rel, "->", json.dumps(j["scope"], ensure_ascii=False))

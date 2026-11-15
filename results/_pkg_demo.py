
import json, sys, urllib.request, uuid
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\项目作业\food_project")
def post(path):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"portion\"\r\n\r\nnormal\r\n").encode()
    body += (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
             f"Content-Type: image/jpeg\r\n\r\n").encode() + path.read_bytes() + b"\r\n" + f"--{b}--\r\n".encode()
    req = urllib.request.Request("http://127.0.0.1:8000/api/predict", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
h = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=30).read())
print("health packaging_head:", h["packaging_head"], "food_gate:", h["food_gate"]["enabled"])
for rel in ("results/_discarded_v3/pizza/IMG_pizza_0003.jpg", "results/_discarded_v3/ice_cream/IMG_ice_cream_0002.jpg"):
    j = post(ROOT/rel); s = j["scope"]
    print("\n==", rel)
    print("  in_scope", s["in_scope"], "not_food", s.get("not_food"), "p_pack", s.get("packaging_prob"),
          "p_food", s.get("food_prob"), "top1", s.get("closest"), s.get("top1_prob"))
    print("  message:", s.get("message"))
    print("  top3:", [i["class"] for i in j["top3"]], "note:", j["note"][:40])

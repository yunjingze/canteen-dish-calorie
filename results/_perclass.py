import numpy as np, json, sys
from pathlib import Path
ROOT = Path(r"E:\项目作业\food_project")
sys.path.insert(0, str(ROOT/"src"))
from foodcls.classes import CLASS_NAMES
def sm(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e/e.sum(-1, keepdims=True)
res = {}
for split in ("test","canteen"):
    d = np.load(ROOT/"results"/f"_ens_resnet18_last_stage_aug_f101n__{split}.npz")
    z, y = d["logits"].astype(np.float64), d["labels"]
    p = sm(z.mean(1))
    pred = p.argmax(1)
    per = {c: {"n": int(((y==i)).sum()), "acc": float((pred[y==i]==i).mean())} for i,c in enumerate(CLASS_NAMES)}
    conf = {}
    for i,c in enumerate(CLASS_NAMES):
        m = y==i
        wrong = pred[m][pred[m]!=i]
        conf[c] = {CLASS_NAMES[j]: int((wrong==j).sum()) for j in np.unique(wrong)}
    res[split] = {"per_class": per, "confusions": conf}
J = {c: res["canteen"]["per_class"][c]["acc"] for c in CLASS_NAMES}
T = {c: res["test"]["per_class"][c]["acc"] for c in CLASS_NAMES}
print(f"{'class':22}{'test':>7}{'canteen':>9}{'drop':>8}")
for c in sorted(CLASS_NAMES, key=lambda c: J[c]):
    print(f"{c:22}{100*T[c]:7.1f}{100*J[c]:9.1f}{100*(T[c]-J[c]):8.1f}")
print()
for c in sorted(CLASS_NAMES, key=lambda c: J[c])[:6]:
    print(c, "->", dict(sorted(res["canteen"]["confusions"][c].items(), key=lambda kv:-kv[1])[:4]))
(ROOT/"results/tables/per_class_domain.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

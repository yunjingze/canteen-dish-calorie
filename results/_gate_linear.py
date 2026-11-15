"""Frozen-feature linear gate: can 138 negatives teach a food/non-food separator that generalises to unseen negative categories?"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageOps
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(r"E:\项目作业\food_project")
sys.path.insert(0, str(ROOT / "src"))
from foodcls.classes import CLASS_NAMES
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from foodcls.models import build_model
from foodcls.utils import pick_device

NORM = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
TF = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), *NORM])

def load():
    ck = torch.load(ROOT/"results/runs/resnet18_last_stage_aug_f101n/best.pt", map_location="cpu", weights_only=False)
    m = build_model(ck["arch"], len(CLASS_NAMES), pretrained=False)
    m.load_state_dict(ck["model"]); m.to(pick_device()).eval(); return m

def feats(model, paths, device, bs=64):
    """Penultimate (512-d pooled) features; one centre-crop view."""
    store = {}
    h = model.avgpool.register_forward_hook(lambda m, i, o: store.__setitem__("f", o))
    out = []
    def run(batch):
        x = torch.stack([TF(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in batch]).to(device)
        model(x)
        return store["f"].flatten(1).float().cpu().numpy()
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            out.append(run(paths[i:i+bs]))
            if (i // bs) % 10 == 0: print("  feats", i, "/", len(paths), flush=True)
    h.remove()
    return np.concatenate(out)

def main():
    device = pick_device(); model = load()
    cats = {}
    with open(ROOT/"data/splits/negatives/sources.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            cats[Path(row["file"]).name] = row["category"]
    neg_paths = sorted((ROOT/"data/splits/negatives").glob("*.jpg"))
    neg_cat = [cats.get(p.name, "v1_unknown") for p in neg_paths]

    rng = np.random.default_rng(0)
    train_all = sorted((ROOT/"data/splits/train").rglob("*.jpg"))
    pos_train = [train_all[i] for i in rng.choice(len(train_all), 4000, replace=False)]
    pos_clean = sorted((ROOT/"data/splits/canteen_clean").rglob("*.jpg"))
    pos_real = sorted((ROOT/"data/splits/canteen_real").rglob("*.jpg"))
    pos_test = [train_all[i] for i in rng.choice(len(train_all), 600, replace=False)]

    print("extracting features...", flush=True)
    Xn = feats(model, neg_paths, device)
    Xp = feats(model, pos_train + pos_clean + pos_real + pos_test, device)
    n_pt, n_pc, n_pr = len(pos_train), len(pos_clean), len(pos_real)
    Xp_tr, Xp_cl, Xp_re, Xp_te = (Xp[:n_pt], Xp[n_pt:n_pt+n_pc],
                                  Xp[n_pt+n_pc:n_pt+n_pc+n_pr], Xp[n_pt+n_pc+n_pr:])
    np.savez(ROOT/"results/tables/gate_features.npz", Xn=Xn, Xp_tr=Xp_tr, Xp_cl=Xp_cl, Xp_re=Xp_re,
             Xp_te=Xp_te, neg_cat=np.array(neg_cat))
    print("features", Xn.shape, Xp.shape, flush=True)

    cats_all = sorted(set(neg_cat))
    res = []
    for hold in cats_all:
        tr = np.array([c != hold for c in neg_cat]); te = ~tr
        if te.sum() < 3: continue
        X = np.r_[Xn[tr], Xp_tr, Xp_cl, Xp_re]; y = np.r_[np.zeros(tr.sum()), np.ones(len(Xp_tr)+len(Xp_cl)+len(Xp_re))]
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
        neg_te = clf.predict_proba(sc.transform(Xn[te]))[:, 1]
        pos_ho = clf.predict_proba(sc.transform(np.r_[Xp_te, Xp_re]))[:, 1]
        auc = roc_auc_score(np.r_[np.zeros(len(neg_te)), np.ones(len(pos_ho))], np.r_[neg_te, pos_ho])
        thr = np.quantile(pos_ho, 0.05)  # keep 95% of food
        fpr = float((neg_te >= thr).mean())
        print(f"  hold={hold:16} n_neg={te.sum():3} AUC={auc:.3f}  FPR@95%food={fpr:.2f}", flush=True)
        res.append({"hold": hold, "n_neg": int(te.sum()), "auc": float(auc),
                    "fpr_at_95_tpr": fpr, "median_neg_p": float(np.median(neg_te)),
                    "median_pos_p": float(np.median(pos_ho))})

    # in-distribution 5-fold on all negatives (upper bound)
    from sklearn.model_selection import StratifiedKFold
    y = np.r_[np.zeros(len(Xn)), np.ones(len(Xp_tr))]
    X = np.r_[Xn, Xp_tr]
    sc = StandardScaler().fit(X)
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    print(f"  5-fold in-distribution AUC={np.mean(aucs):.3f} (+-{np.std(aucs):.3f})", flush=True)
    out = {"loco": res, "indist_auc": float(np.mean(aucs))}
    (ROOT/"results/tables/gate_linear.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/tables/gate_linear.json")

if __name__ == "__main__":
    main()

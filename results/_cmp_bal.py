
import json, pathlib, sys

def load(p):
    return json.loads(pathlib.Path(p).read_text("utf-8"))

a_path, b_path = sys.argv[1], sys.argv[2]
a, b = load(a_path), load(b_path)
print(f"{'':22s} {'base':>8s} {'new':>8s} {'delta':>7s}")
print(f"{'val top-1':22s} {a['best_val_acc']*100:8.2f} {b['best_val_acc']*100:8.2f} {(b['best_val_acc']-a['best_val_acc'])*100:+7.2f}")
for k in ("top1", "top5", "ece", "mean_top1_prob"):
    if k in a["test"] and k in b["test"]:
        print(f"{'test '+k:22s} {a['test'][k]*100:8.2f} {b['test'][k]*100:8.2f} {(b['test'][k]-a['test'][k])*100:+7.2f}")
rows = []
for name in a["classes"]:
    pa = a["test"]["per_class"].get(name, 0.0)
    pb = b["test"]["per_class"].get(name, 0.0)
    rows.append((pb - pa, name, pa * 100, pb * 100))
rows.sort()
print("\nworst 6:")
for d, name, pa, pb in rows[:6]:
    print(f"  {name:22s} {pa:6.1f} -> {pb:6.1f}  {d*100:+5.1f}")
print("best 6:")
for d, name, pa, pb in rows[-6:][::-1]:
    print(f"  {name:22s} {pa:6.1f} -> {pb:6.1f}  {d*100:+5.1f}")
print(f"\nmean per-class: {sum(r[2] for r in rows)/len(rows):.1f} -> {sum(r[3] for r in rows)/len(rows):.1f}")
print("epochs:", [h["epoch_time_s"] for h in b["history"]])

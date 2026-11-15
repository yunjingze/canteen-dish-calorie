import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(r"E:\项目作业\food_project")
data = json.loads((ROOT/"results/tables/real_photos_views.json").read_text(encoding="utf-8"))
def montage(paths, out, cols=5, cell=260):
    rows = (len(paths)+cols-1)//cols
    canvas = Image.new("RGB", (cols*cell, rows*(cell+22)), (245,245,245))
    d = ImageDraw.Draw(canvas)
    for i,p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        img.thumbnail((cell, cell))
        x = (i%cols)*cell + (cell-img.width)//2
        y = (i//cols)*(cell+22) + (cell-img.height)//2
        canvas.paste(img, (x,y))
        d.text(((i%cols)*cell+4, (i//cols)*(cell+22)+cell+4), Path(p).parent.name+"/"+Path(p).name, fill=(0,0,0))
    canvas.save(out, quality=88)
    print(out, canvas.size)

miss = [str(ROOT/r["path"]) for r in data["single_dish"]["rows"] if r["E7"]["fused"] != r["label"]]
montage(miss, str(ROOT/"results/_real_misses.png"), cols=5, cell=300)
neg = sorted((ROOT/"data/splits/negatives").glob("*.jpg"))
montage([str(p) for p in neg], str(ROOT/"results/_real_negatives.png"), cols=5, cell=300)

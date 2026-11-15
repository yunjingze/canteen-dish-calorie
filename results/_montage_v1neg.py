from pathlib import Path
from PIL import Image, ImageDraw
root = Path(r"E:\项目作业\food_project\data\splits\negatives")
files = [root/f"{i:03d}.jpg" for i in range(1,16)]
cell = 300
cols, rows = 5, 3
sheet = Image.new("RGB", (cols*cell, rows*cell), "white")
d = ImageDraw.Draw(sheet)
for k, f in enumerate(files):
    im = Image.open(f).convert("RGB")
    im.thumbnail((cell-8, cell-22))
    x, y = (k%cols)*cell, (k//cols)*cell
    sheet.paste(im, (x+4, y+4))
    d.text((x+6, y+cell-16), f.name, fill="black")
sheet.save(r"E:\项目作业\food_project\results\_v1_negatives.png")
print("ok")
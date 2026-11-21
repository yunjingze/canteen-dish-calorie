"""Copy the 11 demo shots under Chinese names for the screen recording.

The page never shows the file name (the preview is the photo itself), but the
name flashes in Explorer while you drag it onto the drop zone, and a reviewer
watching the video should be able to tell 01_酸辣土豆丝 from 04_包子 without
pausing.  Copies, not moves: data/external/demo_shots stays the canonical set
that check_demo_shots.py replays.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/external/demo_shots"
DST = ROOT / "data/external/demo_upload"

NAMES = {
    "01_hot_sour_potato.jpg": "01_酸辣土豆丝",
    "02_braised_pork.jpg": "02_红烧肉",
    "03_di_san_xian.jpg": "03_地三鲜",
    "04_baozi.jpg": "04_包子",
    "05_pizza.jpg": "05_披萨",
    "06_not_food.jpg": "06_不是菜",
    "07_off_list_cn.jpg": "07_鱼香肉丝_清单外",
    "08_mapo_tofu.jpg": "08_麻婆豆腐_清单外",
    "09_peking_duck.jpg": "09_北京烤鸭_清单外",
    "10_mixed_burger_fries.jpg": "10_西式拼盘",
    "11_mixed_cn_tray.jpg": "11_食堂拼盘",
}


def main():
    DST.mkdir(parents=True, exist_ok=True)
    n = 0
    for src_name, stem in NAMES.items():
        src = SRC / src_name
        if not src.exists():
            print("missing", src_name)
            continue
        shutil.copyfile(src, DST / (stem + ".jpg"))
        n += 1
    print(f"copied {n} files to {DST}")


if __name__ == "__main__":
    main()

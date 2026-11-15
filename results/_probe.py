import sys, torch, numpy as np
sys.path.insert(0, r"E:\\项目作业\\food_project\\src")
from PIL import Image
from torchvision import transforms
from foodcls.models import build_model
from foodcls.classes import CLASS_NAMES
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
tfs = {
 "v0_256crop": transforms.Compose([transforms.Resize(int(IMG_SIZE*256/224)), transforms.CenterCrop(IMG_SIZE)]+norm),
 "v1_square":  transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))]+norm),
 "v2_320crop": transforms.Compose([transforms.Resize(int(IMG_SIZE*320/224)), transforms.CenterCrop(IMG_SIZE)]+norm),
}
img = Image.open(r"E:\\项目作业\\food_project\\results\\_probe_pkg.png").convert("RGB")
img = img.crop((178, 153, 505, 483))
img.save(r"E:\\项目作业\\food_project\\results\\_probe_crop.png")

for tag, ck in [("E4", "resnet18_last_stage_aug"), ("E7", "resnet18_last_stage_aug_f101n")]:
    c = torch.load(rf"E:\\项目作业\\food_project\\results\\runs\\{ck}\\best.pt", map_location="cpu", weights_only=False)
    m = build_model(c["arch"], len(CLASS_NAMES), pretrained=False); m.load_state_dict(c["model"]); m.eval()
    print("==", tag, "temp", c.get("temperature"))
    for name, tf in tfs.items():
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(m(x).float(), 1)[0]
            pf = torch.softmax(m(torch.flip(x, dims=[3])).float(), 1)[0]
        avg = (p+pf)/2
        top = torch.topk(avg, 4).indices.tolist()
        print(f"  {name:12s} maxview={p.max()*100:5.1f} flip={pf.max()*100:5.1f} | " + ", ".join(f"{CLASS_NAMES[i]}={avg[i]*100:.1f}" for i in top))
    # mean of all 3 views with flip
    xs = torch.cat([tfs[k](img).unsqueeze(0) for k in tfs]); 
    with torch.no_grad():
        allp = torch.softmax(m(torch.cat([xs, torch.flip(xs, dims=[3])]).float()), 1).mean(0)
    top = torch.topk(allp, 4).indices.tolist()
    print("  6view-avg   " + ", ".join(f"{CLASS_NAMES[i]}={allp[i]*100:.1f}" for i in top))

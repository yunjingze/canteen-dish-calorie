# 校园食堂菜品识别与热量区间估算

课程项目作业 02 的完整实现：拍照识别 20 类菜品，输出 **Top-3 菜品 + 粗略热量区间**，并配有可交互的网页界面。

- 数据集：**Food-101**（Bossard et al., ECCV 2014）官方划分的子集，20 类，训练 12,000 / 验证 2,000 / 测试 5,000 张。
- 模型：ResNet-18、MobileNetV3-Small（ImageNet 预训练 + 分类头重训 + 最后一阶段解冻微调）。
- 对比实验：冻结 vs 解冻、公开测试集 vs 食堂代理域、颜色增强前后、ImageNet 预训练 vs 随机初始化。
- 额外分析：置信度校准（温度缩放）、份量切换与手动输入重量对热量区间的影响。
- 混合餐盘：多标签模型判断盘子里同时有哪几样菜（薯条 + 鸡翅，或西红柿炒蛋 + 炒青菜 + 米饭这类食堂托盘），每部分单独选份量或填克数，输出各部分与整盘热量区间。
- 指标：Accuracy、Macro-F1、混淆矩阵、跨域性能、ECE、Params、MACs、GPU/CPU 推理耗时。

## 1. 目录结构

```
food_project/
├─ data/                     # 数据（不入库，见第 3 节）
│  ├─ raw_hf/               # HuggingFace food101 parquet 原始分片
│  ├─ raw/food-101/         # 抽取后的 images/ 与 images_test/（20 类）
│  ├─ wide/                 # 116 类（Food-101 101 类 + 15 道中餐）：raw/ 与 splits/{train,val,test}
│  ├─ external/             # brand_menu.json（麦当劳中国官网营养表，见 5.6）、cnfood241_*.zip
│  └─ splits/               # train / val / test / canteen / canteen_clean / mixed / cn241_full
├─ src/foodcls/
│  ├─ classes.py            # 20 类定义：英文名、中文名、每 100 g 热量、常见份量
│  ├─ classes_wide.py       # 116 类定义（前 101 个为 Food-101，按字母序对齐旧权重）
│  ├─ classes_cn.py         # 食堂 20 道家常菜定义（CNFOOD-241 子集，中文名、热量、份量）
│  ├─ plate_classes.py      # 混合餐盘 40 类：20 西餐 + 20 食堂（顺序固定，对齐权重）
│  ├─ calorie.py            # 热量区间估算（低置信度放宽、份量缩放、手动重量）
│  ├─ brands.py             # 连锁品牌营养参考（官网菜单 → 类别对照与可搜索面板）
│  ├─ mixed.py              # 混合餐盘：合成多标签样本（20/40 类通用）、阈值检测、各部分热量合计
│  ├─ data.py               # 数据集、变换、食堂拍摄退化流水线
│  ├─ models.py             # 三种网络 + 三种微调策略
│  ├─ train.py              # 训练/评估入口（AMP、AdamW、余弦退火、按 val Macro-F1 存最优）
│  ├─ utils.py              # 指标、ECE、Params、THOP MACs、延迟测量
│  ├─ app.py                # FastAPI 网页服务（含 /api/qr.png、manifest、sw.js）
│  ├─ predict.py            # 命令行单图预测
│  └─ static/               # 前端单页 index.html、PWA manifest、Service Worker、应用图标
├─ miniprogram/             # 微信小程序版（导入微信开发者工具，改 config.js 的 BASE_URL）
├─ docs/                    # 使用文档：永久公网地址.md、微信小程序上手指南.md
├─ scripts/
│  ├─ fetch_food101_hf.py   # 从 hf-mirror 下载 parquet 分片
│  ├─ extract_food101_hf.py # 抽取 20 类（或 --all 抽全 101 类）到本地目录结构
│  ├─ prepare_data.py       # 生成 train/val/test/canteen 划分
│  ├─ prepare_wide_data.py  # Food-101 101 类的 train/val/test 划分（650/100/250 每类）
│  ├─ fetch_kaggle.py       # 匿名下载 Kaggle 数据集（多线程分段 Range）
│  ├─ add_chinese_dishes.py # 解压 16 道中餐，按自带划分并入 data/wide/splits
│  ├─ fetch_brand_nutrition.py # 抓麦当劳中国官网营养表 → data/external/brand_menu.json
│  ├─ run_all.py            # 一键跑 6 组实验 + benchmark + 画图
│  ├─ train_mixed.py        # 训练多标签混合餐盘模型（--classes plate 时 20 西餐 + 20 食堂）
│  ├─ train_wide.py         # 训练 116 类模型（--init 热启动扩类，三视角 TTA + 温度缩放）
│  ├─ make_mixed_samples.py # 生成网页用的混合餐盘示例图与真值 labels.json（含 cuisine）
│  ├─ eval_mixed_plate.py   # 40 类餐盘的验证盘评估：阈值扫描、逐类阈值、真实照旁证
│  ├─ benchmark.py          # Params / MACs / GPU-CPU 延迟
│  ├─ make_figures.py       # 论文与报告用图
│  ├─ make_paper_numbers.py # 从结果文件生成 LaTeX 数字宏
│  ├─ make_readme.py        # 由结果生成下方表格
│  ├─ fit_temperature.py    # 温度缩放校准（val 上拟合，输出 calibration.json）
│  ├─ eval_real_photos.py   # 用真实食堂照片评估（两种目录布局）
│  ├─ eval_mixed.py         # 混合餐盘阈值细扫（写 mixed_threshold.json / 画阈值曲线）
│  ├─ eval_mixed_plate.py   # 40 类餐盘评估：西餐盘 / 食堂盘各 600 个，扫阈值 + 逐类校准
│  ├─ eval_openset.py       # 开集门控：三视角一致性，误报/检出率与分布图
│  ├─ eval_wide_domain.py   # 116 类模型在 20 类食堂数据上的对比评估
│  ├─ eval_wide_gate.py     # 116 类模式的门控阈值（官方 test 最差分数的 5% 分位）
│  ├─ eval_wide_groups.py   # 116 类 test 评测：单视角 / TTA / ECE / 分组 / 每类
│  ├─ pdl.py                # 多线程 HTTP Range 下载器（大文件直链，12 线程约 12 MB/s）
│  ├─ prepare_cn241.py      # 解压 CNFOOD-241 并抽食堂 20 道 → data/splits/cn241_full
│  ├─ train_cn241.py        # 训练 / 线性探针 20 道食堂模型（--mode probe|finetune）
│  ├─ eval_cn_scope.py      # 食堂模型的门控阈值（清单内保留率 vs 清单外挡掉率）
│  ├─ eval_cn_domain.py     # 食堂模型在真实食堂照（西餐）上的越界行为
│  ├─ eval_tta.py           # 测试时增强（翻转/五裁剪/多尺度）与代理域重抽（种子 7）
│  ├─ finalize_wide.py      # 单次遍历 test 重算单视角/TTA/ECE，写回 results.json
│  ├─ make_wide_figure.py   # 画 20 类 vs 116 类在同一评分下的对比图
│  ├─ start_demo.ps1        # 一键启动网页（监听 0.0.0.0，打印局域网地址，-Qr 出扫码图）
│  ├─ start_public.ps1      # 公网临时链接（Cloudflare 快速隧道，任何网络都能打开）
│  ├─ setup_cloudflare_tunnel.ps1 # 固定地址：创建 Cloudflare 命名隧道并绑定自己的域名
│  ├─ start_permanent.ps1    # 固定地址：启动后端 + 命名隧道（配合上一个脚本）
│  ├─ start_funnel.ps1       # 固定地址：Tailscale Funnel（不买域名）
│  ├─ start_ngrok.ps1        # 固定地址：ngrok 免费固定域名（有配额）
│  ├─ package_demo.ps1      # 打包成可发给他人独立运行的 zip
│  ├─ demo_screenshot.ps1   # 启动服务并截图网页
│  └─ make_icons.py         # 生成 PWA 图标（192 / 512 / maskable）
├─ tests/                   # unittest：热量区间规则、类别表/变换/模型工厂契约
├─ results/
│  ├─ runs/<实验名>/        # best.pt, last.pt, history.csv, results.json, pred_*.npz
│  │                        # （含 20 类的 E1–E6、resnet18_multilabel_*、resnet18_food101、
│  │                        #  resnet18_food101_cn116）
│  ├─ figures/              # 混淆矩阵、曲线、Grad-CAM、开集分布、缩放对比、网页截图
│  └─ tables/               # complexity.json、summary.csv、mixed*.json、openset_gate.json、wide_*.json
└─ paper/                   # IEEE 会议格式英文论文（LaTeX）
```

## 2. 环境

- Windows + Python 3.13，NVIDIA RTX 4060 Laptop (8 GB)，PyTorch 2.11.0 + CUDA 12.8。
- 安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

预训练权重默认缓存到 `food_project/.torch`（由 `src/foodcls/__init__.py` 设置 `TORCH_HOME`），无需写入用户目录。

单元测试用标准库 `unittest`，不需要 pytest，也不需要数据集（21 个用例，约 1 秒）：

```powershell
cd "E:\项目作业\food_project"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

覆盖：热量区间的边界（克数覆盖份量先验、低置信度加宽、取整）、类别表一致性、训练/评测变换的形状、模型工厂与微调冻结策略。

## 3. 数据准备

```powershell
# 1) 下载 Food-101 的 parquet 分片（约 4.7 GB，来自 hf-mirror 镜像）
python scripts/fetch_food101_hf.py --workers 4
# 2) 抽取本实验使用的 20 类
python scripts/extract_food101_hf.py
# 3) 划分：官方 train 750/类 -> train 600 + val 100 + probe 50；官方 test 250/类 -> test
python scripts/prepare_data.py
```

划分原则：

- 训练、验证、测试三部分**互不重叠**，且测试集严格来自 Food-101 官方 test 划分。
- `canteen_clean`（每类 50 张，共 1,000 张）不参与训练和选模，只用于生成 `canteen` 代理域。
- `canteen` 是对 `canteen_clean` 做手机拍摄退化（旋转、透视、裁剪、暖光/荧光白平衡、亮度对比度、高斯/运动模糊、暗角、噪声、JPEG 35–65）得到的**代理域**，用于模拟"食堂实拍"。真实手机照片可直接上传到网页界面，或放进 `data/splits/canteen/<类名>/` 后重跑评估脚本，无需改代码。

另外三套数据不参与课程协议的 20 类划分，只在第 5.5、5.7 节的扩展实验里用：Food-101N 的 20 类清洁标注子集（`data/external/food101n_20cls`，用于 E7 微调）、16 道中餐名菜（`data/external/16chinese.zip` → `data/wide`）与 CNFOOD-241 的 20 道食堂家常菜（`data/external/cnfood241_*.zip` → `data/splits/cn241_full`）。下载命令见第 4 节对应小节，各数据集的许可与归属见根目录 `NOTICE`。

## 4. 训练与评估

```powershell
cd src
# 单组实验（示例：解冻最后阶段 + 颜色增强，12 epochs）
.\.venv\Scripts\python.exe -m foodcls.train --data-root ../data --out ../results/runs/resnet18_last_stage_aug --tag resnet18_last_stage_aug --arch resnet18 --mode last_stage --epochs 12 --color-aug
```

`--mode` 可选 `frozen` / `last_stage` / `full`；`--arch` 可选 `resnet18` / `mobilenet_v3_small` / `efficientnet_b0`。
加 `--no-pretrained` 可从头训练（随机初始化），用于消融"ImageNet 预训练到底贡献多少"：

```powershell
.\.venv\Scripts\python.exe -m foodcls.train --data-root ../data --out ../results/runs/resnet18_scratch_noaug --tag resnet18_scratch_noaug --arch resnet18 --mode full --no-pretrained --epochs 12
```

一键复现全部 6 组实验、复杂度基准与所有图表：

```powershell
.\.venv\Scripts\python.exe scripts/run_all.py
```

训练使用 AdamW（头 1e-3 / 主干 1e-4）、余弦退火、label smoothing 0.1、AMP，按验证集 Macro-F1 选择最优权重，训练结束后一次性评测 val / test / canteen 三个集合。

混合餐盘模型（多标签，二值交叉熵）单独训练，盘面由单菜训练图现场合成（默认 20 道西餐；--classes plate 时再接上食堂 20 道，共 40 类）：

```powershell
.\.venv\Scripts\python.exe scripts/train_mixed.py --classes plate --epochs 10 --n-train 24000 --n-canteen 300 --workers 4 --init-ckpt results/runs/resnet18_multilabel_aug/best.pt --out results/runs/resnet18_multilabel_plate40
.\.venv\Scripts\python.exe scripts/eval_mixed_plate.py
.\.venv\Scripts\python.exe scripts/make_mixed_samples.py
```

每个训练盘以 35% / 45% / 20% 的概率放 1 / 2 / 3 样菜（斜切或三扇区掩膜羽化混合），目标为对应类数的多热向量（20 或 40 维）；阈值在验证盘上扫描取 F1 最优，再按类各校准一遍，同时报告同一权重在单菜测试集上的 Top-1，确保多标签能力没有把单菜识别带崩。细节与 40 类结果见 5.3。

116 类扩展模型（Food-101 101 类 + 15 道中餐；从 101 类权重热启动，全量微调）：

```powershell
.\.venv\Scripts\python.exe scripts/fetch_kaggle.py --dataset patzer0/16-famous-chinese-dishes --out data/external/16chinese.zip
.\.venv\Scripts\python.exe scripts/add_chinese_dishes.py
.\.venv\Scripts\python.exe scripts/prepare_wide_data.py
.\.venv\Scripts\python.exe scripts/train_wide.py --epochs 10 --batch-size 96 --workers 6 --out results/runs/resnet18_food101
.\.venv\Scripts\python.exe scripts/train_wide.py --epochs 4 --batch-size 96 --workers 6 --init results/runs/resnet18_food101/best.pt --lr-head 5e-4 --lr-backbone 5e-5 --out results/runs/resnet18_food101_cn116
.\.venv\Scripts\python.exe scripts/eval_wide_groups.py --run results/runs/resnet18_food101_cn116
.\.venv\Scripts\python.exe scripts/eval_wide_domain.py
```

食堂 20 道家常菜（CNFOOD-241 子集，见 5.7）：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_cn241.py --class-xls data/external/cnfood241_class_name.xls --train-zip data/external/cnfood241_train600x600.zip --val-zip data/external/cnfood241_val600x600.zip --out data/splits/cn241_full
.\.venv\Scripts\python.exe scripts/train_cn241.py --mode probe --data-root data/splits/cn241_full --classes results/tables/cn241_classes_full.json
.\.venv\Scripts\python.exe scripts/train_cn241.py --mode finetune --data-root data/splits/cn241_full --classes results/tables/cn241_classes_full.json --init results/runs/resnet18_last_stage_aug_f101n/best.pt --epochs 8 --batch-size 64 --workers 0 --out results/tables/cn241_full.json --save-dir results/runs/cn241_full
.\.venv\Scripts\python.exe scripts/eval_cn_scope.py
.\.venv\Scripts\python.exe scripts/eval_cn_domain.py
```

两个压缩包用 `scripts/pdl.py` 下载（Mendeley 直链要带 `User-Agent`，单线程只有约 24 MB/min）：

```powershell
.\.venv\Scripts\python.exe scripts/pdl.py --url "<train600x600.zip 的 file_downloaded 链接>" --out data/external/cnfood241_train600x600.zip --threads 12
.\.venv\Scripts\python.exe scripts/pdl.py --url "<val600x600.zip 的 file_downloaded 链接>" --out data/external/cnfood241_val600x600.zip --threads 12
```

## 5. 实验结果

| ID | 模型 / 策略 | 颜色增强 | Test Acc | Test Macro-F1 | Proxy Acc | Cross-domain Gap | Params (M) | MACs (G) | 训练时长 (min) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | ResNet-18 · 冻结 | 否 | 84.44% | 84.43% | 64.00% | 20.4 pt | 11.19 | 1.82 | 4.0 |
| E2 | ResNet-18 · 冻结 | 是 | 84.08% | 84.04% | 64.30% | 19.8 pt | 11.19 | 1.82 | 5.6 |
| E3 | ResNet-18 · 解冻最后阶段 | 否 | 89.74% | 89.76% | 64.00% | 25.7 pt | 11.19 | 1.82 | 3.8 |
| E4 | ResNet-18 · 解冻最后阶段 | 是 | 90.54% | 90.55% | 66.00% | 24.5 pt | 11.19 | 1.82 | 5.9 |
| E5 | MobileNetV3-Small · 解冻最后阶段 | 是 | 81.92% | 81.94% | 57.20% | 24.7 pt | 1.54 | 0.06 | 4.9 |
| E6 | ResNet-18 · 随机初始化（全网络） | 否 | 66.04% | 65.71% | 35.60% | 30.4 pt | 11.19 | 1.82 | 7.4 |

推理耗时（FP32，batch=1，预热 10 次后取 50 次均值）：

| ID | 模型 | GPU 单张 (ms) | CPU 单张 (ms) | GPU FPS |
| --- | --- | --- | --- | --- |
| E1 | ResNet-18 · 冻结 | 3.39 | 16.78 | 295 |
| E2 | ResNet-18 · 冻结 | 3.20 | 15.52 | 312 |
| E3 | ResNet-18 · 解冻最后阶段 | 3.79 | 17.10 | 264 |
| E4 | ResNet-18 · 解冻最后阶段 | 3.70 | 17.04 | 270 |
| E5 | MobileNetV3-Small · 解冻最后阶段 | 5.77 | 9.04 | 173 |
| E6 | ResNet-18 · 随机初始化（全网络） | 4.36 | 16.19 | 230 |

图表位于 `results/figures/`：训练曲线、对比柱状图、测试集与代理域混淆矩阵、Top-3 预测样例、Grad-CAM、可靠性图（校准）、复杂度对比、混合餐盘阈值曲线（`mixed_threshold.png`）、开集门控分布（`openset_gate.png`）与网页截图（`demo_ui.png`、`demo_mixed_preview.png`、`demo_out_of_scope.png`、`demo_wide.png`、`demo_public.png`、`demo_public_qr.png`、品牌热量面板 `demo_brand.png`），116 类扩展的缩放对比（`wide_scale.png`）与食堂 20 道模式的页面截图（`demo_cn.png`）。

### 5.1 置信度校准（温度缩放）

深度网络普遍**过度自信**，页面上的"概率"若直接当可信度用会误导。用 Guo et al. (ICML 2017) 的温度缩放：在**验证集**上拟合单个温度 `T`（最小化 NLL），推理时用 `softmax(logits / T)`。`T` 不改变预测类别，只改变置信度，因此 Acc / Macro-F1 不变、ECE 下降。

```powershell
python scripts/fit_temperature.py --all      # 或 --run results/runs/<实验名>
```

结果写入各 run 的 `calibration.json` 与汇总表 `results/tables/calibration.json`；网页服务启动时会自动读取（存在即生效，不存在则 T=1），健康检查接口会返回当前温度。可靠性图中实线为校准前、虚线为校准后。

### 5.2 用真实照片评估

代理域再像也不是真实拍摄。`scripts/eval_real_photos.py` 支持两种布局：

```powershell
# A) 每个类别一个子目录（推荐）
python scripts/eval_real_photos.py --data data/real_photos --ckpt results/runs/resnet18_last_stage_aug/best.pt
# B) CSV 清单：path,label（label 用英文类名）
python scripts/eval_real_photos.py --manifest data/real_photos.csv
```

脚本会按 EXIF 修正手机照片方向，输出 Top-1 / Top-3、每类样本数与错分清单，写 `results/tables/real_photos.json`。手机随手拍 10–20 张/类即可看出代理域评估与真实场景的差距。

### 5.3 混合餐盘（多标签，20 道西餐 + 20 道食堂菜）

单标签 softmax 无法表达"盘子里既有薯条也有鸡翅"。做法是把 2–3 张图合成到一块盘面（斜切 / 三扇区，或一大块主菜上加小块贴片，接缝用羽化去掉黑角），用同一个 ResNet-18 结构、把最后的 softmax 换成 40 个 sigmoid，以 BCE 训练：

- 类别表是 **20 道西餐（Food-101）+ 20 道食堂家常菜**（5.7 那 20 道；权重顺序 0–19 西餐、20–39 食堂）。这样同一张食堂托盘（西红柿炒蛋 + 炒青菜 + 米饭）也能拆成几部分分别算热量，而不是被硬套成"薯条 / 牛排"；中西混搭（可乐鸡翅 + 炒面）同样一次前向就能拆。
- 训练池 28,717 张单菜图：西餐 `splits/train` 12,000 + 食堂 `cn241_21/train` 的 20 道 16,717（「其它中餐」桶不进拼盘，否则一张糊图会被当成"一样菜"）。每个训练盘现场合成，单样 / 两样 / 三样按 35% / 45% / 20%。
- **20 行西餐头直接继承旧模型**：从 `resnet18_multilabel_aug/best.pt` 热启动，把 20 行分类头搬进新的 40 行头（前 20 行照抄、后 20 行从零学），其余张量原样复制；只解冻最后阶段，10 epoch、batch 64、lr head 1e-3 / backbone 1e-4、约 160 s/epoch（整轮 27 分钟，8 GB 卡够用）。
- 验证盘 1,200 个（西餐 600 + 食堂 600，确定性合成，种子 1234 / 4321），另有 300 个食堂代理盘，以及用 100 张真实食堂照拼成的 80 个盘做旁证——真实照那一组只进报告，不参与挑阈值。
- 盘面两种布局各占约一半：**斜切 / 三扇区**（两样菜各占半边，适合米饭 + 炒菜）与**小块贴片**（一样菜铺满画面，其余菜以 6%–35% 面积的椭圆羽化块贴在角落，模拟"一大盘薯条配几块鸡翅"）。只教前者时，模型会把"占半张图"当成出镜条件，小份配菜直接消失。
- 推理做**翻转增强**（原图与水平翻转各跑一次，逐类取较大分数），再用**每类各自校准的阈值**判断是否算盘中一部分（最多 3 样）；低于 10% 的分数一律不显示，一样都不过线时返回空列表并提示"换角度 / 切到单菜识别"，不再硬给一个 4% 的答案。
- 每部分带 `cuisine`（`western` / `canteen`）：中西混搭时页面提示"两份主食容易都算成整份，可以手动把其中一份改成小份"，纯食堂托盘另有说明，热量仍按各自的分量区间相加。

| 指标 | 数值 |
| --- | --- |
| 类别数 | 40（20 西餐 + 20 食堂） |
| 全局检测阈值（验证盘扫描） | 0.35 |
| 每类阈值范围（40 类各自校准） | 0.10 – 0.65 |
| 验证盘 · 每部分 P / R / F1（西餐 600 盘） | 61.1% / 47.1% / 53.2% |
| 验证盘 · 每部分 P / R / F1（食堂 600 盘） | 62.8% / 50.4% / 55.9% |
| 逐类阈值之后 · 西餐盘 / 食堂盘 F1 | 54.0% / 56.1% |
| 整盘完全正确（西餐盘 / 食堂盘） | 10.5% / 11.3% |
| 走完 `detect_parts`（含 10% 下限）后整盘正确 | 8.2% / 8.5% |
| 整盘一张都没过线 | 0.7% / 0.0% |
| 单菜测试集 Top-1：真实食堂照（西餐）/ 食堂 test | 70.0% / 81.8% |
| 单菜照片平均预测部分数（西餐盘 / 食堂盘） | 1.85 / 1.92 |
| 100 张真实食堂照拼成 80 盘（仅旁证）· F1 / 整盘正确 | 44.0% / 6.3% |

- **和旧的 20 类模型比**：同一批 600 个西餐验证盘上，F1 从 55.8% 降到 53.2%（`results/tables/mixed.json` → `mixed_plate40.json`）：precision 基本没动（60.9% → 61.1%），recall 从 51.5% 掉到 47.1%，每盘报的部分数反而从 2.04 降到 1.85——40 路头要在多出 20 个类上分概率，宁可少报也不乱报。换来的是一张食堂托盘从"根本认不出"到 55.9% F1。多出来的误报集中在 waffles / sushi / boiled_chicken。
- 验证盘那一行有选择泄漏（同一批盘既挑阈值又报数），应把它当上界；食堂 600 盘没参与挑阈值、用同一套阈值仍有 55.9%，说明食堂侧不是靠拟合阈值撑起来的。
- **真实照那一行才是部署水平**：100 张真实食堂照拼的 80 个盘只有 44.0% F1、6.3% 整盘正确，远低于合成盘。真实摆盘的油光、遮挡、米饭压住菜都不在合成分布里；要往上走只能靠 `data/splits/feedback` 攒真实托盘照（见第 6 节反馈闭环），这是当前最大的 gap。
- 示例盘也照出同一个边界：17 张合成示例里整盘完全正确的只有 1 张，mix_01（薯条 + 鸡翅）新旧两个模型给鸡翅都只有 0.02 上下——小块贴片是最难的情形；旧文档里"0.89"那个数字来自另一版示例图，已经不适用。
- 每部分热量 = 该类每 100 g 先验 × 该部分重量（份量档或手填克数），整盘区间 = 各部分区间之和，所以区间仍然覆盖做法差异，而不是把点估计简单相加。
- 复现：`python scripts/train_mixed.py --classes plate --epochs 10 --n-train 24000 --n-val 600 --n-canteen 300 --n-real 80 --workers 4 --init-ckpt results/runs/resnet18_multilabel_aug/best.pt --out results/runs/resnet18_multilabel_plate40`；评估 `python scripts/eval_mixed_plate.py`（写 `results/tables/mixed_plate40.json` 与 `results/figures/mixed_plate40.png`）；示例图 `python scripts/make_mixed_samples.py`。

### 5.4 不认识的菜：开集门控

20 类 softmax 永远会报一个类，所以拍整只烤鸡（Food-101 里没有这道菜）会得到"烤三文鱼 49%"。直接用置信度阈值不够：烤鸡在单视角下的最高概率是 0.49，落在正常菜品的中段。有效的信号是**换取景后是否还认同一道菜**——把同一张图按三种裁剪各跑一次（训练时的 Resize256+CenterCrop224、整帧压成 224×224、Resize288+CenterCrop224），正常菜品三次都稳定，不认识的菜会来回改判。取三个视角里**最差的 top-1 概率**作为稳健分数：

| 分数 | AUC | 误报率 @ 阈值 0.32 | 检出率（20 个未见类） |
| --- | --- | --- | --- |
| 单视角 top-1（原做法） | 0.822 | 2.9% | 13.0% |
| 三视角最差（采用） | 0.830 | 5.3% | 26.7% |

- 阈值 0.32 取在支持菜品分数的 5% 分位附近：900 张支持菜品（val 400 + 食堂代理 500）里 5.3% 会被标黄，代价可以接受。
- 烤鸡那张图的稳健分数是 0.16（页面实测 0.27，取决于裁剪），远低于 0.32，会被标出来。
- 页面上这是**软警告**而不是硬拒识：标题改成"大概不是这 20 道菜"，每个候选仍显示但标"仅供参考"，并提示整只烤鸡/烤鸭、火锅、家常炒菜都不在这 20 类里。这样既不会在食堂正常菜上拒答，也不会假装认识。
- 复现：`python scripts/eval_openset.py`（约 3 分钟 GPU，结果写 `results/tables/openset_gate.json` 并画 `openset_gate.png`；`--cached` 复用已算好的 npz）。

### 5.5 116 类扩展：Food-101 + 15 道中餐名菜

课程协议固定了 20 类，食堂之外的菜（北京烤鸭、西班牙海鲜饭、石锅拌饭、越南河粉……）一律落入“大概不是这 20 道菜”。为了扩大覆盖面，另外训了一个 **116 类**模型，作为网页的第三种模式：完整 Food-101 加上 15 道中餐名菜（麻婆豆腐、宫保鸡丁、回锅肉、水煮鱼、东坡肉等）。中餐数据来自 Kaggle 的 [16 Famous Chinese Dishes](https://www.kaggle.com/datasets/patzer0/16-famous-chinese-dishes)，由 `scripts/fetch_kaggle.py` 匿名直下、`scripts/add_chinese_dishes.py` 并入现有划分。

- **数据**：Food-101 的 101 类（每类官方 train 取 650 / 100 张作 train / val、官方 test 取 250 张，共 65,650 / 10,100 / 25,250 张）＋ 16 道中餐名菜 5,792 张实拍（自带 train/val/test；北京烤鸭并入 Food-101 已有的 `peking_duck`，其余 15 类新增）。合计 70,283 / 10,679 / 25,830 张，是 20 类划分（12,000）的 5.8 倍。中文类名、每 100 g 热量与常见份量在 `src/foodcls/classes_wide.py`。
- **训练**：先按原配方把 Food-101 101 类全量微调 10 epochs（头 1e-3 / 主干 2e-4）；再用 `--init` 从这份权重**热启动**，fc 的前 101 行原样复制、新增 15 行随机初始化，以更小的学习率（头 5e-4 / 主干 5e-5）全量微调 4 epochs。label smoothing 0.1，AMP，batch 96。
- **官方 test（116 路）**：单视角 Top-1 80.58% / Top-5 94.39%；三视角 TTA 82.35% / 95.24%。验证集最好一轮 Top-1 76.52%、macro-F1 77.22%。
- **分组**：原 Food-101 的 101 类 TTA 82.16%（25,285 张，含并入的 35 张真实烤鸭图），新增的 15 道中餐 91.19%（545 张）。热启动让旧类几乎不掉点（原 101 类模型为 82.57%），新类则直接可用。
- **校准**：温度 T = 0.82（验证集拟合），官方 test 的 ECE 从 0.101 降到 0.018。
- **开集门控**：116 类仍覆盖不了火锅、麻辣烫等吃法，按与 20 类门控相同的策略，把阈值放在官方 test 三视角最差分数的 5% 分位（n = 1,740，每类 15 张）：p5 = 0.185，取 **0.18**；0.20 会标记 6.1% 的图，0.30 标记 15.8%。页面的第三模式用 0.18。

把 116 类模型与 20 类模型放在**同一套 20 类评分**下比较（116 类模型在 20 个子类上重归一化，才和 20 类模型可比）：

| 模型 | 训练图 | 官方 test（5,000） | 食堂退化照片（1,000） |
| --- | --- | --- | --- |
| 20 类（E4，课程模型） | 12,000 | 90.54% | 66.00% |
| 116 类（本扩展） | 70,283 | 93.82% | 75.50% |

![116 类扩展对比](results/figures/wide_scale.png)

- 两个域上都有提升，退化域提升更大（+9.5 pt 对 +3.3 pt）：更多类别逼着主干学到更通用的纹理/形状特征，对模糊、旋转、JPEG 失真的照片更稳。
- 需要说明的是，两组训练配置并不完全一致（116 类是**全量**微调，20 类只解冻 last_stage），所以这里是“更大数据 + 全量微调”的组合效果，不能把提升全部归因于数据量。
- 中餐这 15 类只有 545 张 test、每类 22–52 张，91.19% 的置信区间较宽（约 ±2.5 pt），且这批图多为摆盘特写，比食堂随手拍更“干净”；它说明的是新增类能被学会，不等于真实食堂域同样高。
- **交互**：20 类模式下如果被门控标成“大概不是这 20 道菜”，结果下方会出现“用 116 类模型再试一次”按钮，点一下就把同一张图交给宽模型，不用为了看一个菜名重新选模式。
- **真实照片上的表现**：100 张自摄食堂照片上，116 类模型全开 Top-1 **71.0%**、限定到 20 类 **89.0%**（Top-5 99.0%）；20 类模型在同一批图上是 94.0%（六视图）。错法很单向——它只要报出 Food-101 的菜基本就对（官方 test 83.3%），错的情况都是中餐类抢走第一名（炸薯条→佛跳墙、披萨→baklava）。所以“发现两个模型分歧就改判”没用：4×5 组 (top-k, 置信阈值) 的重排网格让真实照片只在 −1.0…0.0 pt 之间晃动，因为失败的不是“Food-101 菜被推翻”而是“中餐类赢错了”。
- **后验融合（已上线）**：把 20 类后验映射进 116 维（其余补零）再混合 p = (1−α)·p116 + α·p20。α = 0.2 时真实照片 71.0% → **79.0%**（修对 8 张、改错 0 张，McNemar p = 0.008），Food-101 官方 test 83.3% → 87.4%（修对 205 / 改错 3），代价是中餐 test（20,830 张）82.9% → 81.9%（改错 212 / 修对 0）。页面第三模式默认 α = 0.2，改变量 `FOODCLS_WIDE_FUSE`（设 0 关掉融合），`/api/health` 里的 `wide_fuse` 会显示当前值。
- 复现：

```powershell
python scripts/fetch_kaggle.py --dataset patzer0/16-famous-chinese-dishes --out data/external/16chinese.zip
python scripts/add_chinese_dishes.py                    # 解压 16 道中餐，按自带划分并入 data/wide/splits
python scripts/prepare_wide_data.py                    # Food-101 101 类 train/val/test 划分
python scripts/train_wide.py --epochs 10 --batch-size 96 --workers 6 --out results/runs/resnet18_food101
python scripts/train_wide.py --epochs 4 --init results/runs/resnet18_food101/best.pt --lr-head 5e-4 --lr-backbone 5e-5 --out results/runs/resnet18_food101_cn116
python scripts/eval_wide_groups.py --run results/runs/resnet18_food101_cn116   # 单视角/TTA/ECE/分组
python scripts/eval_wide_domain.py                     # 与 20 类模型在同一套 20 类评分下对比
python scripts/eval_wide_gate.py                       # 116 类门控阈值（5% 分位）
python scripts/eval_wide_real.py                       # 100 张真实照片上的 116 类 Top-1/5
python results/_rerank_grid.py && python results/_rerank_fuse.py   # 重排网格与后验融合的 α 扫描
python scripts/make_wide_figure.py                     # 画 wide_scale.png
```

> 注：`train_wide.py` 结尾会用 DataLoader worker 再跑一遍 test，在 Windows 上曾因共享内存段耗尽报 `RuntimeError: Couldn't open shared file mapping ... error code: 1455`（该处已降到 2 个 worker）。对 116 类划分，这一步要处理 25,830×3 个视图、约 40 分钟，所以实际指标由 `eval_wide_groups.py` 单独重算（8 worker 一次解码三视图），用的仍是同一套温度缩放与 TTA。

### 5.6 连锁品牌热量参考（麦当劳中国官网）

课程先验描述的是“家常做法”的平均值（炒饭 175 kcal/100 g 这类），对快餐类菜品误差偏大：一个汉堡可能是 294 kcal 的吉士汉堡，也可能是 707 kcal 的培根安格斯厚牛堡，两者差 2.4 倍。官网公布的每份实测值正好补上这一块。

- **数据**：`scripts/fetch_brand_nutrition.py` 抓麦当劳（中国）官网营养计算器（https://www.mcdonalds.com.cn/nutrition_calculator）页面里的内嵌 JSON，得到 89 项、8 个类别（汉堡和卷 13、小食 8、甜品 9、饮品 12、早餐 16、McCafé 23、500 大卡套餐 2、开心乐园餐 6）。字段为每份克重、千焦、蛋白、脂肪、碳水、钠；kcal 按千焦 / 4.184 换算，并同时给出每 100 g 热量。落盘 `data/external/brand_menu.json`（schema 1，含 generated_at 与来源 URL）。`--check` 只校验已有文件（行数下限、50–3000 kcal、10–2000 g），不联网。
- **只有麦当劳**：肯德基中国官网没有可机读的营养表（交互查询、无结构化数据），所以不编造数字。`src/foodcls/brands.py` 的类映射也只引用真实抓到的条目，缺的会跳过而不是补一个相近的。
- **映射**：汉堡 → 巨无霸 / 板烧鸡腿堡 / 麦辣鸡腿汉堡 / 双层吉士汉堡 / 吉士汉堡包；薯条 → 脆脆薯条 / 薯条；鸡翅 → 麦辣鸡翅 / 薄皮焦香 V 翅；冰淇淋 → 圆筒冰淇淋 / 奥利奥麦旋风 / 朱古力新地。每类最多展示 3 条。
- **界面**：每个候选菜品下方多一行蓝色小字的“连锁参考”，例如薯条 → 脆脆薯条 279 kcal（103 g · 271 kcal/100g）；结果区下方另有可展开的“连锁品牌热量参考”面板，89 项按类别分组，支持菜名搜索与分类筛选，每行显示份量、三大营养素、热量与每 100 g 热量。混合餐盘的每一部分也有对应的一行。
- **口径**：品牌数据只作“同样一份在连锁店大概多少”的对照，不参与本项目的区间计算——区间仍由 `k_c × m_c` 与 ±15%（Top-1 低于 0.55 时再 +10%）给出，填了克数则只用 `k_c × 克数 / 100`。
- **局限**：配方与份量会随门店/地区调整；菜单项与食堂菜品也不是同一道菜（麦当劳薯条是细薯，食堂可能是薯角），所以只用于量级对照。抓取日期见 `brand_menu.json` 的 `generated_at`。
- 复现：

```powershell
python scripts/fetch_brand_nutrition.py           # 抓取并写 data/external/brand_menu.json
python scripts/fetch_brand_nutrition.py --check   # 只校验行数与合理性
```
### 5.7 把清单补成食堂真正会出现的菜：CNFOOD-241 的 20 道家常菜

前面的 116 类看着热闹，落到食堂窗口还是不对味。固定的 20 类全是 Food-101 的西餐，"中餐"只有 `fried_rice`、`dumplings`、`ramen` 三个名字，图片还多来自欧美外卖店；5.5 节补的 15 道则是佛跳墙、东坡肉、西湖醋鱼这类宴席名菜。结果是端一盘西红柿炒鸡蛋、酸辣土豆丝或者红烧肉去拍，模型只能硬塞成麻婆豆腐、佛跳墙——而且置信度不低，不是"拿不准"，是"很自信地答错"。这一节就是为了让清单里出现食堂真正天天有的菜。

**数据。** CNFOOD-241（241 类、191,811 张、约 10.5 GB，公开在 Mendeley Data，数据集编号 `fspyss5zbb`），覆盖的正好是食堂菜单的形态：麻婆豆腐、酸辣土豆丝、地三鲜、西红柿炒鸡蛋、糖醋排骨、可乐鸡翅、红烧肉、炒面、包子、馒头都有。从中挑 20 道食堂最常见的：

| 中文 | slug | 中文 | slug |
| --- | --- | --- | --- |
| 酸辣土豆丝 | `hot_sour_potato` | 地三鲜 | `di_san_xian` |
| 手撕包菜 | `hand_torn_cabbage` | 炒青菜 | `stir_fried_greens` |
| 干煸豆角 | `dry_fried_beans` | 西红柿炒鸡蛋 | `tomato_egg` |
| 鸡蛋羹 | `steamed_egg` | 糖醋排骨 | `sweet_sour_ribs` |
| 可乐鸡翅 | `cola_wings` | 白斩鸡 | `boiled_chicken` |
| 黄焖鸡 | `braised_chicken` | 辣子鸡 | `spicy_chicken` |
| 红烧肉 | `braised_pork` | 水煮肉片 | `boiled_pork` |
| 糖醋里脊 | `sweet_sour_pork` | 农家小炒肉 | `farmhouse_pork` |
| 青椒肉丝 | `pepper_pork` | 木须肉 | `mu_xi_pork` |
| 包子 | `baozi` | 炒面 | `fried_noodles` |

官方发布拆成两个压缩包：`train600x600.zip`（9,314,251,486 字节）与 `val600x600.zip`（1,153,239,586 字节），类别名在 `class_name.xls`。这里把 `train` 包当成全部数据：从中抽 21,043 张，按 9:1 切成训练 16,717 / 验证 1,864；`val` 包的 2,462 张原封不动当测试集，训练全程没碰过。每类训练样本 660–1,239 张。唯一的短板是白斩鸡：整个数据集只有 248 张（训练 223 / 验证 25 / 测试 34），所以它的数字只能当参考。

**训练。** 从 5.8 节 E7（Food-101N 清洁标注微调）那版权重（`results/runs/resnet18_last_stage_aug_f101n/best.pt`）热启动，解冻最后一个阶段加新的 20 路分类头，8 个 epoch，batch 64，AdamW（头 1e-3、主干 2e-4，weight decay 5e-4），label smoothing 0.1，余弦退火，AMP。8 GB 的卡上 205 s/epoch，整轮 28 分钟。按验证集准确率存最优，之后在验证集上拟合温度缩放（LBFGS，得到 `T = 0.726`），checkpoint 里连同类别表一起存下。

**结果。** 验证 88.25%，测试 Top-1 87.73%、Top-5 98.38%，校准后 ECE 2.36%（校准前验证集 7.99%）。

| 项目 | 数值 |
| --- | --- |
| test Top-1 / Top-5 | **87.73% / 98.38%** |
| 校准后 ECE | 2.36%（T = 0.726） |
| 平均 Top-1 置信度 | 0.893 |
| 每类最差 | 农家小炒肉 71.9%、糖醋排骨 72.5%、白斩鸡 76.5%（仅 34 张） |
| 每类最好 | 干煸豆角 98.2%、酸辣土豆丝 96.7%、西红柿炒鸡蛋 96.7% |

弱项有规律：农家小炒肉、青椒肉丝、木须肉都是"肉丝 + 配菜"的炒菜，同一盘换个切法就能互相串；糖醋排骨和糖醋里脊共享"糖醋 + 挂汁"的视觉特征；白斩鸡则是样本量的问题。为了确认 8 个 epoch 不是欠拟合，又用一半学习率续训了 10 个 epoch，测试只有 87.29%，没有超过，所以线上用的是 8 个 epoch 这一版（`results/runs/cn241_full/best.pt`）。

**门控。** 沿用前两个模式的做法：对每张图取三视角（原图、镜像、多尺度）最大值中的最小值当"最差视角分数"，低于阈值就回"这道菜不在 20 道清单里"。阈值在两组图上定：清单内是 20 类测试图各取 40 张（白斩鸡只有 34 张，共 794 张），清单外是 116 类测试集里不属于这 20 道的菜各取 8 张（共 320 张），两边都先过温度校准。

| 阈值 | 清单内保留 | 清单外挡掉 |
| --- | --- | --- |
| 0.30 | 97.9% | 22.2% |
| 0.35 | 96.6% | 33.4% |
| **0.40** | **94.8%** | **43.4%** |
| 0.45 | 92.4% | 53.4% |
| 0.55 | 85.0% | 66.9% |

取 0.40，即清单内分数的 5% 分位（0.396），和单菜模式、116 类模式用的是同一个误报预算，三个模式的门控口径因此一致。清单内均值 0.832，清单外均值 0.495、中位数 0.425——两个分布分得开，但重叠区不小，这从"只挡掉 43%"也能看出来。（换成 21 类后这条规则和阈值都改了，见 5.7.1；这张表留着说明 20 类版本的来历。）

**双头交叉验证（veto）。** 20 类分类头只会在这 20 个名字里挑，所以清单外的中餐（麻婆豆腐、北京烤鸭、酸菜鱼）也会被自信地塞进某道菜。第三套权重（116 类宽模型，见 5.5）恰好认识这 15 道中餐，于是拿它当「第二意见」：宽头的**原始**后验（不做与 20 类的融合，免得两头互相认同掩盖分歧）里，若某道不在 20 道清单里的中餐 ≥ 0.50、且不低于食堂头的分数 − 0.02，就推翻食堂头的结论，页面改说「这张大概不在 20 道清单里」。一共 13 类可 veto（麻婆豆腐、宫保鸡丁、水煮鱼、酸菜鱼、西湖醋鱼、清蒸鲈鱼、北京烤鸭、汤包、饺子、春卷、炒饭、拉面、狮子头），故意排除糖醋里脊（就是 20 类里那道糖醋里脊）和回锅肉（和农家小炒肉常常是同一盘）。代价很小：清单内接受 94.8% → 94.4%（2,462 张里只误伤 0.5%），清单外中餐的误接受从 71.1% 降到 40.0%。它挡得住北京烤鸭、麻婆豆腐、酸菜鱼，挡不住宫保鸡丁（宽头只领先 0.005，差一点才触发）和「饺子/汤包 → 包子」——这正是 5.7.1 要补的洞。

**已知局限（写在前面）。** 拿 100 张真实食堂照（`data/splits/canteen_real`，都是本校窗口的西餐：披萨、薯条、寿司等）喂给这个模型，只有 **50%** 被判为"不在清单里"，剩下的会被硬套成炒面、包子之类。原因很直接：这 20 类全是中餐，西餐在它的特征空间里没有"正确的出口"，最近的邻居就是某道炒菜。所以第 4 个模式在网页上是**手动切换**的，不做自动路由；要自动判断"这是一盘西餐还是中餐"，得再训一个域分类头，那是另一件事（第 6 节有说明）。（5.7.1 的 21 类版本把这个比例提到 79%，但同一批照片的类别判断依然不可靠——域的问题还是得靠域分类头。）

**复现。**

```powershell
# 1) 下载两个压缩包（约 10.5 GB，官方 Mendeley 直链，需要 User-Agent）
.\.venv\Scripts\python.exe scripts/pdl.py --url <train600x600.zip 的 file_downloaded 链接> --out data/external/cnfood241_train600x600.zip --threads 12
.\.venv\Scripts\python.exe scripts/pdl.py --url <val600x600.zip 的 file_downloaded 链接> --out data/external/cnfood241_val600x600.zip --threads 12
# 2) 抽 20 类：train 包 9:1 切分，val 包全部做 test
.\.venv\Scripts\python.exe scripts/prepare_cn241.py --class-xls data/external/cnfood241_class_name.xls --train-zip data/external/cnfood241_train600x600.zip --val-zip data/external/cnfood241_val600x600.zip --out data/splits/cn241_full
# 3) 训练（8 GB 显存，约 28 分钟）
.\.venv\Scripts\python.exe scripts/train_cn241.py --mode finetune --data-root data/splits/cn241_full --classes results/tables/cn241_classes_full.json --init results/runs/resnet18_last_stage_aug_f101n/best.pt --epochs 8 --batch-size 64 --workers 0 --out results/tables/cn241_full.json --save-dir results/runs/cn241_full
# 4) 门控阈值与真实照行为
.\.venv\Scripts\python.exe scripts/eval_cn_scope.py
.\.venv\Scripts\python.exe scripts/eval_cn_domain.py
```

许可提醒：CNFOOD-241 以 CC BY-NC 3.0 发布，且其部分图片源自 ChineseFoodNet，**仅限非商业用途**；数据集本身只在本机做实验，不随代码分发，详见 `NOTICE`。

#### 5.7.1 让模型学会说「这是别的中餐」：第 21 类

20 类模型有个结构性问题：清单外的中餐没有「出口」。端一盘宫保鸡丁进去，它只能在 20 个名字里挑最像的那个，答成辣子鸡、置信度 98%——上面那套双头交叉验证能挡掉一部分，但宫保鸡丁差 0.005 就漏了。根子不在阈值，在于模型从来没有被允许回答「这不是清单里的菜」。这一节把它训成第 21 类。

**数据。** 从 CNFOOD-241 剩下的 175 个类里抽「别的中餐」拼成 `other_cn` 桶，每类最多 12 张（防止几个大类主导桶的后验）：训练 2,100 / 验证 350 / 测试 1,396，与 20 类硬链接拼成 `data/splits/cn241_21`（训练 18,817 / 验证 2,214 / 测试 3,858）。`scripts/prepare_cn241_other.py` 排除了三块：20 道本菜、17 个西式/甜点类（汉堡、披萨、蛋糕——那属于「别的菜系」），以及 29 个留出类（宫保鸡丁、麻婆豆腐、北京烤鸭、饺子……），后者单独放进 `data/splits/cn241_21_ood`，专门当「训练时从没见过的清单外中餐」。

**训练。** 流程与 5.7 完全一致（热启动、8 epoch、batch 64、AdamW、label smoothing 0.1），只是分类头 21 路；8 GB 卡上 340–380 s/epoch，整轮约 45 分钟。最优在 epoch 7，验证 81.89%，`T = 0.827`。

**代价。** 多一类就多一份混淆：测试 Top-1 87.73% → 75.40%（Top-5 97.49%），清单内 20 类的单视图平均 87.52% → 83.66%。掉得最多的是白斩鸡 76.5% → 58.8%（测试集只有 34 张）、木须肉 95.9% → 83.5%、糖醋排骨 72.5% → 64.1%、青椒肉丝 78.6% → 70.2%。桶自己的召回 59.2%——它学会了「这不在 20 道里」，但没学会「那到底是哪道」。

**门控要重写。** 桶会分走一部分概率，原来那套「最差视角分数」就会把清单内的菜一起压到线下：同一个 0.34 的阈值，清单内接受率从 94.4% 掉到 87.6%。所以先按视角把桶的质量扣掉、再取最差：

    robust = min_v max_c p_v(c) / (1 - p_v(other))

阈值取验证集清单内照片这个分数的 5% 分位（0.3445），和前面几个模式同一个误报预算。桶另有独立的一道闸：`p(other) >= 0.40`，或者桶直接夺冠，就判「不在清单里」。13 类宽头否决（5.5、5.7）保持不变。

| 规则 | 清单内保留 | 清单外中餐（360 张） | 其它中餐（808 张） | 没见过的中餐（232 张） | 真实西餐照（100 张） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 类 + 13 类否决 | 94.4% | 40.0% | 52.2% | — | 44.0% |
| 21 类，门控只看扣桶分数 | 94.4% | 35.3% | 46.7% | 55.2% | 42.0% |
| **21 类，扣桶分数 + 桶**（上线） | **91.3%** | **24.2%** | **20.4%** | **32.3%** | **21.0%** |

除第一列，表里都是「仍被当成本清单的菜接受」的比例，越低越好。清单内让掉 3 个百分点，换清单外中餐从四成降到两成多、真实西餐照（披萨、薯条这类，本来就不该由这个模式回答）从 44% 降到 21%。分菜看：麻婆豆腐 8.3%、宫保鸡丁 8.3%、北京烤鸭/西湖醋鱼/清蒸鲈鱼/春卷/炒饭 0%、酸菜鱼 4.2%、拉面 12.5%、狮子头 12.5%、水煮鱼 25%；这些以前全靠「长得很像某道菜」混过去。

**还是漏的。** 饺子 66.7%、汤包 54.2% 仍被判成包子，回锅肉 79.2%、糖醋里脊 91.7% 被判成农家小炒肉/糖醋排骨。这不是「不会拒绝」，是「长得真的太像」：包子类的训练图里混着大量饺子、汤包形态，回锅肉和农家小炒肉都是「肉片 + 青椒/蒜苗 + 红油」。要彻底分开只能补拍这两组、或者把桶拆成更细的语义类（面点/川菜/…），那是后续工作。另外 91.3% 也意味着每 11 张支持的菜就有 1 张被劝去 116 类模式，这是这次改动的实际代价。

**复现。**

```powershell
# 1) 生成 21 类：20 类硬链接 + 其它中餐桶；另留 29 类做「没见过的清单外」测试
.\.venv\Scripts\python.exe scripts/prepare_cn241_other.py
# 2) 训练（约 45 分钟）
.\.venv\Scripts\python.exe scripts/train_cn241.py --mode finetune --data-root data/splits/cn241_21 --classes results/tables/cn241_21_classes.json --init results/runs/resnet18_last_stage_aug_f101n/best.pt --epochs 8 --batch-size 64 --workers 0 --out results/tables/cn241_21.json --save-dir results/runs/cn241_21
# 3) 五集评估与规则搜索
.\.venv\Scripts\python.exe scripts/eval_cn_other.py
.\.venv\Scripts\python.exe scripts/search_cn_other_rule.py
```

#### 5.7.2 先问「这是不是食堂菜」：域判别头

上一版把真实西餐照的误接受从 44% 压到 21%，剩下的全是「看起来像某道菜」的：一盘鸡翅被塞进可乐鸡翅，一块蛋糕被当成糖醋里脊。桶救不了它们——桶里装的是**别的中餐**，而披萨根本不是中餐。21 个类全是中餐，「别的菜系」没有出口。

**不训第二个网络。** 主干在倒数第二层已经输出 512 维池化特征，同一个前向就能拿到；在它上面拟合一个逻辑回归（标准化 + `class_weight="balanced"`，C 从 0.003 扫到 3 按验证 AUC 选，选中 0.1），回答「这张像不像食堂中餐」。产物是一个 10 KB 的 JSON（512 个权重 + 均值/方差 + 一个阈值），推理只多一次点积——没有第二个模型、没有第二次前向，8 GB 卡上零成本。门控用的三个视角特征取平均后过分类器。

**训练数据。** 正例 `data/splits/cn241_21/train`（12,000 张 21 类中餐），负例 `data/splits/train`（12,000 张 Food-101 西餐）；验证 2,214 / 2,000，测试 3,858 / 5,000。`scripts/extract_domain_features.py --views 1` 先把 37,550 张图的特征抽出来存 `results/tables/domain_feats.npz`（单视图 84 张/秒，约 8 分钟；平均池化 hook，不额外写任何图片），`scripts/fit_domain_head.py` 再拟合。

**阈值。** 取中文验证集分数的 5% 分位（0.2554）——和 0.32 / 0.18 / 0.3445 用的是同一个误报预算，不是拍脑袋。验证集 AUC 0.974；这个阈值上验证集保留 95.0% 的中餐、拒掉 85.1% 的西餐。单独看这个头（不含门控 / 桶 / 否决）：21 类测试集 91.3% 被判「是食堂菜」，Food-101 测试集只有 14.4%，100 张真实食堂照 34.0%。

**接进决策链之后。** 它是纯否决：`p_domain < 0.2554` 直接判「不是食堂菜」，与其它几道闸并列：

| 规则 | 清单内（2,462） | 清单外中餐（360） | 没见过的中餐（232） | 非中餐（808） | 真实西餐照（100） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 21 类（上一版） | 91.3% | 24.2% | 32.3% | 20.4% | 21.0% |
| **21 类 + 域判别**（上线） | **89.6%** | **20.8%** | **31.5%** | **9.7%** | **13.0%** |

代价是清单内又让掉 1.7 个百分点（94.4 → 91.3 → 89.6，两版合计 4.8 pt），换来非中餐的误接受腰斩、真实西餐照 21% → 13%。第四、五列几乎不动不是巧合：域头判的是**菜系**，不是「认不认识」，清单外中餐仍要靠桶（5.7.1），两者互补。

**分菜看真实照。** 披萨 11 张、汉堡 12 张、巧克力蛋糕 8 张、薯条 7 张、热狗、冰淇淋、千层面、松饼、华夫饼全部 0 接受；剩下的主要是鸡翅 75%（8 张里 6 张仍进）和饺子 30%——鸡翅和可乐鸡翅 / 糖醋里脊在 512 维上确实像，饺子则落在桶和包子的边界上（见 5.7.1）。

**阈值敏感度**（`scripts/pick_domain_threshold.py` 用落盘的每张分数重扫，不用再跑前向）：

| 域头阈值 | 清单内 | 非中餐 808 | 真实西餐 100 |
| ---: | ---: | ---: | ---: |
| 0.05 | 91.0% | 15.0% | 18.0% |
| 0.10 | 90.7% | 12.8% | 15.0% |
| **0.2554**（上线） | **89.6%** | **9.7%** | **13.0%** |
| 0.40 | 87.7% | 7.9% | 11.0% |
| 0.60 | 85.0% | 5.8% | 8.0% |

往右调很诱人，但清单内每让出 1 个百分点，非中餐只再少 0.6 pt 左右——5% 分位恰好在这条曲线开始变缓的地方，所以没有为几个点去牺牲清单内。

**已知局限。** ① 训练时没见过的中餐（`cn241_21_ood` 的 29 类）单独过域头只有 79.7% 被判成中餐——它们是中餐但不在训练分布里，域头会误伤约两成；完整决策链里这组只从 32.3% 掉到 31.5%，因为桶和门控已经先拦住了大部分。② 特征来自当前 21 路主干，**换主干重训要重抽特征、重拟合**。③ 鸡翅这类在 512 维上和中式鸡翅菜重合，域头无能为力。

**复现。**

```powershell
# 1) 抽特征（单视图，约 8 分钟）→ results/tables/domain_feats.npz
.\.venv\Scripts\python.exe scripts/extract_domain_features.py --views 1
# 2) 拟合 + 选 C + 5% 分位阈值 → results/tables/domain_head.json（10 KB）与指标
.\.venv\Scripts\python.exe scripts/fit_domain_head.py --mode mean
# 3) 完整决策链评估（5 集，约 9 分钟）与阈值敏感度
.\.venv\Scripts\python.exe scripts/eval_domain_gate.py
.\.venv\Scripts\python.exe scripts/pick_domain_threshold.py
```

### 5.8 测试时增强与代理域抽样波动

代理域是 1,000 张 clean 图按固定随机种子退化出来的**单次抽样**，因此有两个问题值得单独量化：换一种推理方式（TTA）还能补多少？换一个随机种子重抽代理域，绝对数会漂多少？`scripts/eval_tta.py` 把这两件事一起测了，结果写 `results/tables/tta.json`（每个模型 × 每个 split × 6 种视图组合，含 ECE）。

| 模型 | 推理 | test | canteen（种子 42） | canteen（种子 7） |
| --- | --- | ---: | ---: | ---: |
| E4（无外部数据） | 单视图 | 90.54 | 66.00 | 63.60 |
| E4 | 水平翻转 | 91.20 | 67.90 | 65.80 |
| E7（+Food-101N） | 单视图 | 91.66 | 71.60 | 67.60 |
| E7 | 水平翻转 | 92.46 | 72.60 | 69.20 |
| E7 | 五裁剪 | 92.90 | 71.40 | 69.80 |
| E7 | 多尺度（256/288/320） | 92.10 | 69.40 | 66.00 |

- **水平翻转最稳**：E4 在三列上 +0.66 / +1.90 / +2.20，E7 +0.80 / +1.00 / +1.60。只多一次前向，是性价比最高的一档。
- **五裁剪**对 test 和种子 7 最好（E7 92.90 / 69.80），但对种子 42 的代理域没有收益（71.40）；**多尺度**在代理域上反而掉 2.2 pt（69.40），因为把短边重采样到 320 会抹掉“模糊 + 降采样”这个域特征，等于把两种退化混在一起平均。
- **重抽种子**：E4 66.00 → 63.60，E7 71.60 → 67.60，绝对水平有 2.4–4.0 pt 的抽样波动；但 E7 − E4 的差只从 5.6 变到 4.0，方向不变。所以 71.6% 要按“单次抽样 ±2–4 pt”读，跨域差距本身是稳的。
- **部署取舍**：网页仍用三视角平均（256 / 320 / 方形拉伸），因为用户上传图的尺度不可控；离线评测里翻转最划算，所以命令行推理加了开关：`python -m foodcls.predict 图片 --ckpt ... --tta`（原图与水平翻转的概率取平均，多一次前向，见 `src/foodcls/predict.py`）。

复现：

```powershell
# 先造第二份代理域（种子 7），只写 data/splits/canteen_s7，不动原来的 canteen
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from foodcls.data import make_canteen_proxy; make_canteen_proxy('data/splits/canteen_clean','data/splits/canteen_s7', seed=7)"
# 8 种视图 × 3 个 split × 每个 checkpoint，约 11 分钟/模型
.\.venv\Scripts\python.exe scripts/eval_tta.py --ckpt results/runs/resnet18_last_stage_aug/best.pt --ckpt results/runs/resnet18_last_stage_aug_f101n/best.pt --splits test canteen canteen_s7
```

### 5.9 包装照不是菜：一个否决头

网页上线后被问到的第一类失败不是"认错菜"，而是**商品包装正面照**：一盒冷冻披萨的盒盖上印着披萨，20 类头答"pizza 98%"，食物门控也答"是食物"——两个模型都没错，只是问题问错了。原来的 `src/foodcls/packaging.py` 是一个 numpy 启发式（长直强边缘比例 + 细纹理密度），在 62 张超市包装照上命中 **0** 张。

换成和食物门控同一套做法的线性头：512 维 avgpool 特征（取**整帧**那一路视图，因为包装照是"整件东西"），一层逻辑回归，不需要额外前向——这 512 维本来就在六视图那一次推理里算出来了。

- **数据**：正样本 79 张 = 负样本库里的 17 张包装 + 归档的 62 张商品照；负样本 3,700 张食物照（train 2,000 / clean 1,000 / 真实 100 / test 600）。权重在 `results/tables/packaging_linear.json`，部署副本在 run 目录的 `packaging.json`。
- **留出评估**（对 62 张做 5 折，17 张旧包装留作另一域）：AUC **0.994**，同一操作点（p ≥ 0.80）召回 **79%**；食物被误标包装的比例 0.5–0.7%（train/clean），**真实食堂照 0/100**。
- **端到端**（`results/_pkg_sweep.py`，走 HTTP）：62 张包装照 **0 张再被判进范围**（改之前是 36/62，最离谱的一张报 pizza 98%），其中 50 张给出"这是商品包装"的专门文案，另外 12 张食物门控本来就判非食物；17 张旧包装同样 0 张进范围；真实食堂照仍是 95/100，**没有一张**被误标成包装；300 张代理域 285/300，包装标记 0。
- 否决只在**食物门控已经说"是食物"**时生效：门控说没有食物时，它自己的文案更准确（46 个被标包装的负样本里有 43 个属于这种情况，都是餐巾、菜单牌、空盘）。
- 复现：`results/_pkg_views.py`（抽特征）→ `results/_pkg_final.py`（拟合、写 json）→ `results/_pkg_publish.py`（拷进 run 目录）→ `results/_pkg_sweep.py`（端到端）。

局限：62 张正样本是同一种拍法（商品正面、白底、印刷体），所以"认出这一类包装照"是可靠的，换成实拍货架上的包装则只有 1/17 会被专门标为包装（其余仍被门控以"没找到菜"拒掉，只是文案不专门）。要提升，得补实拍包装负样本，而不是重训这个头。

### 5.10 这些数字有多稳：置信区间与配对检验

前面每一格都是有限样本上的点估计：test 5,000 张、代理域 1,000 张、真实照片 100 张。`results/_stats.py` 直接读缓存的每图六视图 logits（`results/_ens_<run>__<split>.npz`，不用 GPU、不跑数据），给出 4,000 次 bootstrap 的 95% 区间和同一批图片上的精确 McNemar 检验，结果写 `results/tables/stats.json`、说明写 `results/REPORT_stats.md`。

| 划分 | E4 | E7 | 五模型集成 |
| --- | ---: | ---: | ---: |
| test（5,000） | 90.5 [89.7, 91.4] | 91.7 [90.9, 92.4] | 93.4 [92.7, 94.1] |
| canteen 种子 42（1,000） | 66.0 [63.1, 69.0] | 71.6 [68.8, 74.3] | 74.2 [71.5, 76.9] |
| canteen 种子 7（1,000） | 63.6 [60.6, 66.7] | 67.6 [64.6, 70.6] | 71.4 [68.5, 74.3] |
| 真实照片（100） | 85.0 [78.0, 92.0] | 91.0 [85.0, 96.0] | — |

（表中都是**单视图**口径，与 §5.8 一致；网页实际用的是六视图平均，E7 在 test 上 93.4、代理域 73.0、真实照片 95.0。）

- **外部数据的收益是稳的**：E7 比 E4 在 test 上 +1.1 pt（McNemar：223 张修对 / 167 张改错，p = 0.005）、代理域 +5.6（p < 0.001）、重抽种子 +4.0（p = 0.005）。三处都显著，是整个课题里唯一在两种代理抽样上都站得住的改动。
- **去重不显著**：E7 与 E8 在 test / canteen / canteen_s7 上分别翻掉 119/125、65/58、63/72 张，p = 0.49–0.75。也就是说 460 张近重复对结果没有可测影响——和之前"泄漏只值 0.1 pt"一致，可以放心这样写。
- **五模型集成是唯一稳定压过单模型的部署可选改进**：test +1.7、代理域 +2.6 / +3.8，对 E7 的 p 分别是 < 0.0001 / 0.003 / 0.0002。代价是五倍推理，所以网页仍只服务 E7，这一条留在部署 backlog 第一条。
- **真实照片 100 张，区间有 ±5 pt**：E7 91.0% 与 test 91.7% 在统计上不可区分，这正好说明这批图更像"干净网图"而不是食堂托盘照；用它只能看方向。
- **门控**：131 张真负样本拒掉 96.9% [93.9, 99.2]，100 张真实菜保留 95.0% [90.0, 99.0]，两个区间不重叠。

复现：

```powershell
.\.venv\Scripts\python.exe results/_stats.py
```

## 6. 网页演示

> 本节所有命令都在 **food_project 根目录** 执行（先 `cd "E:\项目作业\food_project"`）。
> 或者直接双击项目根目录的 `启动网页演示.cmd`，它会自己切到正确目录并打开浏览器。

```powershell
cd src
.\.venv\Scripts\python.exe -m foodcls.app      # 默认 http://127.0.0.1:8000
```

- 支持点击/拖拽上传照片，也可从示例缩略图中选择（示例图来自未参与训练的代理域）。
- 右侧输出 Top-3：中文菜名、英文类名、类别概率、估算热量区间；置信度低于 0.55 时区间自动放宽。
- **份量切换**：可选"小份 / 常规 / 大份"（0.7× / 1.0× / 1.35×），切换时只在前端重算区间、不重新推理，演示时很快。
- **自定义重量**：在"重量"框里填这一份的实际克数（10–2000 g），每个候选菜品的区间立刻按 `k_c × 克数 / 100` 重算，同样不重新推理；此时份量档自动置灰，点份量档或"清除"就回到按份量估算。知道打了多少克时，这一项比份量档准得多。
- **混合餐盘模式**：点"混合餐盘 · 多标签"切换。模型同时认识 20 道西餐和食堂的 20 道菜（共 40 类，见 5.3），中西混搭也能拆；识别后有哪几样菜就显示几张卡片，每张卡片可单独选"小份 / 常规 / 大份"或直接填克数（填克数时份量档置灰），下方"整盘合计"实时把各部分区间相加；示例缩略图是合成拼盘、标注真值，17 张里 10 张是西餐盘、7 张食堂盘 / 中西混搭，title 上标「食堂」「中西」；纯食堂托盘的结果会说明克数是按食堂常见份量估的。"随机示例"按钮在两种模式下都保留，混合模式下从拼盘示例里抽。
- **没菜的照片**：空托盘、饮料杯、菜单牌、人手这类图，只靠“换视角还认不认同一道菜”拦不住（约一半会被放行）。推理时会先过一个 512 维特征上的食物/非食物线性门控，两个信号都不过才判“这张照片里没找到菜”，并说明不在识别范围里；131 张真实负样本拒掉 127 张，100 张真实菜保留 95 张。
- **商品包装**：盒盖、桶身印着披萨的包装照会让两个模型都说“是食物”。另有一个整帧特征上的包装否决头（p ≥ 0.80），命中时改说“这是商品包装或食品标签”，不给品类和热量。
- **范围提示**：若这张照片三个取景下的判断互相矛盾（见 5.4），标题会变成"大概不是这 20 道菜"，候选标"仅供参考"，并说明哪些菜不在支持范围内；页面上也有"看看这 20 类都是哪些菜"可展开对照。
- 手机照片按 EXIF 自动转正；上传上限 15 MB、分辨率上限 4,000 万像素（超过直接 413，防解压炸弹），非图片或损坏文件会返回中文错误提示；窄屏自动单栏。
- **116 类模式**：第三个标签"全部菜系 · 116 类"切到 Food-101 + 15 道中餐的宽模型（见 5.5），适合拍食堂以外、甚至国外的菜（北京烤鸭、西班牙海鲜饭、石锅拌饭、越南河粉），中餐名菜（麻婆豆腐、宫保鸡丁、回锅肉、水煮鱼等）也能认；示例缩略图来自官方 test 集与中餐实拍。这一模式只用于扩大覆盖面，课程协议与论文里的数字仍以 20 类模型为准。
- **食堂家常菜模式**：第四个标签「食堂家常菜 · 20 道」切到 CNFOOD-241 上训的家常菜模型（见 5.7、5.7.1），拍西红柿炒鸡蛋、酸辣土豆丝、红烧肉、地三鲜这类窗口菜时用它，比前两个模式准得多（20 类版本 test Top-1 87.73%）。分类头是 21 路：除 20 道菜之外还有一类「其它中餐」，专门接住清单外的中餐——`p(其它中餐) >= 0.40` 或它直接夺冠时，页面会说「这张看起来不在食堂模式的 20 道清单里」，并建议切到 116 类标签；门控分数先扣掉桶的质量再取最差视角，阈值 0.3445，清单内接受 89.6%（只算门控 + 桶是 91.3%），清单外中餐只有 20.8% 仍被硬套成某道菜。它**只认识这 20 道**，饺子/汤包仍会被认成包子、回锅肉会被认成农家小炒肉（分菜数字在 5.7.1 末尾）。示例缩略图来自 CNFOOD-241 test 集（未参与训练），深链 `?csample=<类名>__0000.jpg` 可直接打开某道菜的示例。它还多了一道**域判别**（5.7.2）：用同一份主干特征判断「这像不像食堂中餐」，p < 0.2554 就直说「这张看起来不是食堂窗口的家常菜」并建议换标签——披萨、汉堡、薯条、蛋糕基本全挡，真实西餐照被硬套成炒菜的比例从 21% 降到 13%，清单内代价约 1.7 pt（见 5.7.2 的表）。域判别只看菜系、不看「是不是清单里的菜」，宫保鸡丁这类仍是桶的活；**标签仍然要手动切**，自动切换没做。
- **连锁参考**：每个候选下方给出最接近的麦当劳（中国）菜单项实测热量（见 5.6），结果区下方还可展开完整菜单（89 项，可搜索、按类别筛选）；这一栏只作对照，不改变上面的区间计算。
- **结果不对？**：四种模式的结果卡底部都有"结果不对？点这里告诉我们正确的菜"，选好真实类别（或"照片里没有菜""我也说不准"）提交，图片和这一对标签会存进 `data/splits/feedback/`（`labels.csv` 一行 + 同名 jpg）。可选类别跟着模式走：单菜 20 类、全部菜系 116 类、食堂 20 道、混合餐盘**可多选**。食堂模式的纠正以 `mode=cn` 落库，跟着 20 道清单走，可直接用于以后微调（选一样加进小标签，点标签移除）。混合餐盘还会把模型判断出的每样菜列成橙色小标签（"模型说这盘里有"），**点一下就划掉**——用于"这盘里其实没有这道菜"的负反馈，再点一次恢复；划掉的菜和加进来的菜互斥，提交时分别写进 `corrected`（分号连接，可留空）和 `removed`，两者至少填一个。文件名规则：有正确菜时 `<首个类名>+<其余数量>`，只有划掉时 `minus_<首个类名>`。接口是 `POST /api/feedback`（multipart：`file` 或 `sample`、`predicted`、`corrected`、`removed`、`note`、`mode`），`GET /api/feedback` 返回总条数、分模式条数（`by_mode`）和带划除的条数（`with_removal`）。CSV 列：`timestamp,file,predicted,corrected,removed,note,source,mode`；表头与当前列不一致时旧文件会被改名为 `labels_legacy_<时间>.csv` 保留。示例图也能提交（服务端按三个图库依次解析文件名）。这是唯一能把"真实食堂域 + 人工真值"攒起来的入口，所以只记录、不自动训练。结果不确定（门控说“大概不是”或 top-1 低于 0.55）时，top-3 列表本身就是一张表：每个候选后面变成可点的“其实盘里是”，点一下就算提交（`predicted` 是模型的第一名，`corrected` 是点的那道菜），省掉展开表单那一步。攒到一批之后跑 `python scripts/feedback_report.py`，输出 `results/REPORT_feedback.md`（最常被认错的配对、被划掉的菜、按被纠正次数排序的补拍清单）与 `results/tables/feedback_report.json`。
- 指定其他权重：设置环境变量 `FOODCLS_CKPT`（单菜）、`FOODCLS_MIXED_CKPT`（混合）、`FOODCLS_WIDE_CKPT`（116 类）与 `FOODCLS_CN_CKPT`（食堂 21 路权重）；端口：`PORT`；监听地址：`FOODCLS_HOST`（默认 0.0.0.0）；访问口令：`FOODCLS_TOKEN`（不设就是开放的，见下一条）；其它中餐阈值用 `FOODCLS_CN_OTHER`（默认 0.40），21 路门控阈值用 `FOODCLS_CN_SCOPE_21`（默认 0.3445），域判别头路径用 `FOODCLS_DOMAIN_HEAD`（默认 results/tables/domain_head.json）、阈值覆盖用 `FOODCLS_DOMAIN_THRESHOLD`（默认 0 = 用文件里的 5% 分位值）
- **默认没有登录/鉴权**：能连到这个端口的人都能上传图片、看识别结果和热量估算。绑 0.0.0.0 只适合可信的局域网（比如自己宿舍的 Wi-Fi）或自己临时开的隧道，演示结束把进程关掉即可。要在公网放几天，启动前设 `FOODCLS_TOKEN=一串随机字符`：之后第一次访问用 `https://…/?token=那串字符`，服务会种一个 7 天有效的 cookie，以后的请求（含页面里的上传）都带它；没口令的请求一律 401。
- 一键启动（绑定 0.0.0.0，并把本机全部可用 IP 打印出来）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -Open
```

### 6.1 让别人使用这个网页

三种方式，按场景选：

1. **同一 Wi-Fi（课堂演示最快）**：运行 `scripts/start_demo.ps1`，窗口里会打印 `same Wi-Fi : http://192.168.x.x:8000`，把这个地址发给同学，手机/电脑浏览器直接打开，上传照片即可。保持该窗口不要关。若对方打不开，首次需在**管理员** PowerShell 放行端口一次：
   ```powershell
   New-NetFirewallRule -DisplayName 'FoodDemo 8000' -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
   ```
   这是私有地址，仅同一局域网可达；公共 Wi-Fi 若开了"客户端隔离"则连不上，可用手机热点。

2. **对方在自己电脑上跑（交作业/离线评审）**：运行 `scripts/package_demo.ps1` 生成 `dist/FoodDemo.zip`（约 169 MB，含代码、单菜 / 多标签 40 类（20 西餐 + 20 食堂）/ 116 类 / 食堂 21 路（20 道 + 其它中餐）四份 `best.pt` 与各自的温度校准文件、食堂的域判别头 `results/tables/domain_head.json`、四套示例图、启动脚本、小程序工程和 README.txt）。没有食堂权重时打包脚本会跳过它，页面自动少一个标签。对方解压后装 Python 3.10+，执行：
   ```powershell
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   powershell -ExecutionPolicy Bypass -File start_demo.ps1 -Open
   ```
   没有 NVIDIA GPU 也能跑（CPU 单张约 16–25 ms，页面完全够用）。

3. **任何网络都能用（公网临时链接，可选）**：不想让大家连同一个 Wi-Fi 时，用 Cloudflare 快速隧道把本机端口映射成一个公网 HTTPS 地址——不需要公网 IP、不用改路由器、不用买服务器：
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/start_public.ps1
   ```
   脚本先确认服务在跑（没跑就自己拉起来），首次运行自动下载 `tools/cloudflared.exe`（约 50 MB），然后打印 `https://xxx.trycloudflare.com`：手机用移动数据、或换一个完全不同的 Wi-Fi，都能直接打开。关闭窗口即失效。
   - 链接是**临时**的：每次重启脚本都会换一个新地址（免费快速隧道不支持固定子域名）。**要一个永久不变的地址**，见 [`docs/永久公网地址.md`](docs/永久公网地址.md)：三条路线（Cloudflare Named Tunnel + 自己的域名 / Tailscale Funnel / ngrok 免费固定域名）、费用与限制对比，以及配套脚本 `setup_cloudflare_tunnel.ps1`、`start_permanent.ps1`、`start_funnel.ps1`、`start_ngrok.ps1`。
   - 链接是**公开**的：没有登录校验，拿到链接的人都能上传图片并看到识别结果；演示结束按 Ctrl+C 关掉。
   - 流量由 Cloudflare 中转并终结 TLS，本机只监听 127.0.0.1，所以**不用改防火墙**；反过来，此时局域网里的其他人也打不开 127.0.0.1 这个地址，要用隧道的公网地址。

- 一键截图（启动服务 → 截取带识别结果的页面 → 关闭服务）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo_screenshot.ps1
```

命令行单图预测：

```powershell
cd src
.\.venv\Scripts\python.exe -m foodcls.predict ..\data\splits\canteen\fried_rice\<文件名>.jpg
```

### 6.2 手机上用（二维码 / 添加到主屏幕）

桌面版页头右上角有一个二维码，扫码即可打开当前地址（二维码由 `/api/qr.png` 用 segno 现场生成，内容就是浏览器当前地址栏的 origin）：局域网地址只能同网扫，走公网隧道时用移动数据也能扫。要在投影上放大展示，启动时加 `-Qr`，会把二维码写到 `dist/phone_qr.png` 并自动打开：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_demo.ps1 -Qr
```

手机上的页面是同一份单页应用：≤560 px 时自动变成单列、缩略图 3 列，"选择图片"会直接调起相机。页面还带了 PWA 的 `manifest.webmanifest` 与 `sw.js`：在 HTTPS 或 localhost 下可以在浏览器菜单里选"添加到主屏幕"，像 App 一样从桌面图标打开，静态外壳离线也可用（`/api` 请求永不缓存，识别永远走实时推理）。**注意**：局域网 http 地址下浏览器仍能打开并"加书签式图标"，但 PWA 安装提示要求 HTTPS，这是浏览器限制，不是页面没写。

### 6.3 微信小程序版

`miniprogram/` 是一个原生小程序工程（单页，单菜 / 混合餐盘两个 tab，份量、克数、整盘合计都有），用微信开发者工具"导入项目"选这个目录即可运行。它调用的是同一套接口：`/api/predict`、`/api/mixed/predict`、`/api/mixed/total`，所以识别结果与网页完全一致。**完整的上手指南**（装工具、导入、模拟器/真机、公网地址、发布与备案、报错对照表）见 [`docs/微信小程序上手指南.md`](docs/微信小程序上手指南.md)。

使用前只改一个文件 `miniprogram/config.js`：

```js
module.exports = { BASE_URL: "http://192.168.1.23:8000" };
```

- **开发者工具里调试**：填电脑的局域网 IP，并在"详情 → 本地设置"勾选 **不校验合法域名**，直接就能跑。
- **真机 / 发布**：小程序的 `wx.request`/`wx.uploadFile` 只允许 **HTTPS + 已备案域名**，所以需要一个公网 HTTPS 地址（自己的域名 + Named Tunnel，或云服务器），再到小程序后台把该域名加进 request 与 uploadFile 合法域名。局域网 http 地址在真机开发版里可以靠"不校验合法域名"临时用，但正式发布不行（详见上手指南第 10 节）。
- 小程序不需要另建模型：后端仍是本项目的 FastAPI 服务，GPU/CPU 都在电脑上，手机只负责拍照和展示。

## 7. 论文

`paper/main.tex` 为 IEEE Conference 模板的英文论文，正文中的每个数字都由 `paper/numbers.tex` 的宏提供，该文件由 `scripts/make_paper_numbers.py` 从实验结果自动生成，保证论文与代码结果一致。

```powershell
python scripts/make_paper_numbers.py
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## 8. 热量区间怎么来的

每类菜品有两个先验：每 100 g 热量 `k_c` 与食堂常见份量 `m_c`（见 `src/foodcls/classes.py`）。点估计为 `E = k_c * m_c / 100`；区间为 `E * (1 ± s)`，其中 `s = 0.15`，当 Top-1 概率小于 0.55 时再增加 0.10。选"小份/大份"时先把 `m_c` 乘以 0.7 或 1.35 再算（等价于整段区间等比缩放，因为绝对误差主要来自份量而不是每 100 g 热量）；若在网页"重量"框直接填了实际克数 `m`，则用 `E = k_c * m / 100`，份量档不再生效，剩下的 ±15% 只反映做法差异与识别置信度。该数值只反映"常见份量 × 常见做法"的粗略范围，不能替代实际称重或实验室营养分析，页面上也做了同样说明。

## 9. 说明与致谢

- Food-101 数据集仅用于课程学习与研究，版权归原作者所有；使用时请遵守其许可条款。
- Food-101N 的清洁标注来自 CleanNet（作者本人发布在 Kaggle），本项目只用了其中 20 类子集作为下游微调数据，见 5.8 节。
- 16 道中餐来自 Kaggle 数据集 `patzer0/16-famous-chinese-dishes`（见 5.5 节），版权归原作者。
- CNFOOD-241（食堂 20 道，见 5.7 节）以 CC BY-NC 3.0 发布，部分图片源自 ChineseFoodNet，**仅限非商业用途**；本项目只在本地做实验，数据不随代码分发。各项数据的来源、许可以及"本项目未再分发任何图片"的说明集中在根目录的 `NOTICE`。
- ChineseFood172 未使用（获取与许可需自行确认），本项目类别选择偏向国内食堂常见菜品。
- 食堂代理域是对公开图像的合成退化，用于受控的跨域对比；真实食堂照片的接入方式见第 3 节。

# FoodCLS 食品识别与热量估计 —— 项目交接文档

> 用途：本文件是把整个项目对话浓缩成的自包含状态快照。开启新对话时，把本文件全文交给 AI 即可继续开发，无需旧对话上下文。最后更新：2026-11-17（混合餐盘扩到 40 类：20 西餐 + 20 食堂）。

## 0. 项目一句话
Windows 上的食品识别 + 热量估算网页应用：FastAPI 后端 + PyTorch resnet18 多套头 + 前端网页（4 个识别模式标签）。工作目录 E:\项目作业\food_project（注意：venv 在 E:\项目作业\.venv，不在项目内）。

## 1. 当前进度快照（重要）
**四个识别模式全部可用**：
- ① 20 类西餐（Food-101）+ 热量：results/runs/resnet18_last_stage_aug/best.pt，三视图门控阈值 0.32。
- ② 混合餐盘（多标签，托盘上多道菜）：results/runs/resnet18_multilabel_plate40/best.pt，**40 类（20 西餐 + 20 食堂）**，sigmoid 阈值 0.35。
- ③ 116 类宽模型（含 16 道中餐）：resnet18_food101_cn116/best.pt，T=0.819，融合 0.20 的 20 类后验。
- ④ 食堂家常菜 **21 路**（20 道 + 「其它中餐」桶）：cn241_21/best.pt，T=0.827，门控阈值 0.3445（扣桶质量后的最差视图分数），桶阈值 0.40；外加**域判别头**（512 维特征 + 逻辑回归，阈值 0.2554）先挡掉非中餐，见下。

**最新（2026-11-17，第三轮）：混合餐盘从 20 类扩到 40 类（20 西餐 + 20 食堂）**
- 代码：新增 src/foodcls/plate_classes.py（40 类固定顺序 + 中餐热量/份量/中文名）；mixed.py 的 pair_indices / build_eval_set / detect_parts / plate_totals 全部接受 class_names 与三张表（旧 20 类权重照旧可跑）；app.py 的 _mixed_tables() 按 ckpt 里的 class_names 选表、形状不匹配直接拒绝加载，mixed_payload 给每部分加 cuisine，/api/mixed/total 用同一套表，health 加 mixed_classes / mixed_canteen_classes。
- 数据/训练：scripts/train_mixed.py --classes plate 把 cn241_21/train 的 20 道 16,717 张接到西餐 20 个单元之后（「其它中餐」桶不进拼盘），训练池 28,717；从 20 类 resnet18_multilabel_aug/best.pt 热启动，fc 前 20 行搬进 40 行头；10 epoch、batch 64、--workers 4、约 160 s/epoch，27 分钟。val 1,200 盘（西 600 + 中 600，种子 1234 / 4321）。
- 训练结果（best epoch 10）：val plateF1 54.54（P 61.92 / R 48.73 / exact 10.92），逐类阈值 55.47；分菜 F1 西餐 54.80 / 食堂 56.12 / 真实照拼盘 44.00。
- 评估（scripts/eval_mixed_plate.py，600 西 + 600 中）：西餐盘 P 61.06 / R 47.09 / F1 53.17 / exact 10.50 / parts 1.85；食堂盘 62.75 / 50.38 / 55.89 / 11.33 / 1.92；detect_parts 口径 exact 8.17% / 8.50%、empty 0.7% / 0.0%；真实食堂照（16 道西餐实拍 100 张）单图 top-1 70.00%、食堂 test 单图 top-1 81.80%。结果 results/tables/mixed_plate40.json + results/figures/mixed_plate40.png。
- 对照旧 20 类（results/tables/mixed.json）：同一批 600 个西餐盘 F1 55.80→53.17，precision 60.85→61.06（不变），recall 51.53→47.09，parts 2.04→1.85；换来的是一张食堂托盘从 0 到 55.89。
- **修掉一个让训练指标失真的 bug**：mixed.py 的 build_eval_set 先把标签映射到 20..39 再拿去筛图，永远筛不到，回退成 samples[0]，于是所有食堂评估盘都是同一张图。修前同一权重中餐 f1 0.077 / 修后 0.550。复现训练前必须用修好的版本。
- 接入：DEFAULT_MIXED_CKPT 改为 results/runs/resnet18_multilabel_plate40/best.pt；17 张示例盘烘焙进 data/splits/mixed（10 西餐 + 7 食堂/中西），labels.json 带 cuisine，前端缩略图标注「食堂」「中西」。
- 已知边界：真实照拼盘 44.0% F1 / 6.3% 整盘正确，远低于合成盘 55.9%，是最大 gap；小块贴片仍漏（mix_01 薯条+鸡翅，新旧模型鸡翅都 0.02 上下）；17 张示例整盘完全正确的只有 1 张；常多报 waffles / sushi / boiled_chicken。

**再上一轮（2026-11-17，第二轮）：域判别头，先问「这是不是食堂菜」**
- 做法：不训第二个网络，用已部署 21 路主干的 512 维 avgpool 特征拟合逻辑回归（标准化 + balanced，C 扫 0.003–3，选中 0.1）；产物 results/tables/domain_head.json（约 10 KB：512 权重 + mean/scale + threshold），推理只多一次点积。
- 特征：scripts/extract_domain_features.py --views 1 → results/tables/domain_feats.npz，37,550 张（cn_tr 12,000 / cn_va 2,214 / cn_te 3,858 / cn_ood 232 / f101_tr 12,000 / f101_va 2,000 / f101_te 5,000 / real 100 / mixed_real 8 / neg 138），84 张/秒约 8 分钟。
- 阈值：中文 val 分数 5% 分位 = 0.2554（与 0.32 / 0.18 / 0.3445 同一误报预算）；val AUC 0.974，该阈值保留 95.0% 中餐、拒 85.1% 西餐。单头：cn_te 91.3% / f101_te 14.4% / real 34.0% / cn_ood 79.7%。
- 接入（src/foodcls/app.py）：DOMAIN_HEAD_PATH、DOMAIN_THRESHOLD_OVERRIDE、DOMAIN_ZH、load_domain_head()（lifespan 里在 load_cn_model() 之后）、_domain_prob(frames)、scope_of(cn_domain_prob=...)、scope["domain"]、文案分支（在桶文案之前）；payload(cn_frames=...)，两个 cn 端点传 frames；health 加 domain 字段。
- 效果（仍被接受）：in 2,462 张 91.3→89.6% / out_cn 360 张 24.2→20.8% / ood 232 张 32.3→31.5% / 非中餐 808 张 20.4→9.7% / 真实西餐 100 张 21.0→13.0%。真实照分菜：披萨/汉堡/蛋糕/薯条/热狗/冰淇淋/千层面/松饼/华夫饼全 0，鸡翅 75%、饺子 30%。
- 阈值不往右调（敏感度表见 README §5.7.2）：0.40 时 in 87.7% / real 11.0%，清单内每让 1 pt 非中餐只再少约 0.6 pt。
- 脚本：scripts/eval_domain_gate.py（完整决策链跑五集，每张分数落 results/tables/_domain_gate_<tag>.npz）、scripts/pick_domain_threshold.py（离线重扫阈值 → domain_head_choice.json）、scripts/fit_domain_head.py（拟合 + 选 C + 阈值）。
- tests/test_domain_gate.py 9 个用例：sigmoid 方向、多视图先平均、高 prob 保留、低 prob 否决且文案互斥、域文案压过桶文案、不能救回弱图、payload 用同一份 frames、head 为 None 时行为不变。

**上一轮（2026-11-17）：食堂头 20 类 → 21 类，让模型能说「这是别的中餐」**
- 数据：data/splits/cn241_21（train 18,817 / val 2,214 / test 3,858），其中 other_cn（175 个类、每类≤12 张）train 2,100 / val 350 / test 1,396；另留 29 类 232 张在 data/splits/cn241_21_ood 当「没见过的清单外中餐」。生成脚本 scripts/prepare_cn241_other.py，类别表 results/tables/cn241_21_classes.json。
- 训练：8 epoch、batch 64、--workers 0，约 45 分钟（340–380 s/epoch）；best epoch 7，val 81.89%，test top-1 75.40% / top-5 97.49%，ECE 5.36%，T=0.827；日志 results/_train21.log，结果 results/tables/cn241_21.json。
- 规则升级（src/foodcls/app.py）：门控分数改为每视图 p20/(1−p_other) 的最差视图（rob20r），阈值 CN_SCOPE_THRESHOLD_21=0.3445（验证集 5% 分位）；桶闸 p(other)≥CN_OTHER_THRESHOLD(0.40) 或桶直接夺冠即判「不在清单里」；13 类宽头 veto 不变（veto 用的是宽头原始后验）。
- 效果（仍被接受的比例，越低越好）：清单内 91.3%（20 类 + veto 时 94.4%）/ 清单外中餐 360 张 24.2%（原 40.0%）/ 其它中餐 808 张 20.4%（原 52.2%）/ 没见过的中餐 232 张 32.3% / 真实西餐照 100 张 21.0%（原 44.0%）。
- 仍漏（本质边界）：饺子 66.7%、汤包 54.2% → 包子；回锅肉 79.2% → 农家小炒肉；糖醋里脊 91.7% → 糖醋排骨（本来就是同一道菜）。
- 服务在跑：http://127.0.0.1:8000（后台 job pwsh-ede50d69a2fc4b4c，日志 results/_server.log；health 显示 mixed_classes 40 / mixed_canteen_classes 20 / cn_classes 20 / cn_threshold 0.3445 / cn_other 0.40 / domain 0.2554）。

## 2. 待办清单（按建议顺序）
1. `21 类含「其它中餐」重训` **已完成 2026-11-17**（见 §1；README §5.7.1 有完整记录）。
2. **241 类预训练 + 弱类增广**：用 CNFOOD-241 全部 241 类（约 17 万张闲置图）先学表示再微调，约 1.5 小时，预期 +1~2pt；弱类（农家小炒肉/青椒肉丝同色系肉丝）加色温/亮度/容器增广。
3. `中餐-西餐域分类头` **已完成 2026-11-17**（见 §1 与 README §5.7.2）：真实西餐照误接受 21.0%→13.0%，清单内 91.3%→89.6%。**未做**：三套门控统一 / 按域概率自动切标签——现在仍要手动点标签。
4. **托盘多菜扩展 + 反馈闭环**：反馈闭环**已修好（2026-11-17）**（食堂 slug 校验 / cn 画廊解析 / mode=cn 白名单三处 bug）；混合餐盘(②)**已扩到 40 类（20 西餐 + 20 食堂）**并在 8000 端口上线（见 §1 最新）。**未做**：真实托盘照的反馈→再训练闭环（labels.csv 至今 0 条）、UI「一步纠正」只在 top-1 < 0.55 或拒识时出现、一顿饭的份量先验（米饭按整碗算）没有建模。
5. **卡路里份量回归**：现在固定份量先验 ±15%，真实差异 ±50%。
6. **外部测试集**：test 2,462 张全来自 CNFOOD 官方 val 同分布，75.40%（21 路）/87.73%（20 路）偏乐观；用 ChineseFoodNet 对得上的几道做外部测试。
7. **部署**：蒸馏 MobileNetV3 + int8（约 45 MB）适合小程序；服务需守护进程。
8. `重新打包` **已完成 2026-11-17**：dist/FoodDemo.zip 168.9 MB，含四份权重（单菜 / 40 类餐盘 plate40 / 116 类 / 食堂 21 路 cn241_21）+ 各自 calibration.json + results/tables/domain_head.json + 17 张混合餐盘示例 + 四套示例图；包内无 cn241_full 权重。zip 276 项已核对；包内实例此前在 8002 端口跑过，health / predict_sample 正常。

## 3. 目录结构与关键文件
E:\项目作业\food_project\
  README.md                 # 约 75 KB，项目文档（§1 目录/§3 数据/§4 训练/§5 结果/§6 网页与打包/§9 许可）；§5.3 混合餐盘 40 类、§5.7 食堂 20 类、§5.7.1 「其它中餐」21 类、§5.7.2 域判别头
  NOTICE                    # 数据来源与许可声明（仓库不分发任何图片）
  HANDOFF.md                # 本文件
  src\foodcls\              # 应用代码包
    app.py                   # FastAPI 主应用：payload/scope_of/cn_veto/cn_other/domain probe + /api/* 端点
    classes.py               # 20 类西餐 CLASS_SPEC
    classes_wide.py          # 116 类宽模型 WIDE_SPEC
    classes_cn.py            # 20 道食堂菜 CN_*（21 路里第 21 类叫 other_cn）
    plate_classes.py         # 混合餐盘 40 类顺序表：前 20 西餐、后 20 食堂 + 热量/份量/中文名
    models.py                # build_model(arch, n_cls)
    calorie.py               # topk_with_calories / PORTION_LABELS
    foodgate.py packaging_head.py brands.py data.py utils.py
    mixed.py                 # 混合餐盘：合成多标签样本（20/40 类通用）、detect_parts、plate_totals
  scripts\                   # 数据/训练/评估/打包脚本
    prepare_cn241.py prepare_cn241_other.py train_cn241.py eval_cn_scope.py eval_cn_domain.py
    eval_cn_crosscheck.py eval_cn_other.py eval_cn_other_extra.py search_cn_other_rule.py
    eval_gap_cn241.py eval_cn_confusions.py render_cn241_report.py
    extract_domain_features.py fit_domain_head.py eval_domain_gate.py pick_domain_threshold.py  # 域判别头
    package_demo.ps1         # 打包 -> dist/FoodDemo.zip（4 份 best.pt + 每类 2 张样例）
    train_mixed.py make_mixed_samples.py eval_mixed_plate.py   # 混合餐盘训练 / 示例图 / 40 类评估
  tests\test_cn_other.py     # 11 个用例：桶只否决不加分、桶不进热量表、重算门控、端点字段
  tests\test_domain_gate.py  # 9 个用例：域头 sigmoid/平均/否决/文案优先级/缺省
  tests\test_mixed_plate.py  # 9 个用例：20/40 类表切换、cuisine/热量/note、health 字段
  results\
    runs\<run>\best.pt + calibration.json + config.json + history.csv + results.json
    tables\*.json            # cn_other.json / cn_veto13.json / cn241_21.json / domain_head.json / domain_gate.json / domain_head_choice.json 等
    figures\demo_cn.png      # 食堂模式截图
  data\
    splits\cn241_full\{train,val,test}\   # 20 道食堂菜 21,043 张（旧）
    splits\cn241_21\{train,val,test}\     # 21 路：20 道 21,043 + other_cn 3,846（硬链接）
    splits\cn241_21_ood\                     # 留出 29 类 232 张（清单外中餐测试）
    splits\canteen_real\    # 100 张真实食堂照（分类子目录）
    splits\feedback\labels.csv  # 反馈闭环（至今空）

## 4. 数据与许可（详见 NOTICE）
- **Food-101**：HF ethz/food101 镜像，data/raw 20,000 张。许可 ETH 数据集。
- **Food-101N**（噪声标注 310,009 张）：Kaggle kuanghueilee/food-101n；本项目用其中 20 类 69,512 张（去重版 69,972）。论文 CleanNet arXiv 1711.07131。
- **16 道中餐**：Kaggle patzer0/16-famous-chinese-dishes（494 MB，带 YOLO 框），并入 data/wide 共 207,792 张。
- **CNFOOD-241**（241 类中餐 191,811 张）：Mendeley fspyss5zbb（train600x600.zip 9.3 GB + val600x600.zip 1.15 GB）；本项目取 20 道 21,043 张 + 「其它中餐」桶（175 类，train 2,100 / val 350 / test 1,396），剩余 29 类留出做清单外测试。**许可 CC BY-NC 3.0，部分图片源自 ChineseFoodNet，仅限非商用**。方法论文 PLOS ONE doi 10.1371/journal.pone.0358248。
- **麦当劳营养表**：仅作卡路里参考。

## 5. 关键常量与规则（改动前先读 app.py 顶部注释）
- 门控规则（三头一致）：六视图 TTA，报告融合后验；决策用前三视图 worst-view 分数（robust >= threshold）。镜像视图不参与门控。
- 21 路食堂头的 robust 是**扣桶**版本：每视图先 p_dish/(1−p_other) 再取 max_c，最后取三视图最小值（app.scope_of 里 cn_other_prob 非空时生效）。
- SCOPE_THRESHOLD=0.32（20 类）/ CN_SCOPE_THRESHOLD=0.40（旧 20 道版本，保留兼容）/ **CN_SCOPE_THRESHOLD_21=0.3445**（21 路，=验证集 5% 分位，env FOODCLS_CN_SCOPE_21）/ CN_OTHER_THRESHOLD=0.40（桶闸，env FOODCLS_CN_OTHER）/ WIDE_SCOPE_THRESHOLD（116 类）。
- veto：CN_VETO_CLASSES=13 元组（mapo_tofu, kung_pao_chicken, boiled_fish_sichuan, fish_with_pickled_cabbage, west_lake_vinegar_fish, steamed_sea_bass, peking_duck, soup_dumplings, dumplings, spring_rolls, fried_rice, ramen, braised_pork_meatballs）；CN_VETO_THRESHOLD=0.50、CN_VETO_MARGIN=0.02。刻意排除 sweet_and_sour_pork（=20 类同名糖醋里脊）与 twice_cooked_pork（=与农家小炒肉同盘）。
- cn_veto(img) 用宽头**原始**后验（_view_probs，不走 analyse_wide 的融合）；返回 {class, zh, prob, views}。
- other_cn 桶只做否决、不加分：payload() 截断前取 probs[20]，probs 截成前 20 路；桶没有中文热计量，绝不能进 topk_with_calories。closest 仍从前 20 路里取。
- 域判别头（纯否决）：DOMAIN_HEAD_PATH=results/tables/domain_head.json（env FOODCLS_DOMAIN_HEAD）、DOMAIN_THRESHOLD_OVERRIDE（env FOODCLS_DOMAIN_THRESHOLD，默认 0 = 用文件里的 0.2554）；`_domain_prob` 把三视图特征先 mean 再标准化点积，`p < threshold` 即 domain_hit。域头不存在时打印一行并退回「只有门控 + 桶」。
- 混合餐盘：MIXED_CKPT 默认 results/runs/resnet18_multilabel_plate40/best.pt（env FOODCLS_MIXED_CKPT）；_mixed_tables() 只接受「恰好 20 类 == CLASS_NAMES」或「恰好 40 类且前 20 与 CLASS_NAMES 一致」的权重，其它形状 RuntimeError 拒绝加载；每部分 cuisine 由 slug 是否在 CN_CLASS_NAMES 决定；/api/mixed/total 也只接受这 40 个 slug。
- 环境变量：FOODCLS_CKPT / FOODCLS_MIXED_CKPT / FOODCLS_WIDE_CKPT / FOODCLS_CN_CKPT / FOODCLS_CN_OTHER / FOODCLS_CN_SCOPE_21 / FOODCLS_DOMAIN_HEAD / FOODCLS_DOMAIN_THRESHOLD / FOODCLS_GATE / FOODCLS_PACK。

## 6. 模型成绩（权威数字）
- 20 类西餐：六视图 TTA top-1 89.11%（README 的 87.73% 是单视图协议，需统一口径）。
- 116 类宽模型：T=0.819，fuse 0.20；对 15 道中餐 top-1 89.7%。
- 食堂 20 道（旧）：val 88.25%、test top-1 87.73%/top-5 98.38%、ECE 2.36%、T=0.7256；弱类 农家小炒肉 71.9%、糖醋排骨 72.5%、白斩鸡 76.5%（全库仅 248 张）、黄焖鸡 77.1%；强类 干煸豆角 98.2%、酸辣土豆丝/西红柿炒鸡蛋 96.7%。
- **食堂 21 路（现行）**：best epoch 7，val 81.89%、test top-1 75.40%/top-5 97.49%、ECE 5.36%、T=0.827；清单内 20 类单视图平均 83.66%（20 路时 87.52%），other_cn 召回 59.2%。掉得最多：白斩鸡 58.8%、糖醋排骨 64.1%、青椒肉丝 70.2%、木须肉 83.5%。
- **21 路门控对照**（仍被接受）：in 91.3% / out_cn 24.2% / other 808 张 20.4% / ood 232 张 32.3% / real 21.0%；仅门控（无桶）时 in 94.4 / 35.3 / 46.7 / 55.2 / 42.0；20 路 + 13 类 veto：in 94.4 / out_cn 40.0 / other 52.2 / real 44.0。
- **域判别头（现行）**：val AUC 0.9737、阈值 0.2554；单独用 cn_te 91.3% / f101_te 14.4% / real 34.0% / cn_ood 79.7%。进完整决策链后：in 89.6% / out_cn 20.8% / ood 31.5% / 非中餐 9.7% / real 13.0%（域头关掉时 91.3 / 24.2 / 32.3 / 20.4 / 21.0）。
- **混合餐盘 40 类（现行）**：val plateF1 54.54（P 61.92 / R 48.73 / exact 10.92），逐类阈值 55.47；西餐盘 P 61.06 / R 47.09 / F1 53.17 / exact 10.50，食堂盘 62.75 / 50.38 / 55.89 / 11.33；detect_parts exact 8.17% / 8.50%；真实食堂照单图 top-1 70.00%、食堂 test 81.80%。旧 20 类同一批西餐盘 F1 55.80（parts 2.04）。
- 分菜接受率（21 路）：麻婆豆腐 8.3%、宫保鸡丁 8.3%、北京烤鸭/西湖醋鱼/清蒸鲈鱼/春卷/炒饭 0%、酸菜鱼 4.2%、拉面 12.5%、狮子头 12.5%、水煮鱼 25%、汤包 54.2%、饺子 66.7%、回锅肉 79.2%、糖醋里脊 91.7%。

## 7. 运行 / 复现命令
- **启动服务**（后台）：$env:PYTHONPATH='E:\项目作业\food_project\src'; & 'E:\项目作业\.venv\Scripts\python.exe' -m uvicorn foodcls.app:app --host 127.0.0.1 --port 8000
- **健康检查**：curl.exe -s http://127.0.0.1:8000/api/health（含 cn_other{enabled,threshold,gate_threshold}、cn_veto.enabled、cn_threshold）。**不要用 Invoke-WebRequest**（本机报「Windows PowerShell 处于非交互模式」）。
- **食堂模式测试**：GET /api/cn/predict_sample?name=tomato_egg__0000.jpg；POST /api/cn/predict（multipart file）。响应 scope.{in_scope,top1_prob,robust_prob,threshold,closest,label,veto,other{fired,prob}} + top3[]。
- **21 路评估**：python scripts/eval_cn_other.py（五集，~10 分钟）→ results/tables/cn_other.json；python scripts/eval_cn_other_extra.py（扣桶分数落盘）；python scripts/search_cn_other_rule.py（离线搜规则，打印对照表）。
- **域判别头**：python scripts/extract_domain_features.py --views 1（约 8 分钟）→ python scripts/fit_domain_head.py --mode mean → python scripts/eval_domain_gate.py（五集约 9 分钟）→ python scripts/pick_domain_threshold.py（离线扫阈值）。
- **训练食堂头**：python scripts/prepare_cn241_other.py 然后 python scripts/train_cn241.py --mode finetune --data-root data/splits/cn241_21 --classes results/tables/cn241_21_classes.json --out results/tables/cn241_21.json --save-dir results/runs/cn241_21 --epochs 8 --workers 0。
- **混合餐盘 40 类**：python scripts/train_mixed.py --classes plate --epochs 10 --n-train 24000 --n-val 600 --n-canteen 300 --n-real 80 --workers 4 --init-ckpt results/runs/resnet18_multilabel_aug/best.pt --out results/runs/resnet18_multilabel_plate40（混合训练 --workers 4 可用，约 27 分钟）；评估 python scripts/eval_mixed_plate.py（约 3 分钟）→ results/tables/mixed_plate40.json；示例 python scripts/make_mixed_samples.py。
- **测试**：python -m pytest tests -q（当前 67 个用例，约 8 秒；含 tests/test_cn_other.py 11 个、tests/test_domain_gate.py 9 个、tests/test_mixed_plate.py 9 个）
- **打包**：powershell -File scripts/package_demo.ps1 → dist/FoodDemo.zip（自动挑 resnet18_multilabel_plate40 与 cn241_21 权重，缺哪个就回退 / 跳过）。

## 8. Windows / 工程环境坑（新对话必读，避免重蹈）
- DSH 环境：文件策略已是 danger-full-access，审批已关闭——**不要请求沙箱升级**（会自动拒绝）。
- 内层 shell 是 **PowerShell 5.1**：.ps1 必须纯 ASCII；语法校验用 -NoProfile + Parser::ParseFile。
- python -c 传中文路径必被 GBK 控制台打断——**一律写 .py 文件再跑**；控制台打印中文会乱码（即使 PYTHONIOENCODING=utf-8），看结果用 read 工具或写 json。
- 训练 **--workers 0**（分类训练 >0 会挂死；混合餐盘训练 --workers 4 反而快，约 160 s/epoch）；GPU 8 GB，eval batch 用 4，别并行两个训练；warm start 到 40 类头时不能 strict 加载（fc 形状不符），要逐张量复制。
- 仓库里 README.md / HANDOFF.md 都是 **LF**（git 的 core.autocrlf=true 会在 diff 时归一化，所以看不出换行改动）；Python 的 read_text() 会把 CRLF 折叠成 LF，写回时别顺手改掉整份文件的换行符。多行 old_string 用 edit 会 not found，改用按行处理的 Python 脚本。
- JS 里写 Python/markdown：反引号和反斜杠会打架——把 Markdown 放进模板字符串时用 ` 占位反引号、\ 占位反斜杠，Python 里再 replace；Python 单引号 f-string 嵌在 JS 单引号字符串里会破坏 TS 解析。
- 中文文件名 curl 会失败/乱码——上传测试用 python urllib multipart。
- 服务被 kill 后不自动恢复——需要时手动重启（job 日志 results/_server.log）。
- 训练 21 路比 20 路慢一倍（381 vs 205 s/epoch），原因未查明（数据加载线程自旋），不影响结果。

## 9. 近期已解决（不必再碰）
- 环境故障：workspace-write 下 SetNamedSecurityInfoW failed (Win32 5) → 已切 danger-full-access 解决。
- 第 1 项双头交叉验证上线（README §5.7 有说明）。
- 第 2 项 21 类「其它中餐」重训 + 门控重写 + 服务切换 + README §5.7.1/§6 文档（2026-11-17）。
- 域判别头（第 3 项）上线（2026-11-17，README §5.7.2）。
- dist 包重打：cn241_21 权重 + domain_head.json（2026-11-17）。
- 反馈闭环三处 bug 修好：食堂 slug 校验、cn 画廊解析、mode=cn 白名单（2026-11-17）。
- 混合餐盘扩到 40 类（20 西餐 + 20 食堂）+ build_eval_set 标签映射 bug 修复（2026-11-17，README §5.3）。

## 10. 下一步行动建议
混合餐盘已扩到 40 类并上线（西餐盘 F1 53.2 / 食堂盘 55.9 / 真实照拼盘 44.0）。下一步建议：① §2 第 2 项 241 类预训练（+1~2 pt，约 1.5 小时，收益确定）；② 用反馈闭环攒真实托盘照再微调餐盘头（44.0→? 是当前最大 gap，但要有真实数据）；③ §2 第 5 项卡路里份量回归（现在固定份量先验 ±15%，真实差异 ±50%）；④ 三套门控统一 + 按域概率自动切标签（新设计，先想清楚误切代价）。任何改动前先 curl 一次 /api/health 确认服务状态。

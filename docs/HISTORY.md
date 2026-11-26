# FoodCLS 历轮记录与细节（归档，截至 2026-11-21 第十一轮）

> 归档文件：这里是 2026-11-19 第七轮结束时 HANDOFF.md 的完整内容——历轮过程、被否掉的方案、每轮的数字都在这里。**现行状态见 ../HANDOFF.md**。除补写历史细节外不再更新。

> 以下为原文照录（第一段是第七轮结束时的现行状态快照，之后逐轮倒序）。

## 0. 项目一句话
Windows 上的食品识别 + 热量估算网页应用：FastAPI 后端 + PyTorch resnet18 多套头 + 前端网页（4 个识别模式标签）。工作目录 E:\项目作业\food_project（注意：venv 在 E:\项目作业\.venv，不在项目内）。

## 1. 当前进度快照（重要）
**四个识别模式全部可用**：
- ① 20 类西餐（Food-101 课程 + Food-101N 清洁标注微调）+ 热量：results/runs/resnet18_last_stage_aug_f101n/best.pt（**不是**旧 resnet18_last_stage_aug），三视图门控阈值 0.32 + 食物门控 0.20 + 包装否决头 0.80；端到端 123 张真负样本只放过 2 张，100 张真实菜保留 95、top-1 95.0%。
- ② 混合餐盘（多标签，托盘上多道菜）：results/runs/resnet18_multilabel_plate40/best.pt，**40 类（20 西餐 + 20 食堂）**，sigmoid 阈值 0.35。
- ③ 116 类宽模型（含 16 道中餐）：resnet18_food101_cn116/best.pt，T=0.819，融合 0.20 的 20 类后验。
- ④ 食堂家常菜 **21 路**（20 道 + 「其它中餐」桶）：**cn241_21_pre/best.pt**（241 类预训练主干微调，T=0.778），门控阈值 **0.4088**（扣桶质量后的最差视图分数），桶阈值 0.40；外加**域判别头**（512 维特征 + 逻辑回归，阈值 **0.3524**）先挡掉非中餐。第一版（cn241_21，门控 0.3445、域头 0.2554）作为对照保留，见 §1 第七轮。

**最新（2026-11-19，第七轮）：食堂头换 241 类预训练主干（上线）**
- 起因：21 路一直从 Food-101N 主干热启动，主干没见过大多数中餐，弱类（白斩鸡 58.8%、糖醋排骨 64.1%）救不回来。改成先在 CNFOOD-241 全部 241 类上预训练，再回到 21 路。
- 预训练：scripts/prepare_cn241_all.py（train600x600 按 90/10 切）→ data/splits/cn241_all，train 153,746 / val 17,122 / test 20,943；scripts/train_cn241.py --mode finetune 从 f101n 热启动、分类头重初始化，5 epoch、batch 64、--workers 0，783/874/807/783/1723 s，2h32m；best val 70.94%，test top-1 69.27% / top-5 91.54% / ECE 2.56%，T=0.822。run results/runs/cn241_all_pre，表 results/tables/cn241_all_pretrain.json。
- 21 路微调：--init cn241_all_pre/best.pt，8 epoch、batch 64、--workers 0，best epoch 7，val 87.08% / macro-F1 88.45%，test top-1 **79.83%** / top-5 97.64% / ECE 5.11%，T=0.778（第一版 75.40%，+4.43 pt）。run results/runs/cn241_21_pre，表 results/tables/cn241_21_pre.json。弱类：白斩鸡 77.1%、木须肉 83.5%、糖醋排骨 71.0%、青椒肉丝 72.5%、农家小炒肉 76.3%、桶召回 66.5%。
- 门控重测：rob20r（每视图 p20/(1−p_other) 最差视图）验证集 5% 分位 = **0.4088**（rob21 0.4008、rob20 0.2626，scripts/search_cn_other_rule.py → results/_search_pre.log）。
- 域头重拟合（换主干必须重抽）：scripts/extract_domain_features.py --views 1 → results/tables/domain_feats_pre.npz（37,550 条），scripts/fit_domain_head.py --mode mean --tag pre → domain_head_pre.json：C=0.1、阈值 **0.3524**、val AUC **0.9857**（旧 0.974）、keep_cn 95.0%、reject_west 92.6%；单头 cn_te 90.4% / f101_te 6.8%（旧 14.4%）/ real 30.0% / neg 29.0% / cn_ood 72.8%。
- 完整决策链（eval_domain_gate.py，五集）：清单内 **88.9%**（只算门控+桶 91.6%；旧 89.6/91.3）/ 清单外中餐 **16.1%**（旧 20.8）/ 没见过的中餐 **22.8%**（旧 31.5）/ 非中餐 **5.1%**（旧 9.7）/ 真实西餐照 **12.0%**（旧 13.0）。表 results/tables/domain_gate_pre.json。代价：清单内 −0.7 pt，换来四列全面下降，且被接受照片 top-1 +4.4 pt。
- 代码：src/foodcls/app.py 的 DEFAULT_CN_CKPT=results/runs/cn241_21_pre/best.pt、CN_SCOPE_THRESHOLD_21 默认 0.4088（注释同步）；results/tables/domain_head.json 已换成新头，旧头备份为 domain_head_v1.json。评测脚本 eval_cn_other.py / eval_cn_other_extra.py / search_cn_other_rule.py 支持 EVAL_CN_CKPT / EVAL_TAG；fit_domain_head.py 支持 --tag，eval_domain_gate.py 支持 TAG。
- 训练日志：results/_train_all241.log、results/_train_21pre.log（原为 UTF-16LE，2026-11-19 已转 UTF-8）；评测日志 results/_domain_gate_pre.log。
- 睡眠事故：11-18 23:57 进入 Modern Standby、11-19 10:02 才唤醒，21 路 epoch 5 被记成 36,412 s（其余 228–411 s）——这是墙钟不是算力，见 README §5.7.3「踩过的坑」。
- 文档：README 新增 §5.7.3（数据/两次训练/弱类/门控与域头/决策链表/复现/踩坑），§5.7.1 与 §5.7.2 标注为第一版历史。
- 待做：pytest 全量、scripts/package_demo.ps1 换 cn241_21_pre 后重打包、重启 8000 并验 /api/health。
**上一轮（2026-11-18，第六轮）：宽头换主干——认真试过，结论是不换**
- 起因：第四轮把 20 类那条线的主干换成了 Food-101N 清洗标签重训的 resnet18（f101n），于是问 116 类宽头要不要也跟。看着划算：整个应用只剩一个主干家族，而且 15 道中餐每类只有几百张，起点强一点也不亏。
- 三种配方（scripts/train_wide.py，均 6 epoch、batch 96、workers 2、lr 头 1e-3 / 主干 1e-4）：(a) 只用 f101n 的 20 行按类名搬进 116 行头、其余 96 类随机初始化 → results/runs/resnet18_cn116_f101n；(b) 主干 f101n + 现役 116 行头整块搬入（`--init "…f101n/best.pt,…resnet18_food101_cn116/best.pt@head"`）→ results/runs/resnet18_cn116_f101n_hybrid；(c) 在 (b) 上续 4 epoch（lr 5e-4 / 5e-5）→ val F1 76.68 低于 (b) 的 77.12、train@1 已 95.15，过拟合弃用（目录 resnet18_cn116_f101n_hybrid_e10）。
- 对照（单视角；scripts/eval_wide_candidate.py 同一套温度缩放）：官方 test 116 路 Top-1 现役 **80.58** / (a) 79.36 / (b) 79.46；macro-F1 80.77 / 80.12 / 80.30；ECE 1.84 / 1.36 / 1.44；Food-101 分组 80.40 / 79.11 / 79.21；中餐 15 类 88.99 / 91.01 / **91.19**；20 类协议 test（限定 20 类）93.82 / **94.60** / 94.24；食堂退化照片（116 路全开）**95.50** / 90.20 / 89.60；100 张真实食堂照（限定 20 类）89.0 / 94.0 / **95.0**、116 路全开 71.0 / 74.0 / 73.0。现役这一列复算出的数与历史 results.json、wide_real.json、wide_domain.json 完全吻合，说明评测口径可信。
- 结论：**现役 resnet18_food101_cn116/best.pt 不动**（DEFAULT_WIDE_CKPT 未改）。理由：① f101n 只认识那 20 道菜，(a) 等于让另外 96 个 Food-101 类从头学，而现役宽头是 101 类 Food-101 头热启动的；② 把主干单独隔离后（b）f101n 主干本身也不占优，食堂退化照片 116 路全开 −5.9 pt，正是页面最常出错的那一格（脏一点/糊一点的西餐照被拽进中餐类）；③ 真实照 +6 pt 是限定 20 类口径、n=100，±4 pt 不足以下注；④ 中餐 15 类 +2 pt 是真的，真要吃得只在中餐子集上动手，而不是整套 116 路一起换。
- 顺手修掉一个真 bug：`--init` 原来按**位置**把前 20 行搬进 116 行头，而 `WIDE_CLASS_NAMES[:20]` 是字母序、`CLASS_NAMES` 是菜单序，整排错位。现在 scripts/train_wide.py 的 warm_start() 按类名落位，并支持 `--init a.pt,b.pt@head`（逗号分隔按顺序加载，`@head` 只搬分类头），config.json 的 warm_start 记成列表。
- 新工具：scripts/eval_wide_candidate.py（三视图 + 温度，报 top1_restricted / top5_restricted / top1_full116，test 段分 food101 与 chinese 两组；--skip-tta 省内存）、scripts/compare_wide_backbones.py（合成 results/tables/wide_backbone_compare.json 与 markdown 表）；产物 results/tables/wide_candidate_{deployed,f101n,hybrid}.json。
- tests/test_warm_start.py 4 个用例（按类名落位、head_only 不动主干、后一份覆盖头、未知源类跳过），全量 **78 passed**。
- 文档：README 新增 §5.5.1「宽头要不要也跟着换主干：一次否定的实验」（三种配方 / 12 行对照表 / 4 条理由 / 复现命令 / 先接电源的注），§5.5 命令块与 §1 目录树补两个脚本。
- 环境坑（新增）：笔记本用**电池**时 GPU 被功耗墙压到 19 W、SM 利用率 15%，一个 epoch 从 235 s 变成 25 分钟还没跑完，还会报 `OSError: [WinError 1455] 页面文件太小 … cufft64_11.dll`；接上电源后 99% / 49 W / 235–360 s/epoch 正常。commit charge（C 2 GB + D 16 GB pagefile）打满时，三视图 TTA 阶段报 `OpenBLAS error: Memory allocation still failed after 10 retries` → 训练用 --workers 2、评测加 --skip-tta。

**最新（2026-11-18，第五轮）：食物门控接进「116 类」与「混合餐盘」两个标签**
- 起因：门控此前只在单菜 20 类分支生效（scope_of 里写着 `not wide and not cn`），宽头与托盘两个标签对菜单牌/餐巾纸照样给菜名，各放过 70/123 张真负样本。
- 为什么零成本：门控读的是窄主干（20 类）的池化特征，而这两个标签本来就要跑一次窄主干——宽头为了 0.20 的后验融合（analyse_wide 内部调 _view_probs），托盘为了算稳健分数（_packaging_scope → analyse_full）——所以只是把已经算出来的 frames 传给 scope_of。
- 代码（src/foodcls/app.py）：analyse_wide(img, want_feats=False) 在 `alpha > 0 or want_feats` 时一次前向同时拿 p_narrow 与 frames，want_feats 时返回 (views, fused, frames)；新增 _food_prob(frames)（gate 或 frames 为 None 返回 None，否则 gate.prob(frames[0])）；scope_of 的门控条件与 scope["food_prob"] 写入条件都去掉 `not wide`；单菜两个端点改用 _food_prob；宽头两个端点传 food_prob=_food_prob(frames)；_packaging_scope 也传；mixed_payload 在 scope["not_food"] 时把 parts 清空、整盘热量归零，文案改成「这张照片里没找到菜：门控认为画面是菜单牌、饮料杯、包装袋或空盘一类的场景……」。
- 端到端（results/_sweep_wide_mixed_gate.py，走 HTTP，123 张真负样本 / 100 张真实食堂照）：116 类负样本 70→**8**、真实照 90→89；混合餐盘 70→**3**、真实照 89→88。代价是同一张 canteen_real/french_fries/002.jpg（p_food 0.15）在三个标签一起被拒。
- 漏网（宽头 8 / 托盘 3）p_food 在 0.28–0.80：饮料杯、甜品柜、包装食品，和单菜模式那 2 张同类，调阈值分不开，要补负样本。
- tests/test_food_gate_tabs.py 新增 7 个用例（无 frames/无门控为 None、读第一帧、宽头低 p_food 否决且带 food_prob/food_threshold、宽头高 p_food 保留、无门控时宽头行为不变、托盘 not_food 清空 parts 且热量 0、有菜时 parts 保留）；全量 74 passed。
- 文档：README §5.4.1 加「三个模式共用同一道门」小节 + 三行对照表，§6「没菜的照片」补一句两个标签同样生效。

**最新（2026-11-17，第四轮）：部署主干换成 Food-101N，食物门控与包装头真正生效**
- 起因：食物门控（foodgate）与包装否决头（packaging）的权重是在 **resnet18_last_stage_aug_f101n** 的 512 维池化特征上拟合的，而线上 DEFAULT_CKPT 一直指向旧的 resnet18_last_stage_aug，两个 json 又不进分发包——等于两个头全关着，端到端 123 张真负样本被放过 66 张。
- 代码：src/foodcls/app.py 的 DEFAULT_CKPT 改为 results/runs/resnet18_last_stage_aug_f101n/best.pt（顶部注释写明「换主干必须同时换 FOODCLS_GATE / FOODCLS_PACK，否则两个头静默失效」）；新增 GATE_HOME 常量，两条启动告警里带上「this backbone's gate pair lives in ...」；/api/health 新增 checkpoint 字段（= CKPT.parent.name），一眼看出线上跑的是哪个 run。路径同步改成 f101n：src/foodcls/predict.py、scripts/eval_real_photos.py、scripts/start_demo.ps1、scripts/start_public.ps1、scripts/demo_screenshot.ps1、docs/复现路径.md。
- 顺手修的陈旧默认值：start_demo.ps1 / start_public.ps1 的 MixedCkpt 指到 40 类 resnet18_multilabel_plate40，WideCkpt 指到 resnet18_food101_cn116。
- 打包：scripts/package_demo.ps1 的 mainRunName 改为 f101n，best.pt 旁边按 `"calibration.json","foodgate.json","packaging.json"` 逐个拷并打印 bundled；**旧 Food-101 主干不再进包**（打包错主干 + 这两个 json 比两个都不打更糟）。dist/FoodDemo.zip 168.9 MB / 278 项，含 4 份 best.pt（cn241_21、resnet18_food101_cn116、resnet18_last_stage_aug_f101n、resnet18_multilabel_plate40）+ f101n 的 foodgate.json / packaging.json + 3 份 calibration.json + results/tables/domain_head.json，断言「无 resnet18_last_stage_aug/best.pt」通过。
- 端到端实测（results/_api_sweep_port.py 走 HTTP，123 张真负样本 = 138 张里 v1 误标的 15 张已剔除）：旧主干无门控 66/123、旧主干 + 门控 43/123、新主干无门控 57/123、**新主干 + 两个门控 2/123**；真实菜保留 95/100 不变，真实菜 top-1 91.0% → 95.0%。分发包解到 dist/FoodDemo 在 8002 实跑同一批图，数字与源码树一致（2/123、95/100）。
- 门控自身的诚实口径（results/tables/gate_oof.json，按负样本类别留一法）：AUC 均值 0.959（min 0.929 hand / max 0.997 trash_bin），OOF 负样本 vs 100 张真实菜 AUC 0.954；只开门控 t=0.20 拒 92/131、真实菜留 96/100；robust 单用只拒 64/131；合并规则拒 116/131（88.5%）。
- 文档：README 新增 §5.4.1「先问这张照片里有菜吗：食物门控」（数据 / 留一法 / 操作点 / 四行端到端对照表 / 剩余 2 张漏网 / 复现），§5.9 补「包装头也绑在 f101n 特征上」，§5.10 门控区间改成 98.5% [94.6, 99.6] 与 95.0% [88.8, 97.8]（并说明两区间现在有重叠），§6「没菜的照片」与 §6.1 分包描述同步，§6 环境变量补 FOODCLS_GATE / FOODCLS_PACK。

**再上一轮（2026-11-17，第三轮）：混合餐盘从 20 类扩到 40 类（20 西餐 + 20 食堂）**
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
- 规则升级（src/foodcls/app.py）：门控分数改为每视图 p20/(1−p_other) 的最差视图（rob20r），阈值 CN_SCOPE_THRESHOLD_21=0.4088（验证集 5% 分位）；桶闸 p(other)≥CN_OTHER_THRESHOLD(0.40) 或桶直接夺冠即判「不在清单里」；13 类宽头 veto 不变（veto 用的是宽头原始后验）。
- 效果（仍被接受的比例，越低越好）：清单内 91.3%（20 类 + veto 时 94.4%）/ 清单外中餐 360 张 24.2%（原 40.0%）/ 其它中餐 808 张 20.4%（原 52.2%）/ 没见过的中餐 232 张 32.3% / 真实西餐照 100 张 21.0%（原 44.0%）。
- 仍漏（本质边界）：饺子 66.7%、汤包 54.2% → 包子；回锅肉 79.2% → 农家小炒肉；糖醋里脊 91.7% → 糖醋排骨（本来就是同一道菜）。
- 服务在跑：http://127.0.0.1:8000（后台 job **pwsh-04ba4933e39240e0**，日志 results/_server.log）。health 实测：checkpoint=resnet18_last_stage_aug_f101n / food_gate{enabled:true,0.2} / packaging_head{enabled:true,0.8} / temperature 0.7364 / mixed_classes 40 / cn_classes 20 / **cn_threshold 0.4088** / cn_other 0.40 / **domain 0.3524** / **cn_temperature 0.7783** / wide 116。宽头权重仍是 resnet18_food101_cn116（第六轮否掉了换主干）。cn 冒烟：/api/cn/predict_sample?name=hot_sour_potato__0000.jpg → in_scope=true、domain 0.9683、other 0.0724、robust 0.9651。

## 2. 待办清单（按建议顺序）
1. `21 类含「其它中餐」重训` **已完成 2026-11-17**（见 §1；README §5.7.1 有完整记录）。
2. `241 类预训练` **已完成 2026-11-19 并上线**（见 §1 第七轮与 README §5.7.3）：21 路 test 75.40% → 79.83%。**未做**：弱类的色温/亮度/容器增广（241 类预训练已经把弱类从 58–64% 拉回 71–83%，增广是否还有增量未验证）。
3. `中餐-西餐域分类头` **已完成 2026-11-17**（见 §1 与 README §5.7.2）：真实西餐照误接受 21.0%→13.0%，清单内 91.3%→89.6%。**未做**：三套门控统一 / 按域概率自动切标签——现在仍要手动点标签。
4. **托盘多菜扩展 + 反馈闭环**：反馈闭环**已修好（2026-11-17）**（食堂 slug 校验 / cn 画廊解析 / mode=cn 白名单三处 bug）；混合餐盘(②)**已扩到 40 类（20 西餐 + 20 食堂）**并在 8000 端口上线（见 §1 最新）。宽头/托盘门控**已接（2026-11-18，见 §1 第五轮）**。**未做**：真实托盘照的反馈→再训练闭环（labels.csv 至今 0 条）、UI「一步纠正」只在 top-1 < 0.55 或拒识时出现、一顿饭的份量先验（米饭按整碗算）没有建模。
5. **卡路里份量回归**：现在固定份量先验 ±15%，真实差异 ±50%。
6. **外部测试集**：test 2,462 张全来自 CNFOOD 官方 val 同分布，79.83%（21 路）/ 91.66%（20 路单视图，六视图 TTA 93.4%）偏乐观；用 ChineseFoodNet 对得上的几道做外部测试。
7. **部署**：蒸馏 MobileNetV3 + int8（约 45 MB）适合小程序；服务需守护进程。
8. `重新打包` **已完成 2026-11-19（第七轮重打）**：dist/FoodDemo.zip 168.9 MB / 278 项，含四份权重（单菜 f101n / 40 类餐盘 plate40 / 116 类 / 食堂 21 路 **cn241_21_pre**）+ f101n 的 foodgate.json 与 packaging.json + 4 份 calibration.json + 新版 results/tables/domain_head.json（阈值 0.3524）+ 混合餐盘示例 + 四套示例图；包内无 cn241_full 权重、**无旧 resnet18_last_stage_aug 主干、无第一版 cn241_21**。已核对 zip 条目。
9. `宽头从 f101n 重训` **已试并否掉 2026-11-18**（见 §1 第六轮与 README §5.5.1）：三种配方都略逊于现役，保持 resnet18_food101_cn116/best.pt；这一项留下的是 eval_wide_candidate.py / compare_wide_backbones.py 与按类名热启动。

## 3. 目录结构与关键文件
E:\项目作业\food_project\
  README.md                 # 约 85 KB，项目文档（§1 目录/§3 数据/§4 训练/§5 结果/§6 网页与打包/§9 许可）；§5.3 混合餐盘 40 类、§5.4.1 食物门控、§5.5.1 宽头换主干实验、§5.7 食堂 20 类、§5.7.1 「其它中餐」21 类（第一版）、§5.7.2 域判别头（第一版）、§5.7.3 241 类预训练主干（现行）
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
    train_wide.py eval_wide_candidate.py compare_wide_backbones.py   # 116 类训练 / 候选评测 / 对照表
  tests\test_cn_other.py     # 11 个用例：桶只否决不加分、桶不进热量表、重算门控、端点字段
  tests\test_domain_gate.py  # 9 个用例：域头 sigmoid/平均/否决/文案优先级/缺省
  tests\test_mixed_plate.py  # 9 个用例：20/40 类表切换、cuisine/热量/note、health 字段
  tests\test_food_gate_tabs.py # 7 个用例：门控进宽头/托盘两个标签（否决、清空 parts、无门控不变）
  tests\test_warm_start.py # 4 个用例：按类名热启动、head_only、覆盖顺序、未知源类
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
- **主干绑定**：DEFAULT_CKPT=results/runs/resnet18_last_stage_aug_f101n/best.pt；FOODGATE_PATH / PACKAGING_PATH 默认 = CKPT.parent/foodgate.json、CKPT.parent/packaging.json（env FOODCLS_GATE / FOODCLS_PACK）。两个头是在 f101n 的 512 维池化特征上拟合的，换主干必须同时换这两个文件，否则 /api/health 的 food_gate.enabled / packaging_head.enabled 变 false（文件不在）或静默失准（文件在但主干不对）。
- 21 路食堂头的 robust 是**扣桶**版本：每视图先 p_dish/(1−p_other) 再取 max_c，最后取三视图最小值（app.scope_of 里 cn_other_prob 非空时生效）。
- SCOPE_THRESHOLD=0.32（20 类）/ FOODGATE_THRESHOLD=0.20（食物门控，文件里的 threshold 优先）/ PACKAGING_THRESHOLD=0.80（包装否决）/ CN_SCOPE_THRESHOLD=0.40（旧 20 道版本，保留兼容）/ **CN_SCOPE_THRESHOLD_21=0.4088**（21 路，=验证集 5% 分位，env FOODCLS_CN_SCOPE_21）/ CN_OTHER_THRESHOLD=0.40（桶闸，env FOODCLS_CN_OTHER）/ WIDE_SCOPE_THRESHOLD（116 类）。
- veto：CN_VETO_CLASSES=13 元组（mapo_tofu, kung_pao_chicken, boiled_fish_sichuan, fish_with_pickled_cabbage, west_lake_vinegar_fish, steamed_sea_bass, peking_duck, soup_dumplings, dumplings, spring_rolls, fried_rice, ramen, braised_pork_meatballs）；CN_VETO_THRESHOLD=0.50、CN_VETO_MARGIN=0.02。刻意排除 sweet_and_sour_pork（=20 类同名糖醋里脊）与 twice_cooked_pork（=与农家小炒肉同盘）。
- cn_veto(img) 用宽头**原始**后验（_view_probs，不走 analyse_wide 的融合）；返回 {class, zh, prob, views}。
- other_cn 桶只做否决、不加分：payload() 截断前取 probs[20]，probs 截成前 20 路；桶没有中文热计量，绝不能进 topk_with_calories。closest 仍从前 20 路里取。
- 域判别头（纯否决）：DOMAIN_HEAD_PATH=results/tables/domain_head.json（env FOODCLS_DOMAIN_HEAD）、DOMAIN_THRESHOLD_OVERRIDE（env FOODCLS_DOMAIN_THRESHOLD，默认 0 = 用文件里的 0.3524）；`_domain_prob` 把三视图特征先 mean 再标准化点积，`p < threshold` 即 domain_hit。域头不存在时打印一行并退回「只有门控 + 桶」。
- 混合餐盘：MIXED_CKPT 默认 results/runs/resnet18_multilabel_plate40/best.pt（env FOODCLS_MIXED_CKPT）；_mixed_tables() 只接受「恰好 20 类 == CLASS_NAMES」或「恰好 40 类且前 20 与 CLASS_NAMES 一致」的权重，其它形状 RuntimeError 拒绝加载；每部分 cuisine 由 slug 是否在 CN_CLASS_NAMES 决定；/api/mixed/total 也只接受这 40 个 slug。
- 环境变量：FOODCLS_CKPT / FOODCLS_MIXED_CKPT / FOODCLS_WIDE_CKPT / FOODCLS_CN_CKPT / FOODCLS_CN_OTHER / FOODCLS_CN_SCOPE_21 / FOODCLS_DOMAIN_HEAD / FOODCLS_DOMAIN_THRESHOLD / FOODCLS_GATE / FOODCLS_PACK。

## 6. 模型成绩（权威数字）
- 20 类西餐（现行 f101n 主干）：单视图 test 91.66% / 代理域 71.60% / 真实照片 91.0%；六视图 TTA test 93.4% / 代理域 73.0% / 真实照片 95.0%（旧 Food-101 主干 90.54 / 66.00 / 85.0）。README 的 87.73% 是早期单视图协议，口径不同。
- **食物门控（现行）**：留一法 AUC 0.959（min 0.929 / max 0.997），OOF vs 真实菜 0.954；t=0.20 单独拒 92/131、真实菜留 96/100；与 robust<0.32 合并拒 116/131（88.5%）。端到端（/api/predict，123 张真负样本）**2/123**（旧主干无门控 66/123），真实菜 95/100、top-1 95.0%。
- 116 类宽模型：T=0.819，fuse 0.20；对 15 道中餐 top-1 89.7%。
- 食堂 20 道（旧）：val 88.25%、test top-1 87.73%/top-5 98.38%、ECE 2.36%、T=0.7256；弱类 农家小炒肉 71.9%、糖醋排骨 72.5%、白斩鸡 76.5%（全库仅 248 张）、黄焖鸡 77.1%；强类 干煸豆角 98.2%、酸辣土豆丝/西红柿炒鸡蛋 96.7%。
- **食堂 21 路（第一版，第七轮前的数字）**：best epoch 7，val 81.89%、test top-1 75.40%/top-5 97.49%、ECE 5.36%、T=0.827；清单内 20 类单视图平均 83.66%（20 路时 87.52%），other_cn 召回 59.2%。掉得最多：白斩鸡 58.8%、糖醋排骨 64.1%、青椒肉丝 70.2%、木须肉 83.5%。
- **21 路门控对照**（仍被接受）：in 91.3% / out_cn 24.2% / other 808 张 20.4% / ood 232 张 32.3% / real 21.0%；仅门控（无桶）时 in 94.4 / 35.3 / 46.7 / 55.2 / 42.0；20 路 + 13 类 veto：in 94.4 / out_cn 40.0 / other 52.2 / real 44.0。
- **域判别头（现行）**：val AUC 0.9737、阈值 0.2554；单独用 cn_te 91.3% / f101_te 14.4% / real 34.0% / cn_ood 79.7%。进完整决策链后：in 89.6% / out_cn 20.8% / ood 31.5% / 非中餐 9.7% / real 13.0%（域头关掉时 91.3 / 24.2 / 32.3 / 20.4 / 21.0）。
- **混合餐盘 40 类（现行）**：val plateF1 54.54（P 61.92 / R 48.73 / exact 10.92），逐类阈值 55.47；西餐盘 P 61.06 / R 47.09 / F1 53.17 / exact 10.50，食堂盘 62.75 / 50.38 / 55.89 / 11.33；detect_parts exact 8.17% / 8.50%；真实食堂照单图 top-1 70.00%、食堂 test 81.80%。旧 20 类同一批西餐盘 F1 55.80（parts 2.04）。
- 分菜接受率（21 路）：麻婆豆腐 8.3%、宫保鸡丁 8.3%、北京烤鸭/西湖醋鱼/清蒸鲈鱼/春卷/炒饭 0%、酸菜鱼 4.2%、拉面 12.5%、狮子头 12.5%、水煮鱼 25%、汤包 54.2%、饺子 66.7%、回锅肉 79.2%、糖醋里脊 91.7%。

## 7. 运行 / 复现命令
- **启动服务**（后台）：$env:PYTHONPATH='E:\项目作业\food_project\src'; & 'E:\项目作业\.venv\Scripts\python.exe' -m uvicorn foodcls.app:app --host 127.0.0.1 --port 8000（默认主干 f101n，两个门控从 run 目录自动加载；启动日志应有 temperature / food gate / packaging head 三行）
- **健康检查**：curl.exe -s http://127.0.0.1:8000/api/health（含 checkpoint=resnet18_last_stage_aug_f101n、food_gate{enabled,0.2}、packaging_head{enabled,0.8}、cn_other{enabled,threshold,gate_threshold}、cn_veto.enabled、cn_threshold）。**不要用 Invoke-WebRequest**（本机报「Windows PowerShell 处于非交互模式」）。
- **食堂模式测试**：GET /api/cn/predict_sample?name=tomato_egg__0000.jpg；POST /api/cn/predict（multipart file）。响应 scope.{in_scope,top1_prob,robust_prob,threshold,closest,label,veto,other{fired,prob}} + top3[]。
- **21 路评估**：python scripts/eval_cn_other.py（五集，~10 分钟）→ results/tables/cn_other.json；python scripts/eval_cn_other_extra.py（扣桶分数落盘）；python scripts/search_cn_other_rule.py（离线搜规则，打印对照表）。
- **域判别头**：python scripts/extract_domain_features.py --views 1（约 8 分钟）→ python scripts/fit_domain_head.py --mode mean → python scripts/eval_domain_gate.py（五集约 9 分钟）→ python scripts/pick_domain_threshold.py（离线扫阈值）。
- **训练食堂头**：python scripts/prepare_cn241_other.py 然后 python scripts/train_cn241.py --mode finetune --data-root data/splits/cn241_21 --classes results/tables/cn241_21_classes.json --out results/tables/cn241_21.json --save-dir results/runs/cn241_21 --epochs 8 --workers 0。
- **混合餐盘 40 类**：python scripts/train_mixed.py --classes plate --epochs 10 --n-train 24000 --n-val 600 --n-canteen 300 --n-real 80 --workers 4 --init-ckpt results/runs/resnet18_multilabel_aug/best.pt --out results/runs/resnet18_multilabel_plate40（混合训练 --workers 4 可用，约 27 分钟）；评估 python scripts/eval_mixed_plate.py（约 3 分钟）→ results/tables/mixed_plate40.json；示例 python scripts/make_mixed_samples.py。
- **测试**：python -m pytest tests -q（当前 78 个用例，约 7 秒；含 tests/test_cn_other.py 11 个、test_domain_gate.py 9 个、test_mixed_plate.py 9 个、test_food_gate_tabs.py 7 个、test_warm_start.py 4 个）
- **宽头候选评测**：python scripts/eval_wide_candidate.py --ckpt results/runs/<run>/best.pt --skip-tta --out results/tables/wide_candidate_<tag>.json 然后 python scripts/compare_wide_backbones.py（→ results/tables/wide_backbone_compare.json）。**先接电源**：电池下 GPU 被压到 19 W，一个 epoch 跑不完。
- **打包**：powershell -File scripts/package_demo.ps1 → dist/FoodDemo.zip（约 169 MB / 278 项；自动挑 f101n 主干 + foodgate.json + packaging.json、resnet18_multilabel_plate40、cn241_21，缺哪个就回退 / 跳过）。
- **端到端负样本扫描**：起服务后 python results/_api_sweep_port.py 8000 default_f101n（写 results/tables/api_sweep_default_f101n.json），再用 results/_cmp_sweep.py 与历史版本对比（123 张真负样本口径）。

## 8. Windows / 工程环境坑（新对话必读，避免重蹈）
- DSH 环境：文件策略已是 danger-full-access，审批已关闭——**不要请求沙箱升级**（会自动拒绝）。
- 内层 shell 是 **PowerShell 5.1**：.ps1 必须纯 ASCII；语法校验用 -NoProfile + Parser::ParseFile。
- python -c 传中文路径必被 GBK 控制台打断——**一律写 .py 文件再跑**；控制台打印中文会乱码（即使 PYTHONIOENCODING=utf-8），看结果用 read 工具或写 json。
- 训练 **--workers 0**（分类训练 >0 会挂死；混合餐盘训练 --workers 4 反而快，约 160 s/epoch）；GPU 8 GB，eval batch 用 4，别并行两个训练；warm start 到 40 类头时不能 strict 加载（fc 形状不符），要逐张量复制。
- 工作区的 README.md / HANDOFF.md / app.py 都是 **CRLF**（git 的 core.autocrlf=true 在 diff 时归一化，所以换行改动看不出来）；Python 的 read_text() 会把 CRLF 折叠成 LF，改写必须 read_bytes() → decode → split("\r\n") → 写回 "\r\n".join()。多行 old_string 用 edit 会 not found，改用按行处理的 Python 脚本。
- JS 里写 Python/markdown：反引号和反斜杠会打架——把 Markdown 放进模板字符串时用 ` 占位反引号、\ 占位反斜杠，Python 里再 replace；Python 单引号 f-string 嵌在 JS 单引号字符串里会破坏 TS 解析。
- 中文文件名 curl 会失败/乱码——上传测试用 python urllib multipart。
- **门控和主干是绑定的**：foodgate.json / packaging.json 是在 f101n 的池化特征上拟合的，换 FOODCLS_CKPT 必须同时换这两个文件；反过来，打包时也不能只打主干不打 json（旧包就是这么静默失去门控的）。检查方法：/api/health 的 checkpoint 与两个 enabled。
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
- 部署主干换成 f101n + 食物门控/包装头真正生效 + 分发包补两个 json（2026-11-17，README §5.4.1）：负样本 66/123 → 2/123，真实菜 top-1 91.0% → 95.0%。
- 食物门控接进宽头（116 类）与混合餐盘两个标签（2026-11-18，README §5.4.1）：负样本 70/123 → 8/123 与 3/123，真实照各掉 1 张（同一张 french_fries/002）。
- 宽头换 f101n 主干的 A/B（2026-11-18，README §5.5.1）：三种配方都略逊，保持现役 116 类权重；顺带修了 `--init` 按位置搬 20 行头的错位 bug（tests/test_warm_start.py 钉住）。

## 10. 下一步行动建议
部署侧这一轮把「没菜的照片」在四个标签里都压住了：单菜 2/123、116 类 8/123、托盘 3/123（真实照 95 / 89 / 88 张）。下一步建议：① §2 第 2 项 241 类预训练（+1~2 pt，约 1.5 小时，收益确定）；② 补漏网的饮料杯 / 甜品柜 / 包装食品负样本（宽头 8 张里 p_food 0.28–0.80，调阈值分不开）；③ 用反馈闭环攒真实托盘照再微调餐盘头（44.0 → ? 是最大 gap，但要有真实数据）；④ 三套门控统一 + 按域概率自动切标签（新设计，先想清楚误切代价）；⑤ 食堂模式阈值偏严（真实照只留 13/100），可单独再评估。任何改动前先 curl 一次 /api/health 确认 checkpoint 与两个 enabled。⑥ 宽头换主干已试过并否掉（§1 第六轮），想再吃中餐那 2 pt 应只在中餐子集上动手。

## 11. 2026-11-19 第八轮：域头放宽门控 + 外部集核查
- 新增 CN_SCOPE_TRUST（FOODCLS_CN_TRUST=0.90:0.36）：域头 ≥ 0.90 时门控 0.4088→0.36。调参只在 cn241_full/val 上、test 报数：清单内 88.9→90.3（val 90.8→92.0），代价 ood 22.8→23.7、real 12.0→13.0、accepted top1 93.8→93.3。新增 2 个单测（tests/test_domain_gate.py 共 11 个）。
- eval_domain_gate.py 增 val 集与 veto/other_top 字段（旧 npz 缺这两个字段，离线重扫与线上差 0.7 pt）。
- 16chinese 外部核查：与 20 道只有糖醋里脊 1 类重合；md5 + dHash 去重后 392 张，Top-1 88.52%、接受率 92.09%、被接受照片 Top-1 90.58%。新脚本 eval_external_cn.py + 指纹 _fingerprint_cn241_21.json（21,031 张）。
- 教训：16chinese 的 txt 是 Windows 绝对路径，截 .jpg 之后要保留扩展名再拼 images 目录，否则 0 候选。

## 12. 2026-11-19 第九轮：强增广 A/B（否）
- build_transforms 增 crop_scale/jitter/hue/gray_p（默认不变），train_cn241.py --aug strong：裁剪 0.45、jitter 0.5、hue 0.08、10% 灰度。8 epoch 同 init 同日种子：val 87.08→87.13、test 79.83→79.86（+0.03）、top-5 97.64→98.37、ECE 5.11→5.64。
- 分菜：白斩鸡 82.4→79.4、木须肉 90.7→86.6，糖醋里脊 87.8→90.8、青椒肉丝 72.5→74.8、手撕包菜 89.3→91.6，互相抵消。结论：瓶颈是类内差异（切法/配菜/酱色），不是拍照色温；权重未上线（results/runs/cn241_21_strong）。
- 单测：test_contracts 加强增广形状契约（81 passed）。
## 13. 2026-11-19 第十轮：三套门控收口 + 拒识后换标签（上线）
- **收口**：src/foodcls/app.py 新增 _SCOPE_SPECS 一张表（三套 names/zh/threshold/label/tab/examples），scope_of 开头按 mode 查表，判定逻辑一行未动；三个模式的名字/阈值/文案从此只在一处。
- **换标签**：_TAB_ORDER（cn→single,wide；single/wide→cn；tray 不参与，not_food/包装也不参与）、_tab_scope()（在目标标签上跑完整判定链，不是二次意见）、_refusal_reason()（domain/other/veto/weak）、suggest_tab()；payload 在拒识路径带 scope.suggest。前端 index.html 拒识卡片加按钮「换到 X 再试一次」（仅真实上传，点后 setMode + 同一文件重跑）。
- **规则**：只有目标标签自己 in_scope 才给按钮；食堂拒识原因 veto/weak 不给（那张本来就是食堂菜、只是认不出），domain/other 才给。
- **实测**（scripts/eval_tab_route.py，150/150/150/100/150/100 分层抽样，results/tables/tab_route.json）：食堂→(单菜,116 类)：真实西餐 87 拒 / 84 给 / 79 对（94%）；清单内 val 9/4/0、test 9/6/0（误给 10/300，全部答错）；out_cn 113/52、out_other 98/38、out_ood 57/0（外部集类名跨清单不同，这几列不适用）。单菜→食堂：val 36/35（97%）、test 50/47（94%）、out_cn 8/0、真实 0。
- **域概率当开关：否**。被食堂拒掉的 val 照片 p_domain 中位数 0.859、test 0.299、real 0.052，三组重叠；t=0.10 时 val 0/9、real 53/87。改用"为什么拒"之后清单内误给 12→10（val 8→4、test 9→6），西餐少给 1 张（85→84）。
- **延迟**：单次 verdict 均值 cn 67.8 ms / wide 67.4 / single 33.9（最差 320 ms），加在每次拒识上（不是只在点按钮时）。
- **测试**：新增 tests/test_tab_route.py 9 个（refusal reason 顺序、_TAB_ORDER 形状、suggest 门槛与 veto/weak 跳过），全套 90 passed。
- **文档**：README §5.7.5 新增；§5.7 第 484 行"手动切换、不做自动路由"改为指向 5.7.5；HANDOFF 第十轮、§7 地图。
- **教训**：用 read 全文 + write 回写会截断大文件（README 831 行曾被写成 336 行）；改大文档一律用按行处理的 Python 脚本，并先 git checkout 兜底。
## 14. 2026-11-21 第十一轮：类平衡 / focal A/B（否）+ 训练管线三个坑
- **动机**：21 路头最弱类样本很少（boiled_chicken 训练 223 张，多数类 842）。这一轮不动其它超参、只换损失，看能否补弱类。
- **代码**（scripts/train_cn241.py）：新增 FocalLoss(gamma, weight, label_smoothing)；class_weights(dataset, num_classes, cap=4.0, skip=())（反频率、clip 到 [1/cap, cap]、skip 的类钉回 1.0）；CLI 新增 --class-balance {none,weights} / --balance-cap / --balance-skip / --focal-gamma / --debug-every。tests/test_class_balance.py 6 个（γ=0 等价 CrossEntropyLoss、权重生效、focal 压高置信样本、cap、skip）。
- **四个 run**：各 8 epoch、同 init results/runs/cn241_all_pre/best.pt、batch 64、seed 42，只有损失不同。
- **明细**（val 取 best epoch / test 3,858 张；「20 道菜」= 去掉 other_cn 后的 2,462 张）：基线 val 87.08(ep7) / test 79.83 / 菜 87.41 / other_cn 66.5 / top5 97.64 / ECE 5.11 / 宏平均 86.3；bal 86.63(ep8) / 77.66 / **89.24** / 57.2 / 96.35 / 5.88 / **88.0**；focal 86.22(ep7) / 79.78 / 87.86 / 65.5 / **98.65** / **4.72** / 87.0；bal_noother 86.99(ep8) / 79.78 / 88.02 / 65.3 / 97.17 / 4.97 / 87.1。
- **结论**：类权重把菜认准了（白斩鸡 82.4→94.1、糖醋排骨 71.0→77.9、糖醋里脊 87.8→93.1）但把「不是这些菜」认废了（other_cn −9.2，占总 top-1 −2.18，因为兜底类占 test 36%）；把 other_cn 钉 1.0 后兜底类回到 65.3、菜的增益缩到 +0.61、总 −0.05（白干）；focal 只换到校准（top5 +1.01、ECE −0.38，val −0.86）。**三条都不上线**，默认仍是 cn241_21_pre/best.pt。瓶颈确认为类内差异 + other_cn 边界，不在损失函数——与第九轮强增广否掉同一结论。
- **时间黑洞（根因）**：① Windows 下 DataLoader worker 用 sys._base_executable 启动，venv 的 base 是 E:\Python\python.exe（CPU-only torch 2.14.1），子进程 import 错 torch 后**静默死锁**（父等 data、worker 0% CPU、无异常）；解 = $env:PYTHONPATH=<venv>\Lib\site-packages。② torch 默认 24 intra-op 线程空转：10 s 烧 206 CPU 秒、GPU 0–9%；解 = OMP/MKL=1 + workers 4，实测 124 张/s（单进程 51）。③ README 旧复现命令写 --workers 0，照跑慢 2.4 倍（已 7 处改为 4）。④ 电源方案 OEM Silent：CPU 2.2 GHz 无 turbo、GPU P4 / 100 W，epoch 337–438 s 属正常（pre 那轮 228–411 s 同量级）。
- **产物**：results/tables/cn241_21_{bal,focal,bal_noother}.json（权重目录 results/runs/cn241_21_* 被 .gitignore 忽略）；results/_run_bal_suite.ps1 / _run_bal_noother.ps1 / _cmp_bal.py / _cmp_dish.py。文档：README §5.7.6、HANDOFF 第十一轮与 §6 训练坑重写。

## 15. 第十二轮：网图筛选工具与指纹补 test（2026-11-21）

- **动机**：外部测试集只有 16chinese 的糖醋里脊一道菜（392 张，88.5%），报告里缺「20 道菜整体对外多少」这一栏。补它只能自己收图；而收图真正的难点不是下载，是判断哪些图能算「外部」——拼贴、水印菜单、自己 test 里的同一张，混进一张，数字就是假的。
- **先补的漏洞**：results/tables/_fingerprint_cn241_21.json 原来只含 data/splits/cn241_21/{train,val} 的 21,031 张，**没有 test 的 3,858 张**。改了 results/_hash_cn241.py 的 split 循环为 ("train", "val", "test")，重跑得 **24,889 张**（2.8 MB，229 张/s）。用新指纹重跑 scripts/eval_external_cn.py，16chinese 仍是 398 候选 / md5 重 0 / 近似重 6 / 用 392 张、top-1 88.52%、accept 92.09%、accepted top-1 90.58%，结论不变——说明旧结果没被污染，但漏洞是真实的。
- **新工具 scripts/ingest_web.py**（三个子命令，共用参数放到 argparse parent parser，否则 --inbox 等要写在子命令后面才能被识别）：
  - \`scan\`：每图测 md5、64 位 dHash（与 results/_hash_cn241.py 同一配方）、拉普拉斯方差（_gray 到 320 宽）、collage_cues（窄幅近均匀带 _flat_runs + 贯穿全图强边）、frame_frac（外圈 4 条边里纯白/纯黑且低方差的比例）。直接判死：dup_cn241 / dup_batch / near_dup_batch / near_dup_cn241(h≤4) / small(<400) / aspect(>2.6 或 <0.38) / animated / unreadable / unknown_class / no_class。只标记：collage / seam / frame(≥0.75) / blurry(<40) / tiny_file(<15 KB)。输出 review.csv（utf-8-sig，Excel 直接开）+ review/sheets/<slug>_NN.png（每 96 张一页，8 列，编号与 idx 对应，红框=判死、黄框=有 flag）。
  - \`apply\`：只认 verdict 以 k 开头的行；重编码 JPEG（长边 ≤1600、optimize、progressive）并剥 EXIF（RGBA/P 先贴白底）；写 accepted/<slug>/，被拒的 shutil.move 到 rejected/<slug>/（不删，留着看当初为什么扔）；写 manifest.csv（file/label/w/h/kb/md5/verdict/url/note）。inbox/<slug>/sources.csv（file,url）里的 URL 自动进 manifest。
  - \`status\`：按 load_dishes() 的 20 道菜对 --target 打进度条。
- **修掉的三个真 bug**（都是自测抓出来的）：① 参数定义在顶层 parser 上，子命令后面传 --inbox 报 unrecognized arguments；② Path.relative_to(ROOT) 在相对路径入参时抛 ValueError（改为先 .resolve()，并加 _rel 兜底）；③ apply 的 load_dishes 改名后漏改一处，short 列表 NameError。
- **一处语义修正**：类名写错/缺类名原来只是 flags（会被 apply 当成 _unlabeled 收进去），改成 reject:unknown_class / reject:no_class。同时区分 load_dishes()（20 道计分菜）与 load_labels()（含 other_cn，困难负样本要收）。
- **测试 tests/test_ingest_web.py 8 个**：dHash 在半尺寸重压下 ≤4 位差异（用高斯模糊的合成图，比真照片更严苛）、拉普拉斯分得开清晰/模糊、2×2 网格触发 collage_cues 而单张不触发、白边卡片 frame_frac ≥0.75、nearest 的汉明距离与空指纹边界（返回 64）、scan 把重复/过小/极端长宽比/缺类名判成不同 reject 前缀、apply 输出无 EXIF 的 JPEG 且拒收项被移走而保留项原地不动、空目录返回 2。全套 **104 passed**。
- **自测脚本**：results/_web_selftest_build.py 造 9 张假图（正常、副本、半尺寸、2×2 拼贴、300×200、1600×400 全景、未知类、无类名、以及从 data/splits/cn241_21/test/baozi 直接拷的一张），results/_web_selftest_check.py 验输出。实测：借来的 test 图被判 reject:dup_cn241 并指出撞的是哪张，拼贴标 collage，全景 reject:aspect，副本 reject:dup_batch，缩图 reject:near_dup_batch(h=0)。
- **文档**：README §5.7.7（新增）、HANDOFF 待办 5 与复现命令、HISTORY 本节。

## 16. 第十三轮：20 道菜对外测试集（623 张，95.7%）（2026-11-21）

- **动机**：第十二轮只做了工具，没有数字；报告里最缺的一栏仍是「20 道菜整体对外多少」。这一轮把图收齐、逐张过目、跑出诚实数字。
- **收图 scripts/fetch_web.py**：Bing 图片搜索当索引。**关键坑**：普通 `/images/search` 只回首批 34 格，`first=36/71/106` 仍是同一页 → 改打 `cn.bing.com/images/async`（参数 q/first/count=35/mkt=zh-CN/qft=+filterui:imagesize-large，Referer cn.bing.com/images/search），first=35 才拿到新的一页；`search_many` 按 1+page*35 翻页、页间 sleep 0.4。**第二个坑**：查询模板一度写成 `["{zh}", "{zh} 食堂 快餐"]`，后者让 Bing 返回握手照、商务会议、地球插画（hot_sour_potato 联系表 133–150 号全是谈判照片）→ **查询只能用菜名本身**，删掉变体后清空重抓。下载改 ThreadPoolExecutor(max_workers=8)，`_grab` 并行取字节后按提交顺序回放以保持编号确定。实测 20 菜 817 张 **370 s**（旧串行版本同量级要 20+ 分钟）。
- **scan**：817 张 → 730 keep / 87 判死（small 54、near_dup_cn241 3+1、dup_batch 2、near_dup_batch 27），51 张带 flag（collage 26、seam 12、frame 2）。来源域名统计（results/_host_stats.py）：pic.nximg.cn 336、n.sinaimg.cn 160、img95.699pic.com 130、materials.cdn.bcebos.com 35…大量来自图库/资讯站，摆拍与错标风险高。
- **人工清洗（这一轮的主要工作量）**：先出 20 张总联系表看污染，再出 38 张逐类联系表 `review/sheets/_cls_<slug>_<page>.png`（results/_web20_cls_sheet.py，6 列×4 行、tile 250、每格标文件名与模型 top，top==slug 用暖黄、错标用红并加 `X <top>`），把 72 张错例另拼 4 张大图（results/_web20_wrong_big.py → _aud2_1..4.png）逐张核对。
  - **判据**：① 模型判成别的菜且目视确实是别的菜（我自己标错）→ 删；② 非成品菜照片（九宫格拼贴、大字水印/步骤图、卡通线稿、白底产品图、生食材）→ 删；③ **模型判错但图上确实是该类 → 留**，这些是模型真错，必须计入诚实数字。
  - 具体错标（举例）：sweet_sour_ribs 那批 19 张是无骨里脊/咕咾肉；baozi 18 张是烧麦/小笼/素材站棚拍/卡通；boiled_pork 10 张是蒜泥白肉、夫妻肺片、毛血旺、水煮牛肉；braised_pork 4 张是生五花肉/回锅肉；mu_xi_pork 3 张是青椒炒肉/韭黄肉丝；pepper_pork 5 张是农家小炒肉/生肉丝；di_san_xian 4 张是小炒肉/干煸豆角/大字水印；dry_fried_beans 6 张是扁豆/焯水过程/生豆角；farmhouse_pork 5 张是豆角炒肉/葱爆肉。
  - 判定存 `results/_web20_decisions.json`（提交时复制为 results/tables/web20_review.json），由 results/_web20_apply_manual.py + _web20_apply_manual2.py 把 107 张 **移动**到 `data/external/web20/rejected_manual/<slug>/`（不删）。
- **第一版数字（730 张）**：top-1 90.14% / 接受率 80.68% / 被接受 93.21%。清洗后两轮（730 → 630 → **623**）：**top-1 95.67%（596/623）、接受率 84.43%、被接受 96.77%**（results/tables/cn_web20.json）。第二轮又删 7 张：baozi 白底剖面产品图/花卷/卡通线稿、braised_chicken 菠萝咕咾鸡/蒸整鸡、tomato_egg 只有番茄块没有蛋。
- **逐类**：最弱 braised_chicken 75.0%（n=32）、braised_pork 84.6%、mu_xi_pork 89.5%、sweet_sour_pork 92.1%，其余 16 类 94.7–100%。接受率最差 baozi 31.6%（19 张只放 6 张）、boiled_chicken 61.5%、fried_noodles 72.1%、cola_wings 76.2%。
- **误拒分解（results/_web20_reasons.py）**：537 accept / 47 domain / 25 other / 14 weak。域头仍是最大误拒源，集中在 baozi（19 张里 9 张）、boiled_chicken 11、cola_wings 9、fried_noodles 7；这四类 p_domain 均值 0.44/0.65/0.69/0.71，其余 16 类 ≥0.85。反事实（_web20_counterfactual.py）：去掉域头接受率 84.4→90.2%、top-1 不变；去掉 other 头 87.8%；去掉弱门 87.5%。→ **域头在内部五集净赚（§5.7.3），在网图上净亏**，因为网图是浅景深棚拍特写，域头学到的是「食堂托盘/深色桌布」场景而非菜本身（results/_web20_vs_test.py：web20 median_blur 802 vs 内部 test 505，plain_frame 0.003 vs 0.005，mean_frame 0.027 vs 0.32）。
- **27 张错例**：braised_pork→sweet_sour_ribs 6、braised_chicken→sweet_sour_ribs 4（重糖色小块肉）、braised_chicken→farmhouse_pork 3、mu_xi_pork→farmhouse_pork/fried_noodles 4、sweet_sour_pork→cola_wings 2。
- **和内部 test 的关系**：内部 test 79.83% 含 1,396 张 other_cn（36%），只算 20 道菜是 87.41%；外部 95.67% 更高，是因为网图全是单菜特写、主体占满、光线干净——这个差距正好量化「摆拍 vs 托盘」有多难（对应 §5.7.5 托盘 F1 44.0%）。
- **冻结**：图片不进 git（.gitignore 的 data/*），入库 results/tables/web20_manifest.csv（623 行：file/label/top/accept/p_other/p_domain）与 results/tables/web20_review.json（每张被删的理由）。此后不再增删——要扩就新开 data/external/web20_v2，两套都报。
- **测试**：tests/test_fetch_web.py 的 test_query_templates_cover_every_dish 还在引用已删除的 QUERY_TEMPLATES 列表（重命名成单数后没改），修为 QUERY_TEMPLATE 并加注释说明为什么不加语境词。全套 **114 passed**。
- **文档**：README §5.7.8（新增）、HANDOFF（标题/§1 外部集行/§3 待办 5/§5 复现命令/§7 地图）、HISTORY 本节。

## 17. 第十四轮：论文按食堂主线重写（2026-11-21）

- **动机**：课程要求英文 IEEE 论文 6–9 页、正文五节、定量结果必须真实运行。旧 paper/main.tex 全文 **0 行**写食堂线（CNFOOD / 21-way / domain probe / 623 张网图全部 0 命中），只讲 20 类 Food-101 + 1000 张合成代理域，最后构建停在 2026-11-15；10 页也超出 6–9 页建议。六天实验全没进论文。
- **结构改造**：把 paper 拆成 paper/sections/ 十个节文件（00_preamble / 01_abstract / 02_intro / 03_related / 04_dataset / 05_method / 06_experiments / 06_baseline / 07_discussion / 08_conclusion），新增 paper/_assemble.py 按序拼成 main.tex。**paper/_build_tex.py 已删除**（旧的 @→反斜杠真源，全文见 git 历史），改节文件后重拼，别手改 main.tex。
- **主线替换**：标题改为「Chinese Canteen Dish Recognition with Calorie-Range Estimation: Pre-Training Scale, Open-Set Gating and Cross-Domain Evaluation」；摘要/引言/相关工作/数据/方法/实验/局限/结论全部以 21 路食堂头为主线，西餐 20 类降级为第 VI 节基线（压缩后保留 E1–E8 主表与复杂度表）。
- **砍掉的内容**：原 §Distillation 整节（含 tab:distill）、§Qualitative Analysis 整节、Grad-CAM 图、calibration 图、TTA 段，共约 3 页；页数从 10 压到 **9**（达标）。
- **新增图**：results/figures/cn21_confusion.png（21 路行归一化混淆矩阵，红框标 other_cn 行）、results/figures/cn_gate.png（六集接受率双柱，域头开/关）。两张都由 scripts/make_cn_figures.py 代码生成，与 §V-C/§V-D 的数字同源。
- **新增数字宏**：scripts/make_paper_numbers.py 追加 cn*/gate*/web*/ext*/abl*/route* 约 70 个宏，全部从 results/tables/*.json 读，不手写。其中 cnDishAcc / cnOtherRecall / cnMacroFone 由 21×21 混淆矩阵复算（不是读 per_class 均值）。
- **三个编译坑**：① **TeX 控制词不能含数字**——\cn21Test 会被解析成 \cn + 21 + Test，报 Undefined control sequence，全部改名 \cnNoPreTest / \cnNoPreVal；② 表格行尾被写成三个反斜杠报 Misplaced \noalign，用正则收成两个；③ numbers.tex 里的千分位曾写成 18@{,}817（那是已废弃的 _build_tex.py 的转义约定），改为 {,}。
- **数字一致性**：正文里写死的 97.5 / 5.4 / 96.4 / 5.9 / 97.2 / 5.0 / 5.6 全部换成宏；接受率那句「84.4% against 84.4%」是笔误（应为 web 84.4 vs test 90.3），已改。
- **引用**：references.bib 新增 6 条——cnfood241（Mendeley DOI 10.17632/fspyss5zbb.1）、xiong2026cnfood（PLOS One 21(9):e0358248）、chen2017chinesefoodnet（arXiv:1705.02743）、scheirer2013openset、bendale2016openset、patzer16chinese；scheirer 的 @inproceedings 字段用 journal 会报 empty booktitle，改 @article。
- **构建命令**：python scripts/make_paper_numbers.py → python paper/_assemble.py → cd paper; pdflatex main && bibtex main && pdflatex main && pdflatex main。README §7 与 scripts/README_template.md 同步。

## 18. 第十五轮：演示视频分镜与分镜数字校验（2026-11-21）

- **动机**：课程要求 2–5 分钟演示，展示模型在**未参与训练的新样本**上的效果（results/_docx.txt:186）。论文已经 9 页定稿，剩下的交付项就是演示。但演示有个隐蔽风险：分镜里要念的具体数字（"90.4%"、"域头只有 23%"、"汉堡 + 薯条 + 牛排"）只对某一套权重成立，重训或重校准后静默失效，**录到一半才发现**。
- **录屏能力先探清楚**：本机无 ffmpeg；tools/ 只有 cloudflared.exe、ngrok.exe；venv 无 imageio / imageio_ffmpeg / moviepy / cv2 / mss / pyautogui / playwright / requests；pip install websockets 报 "ERROR: Could not find a version that satisfies the requirement websockets (from versions: none)"（无外网 PyPI）。Edge 157.0.4341.60 可 --headless=new --screenshot 截图但不能录屏。→ **结论：演示视频只能用户自己录**（Win+G 或手机拍），我们负责把分镜、素材、口播稿、校验做扎实。
- **11 张镜头**（data/external/demo_shots/，全部未参与训练）：01 酸辣土豆丝、02 红烧肉、03 地三鲜、04 包子、05 披萨、06 不是菜、07 鱼香肉丝（清单外）、08 麻婆豆腐（清单外）、09 北京烤鸭（清单外）、10 西式拼盘、11 食堂拼盘。来源混合：01/02/03/05 取自 623 张网图，06 取自 negatives，07 取自 cn241_21_ood，08 取自 16chinese，10/11 取自混合餐盘的 mix_02 / mix_13。
- **scripts/prepare_demo_upload.py**：把 11 张复制成中文文件名到 data/external/demo_upload/（01_酸辣土豆丝.jpg … 11_食堂拼盘.jpg）。理由很实际——网页上传后**不显示文件名**（src/foodcls/static/index.html:961-983 的 showPreview 只设 img.src），录屏时观众只能靠资源管理器里的文件名判断"我现在拖的是哪道菜"。
- **scripts/check_demo_shots.py**：把 11 张图 × 4 个标签（/api/cn/predict、/api/predict、/api/wide/predict、/api/mixed/predict）重放一遍，只比视频真正展示的字段（in_scope / threshold / domain / suggest / top / prob / plate_kcal），与 results/tables/demo_shots.json 对比；漂移 exit 1 并逐字段打印 a -> b，--update 覆盖基线；服务连不上 exit 2 并提示 scripts/start_demo.ps1。**踩坑**：初版在 main() 里写 global BASE_URL，但模块级 BASE_URL 已被用作 --url 的默认参数 → "SyntaxError: name 'BASE_URL' is used prior to global declaration"；改成把 url 当参数传给 snapshot() 后正常。
- **11 个镜头的实测数字**（results/tables/demo_shots.json）：酸辣土豆丝 cn 95.1% / 阈值 **0.36**（域 0.9992 ≥ 0.90 触发放宽）/ 150–210 kcal；红烧肉 99.3% / 390–520 kcal；地三鲜在单菜标签被拒（top 蛋炒饭 39.1%）且**给出"换到食堂菜"按钮**，换过去 98.1%；鱼香肉丝食堂拒识（其它中餐 80%）、116 类给 53.7%；不是菜那张 not_food=True 无候选；食堂拼盘 炒青菜 65.3 + 干煸豆角 65.1 + 辣子鸡 51.3 / 550–840 kcal；西式拼盘 汉堡 100% + **华夫饼 87.9%（应为薯条）** + 牛排 56.2%（误检）/ 1100–1500 kcal。
- **两个刻意的"不美化"**：① 混合餐盘那一段保留失败案例（薯条被判成华夫饼、多检出一块牛排），因为 plate-F1 只有 54.5，讲"很准"会被现场打脸；② 包子那张别当"阈值放宽"的例子——它的域概率只有 0.857 < 0.90，阈值仍是 0.4088，正好可以讲"域头不确定时保守"。
- **实测延迟**（results/_demo_latency.py，整条 HTTP 往返含上传）：cn 89–138 ms、single 29–76 ms、mixed 37–88 ms。payload 里**没有 elapsed_ms**，网页上那个 1.25 s 是端到端含上传的，随机器变化 → 视频里只说"单张百毫秒级"。
- **产出**：docs/演示视频分镜.md（分镜表 11 镜 + 逐镜口播稿 + 每镜头确切数字表 + 录制前检查 + 现场演示备选）、README §6.4、HANDOFF 第十五轮与待办 8、本节。

## 19. 第十六轮：A/B 改进落地——集成上线、米饭拆第 22 类、域头重拟合（2026-11-26）

- **动机**：第十六轮开头盘了一遍「除食堂实拍数据外还能改什么」（A 类零新数据、B 类现有数据再挖）。用户挑明「先把 A、B 类做完」，于是按 A1–A6 / B1–B3 九项逐条落地。**没有一项依赖新采数据**。
- **A2 先修一个静默 bug**：scripts/eval_mixed_plate.py 的逐类阈值原来把 40 类矩阵配上后 20 个类名（cuts 与类别错位），逐类阈值后的 plate-F1 一直偏低 0.17–0.37；改成两段切片后 **wF1 53.97→54.14、cF1 56.07→56.44**。
- **A1 五模型集成上线**（README §5.7.9 ①）：成员 aug / f101n / f101n_nodedup / f101n_e24 / seed7，全部 resnet18 + 20 类。先补齐五个 run 的 val 集成缓存（results/_ens_collect.py 的 SPLITS 加 val），再用 scripts/fit_ensemble_temperature.py 在 2,000 张 val 上最小化 NLL 拟合**统一温度 0.63**。test 94.40% / ECE **1.52%**（单模型 93.40 / 2.75）；代理域 76.10 / 2.63；重抽种子 73.50 / 4.35。**门控不动**：阈值仍 0.32，且只读 0 号成员（f101n）的池化特征，因为 foodgate.json / packaging.json 是在它上面拟合的。app.py：`_view_probs` 接受模型列表（先平均 logits 再除温度）、新增 `_narrow_models()`、`analyse_wide` 的 narrow 分支改走它、health 加 ensemble 块。`scripts/start_demo.ps1 -NoEnsemble` 与 `FOODCLS_ENSEMBLE=0` 可退回单模型。延迟 4.5 ms → 18.7 ms（六视图）。
- **A5 米饭拆成第 22 类**（README §5.7.9 ②）：米饭 = CNFOOD-241 第 172 类，cn241_all 里 train 811 / val 90 / test 66，原来并进 other_cn 桶。新脚本 scripts/prepare_cn241_rice.py 从 cn241_21 硬链接 20 道菜、从 cn241_all 硬链接 rice、按同一 pool/heldout/CAP 重导桶（只跳过 172）→ **data/splits/cn241_22**（桶 2,088/348/1,388，比 21 路各少 12/2/8，正好是米饭；other_pool 175→174）。**关键约束**：rice 刻意**不进** classes_cn.py 的 CN_CLASS_NAMES——plate_classes.py 用它派生 40 类餐盘表，加进去会变 41 类、废掉 plate40 权重；改为派生 CN_MENU_* 供 22 类头使用。app.py 把「菜数」动态化：_state 里的 cn_names / cn_zh / cn_kcal / cn_portion / cn_dish_count / cn_other_index，scope_of / payload / _tab_scope / /api/cn/classes / health 全部读 _state（回落 20 类常量），桶位置一律取最后一维。
- **22 类头的成绩**：8 epoch，best epoch 7 val 87.53、T=0.774；**test top-1 80.54 / top-5 98.06 / ECE 5.44（n=3,916）**，对照 21 类 79.83 / 97.64 / 5.11（n=3,858）。**米饭类 87.88%**、桶 68.30%（21 类 66.48%）。20 道菜宏平均 87.28→**86.97**（−0.31）：涨的是糖醋里脊 +4.6、青椒肉丝 +4.6、手撕包菜 +3.1、黄焖鸡 +3.1、红烧肉 +3.1；跌的是辣子鸡 −4.6、鸡蛋羹 −3.8、炒面 −3.1、包子 −3.1。
- **22 类的五集决策链**（results/tables/cn_other_22.json / domain_gate_22.json）：清单内 **90.43%** / 清单外中餐 20.56% / 没见过的中餐 25.00% / 其它中餐桶 6.81% / 真实西餐照 10.00%，对照 21 类的 90.33 / 16.11 / 23.71 / 5.20 / 13.00——**每一集都更好**。门控阈值沿用 **0.4088**：22 类 val 5% 分位是 0.4064，几乎相同。
- **域头必须跟着换**（README §5.7.9 ④）：域判别头在主干池化特征上拟合，旧 21 类头套到 22 类骨干上会漂移——实测 AUC 0.9779→**0.9513**，西餐域概率均值 0.100→0.251。用 22 类骨干重抽 37,696 张特征（extract_domain_features.py 新增 --data-root，并修了 cn_ood 路径被前缀重写误伤的 bug）后重拟合：**C=0.03、阈值 0.2868、val AUC 0.9831**（旧 0.9737），保留 95.0% 中餐的同时拒掉 90.25% 西餐（旧 85.1%），f101_te 接受率 14.4%→9.7%。
- **A6 服务侧与文档**：① 装 pillow-heif 1.9.0 并加 _register_heif()，iPhone 的 HEIC 照片能直接上传（health 的 heif 字段可验），五处 400 文案统一成 UNREADABLE_IMAGE；② CORS 中间件（FOODCLS_ORIGINS，默认 *——demo 无 cookie，token 走 query/header）；③ /api/* 请求计数中间件 + health 的 requests 块（n / errors / ms_avg / ms_max / last_status），5xx 或 >3 s 打 WARNING；④ 论文 06_baseline.tex:181 的 20 independent sigmoids 改为 **40**（餐盘头一直是 40 类）；⑤ docs/HISTORY.md 章节号 12/13/14 与 8–11 轮重复，已重排为 11–18。
- **A3 搭配先验：实现了、评测了，不上线**（README §5.3 末尾）：新模块 src/foodcls/plate_prior.py 给 40 类标角色（主食 / 主菜 / 配菜 / 甜点），配对称权重（主食-主菜 1.00、主菜-配菜 0.90、主食-配菜 0.75、主菜-主菜 0.55、配菜-配菜 0.50、主食-主食 0.35、甜点-甜点 0.70、中西混搭 0.25、兜底 0.02），train_mixed.py --pairing-prior 时替换 pair_indices / MixedPlateDataset 的均匀抽样，并新增 scripts/eval_pairing_prior.py 做「同一套 prior=False 验证盘 + 同一阈值」的交叉评测。结果：西餐 0.35 53.17→**54.04**、食堂 0.35 55.89→**56.83**、真实照盘 0.30 44.38→**46.87**；但逐类阈值不稳（炒青菜被校到 0.68），演示托盘 11 号从 3 样掉到 2 样，净收益不到 1 pt。**默认关闭**，这是本轮唯一没上线的一项。
- **A4 反馈回流**：新增 scripts/build_feedback_split.py，把 data/splits/feedback/labels.csv 按 mode 分到四个标签空间（single / cn / wide / mixed），物化成 data/splits/feedback_built/ 并写 results/tables/feedback_split.json；单菜模式要求 corrected 恰好一个、mixed 行支持划掉（纯划掉时回落到 predicted 并计入 weak_positive）。配套 tests/test_feedback_split.py 11 个用例。**但 CSV 里只有 2 条自测样本，还没有数据可训**——管线通了，等真机纠正。
- **B1 canteen_real 逐类拆解**（scripts/eval_wide_real_detail.py → results/tables/wide_real_detail.json）：100 张真实照的 16 个文件夹**全是西餐类名**。宽头全开 116 类 top-1 **71.0%**，限制在 20 个西餐标签内 **89.0%**；29 张全开错、其中 25 张的 top-1 落在 20 类之外。**restricted 口径的 11 张错里 5 张是披萨**（披萨类内 acc 只有 0.545，常被判成 lasagna / hot_dog / pancakes）；答对平均置信度 0.777、答错 0.430。
- **B2 食堂头退化曲线**（data.py 的 degrade_canteen_image 加 strength 参数，scripts/eval_cn_degrade.py）：6 档强度、每类 20 张共 440 张。**top-1 从 0.850 掉到 0.746（−10.4 pt），门控通过率从 0.920 掉到 0.834（−8.6 pt）**；0→0.6 段掉 7.3 pt，0.6→1.0 再掉 3.1 pt，是缓坡不是悬崖。随机数抽取顺序与 strength 无关，同一 seed 只改伤害量，默认 1.0 与旧行为一致。
- **B3 8 张真实混合托盘**（scripts/eval_mixed_real.py → results/tables/mixed_real.json）：5 张有标签的托盘 **P 0.600 / R 0.300 / F1 0.400**；3 张无菜托盘里 2 张误检。三个病灶：① pancakes 从没被单独检出过（4 张煎饼盘全被 waffles 吃掉）；② 中式类会吸走西式托盘（004→fried_noodles、006→cola_wings、008→farmhouse_pork）；③ 空托盘误检（007→donuts 0.94）。
- **测试**：本轮新增 tests/test_feedback_split.py（11 例），全量 **125 passed**。

## 20. 第十七轮：文档收尾、重新打包与提交（2026-11-26）

- **演示分镜按新基线重算**：22 类头 + 新域头改变了 cn 分支的每一个概率，scripts/check_demo_shots.py 首跑 18 处 DRIFT（全是 single / wide / cn，mixed 分支不动——餐盘头没换），已 --update 覆盖 results/tables/demo_shots.json（旧基线备份 results/_demo_shots_before_ens.json 与 _demo_shots_before_cn22.json）。docs/演示视频分镜.md 同步：镜头 3 **95.1%→97.2%**、镜头 4 **99.3%→99.5%**、镜头 5 换台后 **98.1%→97.9%**，域概率 0.9992→0.9981 / 0.9997→0.9992 / 0.9999→0.9998；口播「食堂菜管 21 路」改为 **22 路（20 道菜 + 米饭 + 其它中餐）**。
- **一条注意事项整个作废**：原文说「包子那张域概率只有 0.857 < 0.90，阈值仍是 0.4088，正好讲域头不确定时保守」——22 类域头在包子上给 **0.9334 ≥ 0.90**，阈值自动放宽到 0.36、包子 93.2% 被接受，这条已经反过来。改写为「包子这张现在会放宽阈值」，并把讲保守的例子换成 **08 麻婆豆腐（域 0.8499 < 0.90，阈值仍 0.4088）**。
- **README**：§5.3 末尾新增「搭配先验 A/B（已实现、已评测，未上线）」；§5.7.3 标题标注「21 类，已被 5.7.9 取代」；**新增 §5.7.9「部署侧收尾」**（11 个小节：集成 / 米饭 22 类 / 五集表 / 域头重拟合 / 逐类阈值修复 / 退化曲线 / 真实照逐类 / 8 张 mixed_real / 反馈回流 / 服务侧三件 / 文档订正）；§6 食堂模式整条与打包 bullet 重写；§5.7 / §5.7.2 / §5.7.6 的交叉引用一并指到 5.7.9。
- **HANDOFF 全篇刷新**：标题行改第十七轮；四模式表第①行改成五模型集成（94.40 / ECE 1.52 / 18.7 ms）、第④行改成 22 路（80.54 / 98.06 / 5.44 / 米饭 87.88）；五集数字改 90.43 / 20.56 / 25.00 / 6.81 / 10.00；服务 job 改 **pwsh-5baf0a9a8c9549fa**、health 预期值改 cn_classes 21 / domain 0.2868 / cn_temperature 0.7745 / ensemble 5 / heif true；常量表、env 列表、复现命令全部同步；测试数 114→**125**。
- **重新打包**：scripts/package_demo.ps1 已含四个集成成员与 cn241_22_pre。实跑输出 **327.6 MB / 343,539,726 B / 286 项**（解包 369.9 MB），8 份 best.pt（集成五份 + plate40 + cn116 + cn241_22_pre）+ ensemble_calibration.json + 22 类版 domain_head.json。
- **回归**：pytest → **125 passed, 1 warning in 7.65 s**；scripts/check_demo_shots.py → **all 11 shots match**；/api/health 实测 cn_classes 21 / ensemble 5 (T=0.63) / domain 0.2868 / cn_temperature 0.7745 / heif true。
- **产出**：README §5.3 / §5.7.3 / §5.7.9 / §6、HANDOFF 全篇、docs/演示视频分镜.md、本节。提交 **d07c491**（43 个文件：源码、脚本、结果表、文档、论文 40 sigmoids 订正）+ **a3bb31f**（重打包后的 domain_head.json、mixed_plate 表、paper/main.tex）；dist/FoodDemo.zip 327.6 MB 不入库（.gitignore）。

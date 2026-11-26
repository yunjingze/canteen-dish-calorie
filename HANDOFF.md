# FoodCLS 食品识别与热量估计 —— 项目交接文档（现状）

> 只放「现在是什么状态 / 怎么跑 / 哪里会踩坑」，目标一屏半读完。历轮实验的完整过程、被否掉的方案、每轮数字全在 [docs/HISTORY.md](docs/HISTORY.md)，考古去那里。最后更新：2026-11-26（第十七轮：A/B 改进落地——五模型集成上线、食堂头拆出米饭改 22 类、域头用新骨干重拟合、逐类阈值错位修复、反馈回流脚本、三项诊断；README §5.3 末尾与 §5.7.9）。

## 0. 一句话

Windows 上的食品识别 + 热量估算网页应用：FastAPI + PyTorch resnet18 多套头 + 前端 4 个识别模式标签。工作目录 E:\项目作业\food_project；**venv 在 E:\项目作业\.venv，不在项目内**。

## 1. 当前状态

四个模式全部可用，权重与常数如下：

| 模式 | 权重 | 关键常数 | 权威成绩 |
|---|---|---|---|
| ① 20 类西餐（**五模型集成**） | results/runs/resnet18_last_stage_aug_f101n/best.pt + 另四份（aug / nodedup / e24 / seed7） | 门 0.32 + 食物门 0.20 + 包装否决 0.80；集成 **T=0.63** | test 六视图 **94.40%** / ECE **1.52%**（单模型 93.4 / 12.6）；代理域 76.10 / ECE 2.63；真实照 top-1 95.0%；推理 18.7 ms vs 4.5 ms |
| ② 混合餐盘 40 类 | results/runs/resnet18_multilabel_plate40/best.pt | sigmoid 0.35 | val plate-F1 54.54（西 53.17 / 食堂 55.89） |
| ③ 116 类宽头 | results/runs/resnet18_food101_cn116/best.pt | T=0.819，融合 0.20 | test top-1 80.58%，中餐 15 类 89.7% |
| ④ 食堂 **22 路**（20 道 + 米饭 + 其它中餐） | results/runs/cn241_22_pre/best.pt | T=0.774；rob20r 门 **0.4088**（域头 ≥ 0.90 → 0.36）；桶闸 0.40；域头 **0.2868** | test top-1 **80.54%** / top-5 98.06% / ECE 5.44%；米饭 87.88%、桶 68.30% |

食堂完整决策链（五集，results/tables/domain_gate_22.json）：清单内 **90.4%** / 清单外中餐 20.6% / 没见过的中餐 25.0% / 其它中餐桶 6.8% / 真实西餐照 10.0%；验证集 92.7%。21 类旧版（domain_gate_pre.json）是 90.3 / 16.1 / 23.7 / 5.2 / 13.0，权重 cn241_21_pre 仍在盘上做对照。**换食堂头必须同时换域头**（域头是在主干池化特征上拟合的，results/tables/domain_head.json 现在是 22 类版）。外部集：16chinese 糖醋里脊 392 张 88.5%（cn_external.json）；**自筛网图 20 道菜 623 张 top-1 95.67% / 接受率 84.43% / 被接受 96.77%**（cn_web20.json，README §5.7.8）。

损失函数 A/B 已试并否（README §5.7.6）：反频率类权重把 20 道菜 +1.83（白斩鸡 82.4→94.1），但兜底类 other_cn −9.2，总 top-1 −2.18；把 other_cn 权重钉回 1.0 后兜底类回到 65.3、菜的增益缩到 +0.61、总 −0.05（白干）；focal γ=1 总 top-1 持平、top-5 +1.0、ECE −0.38，但 val −0.86。三条都不上线，默认权重不变。

拒识卡片现在给一个「换到 X」按钮（README §5.7.5，results/tables/tab_route.json）：食堂标签拒掉的真实西餐照 87 张里给 84 个按钮、79 个答对（94%）；清单内 300 张里误给 10 张（0 对，都是食堂标签自己认不出的难图）。单菜→食堂很干净（清单内 94–97%，外部集 0 误切）。域概率当开关已试并否掉——被拒食堂照与西餐照的域概率重叠，改看拒识原因（veto/weak 不给按钮）。

## 2. 服务与分发包

- 服务：http://127.0.0.1:8000，后台 job **pwsh-5baf0a9a8c9549fa**（2026-11-26 夜间重启，health 已验），日志 results/_server.log（UTF-8）。health 应显示 checkpoint=resnet18_last_stage_aug_f101n / **ensemble.members 5**（T=0.63） / **cn_classes 21**（20 道菜 + 米饭，桶不计） / cn_threshold 0.4088（域头 ≥0.90 的图 payload 里报 0.36） / **domain.threshold 0.2868** / **cn_temperature 0.7745** / mixed_classes 40 / wide 116 / **heif true**。
- 分发包：dist/FoodDemo.zip **327.6 MB / 343,539,726 B / 286 项**（2026-11-26 22:37 重打包，解包 369.9 MB），含 **8 份 best.pt**——集成五份（f101n + aug / nodedup / e24 / seed7）、plate40、cn116、**cn241_22_pre** + ensemble_calibration.json + 22 类版 domain_head.json 与各 run 的 calibration.json；**不含**旧 cn241_21_pre、cn241_21、以及其它被否掉的历史 run。`scripts/start_demo.ps1 -NoEnsemble`（或 `FOODCLS_ENSEMBLE=0`）可退回单模型，此时包只需 f101n 一份。
- git：最新提交见 `git log --oneline -1`（第十七轮把 A/B 改进整体入库）；再往前 4951b81、c65a563、a488747、e4c2e4b（演示分镜 + 分镜数字校验脚本）、9a1965e（论文按食堂主线重写、9 页）、ee8384d（623 张网图外部测试集）。
- 论文：paper/main.pdf **9 页**（IEEE conference，要求 6–9 页），正文由 paper/sections/ 十个节文件经 paper/_assemble.py 拼成；**旧的 paper/_build_tex.py 已删除**（全文见 git 历史，别再手改 main.tex）。
- 测试：python -m pytest tests -q → **125 passed**（约 8 s；新增 test_feedback_split.py 11 个用例）。分镜数字另有 scripts/check_demo_shots.py（需服务在跑，11 张 × 4 标签）。

## 3. 待办（只列还没做的）

1. **弱类增广 / 重加权**：241 类预训练把白斩鸡 58.8→77.1%、糖醋排骨 64.1→71.0%。**强增广已试并否**（README §5.7.4：test +0.03，白斩鸡 −3.0、木须肉 −4.1）；**类权重 / focal 也已试并否**（README §5.7.6：菜的 +1.8 全被兜底类 −9.2 吃掉）。两条路都指同一结论——瓶颈是类内差异和数据边界，不是损失函数或色温；只能补真实托盘数据。
2. **三套门控收口 + 换标签按钮**：已完成（README §5.7.5）——三套阈值/文案进 _SCOPE_SPECS 一张表，拒识后按目标标签自己的门控给「换到 X」；域概率当开关已试并否掉（拒绝原因 overlap）。**仍未做**：上传即自动切标签（无感路由）。
3. **真实托盘反馈闭环**：回流**脚本**已做（scripts/build_feedback_split.py 读 labels.csv 按 mode 分到四个标签空间、物化到 data/splits/feedback_built/，tests/test_feedback_split.py 11 个用例）；但 labels.csv 仍只有 2 条自测样本（source=sample，2026-11-18），所以现在还**没有数据可训**——攒够真机纠正再把 feedback_built 接到 train.py --extra-train。「一步纠正」只在 top-1 < 0.55 或拒识时出现；一顿饭的份量先验（米饭整碗）没建模。
4. **份量回归**：固定份量先验 ±15%，真实差异 ±50%。
5. **外部测试集**：16chinese 只有糖醋里脊 1 类（392 张 88.5%）；自筛网图 20 道菜已出 **623 张 top-1 95.67%**（README §5.7.8，results/tables/cn_web20.json + web20_manifest.csv 冻结）。这套只报不改；要扩就新开一套（--out data/external/web20_v2），两套都报。**下一步可做**：把网图上暴露的域头误拒（包子/白斩鸡/炒面/可乐鸡翅）在真实托盘数据上复核，别只按网图调阈值。
6. **小程序部署**：蒸馏 MobileNetV3 + int8（约 45 MB）；服务需要守护进程。
7. 宽头换 f101n 主干 **已试并否掉**（README §5.5.1），不要再试。
8. **演示视频**：分镜、口播稿、11 张镜头与中文名副本都已就绪（docs/演示视频分镜.md，README §6.4），**剩下就是录屏**——本机没有 ffmpeg，Edge 只能 headless 截图，脚本录不了。录前先跑 scripts/check_demo_shots.py。

## 4. 关键常量与规则（改前先读 app.py 顶部注释）

- 门控：六视图 TTA 报融合后验，决策用**前三视图 worst-view**（robust >= 阈值），镜像视图不参与。
- 主干绑定：DEFAULT_CKPT=f101n；FOODGATE_PATH / PACKAGING_PATH 默认 = CKPT.parent/{foodgate,packaging}.json。两个头在 f101n 的 512 维池化特征上拟合，**换主干必须同时换这两个文件**，否则静默失准（health 里两个 enabled 可验）。
- 食堂 robust 是**扣桶版**：每视图 p_dish/(1−p_other) 取 max，再取三视图 min（app.scope_of，rob20r）；22 类头下 p_other 取最后一维（other_cn）。
- 阈值：SCOPE 0.32 / FOODGATE 0.20 / PACKAGING 0.80 / CN_SCOPE 0.40（旧 20 道，兼容）/ **CN_SCOPE_THRESHOLD_21=0.4088**（22 类头沿用：val 5% 分位 0.4064，几乎相同）/ CN_OTHER 0.40 / **域头 0.2868**（22 类版；21 类版是 0.3524）。
- veto：13 类宽头原始后验，阈值 0.50、margin 0.02；刻意排除糖醋里脊、回锅肉。
- other_cn 桶**只做否决不加分**，不进 topk_with_calories。
- 域头：三视图特征先 mean 再标准化点积，p < 阈值即否决；头文件缺失时打印一行并退回「门控+桶」。
- 可信域头：p_domain ≥ 0.90 时门控阈值降到 0.36（FOODCLS_CN_TRUST=0.90:0.36，设 0 关闭）；只在扣桶 robust 之后生效，桶/清单外/veto 三道闸不受影响。
- 换标签：_TAB_ORDER（cn→single,wide；single/wide→cn；tray 不参与），只有目标标签自己 in_scope 才出按钮；食堂拒识原因 veto/weak 不给按钮，domain/other 才给（not_food/包装也不给）。
- 混合餐盘：只接受 20 类 == CLASS_NAMES 或 40 类且前 20 对齐的权重，形状不符直接 RuntimeError。
- env：FOODCLS_CKPT / MIXED_CKPT / WIDE_CKPT / CN_CKPT / CN_OTHER / CN_SCOPE_21 / DOMAIN_HEAD / DOMAIN_THRESHOLD / GATE / PACK / **ENSEMBLE**（默认 1，0/off/false/no 关集成）/ **ORIGINS**（CORS，默认 *）/ **CN_TRUST**（默认 0.90:0.36）。

## 5. 复现命令

- 启动：powershell -File scripts/start_demo.ps1 -Port 8000（或 PYTHONPATH=src python -m uvicorn foodcls.app:app --host 127.0.0.1 --port 8000）
- 健康：curl.exe -s http://127.0.0.1:8000/api/health（**别用 Invoke-WebRequest**，本机报非交互模式）
- 测试：python -m pytest tests -q
- 食堂评测：python scripts/eval_cn_other.py → eval_cn_other_extra.py → search_cn_other_rule.py（EVAL_CN_CKPT / EVAL_TAG 可换 ckpt）
- 域头链：extract_domain_features.py --views 1 --ckpt <头>/best.pt --data-root data/splits/cn241_22 --out results/tables/domain_feats_22.npz → fit_domain_head.py --feats results/tables/domain_feats_22.npz --tag 22 → eval_domain_gate.py（EVAL_CN_VAL / EVAL_CN_IN / EVAL_TAG 指 22 类）
- 外部集：python scripts/eval_external_cn.py（16chinese，仅糖醋里脊；依赖 results/tables/_fingerprint_cn241_21.json）
- 网图外部集评测：python scripts/eval_web20.py → results/tables/cn_web20.json（623 张，约 80 s）
- 收网图做外部集：python scripts/ingest_web.py scan --fresh → 改 review.csv → apply → status（data/external/web20/）
- 训食堂头：prepare_cn241_all.py → train_cn241.py --mode finetune（data-root data/splits/cn241_all）→ 再 --init <all>/best.pt 训 21 路；**现行 22 类**：prepare_cn241_rice.py → train_cn241.py --data-root data/splits/cn241_22 --classes results/tables/cn241_22_classes.json --mode finetune --init results/runs/cn241_all_pre/best.pt --save-dir results/runs/cn241_22_pre
- 打包：powershell -File scripts/package_demo.ps1 → dist/FoodDemo.zip（含集成五份权重，约 328 MB）
- 换标签评测：python scripts/eval_tab_route.py（写 results/tables/tab_route.json，约 50 s）
- 论文：python scripts/make_paper_numbers.py（刷新 paper/numbers.tex）→ python paper/_assemble.py（拼 main.tex）→ cd paper; pdflatex main && bibtex main && pdflatex main && pdflatex main
- 论文新图：python scripts/make_cn_figures.py（21 路混淆矩阵 cn21_confusion.png + 门控六集 cn_gate.png）
- 日志：python scripts/fix_log_encoding.py（把 PowerShell 写出的 UTF-16 日志转成 UTF-8）
- 演示分镜校验：python scripts/check_demo_shots.py（对着 results/tables/demo_shots.json 比 11 张 × 4 标签；漂移 exit 1，--update 覆盖）
- 演示上传副本：python scripts/prepare_demo_upload.py（data/external/demo_shots → data/external/demo_upload，中文文件名）

- 集成温度：scripts/fit_ensemble_temperature.py（读 results/_ens_<run>__<split>.npz，写 results/tables/ensemble_calibration.json；需要 val 缓存，先跑 results/_ens_collect.py）
- 诊断三件：eval_wide_real_detail.py（canteen_real 100 张逐类错误）/ eval_cn_degrade.py（食堂头 6 档退化）/ eval_mixed_real.py（8 张真实托盘）
- 搭配先验 A/B（未上线）：train_mixed.py --pairing-prior → eval_mixed_plate.py --pairing-prior → eval_pairing_prior.py（同一套盘子上交叉对比，写 results/tables/pairing_prior_cross.json）
- 反馈回流：python scripts/build_feedback_split.py（data/splits/feedback/labels.csv → data/splits/feedback_built/ + results/tables/feedback_split.json）

## 6. 环境坑（新对话必读）

- DSH 文件策略已是 danger-full-access、审批关闭——**不要请求沙箱升级**。
- 内层 shell 是 **PowerShell 5.1**：.ps1 必须纯 ASCII；`>` 和 Tee-Object 写 **UTF-16LE**，重定向请用 `| Out-File -Encoding utf8`；存量日志用 scripts/fix_log_encoding.py 转。
- python -c 传中文路径会被 GBK 打断——写 .py 文件再跑；控制台中文乱码不代表数据坏了，看 json 或用 read 工具。
- 训练必设 `$env:OMP_NUM_THREADS=1`、`$env:MKL_NUM_THREADS=1`、`$env:PYTHONPATH=<venv>\Lib\site-packages`，并 `--workers 4`：缺 PYTHONPATH 时 worker 会用 base 解释器的 CPU-only torch，管线**静默死锁**（父等 data、worker 0% CPU、无报错）；OMP 默认 24 线程空转把整机拖死。实测 124 张/s（workers 4）vs 51 张/s（workers 0）。GPU 8 GB，eval batch 4，别并行两个训练。
- 电池供电时 GPU 被压到 19 W（一个 epoch 25 分钟）并报 WinError 1455 / OpenBLAS 内存错——**先接电源**。
- README/HANDOFF/app.py 是 CRLF；多行 old_string 用 edit 会 not found，改用按行处理的 Python 脚本。
- 门控与主干绑定（见 §4），打包时不能只打主干不打两个 json。
- 服务被 kill 不自动恢复，需手动重启；机器睡眠会把 epoch 时间记成墙钟（第七轮 epoch 5 记了 36,412 s，实为睡了 9h44m）。

## 7. 文档地图

- paper/：sections/ 十个节文件（00_preamble … 08_conclusion）+ _assemble.py 拼装 + numbers.tex 数字宏 + references.bib；main.tex 是产物。
- README.md：§5.7.3 食堂 241 类预训练（现行）、§5.7.4 强增广 A/B（否）、§5.7.5 拒识后换标签、§5.7.6 类平衡/focal A/B（否 + 训练管线坑）、§5.7.7 网图筛选工具（ingest_web.py / fetch_web.py）、§5.7.8 20 道菜对外数字（623 张 95.7%）、§5.7.1/§5.7.2 第一版历史、§5.5.1 宽头换主干实验、§5.4.1 食物门控；§6.4 演示视频（3 分钟）。
- docs/HISTORY.md：历轮完整记录（第一~第十二轮、被否方案、每轮数字）。
- docs/演示视频分镜.md（分镜表 + 口播稿 + 每镜头确切数字 + 录制清单）、docs/复现路径.md、docs/微信小程序上手指南.md、docs/永久公网地址.md、NOTICE（数据许可）。

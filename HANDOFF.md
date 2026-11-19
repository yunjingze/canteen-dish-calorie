# FoodCLS 食品识别与热量估计 —— 项目交接文档（现状）

> 只放「现在是什么状态 / 怎么跑 / 哪里会踩坑」，目标一屏半读完。历轮实验的完整过程、被否掉的方案、每轮数字全在 [docs/HISTORY.md](docs/HISTORY.md)，考古去那里。最后更新：2026-11-19（第十轮：门控收口 + 拒识后换标签）。

## 0. 一句话

Windows 上的食品识别 + 热量估算网页应用：FastAPI + PyTorch resnet18 多套头 + 前端 4 个识别模式标签。工作目录 E:\项目作业\food_project；**venv 在 E:\项目作业\.venv，不在项目内**。

## 1. 当前状态

四个模式全部可用，权重与常数如下：

| 模式 | 权重 | 关键常数 | 权威成绩 |
|---|---|---|---|
| ① 20 类西餐 | results/runs/resnet18_last_stage_aug_f101n/best.pt | 门 0.32 + 食物门 0.20 + 包装否决 0.80 | test 单视图 91.66% / 六视图 93.4%；真实照 top-1 95.0%；123 张真负样本只放 2 张 |
| ② 混合餐盘 40 类 | results/runs/resnet18_multilabel_plate40/best.pt | sigmoid 0.35 | val plate-F1 54.54（西 53.17 / 食堂 55.89） |
| ③ 116 类宽头 | results/runs/resnet18_food101_cn116/best.pt | T=0.819，融合 0.20 | test top-1 80.58%，中餐 15 类 89.7% |
| ④ 食堂 21 路 | results/runs/cn241_21_pre/best.pt | T=0.778；rob20r 门 **0.4088**（域头 ≥ 0.90 → 0.36）；桶闸 0.40；域头 **0.3524** | test top-1 **79.83%** / top-5 97.64% / ECE 5.11% |

食堂完整决策链（五集，results/tables/domain_gate_pre.json）：清单内 **90.3%** / 清单外中餐 16.1% / 没见过的中餐 23.7% / 其它中餐桶 5.2% / 真实西餐照 13.0%；验证集 92.0%。第一版（cn241_21，门 0.3445、域头 0.2554）仍在盘上做对照，已被取代；旧五集数字在 domain_gate_pre_backup.json。外部集只有一道菜可测：16chinese 糖醋里脊 392 张 Top-1 88.5%（cn_external.json）。

拒识卡片现在给一个「换到 X」按钮（README §5.7.5，results/tables/tab_route.json）：食堂标签拒掉的真实西餐照 87 张里给 84 个按钮、79 个答对（94%）；清单内 300 张里误给 10 张（0 对，都是食堂标签自己认不出的难图）。单菜→食堂很干净（清单内 94–97%，外部集 0 误切）。域概率当开关已试并否掉——被拒食堂照与西餐照的域概率重叠，改看拒识原因（veto/weak 不给按钮）。

## 2. 服务与分发包

- 服务：http://127.0.0.1:8000，后台 job **pwsh-6e596d591fd947f8**，日志 results/_server.log（UTF-8）。health 应显示 checkpoint=resnet18_last_stage_aug_f101n / cn_threshold 0.4088（域头 ≥0.90 的图 payload 里报 0.36） / domain.threshold 0.3524 / cn_temperature 0.7783 / mixed_classes 40 / wide 116。
- 分发包：dist/FoodDemo.zip 168.9 MB / 278 项，含 4 份 best.pt（f101n、plate40、cn116、**cn241_21_pre**）+ 新版 domain_head.json + 4 份 calibration.json；**不含**旧 cn241_21、不含旧 resnet18_last_stage_aug。
- git：本轮「Loosen the canteen gate when the domain probe is sure」已推送 origin/main；上一轮 9a398ea。
- 测试：python -m pytest tests -q → **90 passed**（约 5 s）。

## 3. 待办（只列还没做的）

1. **弱类增广**：241 类预训练把白斩鸡 58.8→77.1%、糖醋排骨 64.1→71.0%；**强增广已试并否掉**（README §5.7.4：test +0.03，白斩鸡 −3.0、木须肉 −4.1）——瓶颈在类内差异，不在拍照色温。
2. **三套门控收口 + 换标签按钮**：已完成（README §5.7.5）——三套阈值/文案进 _SCOPE_SPECS 一张表，拒识后按目标标签自己的门控给「换到 X」；域概率当开关已试并否掉（拒绝原因 overlap）。**仍未做**：上传即自动切标签（无感路由）。
3. **真实托盘反馈闭环**：labels.csv 只有 2 条自测样本（source=sample，2026-11-18）；「一步纠正」只在 top-1 < 0.55 或拒识时出现；一顿饭的份量先验（米饭整碗）没建模。
4. **份量回归**：固定份量先验 ±15%，真实差异 ±50%。
5. **外部测试集**：16chinese 与 20 道只有糖醋里脊重合，已出 392 张 88.5%（cn_external.json）；要真正的对外数字还需新增拍摄或 ChineseFoodNet 208 类（Kaggle yihfeng/chinesefoodnet，18.6 万张，需登录）。
6. **小程序部署**：蒸馏 MobileNetV3 + int8（约 45 MB）；服务需要守护进程。
7. 宽头换 f101n 主干 **已试并否掉**（README §5.5.1），不要再试。

## 4. 关键常量与规则（改前先读 app.py 顶部注释）

- 门控：六视图 TTA 报融合后验，决策用**前三视图 worst-view**（robust >= 阈值），镜像视图不参与。
- 主干绑定：DEFAULT_CKPT=f101n；FOODGATE_PATH / PACKAGING_PATH 默认 = CKPT.parent/{foodgate,packaging}.json。两个头在 f101n 的 512 维池化特征上拟合，**换主干必须同时换这两个文件**，否则静默失准（health 里两个 enabled 可验）。
- 21 路 robust 是**扣桶版**：每视图 p_dish/(1−p_other) 取 max，再取三视图 min（app.scope_of，rob20r）。
- 阈值：SCOPE 0.32 / FOODGATE 0.20 / PACKAGING 0.80 / CN_SCOPE 0.40（旧 20 道，兼容）/ **CN_SCOPE_THRESHOLD_21=0.4088** / CN_OTHER 0.40 / 域头 0.3524。
- veto：13 类宽头原始后验，阈值 0.50、margin 0.02；刻意排除糖醋里脊、回锅肉。
- other_cn 桶**只做否决不加分**，不进 topk_with_calories。
- 域头：三视图特征先 mean 再标准化点积，p < 阈值即否决；头文件缺失时打印一行并退回「门控+桶」。
- 可信域头：p_domain ≥ 0.90 时门控阈值降到 0.36（FOODCLS_CN_TRUST=0.90:0.36，设 0 关闭）；只在扣桶 robust 之后生效，桶/清单外/veto 三道闸不受影响。
- 换标签：_TAB_ORDER（cn→single,wide；single/wide→cn；tray 不参与），只有目标标签自己 in_scope 才出按钮；食堂拒识原因 veto/weak 不给按钮，domain/other 才给（not_food/包装也不给）。
- 混合餐盘：只接受 20 类 == CLASS_NAMES 或 40 类且前 20 对齐的权重，形状不符直接 RuntimeError。
- env：FOODCLS_CKPT / MIXED_CKPT / WIDE_CKPT / CN_CKPT / CN_OTHER / CN_SCOPE_21 / DOMAIN_HEAD / DOMAIN_THRESHOLD / GATE / PACK。

## 5. 复现命令

- 启动：powershell -File scripts/start_demo.ps1 -Port 8000（或 PYTHONPATH=src python -m uvicorn foodcls.app:app --host 127.0.0.1 --port 8000）
- 健康：curl.exe -s http://127.0.0.1:8000/api/health（**别用 Invoke-WebRequest**，本机报非交互模式）
- 测试：python -m pytest tests -q
- 食堂评测：python scripts/eval_cn_other.py → eval_cn_other_extra.py → search_cn_other_rule.py（EVAL_CN_CKPT / EVAL_TAG 可换 ckpt）
- 域头链：extract_domain_features.py --views 1 → fit_domain_head.py --mode mean --tag pre → eval_domain_gate.py（TAG=pre）
- 外部集：python scripts/eval_external_cn.py（16chinese，仅糖醋里脊；依赖 results/tables/_fingerprint_cn241_21.json）
- 训食堂头：prepare_cn241_all.py → train_cn241.py --mode finetune（data-root data/splits/cn241_all）→ 再 --init <all>/best.pt 训 21 路
- 打包：powershell -File scripts/package_demo.ps1 → dist/FoodDemo.zip
- 换标签评测：python scripts/eval_tab_route.py（写 results/tables/tab_route.json，约 50 s）
- 日志：python scripts/fix_log_encoding.py（把 PowerShell 写出的 UTF-16 日志转成 UTF-8）

## 6. 环境坑（新对话必读）

- DSH 文件策略已是 danger-full-access、审批关闭——**不要请求沙箱升级**。
- 内层 shell 是 **PowerShell 5.1**：.ps1 必须纯 ASCII；`>` 和 Tee-Object 写 **UTF-16LE**，重定向请用 `| Out-File -Encoding utf8`；存量日志用 scripts/fix_log_encoding.py 转。
- python -c 传中文路径会被 GBK 打断——写 .py 文件再跑；控制台中文乱码不代表数据坏了，看 json 或用 read 工具。
- 训练 **--workers 0**（分类）；混合餐盘 --workers 4 反而快（约 160 s/epoch）。GPU 8 GB，eval batch 4，别并行两个训练。
- 电池供电时 GPU 被压到 19 W（一个 epoch 25 分钟）并报 WinError 1455 / OpenBLAS 内存错——**先接电源**。
- README/HANDOFF/app.py 是 CRLF；多行 old_string 用 edit 会 not found，改用按行处理的 Python 脚本。
- 门控与主干绑定（见 §4），打包时不能只打主干不打两个 json。
- 服务被 kill 不自动恢复，需手动重启；机器睡眠会把 epoch 时间记成墙钟（第七轮 epoch 5 记了 36,412 s，实为睡了 9h44m）。

## 7. 文档地图

- README.md：§5.7.3 食堂 241 类预训练（现行）、§5.7.4 强增广 A/B（否）、§5.7.5 拒识后换标签、§5.7.1/§5.7.2 第一版历史、§5.5.1 宽头换主干实验、§5.4.1 食物门控。
- docs/HISTORY.md：历轮完整记录（第一~第十轮、被否方案、每轮数字）。
- docs/复现路径.md、docs/微信小程序上手指南.md、docs/永久公网地址.md、NOTICE（数据许可）。

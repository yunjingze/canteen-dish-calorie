# CNFOOD-241 二十道食堂菜原型：结果报告

日期 2026-11-15。数据源 CNFOOD-241 验证包（val600x600.zip，1.15 GB），
只取其中 20 个“食堂常见”类，其余 221 类留在压缩包里未解压。

## 一、数据

- 抽取 2462 张，按 7:1:2 切分：train 1727 / val 244 / test 491，每类约 92 张。
- 类目：酸辣土豆丝、地三鲜、手撕包菜、炒青菜、干煸豆角、西红柿炒鸡蛋、鸡蛋羹、糖醋排骨、可乐鸡翅、白斩鸡、黄焖鸡、辣子鸡、红烧肉、水煮肉片、糖醋里脊、农家小炒肉、青椒肉丝、木须肉、包子、炒面。

## 二、准确率

| 方案 | 训练成本 | val top-1 | test top-1 | test top-5 |
| --- | --- | --- | --- | --- |
| 冻结 E7 特征 + 逻辑回归（probe） | 约 1 分钟，不反传 | — | 66.2% | 93.7% |
| last-stage 微调 8 epoch | 约 4 分钟（8 GB 显卡） | 79.1% | 76.0% | 95.3% |

每类 test 准确率（约 26 张/类）：

| 菜 | slug | 微调 | probe |
| --- | --- | --- | --- |
| 酸辣土豆丝 | hot_sour_potato | 92% | 81% |
| 包子 | baozi | 92% | 96% |
| 干煸豆角 | dry_fried_beans | 91% | 87% |
| 鸡蛋羹 | steamed_egg | 88% | 88% |
| 糖醋里脊 | sweet_sour_pork | 88% | 88% |
| 地三鲜 | di_san_xian | 86% | 69% |
| 西红柿炒鸡蛋 | tomato_egg | 85% | 81% |
| 木须肉 | mu_xi_pork | 84% | 47% |
| 炒青菜 | stir_fried_greens | 81% | 67% |
| 手撕包菜 | hand_torn_cabbage | 81% | 69% |
| 水煮肉片 | boiled_pork | 81% | 81% |
| 可乐鸡翅 | cola_wings | 77% | 77% |
| 辣子鸡 | spicy_chicken | 77% | 46% |
| 白斩鸡 | boiled_chicken | 71% | 14% |
| 黄焖鸡 | braised_chicken | 69% | 23% |
| 红烧肉 | braised_pork | 62% | 73% |
| 炒面 | fried_noodles | 62% | 69% |
| 青椒肉丝 | pepper_pork | 54% | 50% |
| 糖醋排骨 | sweet_sour_ribs | 50% | 27% |
| 农家小炒肉 | farmhouse_pork | 50% | 50% |

## 三、现状模型“答错还很自信”

在同一批 491 张真实食堂菜上跑线上两个权重（三视图 TTA，与网页一致）：

- **20 类 canteen 模型**：平均 top-1 置信度 0.45，35% 的图给出 >0.5 的答案，10% >0.8。最常给出的答案：fried_ricex81、chicken_wingsx71、ramenx63、greek_saladx56。
- **116 类宽模型**：平均 top-1 置信度 0.41，31% 的图给出 >0.5 的答案，11% >0.8。最常给出的答案：twice_cooked_porkx73、sweet_and_sour_porkx48、pad_thaix28、chicken_wingsx23。

116 类模型里这 20 道只有“糖醋里脊”同名，其余 19 道必然答错，错法集中在回锅肉/糖醋里脊/pad thai 三五个近邻上。

## 四、它知道自己不认识吗

20 类新模型在清单内 491 张上平均置信度 0.72，清单外 600 张（116 类里其它菜）0.46；阈值 0.5 时保留 77% 的清单内图片、挡掉 64% 的清单外图片。

## 五、结论与下一步

- 路是通的：4 分钟训练就到 76% top-1 / 95% top-5，好过现在把番茄炒蛋判成回锅肉。
- 瓶颈是数据量不是模型：每类只有 92 张，最弱的糖醋排骨、农家小炒肉正是最易混的相邻品类。全量 train 包（9.3 GB，每类约 700 张）预计再加 6–8 点。
- 白斩鸡在验证包里只有 34 张，这项指标不可信。

接入网页还需三件事：温度标定、中文名＋热量先验表、模式开关（第 4 模式，不动现有三个）。

## 产物

- 权重 results/runs/cn241_last_stage/best.pt（resnet18，20 类，带 class_table）
- 数据 data/splits/cn241/{train,val,test}/<slug>/，类表 results/tables/cn241_classes.json
- 指标 results/tables/cn241_probe.json / cn241_finetune.json / cn241_gap.json / cn241_scope.json
- 脚本 scripts/prepare_cn241.py、scripts/train_cn241.py、scripts/eval_gap_cn241.py、scripts/eval_cn241_scope.py

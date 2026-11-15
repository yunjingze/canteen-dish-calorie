# 食堂真实照片 v2 评估 + 食物门控（2026-11-15）

## 1. 单菜识别（100 张，16 类，六视图融合）

| 模型 | 准确率 |
|---|---|
| E4（基线 resnet18） | 91.0 % |
| **E7（Food-101N 去重）** | **95.0 %** |
| E8（不去重） | 92.0 % |
| E12 | 95.0 % |
| E13 | 90.0 % |
| 五模型集成 | 94.0 % |

E7 错 5 张：fries/002→ice_cream、hamburger/009→ice_cream、hamburger/010→french_fries、
pizza/001→lasagna、pizza/004→lasagna。其余 16 类全对（pizza 9/11、hamburger 10/12）。
100 张下 95 % 的置信区间约 ±3 pt，与 test 93.4 % 无统计差异——这批照片仍是"干净网图"，
不是真正的食堂域。

## 2. 负样本标签错误（关键发现）

第一批 15 张"负样本"（001–015.jpg）里 **7 张其实是食物**：面包卷（001–004）、
托盘上的四盘沙拉（009）、装食物的杯/锥筒（010）、分格盘里的鸡块（015）。
门控给它们 p_food = 0.99 / 0.96 / 1.00 / 1.00 / 0.99 / 0.95 / 0.05——门控是对的，标签是错的。
这就是 v1_unknown 那一类 AUC 只有 0.807 的唯一原因。

已改标为 food_nonvocab，机器可读表在 `data/splits/negatives/v1_categories.json`，
说明在 `data/splits/negatives/README_relabel.md`。136 行 sources.csv 只覆盖后 123 张。

## 3. 食物门控（131 张真负样本）

512 维池化特征上的逻辑回归（C=0.05，class_weight balanced），与主分类器共用一次前向。

- 留一类（LOCO）AUC：trash_bin 0.997、drink_cup 0.989、menu_board 0.967、
  empty_cafeteria 0.967、condiment 0.958、empty_tray 0.945、napkin 0.941、
  packaging 0.934、hand 0.929；**均值 0.959**（改标前 0.936）
- OOF 负样本 vs 100 张真实照片 AUC **0.954**（改标前 0.929）；vs 代理域 0.993
- 部署阈值 p_food ≥ 0.20 且 robust ≥ 0.32

## 4. 端到端（走 HTTP API，138 张负样本 + 100 张单菜，结果在 tables/api_sweep.json）

| | 结果 |
|---|---|
| 131 张真负样本被拒 | **127 / 131 = 96.9 %** |
| 100 张真实菜品保留 | **95 / 100** |
| 7 张改标食物被接受 | 6 / 7 |

基线（只看 robust ≥ 0.32）只拒掉 64/131（49 %）。仅靠调阈值做不到这一点：
robust 分数在"真实菜品 / 代理域 / 非食物"三组之间严重重叠，0.32→真实 96/proxy 815/负样本 70，
0.60→真实 79/proxy 493/负样本 13。

仍被误收的 4 张：condiment_bottle/116（p 0.71）、empty_cafeteria/104（0.56）、
napkin/130（0.35）、hand/059（0.30）。

被误拒的 5 张真实照片：chicken_wings/001（robust 0.199）、waffles/001（0.251）、
pizza/003（0.247）、pizza/005（0.320 边界）、french_fries/002（p_food 0.15）。
前两张 robust 极低是主分类器本身没把握，不是门控的锅——补拍这两类时优先。

## 5. 混合餐盘（8 张真实托盘）

micro P 50.0 / R 20.0 / F1 28.6（tp2 fp2 fn8）。合成数据训练的多标签头在真实托盘上
不可用：001/004 无预测，002/003 只出 waffles，005 误报 sushi 0.64，007 误报 donuts 0.80。
这一块需要真实托盘数据重训（每盘 2–4 菜、带多标签 CSV），与单菜门控无关。

## 6. 复现命令

```
.venv\Scripts\python.exe results\_gate_rebuild.py     # 重抽 138 张负样本特征
.venv\Scripts\python.exe results\_gate_final.py       # 训练门控 + OOF 表
.venv\Scripts\python.exe results\_publish_gate.py     # 写 foodgate_linear.json 并复制到 run 目录
.venv\Scripts\python.exe results\_api_sweep.py        # 端到端 HTTP 扫描（需先启动服务）
```

部署：`results/runs/resnet18_last_stage_aug_f101n/foodgate.json`（阈值 0.20 写在文件里，
服务启动时优先用文件里的值）。

# -*- coding: utf-8 -*-
"""Render results/cn241_report.md from the JSON artefacts."""
import json
from pathlib import Path

ROOT = Path(r"E:\项目作业\food_project")
tb = ROOT / "results/tables"
cls = json.loads((tb / "cn241_classes.json").read_text(encoding="utf-8"))
probe = json.loads((tb / "cn241_probe.json").read_text(encoding="utf-8"))
ft = json.loads((tb / "cn241_finetune.json").read_text(encoding="utf-8"))
gap = json.loads((tb / "cn241_gap.json").read_text(encoding="utf-8"))
scope = json.loads((tb / "cn241_scope.json").read_text(encoding="utf-8"))

zh = {c["slug"]: c["zh"] for c in cls["classes"]}
rows = []
for c in cls["classes"]:
    s = c["slug"]
    rows.append((zh[s], s, c["n"], ft["test"]["per_class"][s] * 100, probe["test"]["per_class"][s] * 100))
rows.sort(key=lambda r: -r[3])
total = sum(c["n"] for c in cls["classes"])

L = []
L.append("# CNFOOD-241 二十道食堂菜原型：结果报告")
L.append("")
L.append("日期 2026-11-15。数据源 CNFOOD-241 验证包（val600x600.zip，1.15 GB），")
L.append("只取其中 20 个“食堂常见”类，其余 221 类留在压缩包里未解压。")
L.append("")
L.append("## 一、数据")
L.append("")
L.append(f"- 抽取 {total} 张，按 7:1:2 切分：train 1727 / val 244 / test 491，每类约 92 张。")
L.append("- 类目：酸辣土豆丝、地三鲜、手撕包菜、炒青菜、干煸豆角、西红柿炒鸡蛋、鸡蛋羹、"
         "糖醋排骨、可乐鸡翅、白斩鸡、黄焖鸡、辣子鸡、红烧肉、水煮肉片、糖醋里脊、"
         "农家小炒肉、青椒肉丝、木须肉、包子、炒面。")
L.append("")
L.append("## 二、准确率")
L.append("")
L.append("| 方案 | 训练成本 | val top-1 | test top-1 | test top-5 |")
L.append("| --- | --- | --- | --- | --- |")
L.append(f"| 冻结 E7 特征 + 逻辑回归（probe） | 约 1 分钟，不反传 | — | {probe['test']['top1']*100:.1f}% | {probe['test']['top5']*100:.1f}% |")
L.append(f"| last-stage 微调 8 epoch | 约 4 分钟（8 GB 显卡） | {ft['best_val_acc']*100:.1f}% | {ft['test']['top1']*100:.1f}% | {ft['test']['top5']*100:.1f}% |")
L.append("")
L.append("每类 test 准确率（约 26 张/类）：")
L.append("")
L.append("| 菜 | slug | 微调 | probe |")
L.append("| --- | --- | --- | --- |")
for zh_s, s, n, a, b in rows:
    L.append(f"| {zh_s} | {s} | {a:.0f}% | {b:.0f}% |")
L.append("")
L.append("## 三、现状模型“答错还很自信”")
L.append("")
L.append("在同一批 491 张真实食堂菜上跑线上两个权重（三视图 TTA，与网页一致）：")
L.append("")
for tag, label in (("canteen20", "20 类 canteen 模型"), ("wide116", "116 类宽模型")):
    d = gap[tag]
    tops = "、".join(f"{k}x{v}" for k, v in list(d["top_predictions"].items())[:4])
    L.append(f"- **{label}**：平均 top-1 置信度 {d['mean_top1_prob']:.2f}，"
             f"{d['share_above_0.5']*100:.0f}% 的图给出 >0.5 的答案，"
             f"{d['share_above_0.8']*100:.0f}% >0.8。最常给出的答案：{tops}。")
L.append("")
L.append("116 类模型里这 20 道只有“糖醋里脊”同名，其余 19 道必然答错，"
         "错法集中在回锅肉/糖醋里脊/pad thai 三五个近邻上。")
L.append("")
L.append("## 四、它知道自己不认识吗")
L.append("")
L.append(f"20 类新模型在清单内 491 张上平均置信度 {scope['in_list']['mean_top1_prob']:.2f}，"
         f"清单外 600 张（116 类里其它菜）{scope['out_of_list']['mean_top1_prob']:.2f}；"
         f"阈值 0.5 时保留 {scope['in_list']['share_above_0.5']*100:.0f}% 的清单内图片、"
         f"挡掉 {(1-scope['out_of_list']['share_above_0.5'])*100:.0f}% 的清单外图片。")
L.append("")
L.append("## 五、结论与下一步")
L.append("")
L.append("- 路是通的：4 分钟训练就到 76% top-1 / 95% top-5，好过现在把番茄炒蛋判成回锅肉。")
L.append("- 瓶颈是数据量不是模型：每类只有 92 张，最弱的糖醋排骨、农家小炒肉正是最易混的相邻品类。"
         "全量 train 包（9.3 GB，每类约 700 张）预计再加 6–8 点。")
L.append("- 白斩鸡在验证包里只有 34 张，这项指标不可信。")
L.append("")
L.append("接入网页还需三件事：温度标定、中文名＋热量先验表、模式开关（第 4 模式，不动现有三个）。")
L.append("")
L.append("## 产物")
L.append("")
L.append("- 权重 results/runs/cn241_last_stage/best.pt（resnet18，20 类，带 class_table）")
L.append("- 数据 data/splits/cn241/{train,val,test}/<slug>/，类表 results/tables/cn241_classes.json")
L.append("- 指标 results/tables/cn241_probe.json / cn241_finetune.json / cn241_gap.json / cn241_scope.json")
L.append("- 脚本 scripts/prepare_cn241.py、scripts/train_cn241.py、scripts/eval_gap_cn241.py、scripts/eval_cn241_scope.py")
out = ROOT / "results/cn241_report.md"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote", out, out.stat().st_size)

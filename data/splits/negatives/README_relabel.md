# negatives/ 标签更正（2026-11-15）

第一批 15 张"负样本"（v1 网络检索的那批，文件名 001–015.jpg）里有 **7 张其实是食物照片**，
被错标成了非食物。这是 v1_unknown 那一类在门控上只有 0.807 AUC 的唯一原因——
门控判断是对的，标签是错的。

| 文件 | 内容 | 新标签 |
|---|---|---|
| 001.jpg | 一盘面包卷 | food_nonvocab |
| 002.jpg | 烤架上的面包卷 | food_nonvocab |
| 003.jpg | 烤盘上的面包卷 | food_nonvocab |
| 004.jpg | 手掰面包 | food_nonvocab |
| 009.jpg | 托盘上四盘沙拉 | food_nonvocab |
| 010.jpg | 装食物的杯/锥筒 | food_nonvocab |
| 015.jpg | 分格盘里的炸鸡块 | food_nonvocab |
| 005.jpg | 一次性杯具套装 | packaging |
| 006.jpg | 空分格盘 | empty_tray |
| 007.jpg | 手端空托盘 | hand |
| 008.jpg | 咖啡杯 | drink_cup |
| 011.jpg | 空分格盘 | empty_tray |
| 012.jpg | 菜单牌 | menu_board |
| 013.jpg | 菜单灯箱 | menu_board |
| 014.jpg | 点餐灯箱 | menu_board |

机器可读版本在同目录 `v1_categories.json`；`sources.csv` 只覆盖后 123 张
（有 category 列），这 15 张以 `v1_categories.json` 为准。

重新训练后的门控（131 张真负样本 vs Food-101 + 代理域食物）：
留一类 AUC 均值 0.959（最低 hand 0.929），OOF 负样本 vs 100 张真实照片 AUC 0.954。
这 7 张被改标的照片 p_food = 0.99/0.96/1.00/1.00/0.99/0.95/0.05——除 015 的
"盘子里只有几块鸡块"外全部被判为食物，与人工判断一致。

以后新增负样本时请**只收真正没有菜的画面**：面包、沙拉、装食物的杯子都算食物
（哪怕不在 20 类里）。词表外食物属于 "看不清/不在词表"，不属于 "不是食物"。

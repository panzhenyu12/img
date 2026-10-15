# 假设 C：从 V36 移动 alpha 再扣一次耳饰

2026-10-15，隔离实验，无生产修改。固定原始用户图、graph、4× combined-point 官方耳饰 alpha（两个组件均 index2）、V36 inverse map、背景与原始 changed/frozen 写入门禁。

仅替换源物理 alpha：`A_move = A_v36 * (1 - A_accessory)`，耳饰仍作为源坐标固定层保留在背景。先用保存的 samples/backing 精确重放 V36，再用成对采样差值更新原样本，避免 Python/Rust 浮点累加差异混入结果。不扩大写域、不换背景、不添加神经修复。

结果否定该假设：原 V36 physical alpha 在 alpha>=.95 的耳饰核心里已经全为 0。源 alpha 最大扣除 .000245852，变化>.001 的像素为0，总扣除仅 .001412 pixel-alpha。目标 alpha 最大变化为 +100 的 .000227853 / -100 的 .000223210；两方向最终 RGBA 与 V36 完全相同，变化像素0。金属、亮部、耳垂及原写域外变化均0。

因此旧肤色条不是因为移动前景仍含大量耳饰 alpha；继续切除耳饰 alpha 无效。下一步应核对原分割把真实皮肤误排除或冻结的问题。

`outputs/report.json` 保存 source/graph/raw-alpha/inverse/background/sample/output hashes；每个方向保存源/目标 alpha、差值放大诊断和1:1原图对照。`accepted=false`，没有8分声明。

# face anatomical v29\n\n目标 contour signed blend：joint flow 下 face sample 仍按逆流，neck donor 在 source 坐标保持静止；以目标 contour alpha 混合 face/neck，避免旧脸 RGB 穿入新颈部。\n
## 结果（2026-10-15）

+100 原尺寸回放出现粗黑外轮廓和两侧黑三角，目标 alpha 直接切断 face sample 后暴露 scene；视觉失败，未集成。

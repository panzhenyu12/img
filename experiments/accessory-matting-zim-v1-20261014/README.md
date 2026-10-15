# ZIM 饰品物理边缘对照

仅原图的两个饰品，各四个官方 raw alpha；未修改正式 detector、graph、Rust、WASM 或服务。沿用 `experiments/zim-20261014` 固定官方 ICCV 2025 源码和校验过的权重，不重新下载。

```sh
services/background-worker/.venv/bin/python experiments/accessory-matting-zim-v1-20261014/run.py
```

输出目录必须不存在，防止覆盖证据。输入为用户1024原图与既有原图解析；提示由 accessory≥.5 的连通区域自动生成：只取下巴高度以上最大的两个连通件，框向外4px，正点选内部距离最大处。没有编辑图重新检测、手画alpha或人工挑选最佳候选。保留四个候选，官方score argmax两侧均为2。

实测模型初始化1.82s、原图编码7.20s、双饰品解码0.73/0.62s，CPU3线程；不是浏览器或交互滑块耗时。浮点全部输出、模型/源图/graph/脚本hash在 `test/results/accessory-matting-zim-v1-20261014`。

原尺寸及2倍nearest查看可见：旧parser把竖直皮肤带及饰品孔隙中的背景一起划入饰品；ZIM把大部分这类误选排除，保留金属形状。这是实际边缘分解的局部收益，无alpha真值、不代表泛化质量或饰品保护合同已通过。

不能直接用新的二值mask替换旧冻结域就宣称修复：原 face/skin 在误标皮肤条处也可能为零，旧像素仍会作为背景留下；软饰品边缘原本混有旧背景，必须验证前景颜色分离、遮挡顺序、未动核心、供体排除及最终合成。

后续 `full-replay/` 已执行了原图 `faceWidth` 的 `0/-100/-30/+30/+100` 五档回放，分别保存正式 v19、物理支持候选和同候选的旧 attachment 对照。它是诊断性 warp，仍未接入产品。组合结果确实去掉了大部分耳环旁竖直肤色条，但 +100 仍把下颌压成偏 V，并把上颈夹窄，因此不能作为验收候选。另有 `full-replay/low-person-audit/` 对候选中 337 个（+100）和 340 个（+30）原 person confidence<.01 的变化做了原尺寸连通域复核；最大连通域位于左耳外缘，说明 ZIM face support 仍有泄进耳朵保护边界的风险，不能把低 person confidence 直接当作背景。

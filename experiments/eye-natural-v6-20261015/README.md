# 眼距/倾斜自然度 V6（独立研究，未部署）

本目录保留多项因果对照及其失败，不代表全部可用。生产 `core/image-core`、WASM、Go API 均未改。

当前可继续评审的是 [源图解剖间距约束版](feasible-review.md)：联合双眼 SVF + 膜应变 + 明确记录的可行幅度，33 项 geometry 通过，原图 0/±30/±100 与匹配实际位移的 V4b 对照齐全。**仍未判 8 分，不接入生产。**

失败记录：

- `first-rejected`：双眼联合域与软鼻锚，强档鼻根细白线、眉组合串扰。
- `second-source` / `second-qa`：收紧 rigid padding、增强眉锚，仍存在细线。
- `third-source`：只修正原静态 lid envelope 的正反轨迹保护；32 项通过，纯眼输出四档逐字节不变，但图像审美仍失败。最小补丁与生产 V4b 不可直接移植的原因见 `lid-trajectory-only.md`。
- `endpoint-source` / `endpoint-qa`：裸端点映射默认回归折叠，拒绝。
- `membrane-endpoint-source` / `membrane-qa`：膜端点原 32 项通过，独立视觉仍失败；新增眼保护边界回归暴露 12 个折叠。见 `endpoint-review.md`。

`binaries` 保存本机带 SHA 的不可变 worker（已忽略提交）；各 worker JSON 保留身份，原尺寸图片、图参数、测试源码、日志及哈希全部保留。运行方法见 `run_photo.py`，公开 apply_graph_json 执行，不使用脚本重绘或修补结果。

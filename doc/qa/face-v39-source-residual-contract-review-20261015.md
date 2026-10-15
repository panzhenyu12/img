# V39 不透明 joint source residual 组件复核（2026-10-15）

**最终结论：拒绝合入/部署 V39。** 它通过独立公开接口回归与参数合同检查，并修复“几乎零形变也改变原照片颜色”的不连续问题；但独立实图检查确认 user/hair 的旧脸缘肤色/橙色残线明显倒退。数学合同成立不能抵消真实照片退化，也不能作为 ≥8 分证据。

## 实图否决项

独立读取 `outputs/{user-face-contour,portrait-hair,portrait-woman}/{100,-100}/face-1x.png` 全部六张，面板为 source / current v19+fringe-v2 / V39。与 root 结论一致：

- **user +100**：两边旧脸缘细肤色线留在暗背景里，原位置残条比 current 更显眼；轮廓缩窄仍在，但残差没有随前景清除。
- **user −100**：新脸皮内部沿旧轮廓出现肤色/亮色竖线，左右下颊均可辨，图像左侧更明显。current 的对应连接更干净。
- **hair +100**：图像右侧青背景中保留旧脸缘细线，左下颊原轮廓也有肤色残边；无可靠外观改善。
- **hair −100**：图像右侧下颊/颈缘新增明显橙色折线，是最清楚的回退反例。
- **woman ±100**：脸部与 current 接近，衣领边局部差异不构成综合改善，不能挽回另两张的明确退化。

这与已知风险吻合：原始 RGB 中没有被估计 F/alpha 解释的脸缘被当作 stationary residual 留在源位置。独立 review 不将此解释视为唯一完整成因，但直接可见的残线已足够拒绝该候选。后续保留红/绿回归作为连续性要求，不能把 V39 公式直接当成可上线修复。

## 源码范围及边界

比较当前 `core/image-core/src/` 与 `experiments/face-source-residual-v39-20261015/src/`，仅 `face.rs` 两处改变，共 12 行新增、1 行替换：限定 `joint_neck` 且整张工作 crop 的 RGBA alpha 都为 255，再以原 source RGB 为基底只加入被运输前景的差值；其余情况使用原有 premultiplied 合成。

- **透明 RGBA**：crop 内任意非 255 alpha 都使新分支关闭；foreground estimator 的透明回退也未变。下述 12 个公开参数组合与旧 core 输出逐字节一致。这证明没有新增回退行为变化，不代表旧透明路径已达到照片级质量。
- **opacity**：仍在最后 `mask::composite` 一次应用；没有把 opacity 写入 residual 计算再二次混合。0 逐字节原图，opaque 的 50% 与 100% 结果的公开中点合同通过。
- **normal + soft protect**：保护值限制现有 velocity，最终 normal 复合去掉 protectMask 是已有约定；128 保护下仍有真实轮廓收缩，幅度介于 0 与 full。不能要求它等于原图/全效果的 RGB 中点，因为保护首先改变几何。
- **非 normal blend**：既有 selection/protection 权重继续保留在最终颜色混合中；未动/硬冻结像素恢复原图的代码也未变。multiply 的 opacity 和 soft/full protect 通过。颜色混合本身可能改变纹理亮度，V37 的前景 coverage 上界不直接约束此后单独请求的 multiply 操作。
- **局限**：整张 crop 的不透明检查是保守分支；一个无关透明 texel 也会关闭新修复。source residual 同样可能保留未被估计 alpha/F 解释的旧轮廓残差，必须查看实图，不能用本合同测试代替。

## 一条真正红→绿的公开操作回归

新增 `experiments/face-source-residual-v39-20261015/tests/source_residual_contract.rs`，只调用公开 `apply_graph_json` 的 `face.adjust`。输入为独立构造的 160×320 图：76 px 宽柔边椭圆脸、连通颈部、条纹背景；源 RGB 与观测 matte 直接构造，不复用私有估计器或 residual 公式。

用户把脸宽从 0 调至 ±0.001 时，几何位移远小于 .01 px，不能出现多字节颜色跳变。期望最大通道变化 ≤1 byte；同时 +100 必须令可见轮廓两侧各内移至少 3 px，防止 no-op 假通过。

1. 先保存 `old-core-audit/` 的当前生产源码快照及 SHA 清单，原样编译该测试。**失败原因正确：−0.001 时最大跳色 7 byte。** 日志 `old-core-audit/red.log`。
2. 同一测试编译 V39，**±0.001 与实际轮廓内收条件全部通过**。日志 `source-residual-green.log`。

```sh
CARGO_TARGET_DIR=.run/face-source-residual-v39-independent-target cargo test --release --manifest-path experiments/face-source-residual-v39-20261015/old-core-audit/Cargo.toml --test source_residual_contract -- --nocapture
CARGO_TARGET_DIR=.run/face-source-residual-v39-target cargo test --release --manifest-path experiments/face-source-residual-v39-20261015/Cargo.toml --test source_residual_contract -- --nocapture
```

这是针对不连续的行为预期，不是生产公式复算，也没有用“改变像素多”当外观质量分数。

## opacity / protect / RGBA 公开 CLI 对照

`probe_public_options.py` 使用公开 JSON graph 和 native image-render，在 opaque / 透明 RGBA 两种输入下，比较旧 core 与 V39：normal/multiply × opacity 0/50/100，以及 normal/multiply × protect 128/254/255。**48 次输出检查全部通过，12 种透明参数组合逐字节等同旧 core。**

实际结果保存在 `public-options-v2/report.json` 及 PNG 中，旧版 40 次检查的 `public-options/` 保留。初轮 protect254 全部零变化，故后续增加 protect128 检查，确认不是只测了冻结状态。opaque multiply + protect128 的最大通道差为 85，低于该公开保护权重允许的 128；protect255 全部原图。opacity50 是 opacity100 与源图的中点，允许 1 byte 舍入差。

运行：

```sh
services/background-worker/.venv/bin/python experiments/face-source-residual-v39-20261015/probe_public_options.py
```

脚本拒绝覆盖既有结果。保存的二进制 SHA：旧 `d0fe0b5251f2f4e862ee20dc7381497fde0f198369f7f37b01eb012c8b68a434`；V39 `30934e5002b0cd74a43c09b150bf6a83f8760487946cca4fd1500bb7320f5ba1`。本复核未修改生产代码或部署服务。

# Lip lighting V2 / frozen candidate C

本实验只处理 `feature.adjust.adjustments.lipBrightness`。生产补丁已由 root 接入，root 正在完成生产 native/WASM/API 验证；实验报告本身不证明已部署，也不代表所有妆容功能达到 8 分。

## 本次可使用的产物

- `baseline-isolated-render`：独立原生基线，SHA-256 `d0fe0b5251f2f4e862ee20dc7381497fde0f198369f7f37b01eb012c8b68a434`。
- `candidate-c-render`：冻结 C，SHA-256 `adc18a9e67414e1aac6e554afe9bbb14813229ed9a37f5bd41670b2914cd364a`。
- `production.patch`：只改唇亮度块，新增 public `lip_lighting_contract.rs`。SHA-256 `a43e8fc137811d00b84e9a9f8c054465bccce04415bd1471dd8dd6c2c8438c7c`。不要用整份实验 color.rs 覆盖并行生产修改。
- `evidence/frozen-c-manifest.json`：源文件、构建锁文件、脚本、测试、补丁、QA 报告的哈希。
- `candidate/tests/lip_lighting_contract.rs`：五项公开接口回归，旧版 4 失败 / 1 通过，C 5 / 5 通过；既有颜色测试 C 60 / 60 通过。
- `evidence/cli-probes/`：实际 CLI 黑纹、饱和彩色、白高光、透明输入探针，避免仅靠单测而未验证导出二进制。

主报告：`../../doc/qa/lip-lighting-v2-20261015/review.md`。完整五原照 55 组记录：`report.json`；同平均线性亮度消融：`matched-brightness.json`；同目录 `*-contact-100pct.png` 与 `*-matched.png` 为原尺寸对照。独立审查在 `independent-review/`。

## 重现命令

从仓库根目录运行，必须显式指定 `candidate-c`；保留冻结 binary，不能再从共享 target 随意覆盖：

```sh
CARGO_TARGET_DIR="$PWD/experiments/lip-lighting-v2-20261015/native-candidate-target" cargo test --release --manifest-path experiments/lip-lighting-v2-20261015/candidate/Cargo.toml --test lip_lighting_contract
CARGO_TARGET_DIR="$PWD/experiments/lip-lighting-v2-20261015/native-candidate-target" cargo test --release --manifest-path experiments/lip-lighting-v2-20261015/candidate/Cargo.toml --lib color::tests
services/background-worker/.venv/bin/python experiments/lip-lighting-v2-20261015/replay.py candidate-c
services/background-worker/.venv/bin/python experiments/lip-lighting-v2-20261015/matched_brightness.py
```

用于真正原生构建的目录为 `native-baseline-target` 与 `native-candidate-target`，互相独立。每次回放前后校验 binary SHA，A/B 相同即拒绝。

## 明确未解决

1. 极端 −100 的浅色唇和浓妆照片仍有现有唇/内嘴蒙版交界的亮边、材质边界；未以模糊口缝或扩大编辑范围掩盖。
2. 彩色通道已到 255 时，正向没有保持原色度的额外亮度空间。当前优先保材料比例，正向响应因此受限。负向不会冻结这些彩色像素。
3. 该修复保留观测材料比例，不恢复真实反射率/照明；没有生成模型或 SOTA 声明。
4. 五张照片的几何和缓存检测没有重新验收；透明与保护区域另由 public/CLI 合同验证。
5. 原生时间包含 PNG I/O、进程与图校验，不能当 WASM 滑块延迟。浏览器和 API 以 root 的生产验证为准。

## 失效与中间产物，禁止用于采用结论

- 最初 `baseline-render` 与 `candidate-a-render` SHA 相同。第一次 55 组回放的 A/B 差异结论无效并已撤回，完整归档于 `../../doc/qa/lip-lighting-v2-20261015/invalid-shared-target-replay-a/`。
- `candidate-b-isolated-render` 为有透明修复但仍锁死饱和色负向调节的中间版本，不能发布。对应饱和材料 public 回归应失败。
- `evidence/invalid-overflow-candidate-b-saturated-test.log` 是早期 u8 加法断言在 release 下溢出造成误通过的失效日志；已用 u16 修正，再得到 B 真失败、C 通过。不要把这份日志计入通过数。

这些记录保留用于追溯，不作为质量分数。

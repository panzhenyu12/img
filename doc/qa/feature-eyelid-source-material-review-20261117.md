# 五官源材料独立复核（2026-11-17）

## 本轮结论

当前有两个明确、面向普通用户的未完成点，不能把五官整体记为可用：

1. **带眼镜的大眼操作仍可全无响应。** 在保留当前细粒度 parser 的整副眼镜类别输入时，`eyeSize=±30` 两张输出逐像素等于原图。这是已有反例在当前 release 的再次确认；隔离镜框分层实验尚不能代表默认算法已修复。
2. **眼睑操作仍会拉伸眼内材质。** 无眼镜 makeup 样片的 `eyelid=±30/±60` 将瞳孔/虹膜等可见纹理纵向压扁或拉长，未只调整眼睑与其遮挡关系。±30 的眼内局部纵向倍率约 0.922/1.088，横向约为 1；±60 达约 0.853/1.186。这不能作为合格的独立眼睑调整。

两者均不需要整体 build 复现。本轮不改 Rust，不改检测 mask，不调滑杆强度，不部署系统。已将证据交给负责核心算法的 root agent，避免并行修改同一实现。

## 入口与版本边界

复用现存 `test/results/eye-size-photo-audit-20261116/native/eye-photo-tests`。其照片入口调用正式 `apply_graph_json`，只额外提供观察器。检查该快照与当前 `geometry.rs` 的差异：运行算法相同，差异为测试态材料准入模块、诊断观察器与测试入口。当前源码 SHA256 为 `df3ddd3f2c929320a35ccae79e38cadef396352324fc31b59b93409ebcb9af8f`。

另调用现有当前 release 的 `core/image-core/target/release/deps/image_core-87faab56d5357871` 中 `geometry::tests::photographic_brow_flow_diagnostics`，重放 makeup `eyelid=±30` 及 glasses `eyeSize=±30`。当前原生入口输出与上述公共图入口快照输出 **PNG 逐字节相同**。当前测试入口本身是 `local_adjust_kind`，因此不冒充此次直接重新编译并运行了最新公共 validation；用明确的成片 parity 校核核心结果。

当前公共 validation 已拒绝 `eyeHeight`，不能把旧 `eyeHeight=±15` 成片当成当前可导出的坏图。本文新发现是仍获准执行的 `eyelid`；不包含已拒绝功能的评分。

## 真实坏例 A：眼镜实例冻结了眼睛

- 源图：`test/results/face-core-fresh/portrait-glasses-street-source.png`，768×512。
- SHA256：`9e3ec419493b0a6d0f506900bb866cf12d674f96c8fd00112561ac0b15405b41`。
- 图谱：复用 `test/results/eye-size-photo-audit-20261116/glasses-fine-parser-eyeSize{±30}.json`，保留来源细粒度 parser 的整副眼镜类别；没有手绘框线或修改 mask。
- 参数：`feature.adjust`，单控 `eyeSize=-30/+30`。
- 当前输出：`/tmp/feature-independent-audit-20261117/glasses-current-eyeSize{±30}-current.png`。
- 实际：两张输出 RGBA 均与 source 完全相同。对照中眼裂、瞳孔和眼内纹理全部原样，滑杆不产生眼睛大小变化。
- 预期：眼后层应可见缩小/放大，同时静止镜框保持；若源材料不足，应明确拒绝或提示，不应成功返回一张完全未编辑的图。

机制证据：`geometry.rs` 将 `hair/ears/accessories` 全部放入 stationary masks，覆盖 ≥0.5 时固定。该眼镜实例包括透明镜片后的眼睛；两个源 eye polygon 质心 `(328.899,263.492)`、`(381.526,257.528)` 的双线性 accessory coverage 均为 1，流场质心位置保持不变。这与整眼冻结相符，而非参数太小。

[FUNC-066](../func/FUNC-066-eye-occluder-layers.md) 的眼后层/静止前框实验已有单图概念验证，但源框光学材料与默认入口仍未完成。不能用删除整副眼镜 mask 或只降低冻结阈值修复；前者会让框跟着眼睛动，frame-only 单平面障碍的旧实验也已出现眼形受压。

## 真实坏例 B：眼睑把眼内纹理压扁/拉长

- 源图：`test/results/face-core-fresh/portrait-makeup-source.png`。
- SHA256：`0ec4c2eb0ed9cc8da77e5e0a9c47011dc5ebf09fb2c79db1f74e8d01afc952dc`。
- 图谱：同目录 `portrait-makeup-faceWidth-60.json`，仅把 operation 改为 `feature.adjust`、移除不属于 feature 的 `subjectMask`、设置单项 `eyelid`；保留原图自身的检测几何、选区和语义。
- 输出：`/tmp/feature-independent-audit-20261117/makeup-eyelid{±30|±60}.png`；±30 的 `-current.png` 为当前 release 复核。
- 原尺寸脸部对照：同目录 `makeup-eyelid-native.png`（source/-60/+60）。
- 眼内局部：同目录 `makeup-eyelid-eye-texture-3x.png` 和 `makeup-eyelid30-eye-texture-3x.png`。裁剪 `x278..446,y207..253` 后 nearest 放大 3 倍，仅用于定位，不生成新细节。

原尺寸可见 +60 的眼内圆形纹理被纵向拉长，-60 被压扁；下眼缘及眼内纹理也参与变化。±30 程度较轻，但方向与同一材料缩放一致。不能把更多眼球显露与把现有眼球像素拉伸混为一谈；前者需要处理眼睑遮挡及可见眼内材料，后者改变了现有纹理比例。

观察器对源 eye polygon 质心计算 forward Jacobian。两个点分别约为 `(307.332,229.758)`、`(418.781,229.688)`，是眼内纹理探针，**不是已检测的虹膜中心或虹膜真值分割**。其结果与所见纹理畸变一致：

| eyelid | 左眼 Jxx / Jyy | 右眼 Jxx / Jyy |
|---:|---|---|
| -60 | 1.000 / 0.85245 | 1.000 / 0.85474 |
| -30 | 1.000 / 0.92194 | 1.000 / 0.92322 |
| +30 | 1.000 / 1.08798 | 1.000 / 1.08640 |
| +60 | 1.000 / 1.18750 | 1.000 / 1.18393 |

这组数据只解释被检查的可见效果，不以 Jacobian 为正、逆流误差小或 changedPixels 非零作为审美通过证据。

机制可定位到 `geometry.rs` 的 `add("eyelid", ..., sy=-0.2, my=0.02, ...)`：它以整个 eye polygon 构建模板 frame，走普通局部非等比缩放；没有 `eyeWidthMaterial` 的独立可见虹膜材料，也没有 whole-eye 的相似变换约束。`validation.rs` 对缺少 ocular source layer 的 `eyeHeight` 有拒绝保护，但 `eyelid` 仍允许这一同类材料错误。

可证伪的后续修复目标：保留同源图、正负参数和实际眼睑响应；在改变眼睑遮挡时，已见眼内纹理不发生上述纵向缩放，并且不靠把整个眼睛固定成无响应来消除畸变。当前若缺少有序上下睑/canthus 和巩膜/虹膜证据，至少应明确说明未完成，不能将 `eyelid` 现有通用 warp 评分为可用。

## 其他小范围观察与复现材料

本轮另看 woman、makeup 两张图的 `noseLength/mouthWidth/upperLip=±60`，共 12 张原尺寸面部结果，未发现明确新增裂口；不据此判全部鼻唇功能自然度通过。woman 的 `eyelid=±60` 为眯眼样片，视觉差异较轻，不用它覆盖 makeup 的明确反例。没有继续大范围扫滑杆。

本地复现目录 `/tmp/feature-independent-audit-20261117/` 含 `cases.json`、`eyelid30-cases.json`、`current-cases.json`、`provenance.json` 和三份原生运行日志。所有原图、巨大图谱、输出和 binary 留在忽略目录或 `/tmp`，不推送。观察脚本只读取源/输出与原生诊断，没有把 QA 探针加入生产输入。

当前 release 复核命令：

```sh
BROW_FLOW_CASES=/tmp/feature-independent-audit-20261117/current-cases.json \
core/image-core/target/release/deps/image_core-87faab56d5357871 \
geometry::tests::photographic_brow_flow_diagnostics --exact --ignored --nocapture
```

修复验收必须返回这两类原始失败条件，不用数值安全测试或其他照片的通过替代。

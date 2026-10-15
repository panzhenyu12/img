# 唇部明亮 V2：单功能辐射与材质修复

日期：2026-10-15。范围仅 `feature.adjust.adjustments.lipBrightness`；不代表瘦脸、五官、妆容总体合格，也不授予 8 分。最小补丁已经 root 与独立审查通过后由 root 接入；生产 native/WASM/API 验证以 root 的发布记录为准。本实验不据此声称已部署。

## 缺陷与最小变更

旧实现以 `0.32 × (255 − 亮度)` 正向抬亮，暗纹获得更大的加性白雾。实际 public CLI 中黑色 `[0,0,0]` 在 +100 变为 `[82,82,82]`。固定 sRGB 色度再抬亮也会使原红唇变浅粉；原口缝因正确地被内嘴排除而保持黑色，与漂白外唇形成刺眼横条。负向固定 sRGB 色度也会改变原材料的线性通道比例。

候选 C 仅替换这段唇亮度传递：

- 转线性光后以共同倍率调整三通道，保留材料通道比例，黑色仍为黑色。
- 正向采用有界肩部避免通道剪裁；已到 255 的饱和彩色像素没有保持原色度的额外提亮空间，正向在该域会受限，不通过漂白假装仍有余量。
- 负向直接降低线性光，使用已有 `diffuse` 置信度保留中性镜面高光。不会把高饱和红通道误当白高光冻结。
- 亮度操作跳过 alpha=0，保持隐藏 RGB 原字节。非零 alpha 值保持。
- 分割、嘴唇/内嘴边界、保护蒙版、色彩/饱和度算法均未调整。

这是针对已复现缺陷的确定性颜色修复，不是生成式补纹、反射率真实恢复或 SOTA 声明。

## 真正独立的构建与公开接口红绿

基线与候选使用不同的全新 `CARGO_TARGET_DIR`，产物另外冻结：

| 产物 | SHA-256 |
|---|---|
| `experiments/lip-lighting-v2-20261015/baseline-isolated-render` | `d0fe0b5251f2f4e862ee20dc7381497fde0f198369f7f37b01eb012c8b68a434` |
| `experiments/lip-lighting-v2-20261015/candidate-c-render` | `adc18a9e67414e1aac6e554afe9bbb14813229ed9a37f5bd41670b2914cd364a` |

`evidence/frozen-c-manifest.json` 记录源码、Cargo 锁文件、回放脚本、生产补丁和回归测试的哈希。每次 CLI 回放前后均核对冻结二进制，脚本发现 A/B SHA 相同时直接失败。

公开 `apply_graph_json` 回归：旧版 **4 失败 / 1 通过**，C **5 / 5 通过**。候选现有 `color::tests` **60 / 60 通过**。新测试覆盖：

1. 黑色/暗红唇纹不被漂灰。
2. 典型唇色正负调节保留线性材料色度、alpha，并有实际强度。
3. ±100、±30、±1、0 单调，过零无跳变、零值逐 RGBA 不变。
4. 牙齿/内嘴、非唇皮肤、手工保护和选区、透明 texel 保持，受保护编辑仍有有效唇区域。
5. 饱和红/粉材料可负向变暗，中性白镜面高光保持。

实际 CLI 探针和图像见 `evidence/cli-probes/`。C 的 `[255,50,70]` 在 −100 输出 `[188,34,49]`；黑色 +100 仍为 `[0,0,0]`；白高光和透明隐藏 RGB 保持原值。

## 五张独立照片、55 组回放

样本为用户提供的原照，以及 `portrait-makeup`、`portrait-woman`、`portrait-hair`、`portrait-dark` 四张独立照片。没有拿旋转/派生图充当额外样本。

每张：亮度 −100 / −30 / 0 / +30 / +100，brightness=0 的 lipColor 和 lipSaturation 两组对照，以及四组亮度与颜色/饱和度组合，共 55 组、110 次冻结原生 CLI 输出。输入、已有检测图、操作图、结果 SHA 均记录于 `report.json`。

- 40 组非零亮度输出实际有新旧差异。
- 15 组零值/单独颜色/单独饱和度，新旧结果逐 RGBA 一致。
- 55 组 ROI 外、远离嘴唇的皮肤/背景、内嘴牙齿 polygon、alpha 均为 0 个改变像素。
- 五张的五档亮度逐通道单调，0 个反向变化像素。
- 实拍照片没有透明区域；透明和手工保护合同由上述 public 测试与实际 CLI 探针验证，未把无透明像素的照片冒充透明域测试。
- 使用已有检测结果，未重新检测；四张旧缓存缺新版耳部/饰品分割，与本次唇亮度的验证范围无直接关系。

原尺寸对照：`*-contact-100pct.png`。列为原图/旧版/C，行为 −100、−30、0、+30、+100。已逐张实际查看，未用缩小全图替代嘴唇原尺寸核对。

## 强度与等亮度消融

正向 C 比旧 +100 的提亮幅度小，这一点必须承认。为区分“只有调弱”与“材料处理改变”，另用旧版强度二分搜索匹配 C 的平均线性相对亮度 Y；所有比较使用同一组实际编辑像素（旧/C 变化区域并集，内嘴和保护域因无变化而排除）。Y 以 sRGB 解码、Rec.709 权重计算；表内单位是 **Y 百分点**，不是评分。

| 样本 | +100 旧版 ΔY | +100 C ΔY | 相近平均 Y 的旧版强度 | 等亮旧版/候选的材料色度 L1 中位数 |
|---|---:|---:|---:|---:|
| 用户 | 13.933 | 4.100 | +37.11 | 0.1191 / 0.0024 |
| 浓妆 | 13.918 | 2.917 | +22.27 | 0.0266 / 0.0019 |
| 张嘴微笑 | 12.219 | 4.105 | +37.89 | 0.0333 / 0.0024 |
| 浅色唇/头发 | 12.789 | 2.885 | +23.83 | 0.0162 / 0.0020 |
| 深肤色 | 12.113 | 3.476 | +37.11 | 0.1068 / 0.0037 |

五张正向平均 Y 匹配误差均小于 0.035 个百分点。同平均 Y 下，原尺寸 `*-brightness-100-matched.png` 仍可见旧版暗纹灰化、材料变粉；C 保留原红/棕材料关系。色度统计支持这一可见差异，并非单靠总体变化量减少认定改进。

负向 −100：浓妆/张嘴微笑/浅色唇三个样本可在旧 −100..0 匹配 C。用户与深肤色 C 比旧 −100 更暗，超出旧版该范围，不称为等亮度对比。完整正负统计、实际匹配强度、误差和状态见 `matched-brightness.json`。

## 已确认的效果与未解决限制

- 用户与深肤色的 +100：旧版大片漂白及横黑口缝明显减轻；唇纹未被抹成统一浅粉。相近平均亮度对照也保留此改善。
- 浓妆正向：原彩色高光和色素关系更稳定，但高饱和/接近白色的唇材质提亮余量小，滑杆响应受限。
- −100 极限值：浅色唇与浓妆仍会显露现有 lip/teeth 蒙版交界的亮边和材质边界。旧版同样存在；本次未通过扩大嘴唇区域或抹平口缝掩盖。此处自然度未完成，必须作为后续几何/语义边界课题。
- 同亮度色度更稳定不等于主观一定更漂亮，也不代表高精度反射模型。本次只接受为可复现的唇亮度单项改善，不给予妆容或全产品 8 分。

原生全流程时延（40 组非零亮度操作）中位数旧 245.93 ms、C 244.77 ms；包含进程、PNG 读写、图校验与合成，并与本地编译竞争资源，不能当成纯算法或 WASM 的实时延迟，更不据此宣称性能提升。WASM/浏览器/API 由生产集成阶段另行验证。

## 交付与复现

最小生产补丁：`experiments/lip-lighting-v2-20261015/production.patch`，仅 `color.rs` 的唇亮度块与新增 `core/image-core/tests/lip_lighting_contract.rs`；`git apply --check` 已通过。未替换整份 color.rs，避免覆盖并行工作。

```sh
CARGO_TARGET_DIR="$PWD/experiments/lip-lighting-v2-20261015/native-candidate-target" cargo test --release --manifest-path experiments/lip-lighting-v2-20261015/candidate/Cargo.toml --test lip_lighting_contract
CARGO_TARGET_DIR="$PWD/experiments/lip-lighting-v2-20261015/native-candidate-target" cargo test --release --manifest-path experiments/lip-lighting-v2-20261015/candidate/Cargo.toml --lib color::tests
services/background-worker/.venv/bin/python experiments/lip-lighting-v2-20261015/replay.py candidate-c
services/background-worker/.venv/bin/python experiments/lip-lighting-v2-20261015/matched_brightness.py
```

## 验证过程中的错误与撤回

首次共用 Cargo target 后复制二进制时，所谓 baseline 实际取到了候选产物，二者 SHA 相同。那批 55 组及主观对比结论已撤回，完整保存在 `invalid-shared-target-replay-a/` 并显式标为无效。本报告只引用后续独立编译且 CLI 探针证实不同的产物。

新增饱和材料测试最初用 u8 加法判断，release 溢出使断言误过；实际 CLI 探针抓到了不一致。测试已改用 u16，记录 B 真实失败、C 通过。错误日志保留为 `evidence/invalid-overflow-candidate-b-saturated-test.log`，不作为验收证据。

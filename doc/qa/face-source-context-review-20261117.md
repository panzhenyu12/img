# 完整源上下文的脸部轮廓独立成片复核（2026-11-17）

## 结论与适用范围

在本次正确编码的同源 alpha、配对背景及源材质权限输入下，`source-domain-continuous-v1` 的 16 张 `faceWidth` 成片、另 20 张下脸/颧侧独立控件成片及 12 张上脸控件成片，未发现本轮所查的明确新增裂口、旧轮廓残线、耳饰脱落或不连续台阶。各控件在已测正负档位均有可见的轮廓响应，可记录为**这批输入与档位的所查项目通过**。

这不是整个算法或产品已可用的结论，也不是 8 分评分。输入经过专门的源层准备；本次未验证默认端到端检测能在任意照片上提供同等质量输入，未验证大侧脸、遮挡、多人、多控叠加、全强度连续拖动、身体、眼鼻嘴、皮肤或妆容。`metrics.completed=true` 仅说明回放完整，不能替代画质判断。

本记录由独立成片审计 agent 形成：不修改 Rust，不跑模型，不整体构建。检查方式为 source/result 对照、原尺寸成片与局部放大；五个独立控件额外逐控检查原尺寸的脸颈/耳侧区域。用户样片带有“AI 创作”水印，因此统一称来源图片或样片，不把全部输入称为相机实拍。

## 本次有效输入和成片

根路径为 `/Users/modao/work/img`；临时回放根为 `/tmp/face-replay-20261117`。临时图像、mask、trace 只供本地复核，不复制进仓库或推送。

| 来源 | 原图（相对仓库） | 尺寸 |
|---|---|---|
| user | `test/results/user-face-contour-20261014/768/canonical-source.png` | 768×768 |
| woman | `test/results/native-face-contour/portrait-woman-source.png` | 683×1024 |
| dark | `test/results/native-face-contour/portrait-dark-source.png` | 683×1024 |
| Lupita | `test/results/face-usable-20261112/motion-interface-20261116/holdout-lupita-romano/canonical-source.png` | 681×1024 |

| 有效批次（相对临时根） | 范围 | 张数 |
|---|---|---:|
| `neural-source-pair-more-alpha255` | user/woman/dark，`faceWidth=+30/-30/+60/-60` | 12 |
| `lupita-complete-source-context` | Lupita，`faceWidth=+30/-30/+60/-60`，保留源耳饰随动权限 | 4 |
| `neural-pair-other-contours` | user/dark，jaw、chin、cheekbones、chinWidth、chinPoint，逐项 `+60/-60` | 20 |
| `neural-pair-upper-contours` | user/dark，forehead、foreheadWidth、temples，逐项 `+60/-60` | 12 |

四个目录的 `metrics.json` 均已记录 `completed=true`。只审查文件名以 `candidate.png` 结尾的当前候选；历史 RBF 的 `baseline.png` 不作为当前算法的通过证据。

## faceWidth：四来源、十六张

| 来源 | 有效档位 | 独立观察 |
|---|---|---|
| user | ±30、±60 | 原错误 alpha 批次在颈部产生的两块浮雕/锯齿伪纹理消失；耳环、项链未见新的裂口或脱落；脸宽正负方向均可见。 |
| woman | ±30、±60 | 下颌接颈部的旧细线消失；发丝、衣领及脸颈衔接未见明确新增接缝。 |
| dark | ±30、±60 | 左颈 `x235..265,y520..660` 未见独立留在原位置的旧竖轮廓。源图本有斜向颈部阴影，未将其误判成新缝。 |
| Lupita | ±30、±60 | 保留同源耳饰随动 support 后，扩脸左耳下的 S 形内凹/垂直平台消失，四档未见复发；两侧耳钉连接完整。 |

这批没有找到可复现的新增不可用缺陷。它不撤销其他未按相同正确输入回放的 holdout 问题，也不能把先前 fresh-hair 的几何回折自动记为已修复。

## 五个独立控件：两来源、二十张

Spec 为 `/tmp/face-replay-20261117/neural-pair-other-contours-spec.json`。每次只调整一个参数，其他参数不叠加。全组对照为 `/tmp/user-other-contours-sheet.png` 与 `/tmp/dark-other-contours-sheet.png`；原尺寸局部对照为 `/tmp/other-contours-{user|dark}-{key}-native.png`，每组三列依次 source、+60、-60。

| 控件 | 约定方向 | user/dark 的 ±60 观察 |
|---|---|---|
| `jaw` | 正值下颌内收，负值外扩 | 下颌两侧有方向一致的可见变化；未见新增耳下断点或下颌台阶。 |
| `chin` | 正值加长，负值缩短 | 下巴长度有明显变化；下唇以下与颈部保持连接，未见新增裂口。正值会拉长既有下巴及其阴影，不以参数意图内的加长本身判失败。 |
| `cheekbones` | 正值颧侧内收，负值外扩 | 颧侧有可见响应；耳/耳饰附近过渡未见明确新断口。 |
| `chinWidth` | 正值加宽，负值收窄 | 下巴宽度按方向变化；没有仅背景变化、下巴完全不动的情况；未见新增底部台阶。 |
| `chinPoint` | 正值侧缘收尖，负值外扩 | 下巴尖圆轮廓有可见变化；未见新增尖角裂口或连接中断。 |

本表判断是逐图可见响应和成片缺陷检查，不是各控件解剖效果、风格偏好或操作曲线的完整验收；也未证明不同控件在组合使用时没有耦合。

## 追加上脸三控：两来源、十二张

Spec 为 `/tmp/face-replay-20261117/neural-pair-upper-contours-spec.json`。单控 `forehead`、`foreheadWidth`、`temples` 分别在 user/dark 上回放 ±60。除全组对照，还检查了每个控件 source/+60/-60 的原尺寸发际线、两侧太阳穴及完整外发缘：`/tmp/upper-contours-{user|dark}-{key}-native.png`。user 局部范围为 `x220..549,y0..354`，dark 为 `x145..564,y170..559`，裁剪未缩放。

| 控件 | 实际观察 | 本批所查缺陷 |
|---|---|---|
| `forehead` | 正值使皮肤/头发内界面上移，负值下移，额头高度有可见变化；既有发丝随内界面连接。 | 未见新增发际线裂口、重复残线或外发缘断层。 |
| `foreheadWidth` | 正值额头两侧加宽，负值收窄；变化位于额头侧缘，不是只动背景。 | 未见新增太阳穴平台、皮肤/头发硬接缝或外层发丝断口。 |
| `temples` | 正值太阳穴内收，负值外扩；侧缘与眼眉外侧皮肤有连续过渡。 | 未见明确新增耳上连接台阶或太阳穴裂口。 |

这些结论来自实际的正负位移和接点观察，不是因为幅度小或 changedPixels 非零就判自然。尚未做独立主观审美盲评，不能据此给三控 8 分，也不外推到侧脸、刘海遮挡、其他强度或多控叠加。

五官保护另做了成片像素核对：以各自公共 graph 的 `leftEye/rightEye/leftBrow/rightBrow/nose/lips/teeth` 多边形为准，按像素中心 `(x+.5)/width,(y+.5)/height` 的 point-in-polygon 判定内部，将输出 RGB 与 source 逐像素比较。全 12 张结果的七类区域内变化均为 0。

| 来源 | 左眼/右眼像素数 | 左眉/右眉像素数 | 鼻/唇/牙像素数 | 十二张中的对应输出 |
|---|---|---|---|---|
| user | 496 / 505 | 702 / 705 | 1990 / 1866 / 23 | 六张均 0 变化 |
| dark | 436 / 421 | 663 / 654 | 2397 / 2458 / 143 | 六张均 0 变化 |

这验证已提供的语义多边形内部保护，不保证检测多边形囊括所有真实眉毛或整个解剖结构。初查的宽矩形 ROI 包含可变的眉旁/眼旁皮肤，存在非零差异；不能把矩形差异直接误报为五官保护失败。未为得到 0 变化而改动输入 polygon。

## 两类无效旧输入必须排除

### alpha 值域错误

旧 `neural-source-pair-more` 与旧 `neural-source-pair-extra` 的 16 张结果不能计入当前画质通过或回归失败。导出的 `physicalAlpha.data` 曾为 0..1，而 `PreparedMask` 的 raster 合同会再除以 255；opaque 的 1 被读为约 0.0039，正常皮肤被错误混入配对背景。后续 source scene 可行下界不能修复这一输入合同错误。

user 旧成片的明显伪纹理位于左颈 `x306..349,y468..490` 与右颈 `x411..457,y466..490`，局部证据为 `/tmp/user-neural-pair-neck-three.png`。修正成 0..255 并核验 renderer 解码后重跑，上述伪纹理消失。因此不能再据旧输出调整形变阈值，也不能把修复记成新的几何算法收益。

### Lupita 缺失源耳饰随动权限

`neural-source-pair-extra-alpha255` 的 alpha 已正确，但仍省略了源 cache graph 的 `__sourceMovingAttachmentSupport`。因此其 Lupita 结果不作为最终完整输入结果。`-60` 在左耳下约 `x224..235,y345..389` 出现 S 形内凹/平台：源边界随 `y352..385` 从 x229 平顺变到 x237，输出却大致停在 x229；-30 程度较轻。

只加入 interface 的消融没有修复该平台；恢复 attachment support 明显修复。`lupita-source-permissions-ablation` 中 attachment-only 与 both 的 `-60` PNG 经逐字节比较相同。因此本例改善归于**恢复同源耳饰随动权限**，不能记成 interface 的收益。

消融局部对照为 `/tmp/lupita-moving-ear-left.png` 与 `/tmp/lupita-moving-ear-right.png`，三列为 source、省略 support、恢复 attachment。最终 `lupita-complete-source-context` 四档已另行审过，未见上述平台复发或新的耳钉断口。缺少既存源权限的结果是合同缺失的反例，不能拿来证明完整合同下的最终结果失败。

## 复核指纹与未覆盖项

来源 SHA256：

```text
user    0076be94ee7650c06bdf5f7850e7ae92e0f95305af790351e8f103df66739397
woman   0dd2e17b2195062a8507978bf2e3c1f8b48f37c4cfabe3577aa4bc3efa0fbfd6
dark    c1999306aa24f649f2aff3c078cbb5f33f060903f88b3326ab2365657336a548
lupita  334e509ed36e28c5f9df46eecfac983e817b67fd0d5aabff8e32ed1bf2a39c8f
```

为防止后续覆盖输出后仍引用本次审图，记录每批候选 PNG 的清单摘要。计算规则：按文件名排序，各行 `文件名 + 空格 + 文件SHA256 + 换行`，再对整个 UTF-8 清单计算 SHA256。

```text
neural-source-pair-more-alpha255  6406e5f7198853cee75e0bce4efb936d095d4387dd684756ca3f165bd8d52e13
lupita-complete-source-context   a0cf0670d576b68b70e2360591dc43c093e7f51fee3592ddc72741e144fabf4a
neural-pair-other-contours       0c1e2a4b51c9ba76417c8a2e4e406917f3f5413bcda18ceb499ea9252b9feb85
neural-pair-upper-contours       bff237eb17f762e1605142e662dc536347088c3d1bc3d8656f87071cec324487
```

后续若只删未使用路径，成片字节一致可沿用本批审图；若有效输出发生变化，应重新检查对应结果。后续验收仍须覆盖：默认源层准备能否满足合同、额外人物与姿态、发丝/胡须遮挡、未测强度及多控叠加。此次未做浏览器接入、整体 build 或性能测量，不据此宣称服务已经更新、实时性达标或整个系统可用。

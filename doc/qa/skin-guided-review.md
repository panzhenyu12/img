# 皮肤引导滤波与肤色实拍复核

日期：2026-10-13。本次只读取既有输出和参数，不重跑算法或测试。**工程保护检查成立，但皮肤/肤色/妆容尚未完成逐功能 8 分验收；强磨皮已发现明确自然度失败。**

## 数据与对照真实性

`test/results/skin-guided/` 为六个样片版本 × 八种参数，共 48 个 `skin.retouch` 输出，最长边 1024px。`portrait-roll` 是 `portrait-makeup` 顺时针旋转 18° 的派生图，因此只有五张独立原照，不能称六种独立姿态。`fixture-portrait` 来自 Web 测试实拍 fixture，不是合成皮肤测试。

已读取 48 项 graph 与图片，并核对每个 comparison：左栏逐像素等于 `test/results/native-face-contour/<sample>-source.png` 按 graph 的 region 裁切，右栏逐像素等于同名编辑输出，画布只有这两栏。目录没有 baseline 文件；**本报告只比较原图与本次结果，不作“优于旧 Gaussian”的实拍结论，也不使用未经验证的旧二进制输出作对照。**

八种参数为 smooth=60/texture=60、smooth=100/texture=0、smooth=80/texture=100、whiten=60、warmth=±40、redness=±40。`portrait-makeup` 是带妆样片名称，48 项里没有唇色、眉毛、牙齿或眼下妆容操作。

[既有自动报告](skin-guided-validation.json)中 48 项支持域外变化与 alpha 变化均为 0；native 总进程时间 229.43–662.15ms，包括加载/PNG 编解码和处理，未计模型初始化与解析、没有热/冷分离或峰值内存。此指标只证明该支持域和透明度合同，不证明所有五官、头发、耳饰均正确保护。

六张 graph 有 raster skin/hair/neck，但没有当前新版独立的 accessories/ears 蒙版；因此不能用这批缓存证明新增耳饰/耳部保护的实拍质量。解析 confidence 是编辑覆盖，不是真实皮肤/发丝 matting。

## 实际目视结果

已查看六张九格图的全部参数，并对深肤色、浅肤色、带妆和斜图的自然磨皮/强磨皮/美白做原尺寸对照。九格图只为扫描，细节判断采用原尺寸 comparison。

| 样片与证据 | 发现与限制 |
|---|---|
| [portrait-dark](skin-guided-review/portrait-dark-grid.png) | smooth60/texture60 稍减额头、颊部小纹理，保留主要明暗；smooth100/texture0 明显抹去毛孔、皮肤成为连续色面，有蜡感。texture100 的保留方向正确，但不是智能区分瑕疵与正常细节。美白仍保留深阴影，冷调偏粉紫、暖调偏橙；不能以亮度变化证明自然肤色。 |
| [fixture-portrait](skin-guided-review/fixture-portrait-grid.png) | 自然档改变较轻，明显细点仍保留；强档额头/脸颊均过度平滑，与眼周原细纹形成材质差异。牙齿、嘴唇与背景目视未发生整体漂白；本批不是牙齿美白操作。 |
| [portrait-woman](skin-guided-review/portrait-woman-grid.png) | 笑容及主要五官保留；自然档对轻微细纹作用保守。强档额头、颊部和下巴过于光滑，眼角仍有原细纹，呈现不一致的肤质。美白60整体较亮，不能替代局部光影和肤色均匀化。 |
| [portrait-hair](skin-guided-review/portrait-hair-grid.png) | 发丝和牙齿总体固定；强档脸和脖颈变成平整色面，鼻/唇周保留纹理，质感不协调。美白加亮额颊及颈部，较浅肤色的立体层次减弱。自动 skin 当前包含可见 neck，需要按实际肤质与范围精调。 |
| [portrait-makeup](skin-guided-review/portrait-makeup-grid.png) | 原照已经有妆且肤质平滑，自然档变化轻；强档进一步抹去细节，不能作为修瑕成功的例子。美白60在本来明亮的额头/鼻/下巴增加亮度，局部层次偏平；冷调-40有粉紫倾向。唇色保持只是皮肤排除嘴唇的辅助证据。 |
| [portrait-roll](skin-guided-review/portrait-roll-grid.png) | 与原带妆图旋转版本相同的强磨皮塑料感仍存在；说明旋转后能定位，不等于覆盖真实侧脸、俯仰或三维姿态。美白与冷暖方向成立，未证明肤色自然度达到 8 分。 |

强档失败不能被自然档的微弱变化掩盖。当前算法同时存在“保留纹理时对瑕疵不够有选择性”和“极值去纹理过多”的问题；缺少毛孔/痘/斑的材质分类或可校正瑕疵 mask。身体、颈部和面部的肤色统一、极暗光、胡须、眼镜、多人及多尺度仍需独立验收。

## 妆容证据的准确范围

另读取 `test/results/feature-skin-algorithms/` 的独立 feature 缓存：portrait-makeup/dark/roll 的结果时间为 2026-10-13 09:24–09:26 UTC（本地 17:24–17:26），不是 skin-guided 的 48 项，也不当作其 baseline。这里只补充目视，不复跑或外推当前 native/WASM 一致性。

- `portrait-makeup/feature-lipColor-60.png` 与 [portrait-dark 唇色60](skin-guided-review/feature-portrait-dark-lipColor.png)：着色基本限制在唇部，深肤色唇部有明显红色变化；当前是 RGB/HSL 调节，缺少真实口红材质、任意色号、唇纹反射和边缘自然过渡的多图证明。
- [portrait-roll 眉深浅60](skin-guided-review/feature-portrait-roll-browColor.png)：跟随倾斜眉毛，但整体变深接近均匀填色，尚未证明毛发逐根质感或自然妆效果。
- `portrait-makeup/feature-teethWhiten-60.png`：闭嘴原图没有可见牙齿，输出零变化合理，只能证明没有误漂白嘴唇，不能当有效牙齿美白验收。

本次没有有牙样片的独立美白质量复核、深浅肤色眼下阴影多图评审，也没有自动腮红/眼影/修容材质模型。不能把旧 UI 的参数存在或缓存的局部差分换成妆容质量通过。

## 下一步验收

保留自然档与强档失败，先核对瑕疵弱化与毛孔保留的可控性，再验证肤色在曝光/肤色/光源变化下是否自然。新增独立样片覆盖极暗光、胡须/眼镜、真实侧脸和牙齿，对当前 accessories/ears mask 与每个妆容参数分别回放、逐部位查看原尺寸。眼袋与泪沟结构仍无已验证算法，不能靠眼下提亮冒充结构修复。

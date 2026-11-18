# FUNC-071 身体内部孔与外轮廓测量分离

状态：已进入 Rust 核心；定向回归通过。`tilted-shirt` 强档仍未达到视觉验收，不代表 Web 已部署或通用美体可用。

## 真实失败与根因

`tilted-shirt` 收腰 `-60` 的同一件不透明白衣，在 `(92,400)→(92,401)` 出现逆位移 `0→14.3323 px`。旧轮廓扫描从骨轴出发，遇到第一个 alpha < 0.5 就停止，把内部语义孔误当成外边缘。截面半径因而从约 53 px 跳到 131 px，位移的外侧支持硬截断切开了仍然可见的衣料。

只把 part alpha 换成整个人物并不能解决：旧版会把错台移到更上方。问题需要在测量与光学层之间建立边界，而不是填掉输入透明度。

## 实现

- `body.rs::exterior_contour_support` 对 alpha < 0.5 的外部背景做八连通搜索。仅在测量外轮廓时跨过不与图像外部连通的内部 aperture。
- 原 optical alpha、前景颜色、背景取样与显式保护保持不变。真正通向外界的开口，包括单像素斜向通道，继续保留。
- 骨轴中心缺少材料时仍保留原锚定，避免把穿过中心的遮挡自动变成可动材料。
- 此拓扑处理只消费已经选定的人物/部位材料；封闭不等于实例或材质归属，不能据此合并多人或前后遮挡层。

## 验证与边界

`body_internal_aperture_regression.rs` 的六项回归覆盖不透明衣料的内部孔、通外界开口、输入 alpha 不变、单像素斜向开口、显式固定遮挡层及真实 aperture 的可见下层。修复前合成衣料有 `6.4486847 px` 单像素位移跳变；修复后通过。根线程统一运行 `cargo body::`：29 通过、0 失败、0 ignored。

真实照片修复后，`x110,y398/400/401` 位移为约 `10.437/10.691/10.815 px`，旧断层消失。独立视觉复核确认 `(85..135,395..405)` 横向衣料错台改善，原尺寸 `+30` 未见旧明显断层。用于对照的 source alpha 和 foreground 保持一致；同依赖重编旧 body 的三档输出与旧 binary 逐字节相同，排除其他 matting 修改带来的混淆。

已接受样片回归：fashion-standing `±30` 的 field/alpha 逐字节相同；fullbody-skirt 的 frozen source-layer 实图 `-30/+30` 只变化 `25/27` 像素、最大 RGB 差 `1/2`，无不透明前层变化。这不是质量评分。

仍未解决：强档左上缘 `(88..97,358..366)` 有新增轮廓台阶/折角；右腰约 `(280..310,485..530)` 仍需后续验证。因此不宣称强档可用或达到 8 分。

## 后续上缘单变量诊断

独立 tracer 固定全部几何控制、位移场、选择与保护，只将 source optical part 改为现有 subject alpha。两档完整场逐字节相同，原 baseline 复现也逐字节相同，但上缘台阶仍在。它排除了“只换整个人物光学层即可修复”的假设。

源 RGB 显示连续白衣，但旧 alpha 在 `x89,y361→362` 从约 `0.0342→1.0`，更上方已呈白衣 RGB 的像素却为 0。该错误源边缘仍被当作固定背景。

复用同源照片的既有 MEMatte 缓存，只替换 optical alpha 并重算配对 F/B：完整 field 仍逐字节相同，实际消费 alpha 与缓存逐值相同，范围 `[0,1]`，没有把 `[0,1]` 误当作 raster `[0,255]`。`x89,y356/360/361/362` 的新 alpha 为 `.97066/.92695/.91773/.93259`。两方向的旧左上台阶明显消失。若只把 contraction 的旧 `target<=.01` 早退门冻结回去，白衣残影重新出现，说明错误背景判断也会阻止正确材料被渲染。

这是一因子的 source-layer 诊断，不是正式整图修复：alpha 改变后配对 F/B 也按既有算法重新估计；局部恒等重合成残差最大 `2.288`、平均 `.096` 个码值，不能称严格恒等。缓存本身是 shirt-only，右腰裙子 `(293,510/511)` 从旧 alpha=1 变成 0，不能把不再移动裙子的结果算作裙子算法改善。两像素 color-line trimap 的独立反证也未解决该处错台。默认 alpha 规则尚未更改。

独立视觉复核在原尺寸与一次局部对照中确认：MEMatte 两方向的旧短水平错台均明显消失，同处没有旧 gate 诊断图的白色竖条/残影。更上方 `y346..349` 仍有较轻的方向变化，尚不足以判作裂缝。结论仅接受该段外轮廓的改善，不包括袖子/躯干内部遮挡和整图可用性。

## 本地证据

测试照片和生成图保持本地，不上传仓库：

- `test/results/body-current-diagnosis-20261117/tilted-shirt/`：原/旧/新对照、trace、`same-dependencies-proof.json`、`fixed-geometry-source-roles.json`。
- 同目录 `fixed-geometry-mematte.png/json`：固定场 source 光学层反证与实际消费/恒等残差核对。源照片 SHA `c10d5c84643c857d1476d2954b026bb09eb6020122e94c90502c8c5ed78a2575` 与既有模型运行报告一致，未重跑推理。
- `test/results/body-current-diagnosis-20261117/accepted-layer-field-regression/`：已接受照片回归。
- `test/results/body-current-diagnosis-20261117/native-aperture/build.json`：核心 SHA `0be65d45c23e5cd75e9ea3c79bda7c736e97d1fb750f30e9da974f303398dded` 与构建命令。
- `experiments/body-internal-aperture-20261117/source_roles.py`：固定 geometry、单独更换 optical source 的诊断构建。此实验不是新的默认引擎。

# FUNC-057 邻发人像常用档审查

日期：2026-11-10。状态：**复核既有真实成片；portrait-hair 与 portrait-makeup 的 ±30 常用档当前可继续作为候选，未发现新大裂缝。未接入默认算法，也未因此宣称8分。**

## 样本和范围

本轮只读取已有实验输出和源图，不重新检测、不构建、不改变核心实现。样本为：

- `test/results/face-core-fresh/portrait-hair-source.png`，588×768；
- `test/results/face-core-fresh/portrait-makeup-source.png`，768×677；
- 既有 `shared-cheek-curve-v4`（外发缘保护、共享邻发域、源相对曲线、容量约束）的真实输出：`/tmp/face-shared-cheek-local-20261106`，±30；强档输出在 `/tmp/face-shared-cheek-strong-20261106`，±60。

查看固定源坐标的局部并建立整脸面板，覆盖脸颊—发丝边、皮肤纹理、眼眉附近发丝、耳边和背景。另用已有 forward field trace 检查每个样本在 skin/hair 源区域的局部最小采样尺度。

## 观察结果

### portrait-hair

±30 的脸颊收缩/扩展连续，邻发边缘随局部曲线变化，没有矩形切口或黑缝。发丝贴脸区域的纹理没有被截断，蓝色背景边界保持连续。最小局部采样尺度：

- skin：−30 为 0.770，+30 为 0.746；
- hair：−30 为 0.716，+30 为 0.712；
- skin/hair 采样单元中没有尺度小于 0.5 的面积。

放大裁图中的差异是连续纹理移动，没有发现新断裂。±60 仍可渲染并看起来连续，但不把强档自然度等同常用档。

### portrait-makeup

±30 的脸颊、眼下皮肤和眉旁发丝边缘连续，没有观察到裂缝。最小局部采样尺度：

- skin：−30 为 0.758，+30 为 0.727；
- hair：−30 为 0.475，+30 为 0.420；
- 发丝区域只有极少数（约 `2.08e-5`）采样单元低于0.5，固定源局部放大未见对应断裂或重复边。

该图源 graph 的 `ears`/`accessories` 为空，虽然照片可见悬垂细耳饰，故本轮不能将耳饰保持、耳根随动或材料alpha宣称通过。它只验证脸颊/邻发常用档的实际连续性。

## 路线判断

两张图常用 ±30 当前已有共同域/容量约束候选，实际输出暂未显示必须更换为新的 bounded bilinear solver 的具体失败。新 solver 可以作为后续统一场引擎，但在没有可复现成片问题前接入会混淆“曲线效果”和“求解器变化”。本轮没有擅自替换 attachment_motion 的实验代码。

当前问题边界：

- `portrait-hair` 与 `portrait-makeup` 的常用档连续性可接受，但没有人工审美真值，不能换算成8分；
- `portrait-makeup` 的耳饰空语义是材料/附件类别缺口，属于另一条源类别路径，不能靠共享头发域解决；
- ±60/更强档需要独立审美审查，不能由±30通过推出；
- 身体、五官、皮肤、妆容和去人功能不在本次证据范围。

## 本地证据

- `/tmp/face-neighbor-material-review-20261110/portrait-hair-mild.png`、`portrait-makeup-mild.png`：两张原尺寸脸颊邻发 ±30 对照；
- `/tmp/face-neighbor-material-review-20261110/*-strong.png`：±60 强档对照；
- `/tmp/face-neighbor-compression-review-20261110/`：skin/hair 最小局部采样尺度和源坐标放大；
- `/tmp/face-shared-cheek-local-20261106/`、`/tmp/face-shared-cheek-strong-20261106/`：既有原生 trace、forward/inverse 与实际成片。

所有输出和照片留在 `/tmp`，不提交。下一步若要推进，应补充多张有非空耳饰/复杂发丝的手工源类别验证，再决定是否把 bounded solver 作为公共形变引擎。

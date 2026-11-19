# 人工源材料修正：隔离算法接口

此接口为自动材料识别提供可审查的人工修正，不接 Web、Rust core 或 MCP。它不要求人工提供光学 alpha 或被遮住的眼球底色，也不把人工路径成功写成自动算法通过。

2026-11-19，两个人工标注 known-layer 反例通过；自动 V5/V6 的既有失败仍保留。用户 768 px 源图的眼球身份修正、makeup 的一根可见细睫毛支撑修正分别完成有限实拍审查。用户左眼仍有残差，makeup 的收益细微。不是整体 8 分或任意照片可用。

## 三类材料标签

| 标签 | 人工声明 | 算法行为 |
|---|---|---|
| `ocularAnchors` | 这些源像素确认是直接可见眼球，包括虹膜、瞳孔、反光或眼白 | 覆盖自动错误归属，设置 `B=C, E=1`；撤销该处自动睫毛标记。这是材料身份声明，不是用户画出的半透明 alpha |
| `movingLashSupport` | 指定细睫毛及其与同动眼睑相接的混合边支撑 | 仅在该支撑内推断前景/眼球层；可以纠正自动漏检。曝光比例与隐藏底色由源 RGB 估计，不从标签数值复制 |
| `unknown` | 此处材料尚不确定 | 不当作供体；只要非零运动的目的像素或反采样足迹依赖该处，明确拒绝编辑。零值仍逐 RGB 相同 |

未标注区域保留选中的自动估计器结果。不同标签重叠、两个眼睛的标签争用同一源像素、错误原图绑定或尺寸均拒绝；没有“后写覆盖前写”。

`ocularAnchors` 必须在当前源眼球几何域内。几何本身错误时需要先显式更正源几何，不能在无睫毛分支悄悄越域、带睫毛又拒绝。域外移动支撑仍属于结构未知，不把完全隐藏眼球宣称为已知。

## 原图绑定与坐标

`--materials` 是显式 JSON 文件。使用原始输入文件的 SHA-256、完整解码尺寸和原图 pixel-index 坐标；整数 `(x,y)` 指数组中该像素中心，与显示图片的数组索引一致。不要把 native graph 的半像素转换再应用到材料标注。

格式示例（只展示一个眼睛；本地实拍验收 JSON 包含两个眼睛）：

```json
{
  "version": "1.0",
  "source": {
    "sha256": "0076be94ee7650c06bdf5f7850e7ae92e0f95305af790351e8f103df66739397",
    "width": 768,
    "height": 768,
    "coordinates": "pixel-index"
  },
  "eyes": {
    "leftEye": {
      "ocularAnchors": [{
        "shape": "polygon",
        "points": [
          {"x": 324, "y": 273}, {"x": 325, "y": 270},
          {"x": 328, "y": 269}, {"x": 332, "y": 270},
          {"x": 334, "y": 273}, {"x": 332, "y": 277},
          {"x": 327, "y": 276}
        ]
      }],
      "movingLashSupport": [],
      "unknown": []
    }
  }
}
```

每类是若干简单形状的并集，沿用 `shape/region/points/strokes` 概念，仅用硬标签：

- `rect`：`region={x,y,width,height}`，选择 `[x,x+width) × [y,y+height)` 内的像素中心。
- `polygon`：`points=[{x,y},…]`，至少三点；选择周界及内部的像素中心。
- `brush`：`strokes=[{x,y,radius},…]`，圆形笔触的并集，半径单位为源像素。

不接受 opacity、strength、alpha raster、非零 feather 或 invert。形状不得离开原图或变成空选择；眼睛标签不得越出其本次准备的源 crop，不无声丢弃。源图仍须为不透明 RGB，带 EXIF 旋转的输入先显式归一化并重新观测。

## 源推断与未知的边界

`source_material.apply_materials` 在请求场生成前执行。每眼源层只准备一次，所有强度复用，材料 JSON 和代码均记录运行前后 SHA-256。

人工移动材料使用 `explicit_source_layers.infer_explicit_moving_source_layer`：眼球底色由显式 `ocularAnchors` 提供附近源 RGB，稳健局部表面不得用移动支撑或 unknown 当供体；前景颜色提议来自该支撑的源睫毛和现有源估计的 `skinPrior`。三色模型表示眼球、细睫毛与同动眼睑的混合。没有传入真 alpha、真底色、真皮肤层、目标强度或编辑后图。

`estimated` 只表示供体足够、源层有限有界且可重建源 RGB，不等于恢复了光学真值。颜色近似误差超过源模型经验尺度时单列 `modelUncertain`；已由人工确认的材料身份不因这个经验尺度被重新改成未知。该许可只属于明确人工身份，不能授予自动假 track。

缺少供体、域外或无法满足数值物理约束则是 `unknown`，保持阻断。不能通过 RGB 裁剪、冻结整块图或偷偷改成 `E=1` 来消除失败。`validate_material_transport` 同时检查目的地和双线性反采样足迹；接入新的调用者时不能省略该检查。

## 固定反例结果

人工绿色测试在独立 `test_source_material.py`，不替换自动 `test_observed_ocular.py`。已知真层只用于生成源图与期望输出，估计器仅收到源 RGB、离散人工标签和既有源几何/颜色提议。

| 固定反例 | 人工输入 | 结果 |
|---|---|---|
| 正常虹膜纹理被自动认作睫毛 | 源图中的一个 36 像素可见眼球矩形 | 4 个仍可见亮/暗纹的最大 RGB 误差 `170.285931 → 0` |
| 自动漏判细睫毛为眼球 | 固定粗笔划共 27 个 moving 像素，4 个可见眼球矩形共 132 个 donor 像素 | 最终 RGB RMSE `16.132596 → 2.733173`，低于原门槛 `5.650029`；暗线质心误差 `.081089 px < .25 px` |

第二项没有用真 alpha 描边。笔划中心线 `[31,16]→[38,20.5]`、半径 1.4、`y>=16`，在首次测试前固定；它标的是可见眼内线身与混合边，不据此声称完整睑根所有情形已验收。27 个像素均有估计，1 个 `modelUncertain`，0 个结构 unknown。原独立 RGB 与质心门槛没有放宽。

其他合同覆盖未知/缺供体依赖拒绝、零值恒等、原图/尺寸绑定、类别冲突、跨眼冲突、非 alpha 输入、RGB 0..255 与各分支域一致性。合计 5 项人工测试通过；它们不等于 5 个产品功能全面可用。

```sh
services/background-worker/.venv/bin/python -m unittest discover \
  -s experiments/eye-closing-source-20261119 -p test_source_material.py -v
```

可视图与完整数值在本地忽略目录 `test/results/eye-closing-source-20261119/manual-materials-known/`。源图、离散标注、自动失败、人工结果与独立 once-only oracle 并排保存。指标使用精确合成 float64 RGB，PNG 仅是取整显示。

## 用户原图的局部修正

原图为 `test/results/user-face-contour-20261014/768/canonical-source.png`，SHA-256 见格式示例。先看未修改源图的眼部坐标网格，再画两处可确认的眼球 polygon（61/68 像素）；JSON 在出图前冻结。没有从输出坏点差分生成标注，也没有在看到残差后扩圈。

同一次回放由旧 V1 源准备产生自动控制组和人工组，共用完全相同的 native inverse 与 `0/−15/−30` 请求。原图、原图上的标签、完整眼部及全图对照都保留，不能只用两个采样点验收。

在原先坐实的两个 glint 位置，−30 的最大 RGB 源差从 `29→5` 与 `24→0`，−15 为 `14→2` 与 `12→0`。**左眼仍有残差，不能写成两点完全恢复；这只说明人工身份修正具有实际作用。** 原图上未被确认的邻近采样材料继续使用基线估计，不为消除数值差事后扩大标签。

```sh
services/background-worker/.venv/bin/python experiments/eye-closing-source-20261119/run_closure.py \
  --source test/results/user-face-contour-20261014/768/canonical-source.png \
  --observations test/results/eye-height-layers-20261117/raw-source-observations/user/observations.json \
  --output /tmp/user-eye-manual-material-replay \
  --field-runner test/results/eye-height-layers-20261117/canthi-field-candidate/eye-field-tests \
  --estimator source_material:estimate_legacy_source \
  --materials test/results/eye-closing-source-20261119/manual-materials-user/materials.json
```

本地证据在 `manual-materials-user/replay/`，完整说明根目录为 `test/results/eye-closing-source-20261119/`。验收须检查整个虹膜、睑缘和材料边界，防止局部标签接界生成暗环或断线。照片、观测、标注 JSON、数组与输出不纳入代码提交。

独立 reviewer `/root/global_review_1119` 已查看两眼 −15/−30、−30 全脸、完整虹膜 10× 与 polygon 边界：高光明显改善，未见新 polygon 硬边、暗环、双睫线或眼睑断裂。结论仅为该用户照片轻档的 `ocularAnchors` 修正有限成功；此例 `movingLashSupport=0`，不能替代实拍移动支撑验收。

## 一根实拍细睫毛的移动支撑

随后只用 makeup 原图，在其右眼眼白前选择一根源图清楚可见的细线身及相接混合像素。先保存原图坐标网格与 source overlay，再冻结材料 JSON，之后才生成同场自动 V5/人工对照。没有按目标差分选择 ROI 或调整模型参数。

固定笔触中心为 `[676,375]、[676,376]、[675,377]、[675,378]、[674,379]、[674,380]、[673,381]、[673,382]`，半径均为 1.3 源像素，合计 26 个离散 moving 像素。三个相邻可见眼白供体框为 `(669,379,2,5)`、`(678,374,2,9)`、`(673,384,5,2)`，合计 38 个 anchor。原始 JPEG 的 SHA-256 为 `1b2a0147296ce870f74cbace5bfa2e2d147a8c0b384d79794bc4423266f3fd70`，材料 JSON SHA-256 为 `f8149a861abaf236ef34d017dfe9481d31775043bc3d5c2bef299edc43ddf63a`。

三档共用一次源准备与同一组 native field；没有新模型、整体 build 或修改 V5。26 个 moving 像素全部得到有限有界源层，结构 unknown 为 0；**13 个仍为 `modelUncertain`**，保留近似颜色误差，不改写成已观测真 alpha。源 RGB 重建误差为 0，没有裁切输出。

独立 reviewer 查看源标注、rightEye 完整 −15/−30 对照、−30 全脸及 12× 线身/睑根接界，确认标注内灰黑细线随轻闭有小幅移动，未见新增独立第二条睫毛、旧位置明显黑块或标签硬边，完整虹膜和高光无新损伤。接受为**这一根可见眼白前睫毛在轻档的有限实拍验证**；收益细微，不能宣传大幅画质提升、完整睑根通过或其他材料场景通用。

证据位于 `test/results/eye-closing-source-20261119/manual-materials-makeup/`：`materials.json` 和出图前的 source overlay；`replay/` 内有完整自动/人工结果、source review、各档眼部对照、固定源坐标 crop 的 `shaft-comparison-minus15.png` / `shaft-comparison-minus30.png` 与完整报告。所有运行前后输入/代码 hash 稳定；标注与估计器均未按结果改动。

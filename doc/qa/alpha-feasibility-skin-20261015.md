# 皮肤 alpha 色域下界：条件成立，暂不接入

已完成隔离实现与红绿测试，详见 `experiments/alpha-feasibility-skin-v1-20261015/README.md`。只在已确认 skin-over-known-scene 的材料下，从 `C=aF+(1-a)B` 与 RGB 色域推出 alpha 下界，将原 prior 投影到可行区间。

已知背景高光测试：alpha 低估 `.02` 的旧皮肤残线最大4.9215 byte，修正后约0；5项测试均通过。但可行不代表真值：另一个空间变化背景反例源重建约0，移动后仍有2.1165 byte误差。错误暗背景仅5 byte差即可把 alpha `.0021` 错推至 `.6667`。

实际用户原图中，独立观察的235个皮肤路径像素在新版 direct alpha 下均无违反，修正0像素，没有可视提升。宽域79个违反大量属于暗背景/耳饰孔洞，不能当皮肤修复。因此未接入 Rust/WASM 或部署；报告 `algorithmAccepted=false`。

另对 `composite-delta-v2` 完成精确 replay：输出等于目标分层+固定源残差，原图±100均0像素重放差异。零位移恒等不证明物理分层正确。下耳侧 `(630,503)` 的 direct alpha=.974 却因旧/新交界混合降为.275，存在88 byte固定残差和8.72px位移；该交界需要独立核查。

所有失败样本保留。原 delta audit 的 no-clipping 对照曾实际包含低alpha clipping/fallback，本次已用明确范围修正对照并保留原失败记录，不以错误测试结论否定差分合成本身。

## V37 原生层追踪：旧混合缺陷已解，当前灰白条来自 B

只在隔离 `experiments/face-direct-physical-v37-20261015/src/face.rs` 新增 source/F/B、samples、frozen trace；此前源码快照保存到 `experiments/alpha-feasibility-skin-v1-20261015/qa/v37-trace/face-before.rs`。独立 `.run/feature-eye-v37-trace-target` 构建，明确 `FACE_V37_SOURCE_RESIDUAL=1`，新结果不覆盖旧候选。±100 与既有 `outputs-delta` PNG **0 像素差异**、geometry hash一致，按 f32 运算重放最终changed像素也全部0差异。

| 原图坐标 | source alpha | +100 target alpha | F RGB | B RGB | 固定残差 RGB（byte） | +100 output |
|---|---:|---:|---|---|---|---|
| 630,503 | .97409 | ≈0 | 158,132,119 | 103,95,92 | -1.57,-.04,.70 | 101,95,93 |
| 631,501 | .97067 | ≈0 | 158,133,118 | 103,95,92 | -1.39,.11,.76 | 102,95,93 |
| 375,450 | .99456 | .00000646 | 139,116,105 | 238,227,208 | .46,1.40,1.44 | 238,228,209 |
| 376,460 | .99252 | .00000145 | 138,114,108 | 221,209,190 | -.62,-1.71,-.61 | 220,207,189 |
| 640,460 | .99840 | .000588 | 160,127,110 | 119,120,123 | .066,.011,-.021 | 119,120,123 |
| 638,480 | .99714 | ≈0 | 168,139,121 | 105,98,96 | .18,.12,.07 | 105,98,96 |

上述点均 `changed=true`、`frozen=false`。`(630,503)` 已不再是旧混合 alpha=.275/固定残差88 byte，直接alpha已修复该部分。当前收脸露出的灰白区域主要跟随 **B 层灰白色估计**，并非没有清理源像素、门禁冻结或大源残差。金属旁B图存在耳饰颜色向脸下扩展的现象；具体donor机制仍应通过下一轮因果测试确认，不能只因颜色相似便认定唯一来源。

代码上的可疑路径：`face_matting::estimate` 输入仅 RGB 与 skin alpha，无单独饰品/scene材料标签；当不可达背景洞fallback到该B估计时，饰品像素可能被当成非skin背景参与传播。修复方向应检查背景材料供色合同，不应继续调整上述点的skin alpha或以低残差宣称通过。

完整原生trace、输入/源码/二进制 SHA、像素坐标与1×/5×图在 `experiments/alpha-feasibility-skin-v1-20261015/qa/v37-trace/report.json` 和 `100/`、`-100/`。本次没有改变生产算法，也没有声称达到8分。

## B-only 因果修复（0 / ±100）

使用已审核且内部一致的 `experiments/joint-skin-scene-support-v1-20261015/outputs-v2-consistent/scene-support.npz`（SHA `80068202929bc5ab75435eb17563a9a87a1e5934732eeade62c5754693aad4a6`），只在 `sceneDistance` 有限、原生changed、非frozen、非既有opaque金属域替换B。保持rawalpha、F、native samples、geometry完全冻结，采用相同f32差分合成式。没有使用旧不一致scene版本，也未外扩finite域。

- v1：硬有限写域，±100分别改变2292 / 1875像素；0原图exact。
- v2：只对B写域加入有限区域**内部8px smoothstep**过渡，alpha和F不变，±100分别改变2269 / 1847像素；0原图exact。
- 六组圈外、finite域外、opaque金属、frozen、输出alpha均0变化，原alpha/F/samples/geometry SHA保持。独立保存结果验证见 `experiments/alpha-feasibility-skin-v1-20261015/qa/v37-b-only-verification.json`。

核心可视效果：左`(375,450)`的银白输出 `[238,228,209] → [20,18,14]`；右`(640,460)`的灰输出 `[119,120,123] → [31,24,18]`；下耳侧`(630,503)`的灰输出 `[101,95,93] → [23,20,17]`。在几何/alpha/F完全不变时，宽灰白条消失，支持B污染是该现象直接原因。

仍未通过整体可用验收：

1. finite背景map在左y532、右y535附近结束，v1有矩形跳变，v2的内部过渡只能缓解，仍留局部小楔。不能在没有场景依据时越界假装补全；需更完整且可信的背景供色域。
2. 原轮廓存在极细残线。右`(641,470)`：C=[85,68,61]、rawalpha=.3814、F=[155,125,112]、新B=[31,24,18]、target≈0，固定残差[6.7,5.5,7.1]byte，输出[38,29,25]；邻`(642,470)`蓝通道残差6.5byte。大B污染消除后，冻结F/rawalpha相对新B的源分层不一致暴露出来。单变量实验保留该问题，没有偷偷改F/alpha。

可视对照：

- `experiments/alpha-feasibility-skin-v1-20261015/qa/v37-scene-b-only-v1/{0,100,-100}/`
- `experiments/alpha-feasibility-skin-v1-20261015/qa/v37-scene-b-only-v2-feather/{0,100,-100}/`
- 各目录包含face/left/right 1×、4×原图–V37–B-only对照，allowed/changed mask、新B和report。

这是有可视改善的因果实验；`accepted=false` 继续保留，未部署、未声称8分。

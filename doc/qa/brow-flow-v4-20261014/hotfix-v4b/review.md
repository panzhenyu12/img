# V4b 旋转面积与透明 identity hotfix

当前两项修复及对应回归通过。此前眉毛眼睑、边界和跨零修复仍保留，但首次接受漏测眼倾斜的局部面积串扰与透明冻结RGBA。旧源码与失败记录见[归档](../../../../experiments/feature-svf-v4-initial-audited-20261014/README.md)。**没有达到逐功能8分验收**；本轮只接受这两项可证实的几何/采样修复。

## 原因与实际改变

旧SVF把有限旋转位移当速度，`cos(a)-1`径向项在积分中持续收缩inverse map。单独`eyeTilt±100`、眼大小/宽高为零，实际检测眼中心inverse Jacobian约0.91455，使forward局部面积额外放大约9.34%。现在SVF用正确的旋转生成元`(-a*v,+a*u)`；旧无语义直接warp公式没有改变。

真实检测fixture、192×192最终Rust mapper、h=0.1px中心差分：修后inverse Jacobian约0.999938、forward局部面积比约1.000062，角度±0.300058rad，请求±0.300000rad，中心没有移动。独立圆形支持域测试同时验证半径、中心面积和预期角度，先在旧源码明确失败。

**中心面积稳定不能视作整眼刚体旋转。** 椭圆支持域旋转权重随位置改变。检测polygon顶点forward积分后的shoelace面积比为0.980707–0.998189，仍有约0.18%–1.93%面积减小；它是此fixture边界顶点的量化，不是所有眼型或连续眼域的严格面积积分。本轮消除了已证实的中心9.34%放大串扰，没有宣称整眼严格保面积。

旧remap对identity也做预乘alpha采样，alpha=0时丢hidden RGB。现在映射精确等于原坐标且位于source bounds内，直接复制四个原字节；真正移动的像素仍预乘alpha插值。扩展画布越界不读取源内存。公开`apply_graph_json`测试涵盖alpha=0、1、梯度，以及hair/ears/accessories、硬保护/选区，核对所有actual identity位置的最终RGBA；移动区域允许合法alpha重采样。

## 当前结果与历史范围

| 检查 | 结果与范围 |
|---|---|
| geometry默认回归 | 31 passed、0 failed、1 ignored，13.10s；当前源码 |
| 极值/边界实际mapper | 原82场景加4个对称组合，共86；正Jacobian、逆流、冻结供体/位置、眼包络、目标位移、像素/恒定alpha |
| 跨零连续性 | 原4组加5组对称/五官组合，共9；全图mapper<0.01px、RGBA≤1/255 |
| 新增独立回归 | 旋转半径/面积/角度、检测眼中心、public graph非均匀alpha、identity扩展bounds；旧版本3项明确失败日志保留 |
| native/WASM构建 | `make native wasm`通过；immutable副本保留 |
| 受影响真实UI导出 | eyeTilt+100、1024×1024全RGBA差异0、page errors0；上传→工程导入→控件→PNG导出 |
| raw alpha parity | 5例全图native/WASM差异0、冻结hidden RGB变化0；每例另核对81个固定像素 |
| 未受影响输出 | 30眉毛+原UI18例，共48次fresh native PNG hash全不变；重复控制/派生不增加独立样本数 |
| 旧42原尺寸mapper诊断 | 本轮未重跑，保留历史；所有为零rotation、非零眉场，速度公式未变，并由相关native hash回放支持保留 |

当前报告：[geometry回归与旋转量化](geometry-regression.json)、[实际UI](ui-parity.json)、[raw alpha](alpha-parity.json)、[48次hash重放](unchanged-output-hashes.json)、[源码/产物provenance](../provenance.json)。旧19例UI与全仓Rust/E2E通过只证明旧版本，不声称全数对新源码重跑。主任务负责当前全仓检查。

浏览器Canvas解码PNG可能丢弃完全透明像素的hidden RGB；raw alpha parity绕过Canvas，直接把RGBA交给WASM，验证核心边界。本轮不宣称编辑器PNG导入全程hidden RGB无损。

## 目视与限制

[原尺寸眼倾斜Original / V4 / V4b裁切](eyeTilt-comparison.png)已目视复核：旧倾斜同时略放大眼睛，新生成元消除这项中心串扰。极值仍为二维局部眼形变化，不能称专业重建。原眉毛裁切输出hash不变，可继续按上级历史review目视；強眉峰仍夸张，眉尾受眼域/遮挡限制，邻近纹理仍可拉伸。

UI导出1249ms为本次单个墙钟时间，不是稳态热帧或实时benchmark。有限RK4、f32、光栅蒙版与紧凑场没有普遍无折叠证明；4K、侧脸、重遮挡、密集组合与跨分辨率仍需扩大验证。

## 复现与产物

```sh
cargo test --release --manifest-path core/image-core/Cargo.toml geometry::tests -- --nocapture
make native wasm
```

普通回归的42KB真实检测fixture已跟踪，纹理由测试生成，无模型、下载或本地results依赖。observer仅cfg(test)编译，无新增生产QA/env钩子。可选回放先冻结worker，再运行`scripts/audit_feature_svf_hotfix.py`、`scripts/audit_feature_svf_alpha.mjs`；UI脚本用`FEATURE_QA_OUTPUT/DOC/CASES/LABELS`选择eyeTilt。所有新产物写入`test/results/feature-svf-v4b-20261014`，没有覆盖旧接受证据。

| 当前产物 | SHA256 |
|---|---|
| geometry.rs | `864c6f89c7683537270248211dcd007a9ecf133a2811ffa1be1003585df74424` |
| fixture | `fbe960b7b04aee2c6b756b62aa0df0f5b1b7550163bbbd740cf10459b24b096b` |
| immutable native | `8b6e8ba027a42c767fe405bfdbd7eb2633b77b365b108877a414aa59f4545e95` |
| immutable WASM | `64419a6d260e33fab6cfee98cdcc9c586d0252cd2b6b5332a62bd417c0ee4190` |

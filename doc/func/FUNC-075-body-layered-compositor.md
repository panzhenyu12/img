# FUNC-075 Rust 内存前后材质分层合成

状态：**普通 Rust rlib 可调用，三张既有照片共11个输出与研究合成器逐 RGBA 字节一致。未接入系统，不据此宣布自动分层、隐藏躯干补全或通用收腰达到8分。**

## 能力与边界

公开 `BodyLayeredSource` 和 `composite_body_source_layers(source, layers, inverse_displacement)`。调用者传入完整分辨率的静止前层与可动后层，不把袖子和躯干混成一个图层，不再将含旧袖子的原图搬动后重新贴一次袖子。

输入包含：

- `width/height`：必须与不透明源 RGBA 图片相同。
- `front_alpha`、`rear_alpha`：每像素一个 `[0,1]` 范围 f64 覆盖。
- `front_rgb`：每像素三个 f64 前层颜色，范围 `[0,255]`。
- `rear_source_rgb`：每像素三个 f64 后层观测颜色，包含其固定场景贡献。
- `scene_rgb`：同尺寸、同编码空间的已配对固定场景 f64 RGB。
- `inverse_displacement: &[[f32;2]]`：调用者已计算并固定的逆位移场，像素中心 `(x,y)` 读取 `(x+dx,y+dy)`。

f64 源层保留混合边缘的精确源重构，避免把后层先存成 RGBA8 再采样的二次量化。源准备、模型、文件、MCP 参数均不在此算法接口内；现有 `body` 几何驱动不变。调用者负责场的 selection/protect 权限与物料角色，传完整场不代表可以移动所有部件。

## 算法合同

即使逆场全零，也先检查维度、长度、有限值、透明源画布、覆盖范围、颜色范围和源重构：

`C = a_front F_front + (1−a_front) C_rear`。

同时检查后层预乘残差 `P_rear = C_rear − (1−a_rear) B` 在 `[0,255 a_rear]` 内。只允许由 f64 表示和显示公式中乘加累计误差解释的机器精度范围；不设可容忍1个RGB码值的语义容差，不修改alpha或底图来让不一致数据通过。

独立代码审查发现并修复一项误差传播问题：亮源底图处约−1.3e−12的残差在源验证允许范围内，搬到黑目标底图后不能重新用更小的最终RGB尺度拒绝它。新增回归实际先失败，现把每个源样本已接受的计算误差随相同插值权重传播到输出检查。没有规范化/裁掉源残差，也没有扩大为任意RGB容差。

渲染只将 `P_rear` 和 `a_rear` 按相同双线性权重搬运，使用目标位置固定 `B` 合成后层，再覆盖静止前层 `a_front F_front`。越界坐标按现有body照片语义clamp；全零场和前层alpha1位置保持原像素。最终只量化一次，不对不合法RGB输出做裁剪。

该接口**不估计衣料归属、遮挡次序、不可见腰轮廓，也不生成隐藏纹理**。错误模型身份或错误补全仍可能输出不美观照片，数值合同不会把它们自动变正确。

## 验证

普通 rlib 的9项集成测试覆盖：混合边缘的零值/双向微量连续；后层双向真实移动且不复制旧袖；不透明前袖和零场像素保持；不一致源配对/非法隐藏场景在零请求也拒绝；非法尺寸、非有限值、透明图拒绝；软后缘在目标固定场景上合成的独立解析期望；画布越界无透明条带；亮底到黑底误差传播；二维双线性与软前后层的解析真值。

例程 `core/image-core/examples/body-layered-replay.rs` 直接链接普通库，不启用 `cfg(test)` 研究开关。实图源层由既有源准备重新配对一次后冻结，所有照片/数组/缓存逆场在本地ignored目录；算法不读取参考成片。回放后独立读取保留的参考图比较：

| 既有照片 | 档位 | RGBA差异像素 | 质量边界 |
| --- | --- | --- | --- |
| fashion-standing | 0/−30/+30 | 全部0 | 此前接受的轻档袖衣回归，不是新盲验 |
| fullbody-skirt | 0/−30/+30 | 全部0 | 此前接受的轻档衣裙回归，不是新盲验 |
| tilted-shirt | 0/±30/±60 | 全部0 | 手工前袖、ObjectClear下层、固定rear alpha的因果对照；不能称通用收腰通过 |

11张图全部逐RGBA相同；PNG编码器不同，未要求压缩文件哈希相同。单次本机native合成含全层验证约18.5–21.9ms，不包含源模型准备/几何场计算/文件解码，不是浏览器60fps承诺。

`tilted-shirt` 已由独立视觉确认前袖固定、旧棕楔消除，但下层模型重生了另一只袖子/手；后垂袖完整肩胸分界不可可靠观测，原场内也缺少可测双侧腰轮廓。此图只承担遮挡合成回归，不用于腰宽评分。详细证据见 `doc/qa/body-front-sleeve-source-review-20261118.md`。

代码：`body_layered_compositor.rs`、对应integration tests与example。离线导出与对等核对脚本：`experiments/body-internal-aperture-20261117/export_layered_sources.py`、`run_layered_rust_parity.py`。本地记录：`test/results/body-current-diagnosis-20261117/rust-layered-parity/report.json`。

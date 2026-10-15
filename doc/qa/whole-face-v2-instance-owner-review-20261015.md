# Whole-face V2 scoped-owner 独立审查

日期：2026-10-15。结论：**不能把当前连通组件保护视为人物实例保护；暂不接入生产。** 本审查只验证实例归属与覆盖层契约，不评价整体美观分数，也没有修改候选源代码。

## 冻结对象与复现

- 候选：`experiments/whole-face-shape-v2-20261015/scoped-owner-source-manifest.json`。
- 所有 manifest 中源码 SHA-256 与读取时文件一致；冻结二进制 SHA-256：`2355e919c57a3dbd30041108eeaa7f31f4e582e9254eb9452cad5f045baa0880`。
- 独立目录：`experiments/whole-face-shape-v2-20261015/independent-owner-audit/`。包含冻结二进制、manifest、`frozen-face.rs`、复现脚本、每项输入 PNG/graph、输出 PNG、debug masks 和逐项指标。
- 执行：`services/background-worker/.venv/bin/python experiments/whole-face-shape-v2-20261015/independent-owner-audit/reproduce.py --consistent-only`。
- 全部使用公开 `image-render --input --graph --output` 入口，没有插桩替换候选算法。

输入是独立生成的 160×320 合成契约场景：选中人物皮肤为椭圆并连有颈部，右侧未选中人物用可辨识彩色纹理表示。第二人物测试域为 `x=[123,145), y=[88,136)`，共 1,056 像素。它用于验证像素归属，**不是照片审美质量证据**。

## P1：头发接触会把另一人物整体认作当前人物

`frozen-face.rs:1618` 从与选中脸部相交的种子，在语义 head 的 `alpha > 0.5` 上执行无实例限制的四邻域 flood fill。`head_owner_support` 又由该组件及关联材质扩展产生。两人的头发在投影中接触时，一个连通组件能够包含两个人。

接着 `frozen-face.rs:1721` 的 `unselected_head` 仅检查是否不属于该连通组件。被错误连通的第二人物不再冻结，并在 `frozen-face.rs:1791` 附近作为 `joint_owned` 种子进入实际移动覆盖层。

复现只需给原本分开的两个头部增加 `x=[112,124), y=[110,114)` 的头发连接。最终对照重建了 **head 自己的真实 0.5 等值轮廓** 和独立 faceContour，像素中心约定与前端 marching-squares 一致；连接内头发和皮肤互斥，满足头发属于 head、head 属于 person 的类别包含关系。因此失败不依赖把旧 ellipse 错当成完整 headOutline。

| 输入与强度 | 未选中人物变化像素 | 最大单通道差值 | 未选中域 owner | 未选中域 frozen | 未选中域 moving alpha > 0 |
|---|---:|---:|---:|---:|---:|
| 分离，+100 | 0 / 1,056 | 0 | 0 | 1,056 | 0 |
| 分离，−100 | 0 / 1,056 | 0 | 0 | 1,056 | 0 |
| 头发连通，+100 | 645 / 1,056 | 152 | 1,056 | 0 | 1,056 |
| 头发连通，−100 | 650 / 1,056 | 156 | 1,056 | 0 | 1,056 |

最终数据见 `report-consistent.json`；原尺寸图见 `connected_hair_consistent/{100,-100}/comparison-native-1x.png`。所有 0 强度输入逐字节保持不变；输出 RGBA alpha 通道均保持不变。

补充控制：早先仅采用原测试轮廓的连通输入，+100/−100 分别改动第二人物 644/649 像素；添加显式 `protectMask` 后两方向均为 0。该结果见 `report.json`。显式保护有效，但不能据此宣称自动人物实例隔离有效。

## 为什么 faceOutline 不是修复依据

`apps/web/src/core/parsing.ts:9` 的 HEAD 是多个语义类别的并集；`decodeFaceClasses` 对裁图中的类别概率累加，未输出人物实例 ID。`parseFace` 在检测人脸周围裁图，长发触边还会扩展裁图。`parsing.ts:161` / `:162` 的 `faceOutline` 来自 **同一个语义 head mask** 的最大等值轮廓。

因此被连通的相邻头发既会进入 head，也会进入这个 head 推导的 outline。用它硬裁没有提供独立身份信息。虽然 `parseFace` 收到 `DetectedPerson` 用于裁图，返回图层本身没有证明只包含该人物。现有两项 `whole_head_contract.rs` 测试只验证了断开的高置信第二人头，无法覆盖这类歧义。

接入前应消费明确的选中人物实例/其他人物保护信息，并同时约束 owner、移动 alpha、采样供体与写入区域。缺失实例证据且出现多人物接触歧义时，应给出可解释拒绝或要求用户明确选区，不能把几何连通视为身份。实例 mask 仍不能当物理 alpha 使用；物理边缘覆盖需要独立估计。

## Owner 与物理 coverage 分离检查

追加了“语义耳部延伸到 head matte 外”的诊断输入。341 像素耳部域全部获得 owner support，但 +100 时只有 37 像素有移动 physical alpha，217 像素被 person/context 规则冻结；原尺寸可见耳部主体停留，靠脸一侧出现分离。

这说明 **owner=true 本身没有被直接当成 alpha=1**，角色分离仍在；同样也说明仅验证 owner 或脸宽变化，不能证明耳部实际跟随。但该输入 `ears > head` 不满足当前前端 parser 的类别包含关系，属于自定义/相互矛盾的图层诊断，**不把它列作正常 parser 路径已复现的产品 bug**。未来若支持独立物理 head matte 与语义 ears，各自来源、可信范围和不一致处理需要显式契约，不能简单并集冒充物理覆盖。

## 审查边界

本轮没有修改或部署候选，也没有为了通过而降低强度、缩图或改变被比较区域。合成反例证明的是自动实例隔离契约失败；它不估计真实照片出现频率。所有人物功能达到 8 分仍需要独立的真实多人照片、人脸/头发遮挡和各操作视觉验收。

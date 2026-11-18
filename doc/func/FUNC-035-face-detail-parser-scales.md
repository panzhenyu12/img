# FUNC-035：耳朵、耳饰细尺度解析与头部上下文分离

2026-10-29 核心供体保护增量：头发与耳饰或耳朵的语义掩码可能重叠。旧 `donor_frozen = attachment_fixed && !hair` 把重叠处误当作允许的头发背景；当人物蒙版也遗漏该处时，附件颜色可能进入新露出的背景。现分别保留刚性耳朵/耳饰身份，重叠时仍禁止作为背景供体，纯头发保持原有处理；缺失/不连通颈部的回退路径也使用相同规则。

实际渲染路径回归在旧代码中确认重叠附件 `donorFrozen=false`、`donorExcluded=false`；新代码保护耳饰和耳朵，并用真实背景填补函数验证亮色附件不会污染洞，纯头发仍按既有规则处理。人脸模块142项定向测试通过、3项忽略。用户、dark、makeup三张各±60的主native复核中，五组与此前RGBA相同，makeup +60仅耳侧24像素变化、最大通道差8；局部照片对照未见明显新增破坏。没有Web/WASM或服务部署，也不代表完整瘦脸、前景采样或柔边问题已解决。

状态：解析算法修正与聚焦回归已完成；dark 同请求实片局部改善，makeup 细长耳链仍漏检。不是完整人像自然度验收。本轮未构建 Web/WASM、未部署。

## 问题与原因

`parseFace` 为避免头发在内部裁切边界被截断，会将原人脸裁切扩大 1.7 倍重新推理。过去直接返回最后一轮的所有类别，初轮已识别出的耳朵、耳饰随之被低分辨率结果覆盖。

2026-10-28 使用现有本地 `faceparser.onnx`，通过实际浏览器 Canvas + ORT-Web WASM 保留每轮概率，复现了该问题。没有改模型、阈值、原照片、landmarks 或 faceWidth 请求，也没有人工绘制保护蒙版。

| 样例 | 初轮/上下文源裁切宽度 | 初轮耳部源像素 ≥0.5 | 上下文耳部源像素 ≥0.5 | 初轮耳饰源像素 ≥0.5 | 上下文耳饰源像素 ≥0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| dark | 527.417 / 896.609 px | 3163 | 493 | 283 | 0 |
| makeup | 537.799 / 914.259 px | 288 | 0 | 16 | 0 |

两图初轮均完整覆盖原始人脸 landmarks 的包围框；dark 四边余量为 125–149 px，makeup 为 130–152 px。初轮截断的是额外头发上下文，不是面部区域。初轮 crop 与最后 crop 均使用 512×512 模型输入。

## 算法契约

实现：`apps/web/src/core/faceParsingDetails.ts`，由 `parsing.ts` 复用原推理轮次调用，不增加模型或额外推理。

1. 每轮保存自己的源图 crop、模型尺寸及耳组 `[7,8]`、饰品组 `[6,9]` 概率。
2. 对耳组和饰品组分别选择：完整覆盖原人脸包围框、该类别未触及内部裁切边界的轮次。边界判断使用与 coverage 相同的概率起点 `>0.4`；原照片外边缘不视为内部裁切。
3. 在合格轮次中按源像素采样间距最小选择，不根据最高置信度、修图结果或 QA 选择。一个类别被截断不应阻止另一个完整类别采用细尺度。
4. 使用所选轮次自己的 crop 进行原图坐标投影。输出栅格尺寸、像素中心约定、双线性采样和 0.4–0.6 平滑覆盖转换均保持原约定。
5. face、faceContour、skin、neck、hair、subject 保留最后完整头部上下文的结果。不同尺度的语义不是联合分类分布，不做跨尺度概率求和或全图取最大，更不能直接当成物理透明度。

`parseFace.detailSemantics` 区分三种状态：

| 状态 | 返回的对应 mask | 含义 |
| --- | --- | --- |
| `detected` | 有非零覆盖 | 模型观察到类别证据；不保证完整或无误检 |
| `no-detection` | 合法全零栅格 | 有合格推理输入但没有检测证据；不代表图中不存在该物体 |
| `unavailable` | 缺少该 key | 没有满足源范围/裁切条件的轮次；不能把缺失解释成零保护区域 |

所选 crop 与 sourcePitch 随状态返回。当前 `detectPeople` 仍只消费解析 mask，详细诊断由 `parseFace` 提供；未做 UI/MCP 状态接入。

## 验证

新增 4 项聚焦行为回归，加上原分类映射回归，共 5 项通过。首先使用旧的“始终返回最后一轮”策略，复现了应选择宽度 6 却选择 12 的失败，然后修正。覆盖以下行为：

- 完整头发上下文不抹掉初轮耳朵与饰品证据。
- 不完整脸部 crop 被拒绝；两类别独立检查内部裁切边界。
- 合法全零证据与缺失来源有区别；不按较高置信度选择粗尺度。
- 细尺度模型像素按自身 crop 落回正确的源图位置。

运行命令（在 `apps/web`）：

```sh
./node_modules/.bin/playwright test tests/face-parsing.spec.ts --reporter=line
./node_modules/.bin/tsc --ignoreConfig --noEmit --skipLibCheck --target ES2022 --module ESNext --moduleResolution Bundler --lib ES2022,DOM,DOM.Iterable --strict src/core/parsing.ts src/core/faceParsingDetails.ts src/vite-env.d.ts
```

两张真实照片的新旧解析器独立比较：face、faceOutline、faceContourMask、faceContour、skin、neck、hair 与 subject 逐值一致；新 ears/accessories 逐值等于先前独立审查的细尺度候选。

同一冻结 Rust 二进制、同一原图、同一 faceWidth=+60、仅替换 ears/accessories 的回放：dark 右耳饰附近的白色缺口明显减少，耳饰轮廓与耳根衔接恢复，左耳拉扯纹理减轻。makeup 仅 1461 像素变化，最大通道差 8；长耳链大部分仍未被解析，不能宣称已解决细长饰品。

本地证据在 `/tmp/fresh-ear-semantics-20261028/`：`{dark,makeup}/scales/report.json` 保存每轮范围、概率统计；`crop-*/` 保存模型输入、实际浮点概率压缩文件及叠加图；`production-parser-verification.json` 保存新旧解析逐值比较；`compare-left-ear.png`、`compare-right-ear.png` 是同参数成片。照片、蒙版数组、模型、渲染二进制和生成图片不提交。

回放二进制 SHA-256：`24dffacba7cdc2429daba8d0d3dc7679a6ba83ac9b58e5041a4671bad82362de`。原解析器 SHA-256：`db3084f3d650ce305f377918e65771806b61d0edf9b3884e882b1d9f5d3c4a94`。

## 未解决边界

跨尺度存在未决语义冲突：以回放输入栅格的覆盖值 ≥128 统计，dark 的细尺度耳部与原 hair 重叠 188 像素、饰品与 hair 重叠 0；makeup 分别为 110 与 6。makeup 的细尺度耳部共 131 个 ≥128 像素，其中大部分与粗尺度 hair 冲突。本轮登记冲突，不强制把其中任何一层改为真实材质归属。

本修正恢复曾被扩大裁切丢弃的源证据，不能替代耳饰材质 alpha、透射层、遮挡顺序或刚体随动模型。裁切边缘没有类别证据也不能证明模型没有漏检。makeup 的长耳链说明最细现有 crop 仍可能不足；本轮未按照片手调尺度、降低阈值或补造 mask。更细类别/实例检测与材质分解需另行验证。当前没有依据把人像功能整体评为 8 分。

关联：[FUNC-026 多尺度眼镜语义候选](FUNC-026-multiscale-eyewear-semantics.md)、[FUNC-027 小耳圈材料实验](FUNC-027-small-accessory-material.md)。

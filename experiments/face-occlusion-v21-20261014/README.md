# 静止饰品遮挡 v21：隔离单变量因果实验

当前正式v19为基线。不是v20源曲线QP/颈控制的后继组合；不修改正式核心、公共WASM、前端协议，不提交或推送。

只改变联合脸颈流的解剖运动附件距离：hair、ears保持连接taper；accessories作为静止前景遮挡，不再把其旁边约43px脸颊当作人体固定连接。原三类 `attachment_fixed` 仍用于联合alpha的源像素障碍与final RGBA保护，饰品不会进入4px context taper。原faceWidth请求、旧QP、RBF/SVF、matting、reconstruction及其他参数保持v19。

原图脸侧y455–500紧邻的固定物主要是悬垂耳环，来源独立叠图 [`../../doc/qa/anatomical-curve-review-20261014/source-boundary-semantics.png`](../../doc/qa/anatomical-curve-review-20261014/source-boundary-semantics.png)。仅用户同一原尺寸1024²图+30/+100首图。后续正负运动需分别验证饰品采色污染、耳根连接与真实轮廓。

旧v19显式reconstruct donor_frozen禁止饰品/耳供体，但alpha颜色校正和全图F/B估计没有供体许可mask，禁止把局部donor_frozen称为全流程供体保证。本实验加饰品RGB反事实诊断；若输出外部对饰品色值有依赖，必须补内部供体隔离，再用同样供体隔离的v19目标作单变量对照。

首版source physical alpha与联合owned沿v19保持；源parser全零导致y547–552亮片被当作背景的独立问题见 `skin-island-diagnosis.json`，不混入本轮目标变量。请求/preQP/可行/实际、耳/发/饰品/五官/ROI/alpha/scene、真正外皮肤行宽与切线均单列。没有自动审美评分。

## 两个已经保留的阶段

`outputs/initial` 是仅改变motion附件距离的首版；源码保存为 `src-field-only-first`，immutable binary及source/binary/output SHA记录于该阶段report。它严格复现单变量，供体反事实却失败：饰品外800–1558像素依赖饰品颜色，最大通道差138–143。

`outputs/guarded` 是随后的供体隔离阶段。内部颜色工作层在供体禁止的coarse accessory硬像素上，用最近的非饰品观测颜色延拓；它是隐藏材质估计，不是恢复纹理。alpha颜色校正、F/B估计与直接前景取样沿该工作层；reconstruct也禁止hair重叠的饰品作为供体。原canonical、原检测缓存、最终冻结域的RGBA保持权威。这个阶段改变了344个物理source alpha像素（247在三类固定域之外），不能说物理alpha完全不变；其同供体隔离的all-attachment v19对照source alpha与candidate全字节一致，用它作运动变量的对照。原source/graph/最终原RGBA alpha没有变化。

guard阶段两档×绿/品红的全流程反事实全部为饰品外0变化；这是针对coarse硬饰品区域的测试，不证明软物理边缘和未知配饰。公开协议没有新增参数；两个native私有环境开关仅用于隔离回放：恢复所有附件运动固定、恢复旧供体通道。未将其接入AI/MCP。

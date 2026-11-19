# FUNC-081：源图眼镜与耳饰的完整组件运动许可

状态：源准备修复及定向验证已完成；修正侧脸曲线 QP 的浮点装配矛盾后，新眼镜正脸的普通 `faceWidth=0/±30` 已得到成片并经两名审查者有限接受。仅覆盖本图轻度脸宽调整，不能据此称任意照片、整个人像编辑器可用或达到 8 分。

## 实际问题

本地 `portrait-man.jpg` 是一张正脸、黑框眼镜、胡须的常见照片。冻结为 683×1024 后，从该原图重新运行检测和源分类；没有读取这张图的历史 prepared 数组。

旧准备流程把 BiSeNet class 6（眼镜）和 class 9（耳饰）合并为 `accessories` 后，将所有完整组件交给随耳运动许可。该图眼镜支持 15,821 像素，与左右眼分别重叠 553、559 像素，还覆盖部分眉毛和鼻梁，因此 `attachment-motion-plan` 在 `protected leftEye` 处拒绝整图。这是运动角色未区分，不能通过删眼保护、清空眼镜、裁掉重叠像素或改阈值消除。

## 修正合同

普通 Rust 已有固定 accessories 与可选随耳运动许可。本次不修改公共 Rust schema、原 accessories 栅格、物理 alpha 或瘦脸幅度，只修正 Python 源准备中的许可分配。

- 从**同一原图、同一已选后验**分别读取 class 6 / class 9，复用既有投影、覆盖转换和原生栅格采样。保留 observed-support 的有效单元范围，不新增模型推理。
- 沿用原 `core>=.5`、`soft>.02`、至少 30 个核心像素的拓扑规则。多个核心连在同一 soft 支持域时合并处理，不把同一实例的支路分别授权。
- 完整眼镜组件保持固定；完整且类别明确的耳饰组件可申请随耳运动。耳饰碰到用户、五官、头发或衣服保护仍拒绝。
- 同一组件同时有两种类别证据，或支持域未被完整观察，记为 `fixed/unresolved`。不割裂组件，不将它宣传为成功识别。
- 源类别判定使用组件内既有 `>=.5` 覆盖证据，仍可能误分类。固定透镜投影区域是本次 `faceWidth` 的几何保护合同，**不是镜片物理 alpha，也不是准确的通用眼镜实例识别**。

没有独立核心的离散弱支路不获得运动许可。模型漏检或真实断开的镜框片段不能仅靠连通域补出；这属于未覆盖边界。

## 绑定和传递

`attachment_subtypes.py` 生成独立 `partition.json`，绑定原图、原始准备 graph、检测 graph、所选完整后验、模型、有效域、完整组件、保护数组、许可数组及实现指纹。公开许可仍是原有 full-canvas 0/255 raster；普通 graph 不增加材料类别字段。

`prepare_attachment_source_plan.py --subtype-cache` 导出显式空许可或子集许可。`rebuild_head_source_cache.py` 重新核验同一分区，cache provenance 保留侧车路径和哈希；旧实验 `run.py → geometry.py` 同样只消费被许可的完整 moving 组件，不在后段重新 union 全部 accessories。缺失侧车不能绕过旧严格校验，缺失显式许可也不能进入 typed 路径。

缺少 `protectMask` 与用户显式提供全 0 `protectMask` 分别保留其输入身份。真实调用回归覆盖准备和 rebuild 两端，避免一端凭空添加全 0 user barrier 导致绑定漂移。

附件实现位于 `experiments/face-attachment-motion-20261110/`。没有修改人脸保护或缩小形变幅度以放行眼镜。随后普通入口暴露的 QP 数值矛盾另行修正，见下文。

## 可提交源码范围

本次附件功能收敛为独立的轻依赖工具，避免提交完整历史研究运行目录。必要实现共六个文件：`source_masks.py`、`attachment_source_plan.py`、`attachment_subtypes.py`、`prepare_attachment_source_plan.py`、`consistent_detail_control.py`、`apply_consistent_source_roles.py`；另有 `test_attachment_source_plan.py`、`test_attachment_subtypes.py` 两个测试文件。八个 Python 文件共约 48.8 KB，依赖标准库、NumPy、SciPy 和 Pillow。用法和输入合同见同目录 `README-source-attachments.md`。

`source_masks.py` 原样抽取既有原生 raster 采样和 mask 分派，旧 `geometry.py` 重导出相同入口。四组非方形、亚像素、反转和单像素轴输入的新旧入口 f32 bit 差异均为 0；由于两个入口最终使用同一函数，这只是抽取接线一致性检查，不是独立数学正确性证明。真正的 rebuild 准入函数 `attachment_release_from_source` 移入 `attachment_subtypes.py`，旧 rebuild 消费同一实现，测试可在无历史运行依赖时检查它。

仅复制上述八个 Python 文件到隔离临时目录，两组测试共 13 项通过（0.623 秒）。隔离过程中不导入旧 `geometry.py`、`run.py`、QP 实验、vendor 模型或完整 pipeline。旧 `face_pipeline.py`、`rebuild_head_source_cache.py`、`run.py`、`geometry.py` 的本地接线用于已有实验验证，不包含在这一独立交付闭包内；因此此交付不能被解释为完整编辑器或远端端到端源准备已经可运行。

轻依赖抽取改变实现指纹，并将 sampling、projection 和 role application 的代码一并绑定。已经冻结的本图 `partition.json`、metadata、源图及模型结果保持原样，它们记录的是抽取前实际运行版本；没有重写指纹冒充新版本生成，也没有为指纹变化重跑模型。源码隔离测试与照片成片分别报告，不混淆证据版本。

主任务另以交付的轻依赖程序读取相同原图、graph 与已选后验，生成新的 `source-attachment-standalone/` 分区。原图 SHA、完整组件角色与公开许可数组均与抽取前完全相同；实现指纹按预期不同。没有改写旧证据或重跑模型。核对记录为 `source-attachment-standalone-parity.json`。随后 `±60` 也完成独立实图审查，范围见 [FUNC-082](FUNC-082-face-curve-source-witness.md)，不扩大本源分类器的识别保证。

## 验证结果

定向命令：

```sh
cd experiments/face-attachment-motion-20261110
../../services/background-worker/.venv/bin/python -m unittest \
  test_attachment_source_plan test_attachment_subtypes
```

实际运行使用仓库内虚拟环境的绝对路径；共 13 项通过。首个回归先在原 `protected leftEye` 拒绝处失败，再验证眼镜整体固定、独立耳饰继续移动。其余覆盖混合 soft 连通组件、未知类别/域、耳饰保护冲突、禁止重放时恢复全附件许可、原图/graph/后验变更拒绝，以及缺省/显式零保护的真实 prepare→rebuild 调用。

独立旧用户图对账：两侧耳饰仍为 moving，支持分别为 1,195 和 1,237 像素，总计 2,432；与之前已冻结许可逐像素差异为 0。只复用了相同源后验，没有重新检测或推理。

新眼镜正脸的 typed 源准备得到一个 eyewear/fixed 组件，15,821 像素完整支持保持，显式 moving 许可为 0。原 source mask 不变，源重建通过。后续采用已审查的固定 outer-unknown 4px 与局部场景 inner32/outer4 策略；17 个 256px/128px stride tile 完整覆盖接受域，没有按成片改参数。

**首次普通入口失败已保留：** `faceWidth=0` 输出逐 RGBA 等于原图；`+30` 和 `-30` 均返回 `side curve QP status PrimalInfeasible`。旧 `ordinary-current/manifest.json` 不覆盖。历史测试程序用于源准备的成功不替代普通入口验收；失败也不能直接解释为眼镜与瘦脸物理冲突。

随后捕获的真实两方向失败输入均有 28 个曲线样本、8 个固定变量；原图见证仅在曲率约束 row 95 违反 `1.2348133396e-6`。该行的三个变量全部固定，等式残差为 0，因此这是严格的装配矛盾，而非用户请求过大。所有输入有限，高度无重复。两名审查者独立复算得到相同结论。

修复将 QP 写为 `d=target-source`，用一致的 f64 源差分装配约束，固定点仍为 `d=0`，区间、一阶和曲率保护保留，零位移成为明确可行见证。没有放松求解容差、迭代上限、间距下界或强度。四项曲线定向测试通过，其中包含真实反例的失败到修复验证。具体代码审查和成片对账见 [新眼镜正脸独立审查](../qa/face-glasses-source-independent-20261119.md)。

修复后普通输出写入 `candidate-source/ordinary-curve-fixed/`。零档逐 RGBA 恒等；`+30/-30` 分别改变 36,025 / 36,117 像素，眼镜核心 14,607 像素及双眼、双眉、鼻、唇保护核心逐 RGBA 不变。两名审查者查看了完整 683×1024 原图和双向成片、原尺寸全脸、2 倍镜圈/双耳颊/胡须颈部对照：有可见轻度双向脸宽响应，未见新增镜框弯曲或双影、镜圈下硬缝、脸颊台阶、胡须断裂或颈边色线。变化像素数本身不证明自然度；本结论限于该图这两个档位，未扩展至强档、侧脸、交叉遮挡或眼部编辑。

本地忽略证据统一在 `test/results/face-new-frontal-20261119/`：`fixed-input.json`、`source-conflict-report.json`、`source-attachment-conflict.png`、`source-attachment-typed/partition.json`、`user-earring-partition-regression.json`、`attachment-tests.log`、`minimal-delivery-verification.json`，以及首次失败 `candidate-source/ordinary-current/manifest.json` 和修复后 `candidate-source/ordinary-curve-fixed/manifest.json`。照片、后验、模型输出、trace、二进制均不随代码提交。

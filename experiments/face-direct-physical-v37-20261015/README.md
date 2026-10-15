# V37：直接软alpha与源残差合成对照

状态：隔离诊断，整体画质未验收，不接入生产。与当前生产v19不同，本实验基于V36并使用direct skin实验输入。

## 控制变量

几何先按 `experiments/joint-skin-matting-v1-20261015/native-v2-*` 图操作构建，再由仅native实验环境变量 `FACE_V37_PHYSICAL_ALPHA` 替换物理alpha。原图、目标曲线、实际逆映射不重新检测、不降低滑杆。`outputs/` 使用绝对合成；`outputs-delta/` 使用 `FACE_V37_SOURCE_RESIDUAL`：

`C_out = C_source + moved_premult - A_source * F + (A_source - A_target) * B`

源残差公式仅用于不透明输入。全部输出都是已标注候选，不能把“直接神经软mask”称为真实光学alpha。

## 结果

- user/woman/hair三原片0、±100完成，0与原图相同，几何诊断与native-v2一致。
- 原绝对合成会在极小神经alpha区域大幅改写背景。`test_opacity_bound.py`记录红与绿：user3220/3144、woman271/138、hair602/502个违例在差分合成下均归零。此测试是在初次公式试验后补写，不追称为完整初始TDD。
- 独立审计 `audit_independent.py` 检查整个ROI的 `255*max(A_source,A_target)+1` 变化上界、真实位移、圈外及零值；六组通过，见 `independent-audit/report.json`。
- 原尺寸视觉仍有旧脸缘细线、耳根衔接与灰银色露底。上界合同通过不能代表自然度通过。

## 已进一步分离的原因

精确native trace在 `../alpha-feasibility-skin-v1-20261015/qa/v37-trace/`。重新合成与冻结PNG逐字节一致。原图(630,503)、(375,450)、(640,460)在脸移走后，主要输出来自错误背景B，而非大源残差或冻结像素。

只替换有限且已验证的场景B、保持F/alpha/几何后，两侧灰银条明显减少。`qa/v37-scene-b-only-v2-feather/`保留对照；下端覆盖边界与细线仍在，尚非完整修复。后续F/B一致性实验保存在 `../source-skin-color-consistency-v1-20261015/`，不能用已知层的合成反例成功替代实图验收。

注意：`debug/source-physical-alpha.png`是覆盖前的调试图；覆盖后实际alpha以trace的float32文件为准。所有带调试导出的binary重新核对最终PNG；不可混用旧target目录中的同名binary。

生产候选V39单独把残差公式移到生产副本后出现照片旧边恶化，已拒绝，见 `../face-source-residual-v39-20261015/README.md`。本实验不是可直接替换生产的实现。

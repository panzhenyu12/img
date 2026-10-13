# 多样本身形效果记录

日期：2026-10-13。最长边768px。输出重新Pose分割，测量原部位轴线的连续前景宽度，测量只作辅助判断；表中UNVERIFIED必须视觉检查，不能当成收缩验收通过。

| 照片 | 部位 | 状态 | 宽度变化px（重新分割） | 原背景变化 |
|---|---|---|---|---|
| fashion-standing | waist | UNVERIFIED-no-measured-shrink | 0 | 0 |
| fashion-standing | left-upper-arm | UNVERIFIED-no-contour-measurement | 未测得 | 0 |
| fashion-standing | left-thigh | UNVERIFIED-no-contour-measurement | 未测得 | 0 |
| fashion-standing | right-upper-arm | UNVERIFIED-no-contour-measurement | 未测得 | 0 |
| fashion-standing | right-thigh | UNVERIFIED-no-contour-measurement | 未测得 | 0 |
| fullbody-skirt | waist | UNVERIFIED-no-measured-shrink | 0 | 0 |
| fullbody-skirt | left-upper-arm | UNVERIFIED-no-measured-shrink | 0 | 0 |
| fullbody-skirt | right-upper-arm | contour-resegmentation-shrank | -5 | 0 |
| fullbody-side-fashion | waist | UNVERIFIED-no-measured-shrink | 0 | 0 |
| fullbody-side-fashion | right-upper-arm | contour-resegmentation-shrank | -9 | 0 |
| tilted-shirt | waist | contour-resegmentation-shrank | -4 | 0 |
| tilted-shirt | left-upper-arm | UNVERIFIED-no-contour-measurement | 未测得 | 0 |
| action-suit | waist | UNVERIFIED-no-measured-shrink | 8 | 0 |
| action-suit | left-upper-arm | contour-resegmentation-shrank | -3 | 0 |
| action-suit | right-upper-arm | FAIL-render | 未测得 | - |

缺失部位/检测失败：

- fashion-standing：left-calf, right-calf
- fullbody-skirt：left-thigh, right-thigh, left-calf, right-calf
- fullbody-side-fashion：left-upper-arm, left-thigh, right-thigh, left-calf, right-calf
- tilted-shirt：right-upper-arm, left-thigh, right-thigh, left-calf, right-calf
- action-suit：left-thigh, right-thigh, left-calf, right-calf

原图、输出、差分图与操作图保存于test/results/body-algorithms。

人工目视复核与能力边界：

- fashion-standing的腰部有细微轮廓收缩，遮挡的手保住；重新分割没有量到收窄，保持UNVERIFIED。
- action-suit的腰部外套边缘有收缩，重新分割的+8px不能当作真实扩张。右上臂因无可靠补洞供体拒绝，属于本轮实际失败。
- fullbody-skirt的大腿和小腿均被衣裙/交叠遮挡，本轮不开放这些自动部位；旧版本拉动裙摆的输出不能当作腿部验收。
- 身体前景分割只能区分人物与背景，不能可靠区分衣裙和真实肢体。这轮的间隙/半径连续性检查是保守限制，完整服装与身体解析仍未完成。

15项中14项实际输出均通过原背景与支持区外像素保持检查；其中有多项效果测量为UNVERIFIED。保留这些状态，不将“有像素变化”转换成轮廓自然度通过。

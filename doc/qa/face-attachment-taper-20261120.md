# Face attachment taper 真实回归（2026-11-20）

## 输入

- source：`test/results/face-usable-20261112/motion-interface-20261116/cross-condition-20261117/portrait-hair/canonical-source.png`（784×1024）
- graph：同目录 `pipeline/detection/face-graph.json`
- 操作：`faceWidth=+60/-60`
- 输出：`/tmp/face-replay-20261119-face-taper-profile-2/`

## 结果

| 档位 | changed pixels | max RGBA delta | 区域外变化 | 显式保护区变化 |
|---:|---:|---:|---:|---:|
| +60 | 27,593 | 156 | 0 | 0 |
| −60 | 27,631 | 180 | 0 | 0 |

对照旧 fresh-source 回放（+60 34,915、−60 35,365）后，边界写入范围收窄；原尺寸成片中耳下/脸颈接入更平顺，脸颊的主要收窄仍保留。输出仍需人工审图，不能将 changed-pixel 下降直接换算为整体可用或 8 分。

## 单测

```text
cargo test --manifest-path core/image-core/Cargo.toml --lib source_paired_motion_domain_uses_a_wider_attachment_taper --quiet
1 passed

cargo test --manifest-path core/image-core/Cargo.toml --lib fixed_attachment_transition_spreads_motion_without_a_platform --quiet
1 passed
```

完整 lib 未重复运行；已有 body 与 face 回归计数保持不变。

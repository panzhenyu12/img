# 离线去人候选实验

这些算法未通过实拍质量验收，不在 Web/API/MCP 正式 Operation 里开放。失败图及采用决策记录在 `doc/qa/inpaint-refinement-review.md`。小误差或合成回归通过不等于真实画质达到8分。

PatchMatch 候选：

```sh
cargo test --release --manifest-path experiments/inpaint/Cargo.toml
cargo build --release --manifest-path experiments/inpaint/Cargo.toml
python scripts/check_patch_refinement.py
```

MI-GAN ICCV2023 候选，在独立 Python3.11 环境安装本目录 `requirements.txt`，然后：

```sh
python scripts/review_migan.py --download
```

`--download` 只下载固定 revision/校验 SHA256 的29.5MB ONNX；模型不提交 Git。RGB/蒙版使用一致正面积缩放、固定512对称补边，再送给原 pipeline，避免内部 resize footprint 混色。输入 RGB uint8 NCHW；mask uint8，255已知、0待补全；输出 RGB uint8 NCHW。保护/目标外/原 alpha 在原尺寸精确还原，仍无法修复复杂结构的模型失败。

PatchMatch 使用候选初始化、已知背景高权重、传播/随机搜索与原图补丁投票，512分析尺寸，原尺寸只采合法原图 donor。只允许不含目标/保护/非全不透明像素的原补丁；它可能重复纹理或复制错误结构，不能作为默认 LaMa 后处理。

MAT Places-512 候选另使用 `scripts/review_mat.py`。Python需要Torch2.2.2、safetensors0.5.3和本目录的NumPy/Pillow；架构来自 `Acly/comfyui-inpaint-nodes` revision `5b7681746fe19e563113d27366ba85236bc1ba5b`，解压到 `.run/comfyui-inpaint-nodes-5b7681746fe19e563113d27366ba85236bc1ba5b`。模型从 `spacepxl/MAT-inpainting-fp16` revision `bef6e8b7d535abe2d42fdaeafec903abdfc638a3` 下载 `MAT_Places_512_fp16.safetensors` 到 `.models/background/`。脚本会校验SHA256 `309dd6ce6e04034dc4b6b15c7bd2a4844d158e03eb2a139e0eca6b366e40c0de`；使用float32 CPU4线程、固定seed0，三图真实运行仍有明显伪影，未进入生产。

```sh
python scripts/review_mat.py
```

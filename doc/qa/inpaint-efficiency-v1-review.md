# 传统 exemplar 去物等价加速 v1

日期：2026-10-14。四个固定案例的完整 RGBA 与 `27f43bf` 基线逐字节一致，传统去物计算明显缩短；本轮没有改变去人自然度评分，也没有改变 LaMa 模型。

## 实现与正确性

`inpaint.rs` 预编译目标和保护蒙版，用有序 frontier 集合代替每步整图扫描，并在每次填充时只提取一次已知目标上下文。donor 的非负 SSD 累积已经不可能胜出时提前停止。保持原 raster 顺序、相同 f32 求和与最终得分运算以及同分时优先选旧候选的规则，参数、搜索范围和填充内容保持原语义。

新增 `bounded_search_retains_exhaustive_ssd_winner_and_float_ties`，与未剪枝 reference 比较稀疏上下文、裁切窗口、候选胜者及浮点近同分情况；保留棋盘/横纵条纹、保护与透明 donor、窄背景、自适应窗口和无 donor 错误回归。复核发现已有语义蒙版构造器会把未写shape但有points的inline蒙版当polygon；新增保持旧默认rect的预编译入口及先失败再通过的回归，同时核对目标和保护范围、整图无donor错误。全量 Rust release 为188项库测试、10项黑盒集成通过，2项手动检查/性能基准 ignored；all-targets/all-features clippy 通过。

## 固定案例实测

街景与人群来自既有实拍，建筑是明确的合成遮挡实验。三图缩至最长边512px并使用 raster；街景另按768×576原尺寸使用既有 polygon 和矩形保护区。每图固定 source、graph、radius=.008、iterations=32；先首次运行，再在同一进程重复五次取热中位数。

| 案例 | 首次旧→新 ms | 热中位旧→新 ms | 热加速 | RGBA差异 |
|---|---:|---:|---:|---:|
| 街景 512×384 | 273.41→66.98 | 248.28→62.50 | 3.97倍 | 0 |
| 人群 512×342 | 287.98→84.85 | 336.61→70.48 | 4.78倍 | 0 |
| 建筑合成遮挡 512×341 | 358.93→88.71 | 331.29→88.34 | 3.75倍 | 0 |
| 街景 polygon 768×576 | 671.79→77.81 | 736.41→71.17 | 10.35倍 | 0 |

四例选区外、保护区域、alpha改变均为0；768街景输出PNG SHA仍为 `83e241eec83b2d1e16f4290cb241282b4949c64c0e704896b2104a6bbb5903a6`。输入、graph、二进制、源码和输出SHA及每次热耗时见 [原始报告](inpaint-efficiency-v1.json)。完整图片在忽略目录 `test/results/inpaint-efficiency-v1`，历史街景结果见 [原去物审计](algorithm-audit-inpaint.md)。

这次计时在最终兼容修复后、本任务其他构建、模型推理和回放停止后单独执行，但未控制电脑其他活动；只有四个固定案例和每例五次热重复。旧版部分热重复285–431ms、768px为681–803ms，波动保留；前一次768px热中位562→65ms，说明本次10.35倍不能当作稳定加速承诺。范围为 native 同进程 graph解析、校验、填补及合成，排除图片读写、识别和模型加载。未测浏览器/WASM热耗时、大图、内存峰值或全链路，不据此保证所有滑杆实时。传统去物原有接缝和复杂结构失败保持原样。

## 复现

基线与候选应分别从对应源码构建 `graph_quality_bench`，复制到独立路径防止构建覆盖，并核对报告SHA；本次只读副本分别为 `.run/core-bench-before-20261014` 和 `.run/inpaint-bench-final-20261014`。

```sh
services/background-worker/.venv/bin/python scripts/check_inpaint_efficiency.py \
  --baseline .run/core-bench-before-20261014 \
  --candidate .run/inpaint-bench-final-20261014
```

脚本将任何 RGBA 差异或选区外/保护/alpha改变视为失败。执行成功表示本组等价与耗时核对完成，不能替代去物效果验收。

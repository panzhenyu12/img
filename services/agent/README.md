# 可选 DeepAgent 服务

运行 `make agent` 启动 http://127.0.0.1:8090 。设置 `OPENAI_API_KEY`、可选 `OPENAI_BASE_URL` / `IMG_AI_MODEL` / `IMG_MCP_URL`。密钥只读环境变量，不进入 Web 构建和 EditGraph。

`GET /healthz` 返回服务和配置状态；`POST /recommend` 接受 `{ "sessionId": "...", "prompt": "自然语言修图意图" }`。DeepAgent 通过 Go 的 Streamable HTTP MCP 查询能力和图片元数据、获取真实图像预览、提交可审阅提案。应用提案由 Web 确认并使用版本校验完成。

没有配置模型时接口返回 503；手工修图、WASM、Go 渲染和 MCP 都独立运行。Agent 已开放只读 `render_preview`（MCP ImageContent）给支持图像的模型；没有配置密钥时，真实模型理解和推荐效果尚未验证。未接入生成式去人/去物模型；手工仿制、修复与近邻填补独立可用。

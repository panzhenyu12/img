"""Optional AI service. Manual editing has no dependency on this process."""
import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Retouch DeepAgent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class Recommendation(BaseModel):
    sessionId: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=4000)


@app.get("/healthz")
def health():
    return {"status": "ok", "configured": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/recommend")
async def recommend(request: Recommendation):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "AI 模型未配置，手工编辑可以继续使用")
    from deepagents import create_deep_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI

    client = MultiServerMCPClient({
        "image_editor": {
            "transport": "http",
            "url": os.getenv("IMG_MCP_URL", "http://127.0.0.1:8080/mcp"),
        }
    })
    try:
        available = await client.get_tools()
        # The agent can inspect and propose. Applying changes is a UI action
        # with an explicit accepted flag and a revision checked by the Go API.
        allowed = {"inspect_asset", "render_preview", "propose_operations"}
        editor_tools = [tool for tool in available if tool.name in allowed]
        model = ChatOpenAI(
            model=os.getenv("IMG_AI_MODEL", "gpt-4.1-mini"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        agent = create_deep_agent(
            model=model,
            tools=editor_tools,
            system_prompt=(
                "You are a photo retouching assistant. Use inspect_asset to inspect "
                "the supplied session and its current revision, read the operation catalog, "
                "call render_preview when photo content is needed (it returns an image), "
                "then use propose_operations to create a small, reversible EditProposal. "
                "Use only supported types and parameter names in the catalog. "
                "Always provide operation id, type, version='1.0', enabled=true, "
                "origin='ai', label, params and controlPoints. "
                "All region, stroke and control-point coordinates are normalized 0..1. "
                "Slider values are percentages. Never apply edits, overwrite the source, "
                "claim to see pixels that have not been supplied, or claim generation "
                "or detection capabilities unavailable in the tools. "
                "Return the proposal ID and a brief Chinese explanation."
            ),
        )
        result = await asyncio.wait_for(agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"Session ID: {request.sessionId}\nRequest: {request.prompt}",
            }]
        }), timeout=90)
        message = result["messages"][-1]
        return {"status": "proposed", "message": message.content, "sessionId": request.sessionId}
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "AI 推荐超时，手工编辑可以继续使用") from exc
    except Exception as exc:
        # Logs may contain provider failures; never expose credentials or
        # network/internal file paths through an end-user response.
        import logging
        logging.getLogger(__name__).exception("DeepAgent recommendation failed")
        raise HTTPException(502, "AI 推荐暂不可用，手工编辑可以继续使用") from exc

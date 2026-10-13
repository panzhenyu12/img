import { useEffect, useRef, useState } from 'react';
import type { EditGraph, Operation } from '../core/operations';
interface Proposal { id: string; baseRevision: string; state?: string; status?: string; summary?: string; operations: Operation[] }
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8080').replace(/\/$/, '');
const AGENT_BASE = (import.meta.env.VITE_AGENT_BASE_URL ?? 'http://127.0.0.1:8090').replace(/\/$/, '');
async function responseJSON(response: Response) {
  const body = await response.json();
  if (!response.ok) throw new Error(body.error?.message ?? body.message ?? (typeof body.detail === 'string' ? body.detail : 'AI 服务当前不可用'));
  return body;
}
function fingerprint(graph: EditGraph): string {
  const sorted = (value: unknown): unknown => Array.isArray(value) ? value.map(sorted) : value && typeof value === 'object' ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, sorted(item)])) : value;
  return JSON.stringify(sorted({ sourceAssetId: graph.sourceAssetId, operations: (graph.operations ?? []).map((item) => ({ id: item.id, type: item.type, version: item.version ?? '1.0', enabled: item.enabled ?? true, params: item.params, controlPoints: item.controlPoints ?? [] })) }));
}
export function AIPanel({ sessionId, graph, onApply, onPreview, onNotice }: { sessionId: string | null; graph: EditGraph; onApply: (graph: EditGraph) => void; onPreview: (operations: Operation[]) => void; onNotice: (message: string) => void }) {
  const graphRef = useRef(graph); graphRef.current = graph;
  const [prompt, setPrompt] = useState(''); const [busy, setBusy] = useState(false); const [proposals, setProposals] = useState<Proposal[]>([]); const [previewId, setPreviewId] = useState<string | null>(null);
  const refresh = async () => { if (!sessionId) return; try { const body = await responseJSON(await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/proposals`)); setProposals(body.proposals ?? []); } catch (error) { onNotice(error instanceof Error ? error.message : '读取提案失败'); } };
  useEffect(() => { setProposals([]); setPreviewId(null); }, [sessionId]);
  useEffect(() => { setPreviewId(null); }, [graph]);
  const recommend = async () => {
    if (!sessionId || !prompt.trim()) return; setBusy(true); onPreview([]); setPreviewId(null);
    try {
      const session = await responseJSON(await fetch(`${API_BASE}/api/v1/sessions/${sessionId}`));
      const requestedGraph = graphRef.current; const requestedFingerprint = fingerprint(requestedGraph);
      const { source: _source, ...protocolGraph } = requestedGraph;
      await responseJSON(await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/graph`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ baseRevision: session.revision, editGraph: protocolGraph }) }));
      const body = await responseJSON(await fetch(`${AGENT_BASE}/recommend`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sessionId, prompt }) }));
      if (fingerprint(graphRef.current) !== requestedFingerprint) onNotice('生成期间图片编辑已改变，请重新生成 AI 提案'); else onNotice(body.message ?? 'AI 提案已生成'); await refresh();
    } catch (error) { onNotice(`${error instanceof Error ? error.message : 'AI 请求失败'}，可继续手工编辑`); }
    finally { setBusy(false); }
  };
  const accept = async (proposal: Proposal, accepted: boolean) => {
    if (!sessionId) return; setBusy(true); const localFingerprint = fingerprint(graphRef.current);
    try {
      let requestRevision = proposal.baseRevision; if (!accepted) { const session = await responseJSON(await fetch(`${API_BASE}/api/v1/sessions/${sessionId}`)); requestRevision = session.revision; }
      if (accepted) { const currentSession = await responseJSON(await fetch(`${API_BASE}/api/v1/sessions/${sessionId}`)); if (fingerprint(graphRef.current) !== fingerprint({ ...currentSession.editGraph, operations: currentSession.editGraph.operations ?? [] })) { onPreview([]); setPreviewId(null); throw new Error('手工编辑已改变，请重新生成提案后应用'); } }
      const body = await responseJSON(await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/proposals/${proposal.id}/${accepted ? 'apply' : 'reject'}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ accepted, baseRevision: requestRevision }) }));
      if (accepted && fingerprint(graphRef.current) !== localFingerprint) throw new Error('应用期间编辑已改变，已保留本地手工编辑，请重新生成提案');
      if (accepted) onApply({ ...body.editGraph, source: graphRef.current.source, operations: body.editGraph.operations ?? [] }); onPreview([]); setPreviewId(null);
      onNotice(accepted ? 'AI 提案已应用，可在历史中撤销' : '已拒绝 AI 提案'); await refresh();
    } catch (error) { onNotice(error instanceof Error ? error.message : '提案更新失败'); }
    finally { setBusy(false); }
  };
  return <details className="ai-panel"><summary>AI 助手 <span>可选</span></summary><p className="panel-tip">描述希望调整的效果。AI 生成操作提案，确认后加入历史。</p><textarea aria-label="AI 修图指令" value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="例如：轻微瘦脸，低饱和胶片色调" /><div className="button-grid"><button disabled={busy || !sessionId || !prompt.trim()} onClick={recommend}>{busy ? '处理中…' : '生成提案'}</button><button disabled={busy || !sessionId} onClick={refresh}>刷新提案</button></div>{proposals.filter((proposal) => (proposal.state ?? proposal.status) === 'proposed').map((proposal) => <div className="proposal-card" key={proposal.id}><p>{proposal.summary ?? `${proposal.operations.length} 项编辑`}</p><div className="small-note">{proposal.operations.map((item) => item.label ?? item.type).join(' · ')}</div><div className="button-grid"><button disabled={busy} onClick={() => { const isPreview = previewId === proposal.id; onPreview(isPreview ? [] : proposal.operations); setPreviewId(isPreview ? null : proposal.id); }}>{previewId === proposal.id ? '取消预览' : '预览效果'}</button><button disabled={busy} onClick={() => accept(proposal, true)}>接受</button><button disabled={busy} onClick={() => accept(proposal, false)}>拒绝</button></div></div>)}</details>;
}

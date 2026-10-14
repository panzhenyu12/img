import type { Operation } from '../core/operations';
import { TOOLS, type ToolId } from '../core/registry';

const TOOL_GROUPS: Array<{ id: string; label: string; tools: ToolId[] }> = [
  { id: 'start', label: '基础', tools: ['browse', 'transform'] },
  { id: 'portrait', label: '人像美颜', tools: ['face', 'feature', 'skin', 'body', 'clothing'] },
  { id: 'reshape', label: '局部塑形', tools: ['liquify', 'grid'] },
  { id: 'repair', label: '修复与移除', tools: ['repair'] },
  { id: 'color', label: '色彩与风格', tools: ['color', 'curves', 'hsl', 'film'] },
  { id: 'mask', label: '蒙版', tools: ['mask'] },
];

export function Sidebar({ tool, onTool, operations, selectedId, onSelect, onToggle, onDelete, onMove }: {
  tool: ToolId; onTool: (tool: ToolId) => void; operations: Operation[]; selectedId: string | null;
  onSelect: (operation: Operation) => void; onToggle: (id: string) => void; onDelete: (id: string) => void; onMove: (id: string, direction: number) => void;
}) {
  const toolById = Object.fromEntries(TOOLS.map((item) => [item.id, item]));
  return <aside className="left-panel">
    <div className="panel-title">编辑工具</div>
    <nav className="tool-list" aria-label="编辑工具分组">
      {TOOL_GROUPS.map((group) => <section className="tool-group" key={group.id} aria-label={group.label}>
        <h3 className="tool-group-label">{group.label}</h3>
        <div className="tool-group-items">{group.tools.map((id) => { const item = toolById[id]; return <button key={item.id} className={`tool-item ${tool === item.id ? 'active' : ''}`} aria-label={item.label} onClick={() => onTool(item.id)} type="button"><span aria-hidden="true">{item.icon}</span><span>{item.label}</span></button>; })}</div>
      </section>)}
    </nav>
    <div className="panel-divider" />
    <div className="panel-title history-heading">操作图层 <span className="counter">{operations.length}</span></div>
    <div className="operation-list">{!operations.length && <div className="empty-history">尚未编辑<br /><small>选择工具，开始修图</small></div>}{[...operations].reverse().map((operation) => <div key={operation.id} className={`operation-row ${selectedId === operation.id ? 'selected' : ''} ${!operation.enabled ? 'muted' : ''}`}><button className="operation-eye" aria-label={`${operation.enabled ? '关闭' : '启用'}${operation.label ?? operation.type}`} onClick={() => onToggle(operation.id)}>{operation.enabled ? '◉' : '○'}</button><button className="operation-name" onClick={() => onSelect(operation)} title={operation.type}>{operation.label ?? operation.type}</button><div className="operation-actions"><button title="上移图层" aria-label="上移图层" onClick={() => onMove(operation.id, 1)}>↑</button><button title="下移图层" aria-label="下移图层" onClick={() => onMove(operation.id, -1)}>↓</button><button title="删除操作" aria-label="删除操作" onClick={() => onDelete(operation.id)}>×</button></div></div>)}</div>
  </aside>;
}

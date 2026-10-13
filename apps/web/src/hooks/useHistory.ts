import { useCallback, useRef, useState } from 'react';
import { cloneGraph, EMPTY_GRAPH, type EditGraph } from '../core/operations';
export function useHistory() {
  const [graph, setGraph] = useState<EditGraph>(EMPTY_GRAPH);
  const [counts, setCounts] = useState({ past: 0, future: 0 });
  const data = useRef({ graph: EMPTY_GRAPH, past: [] as EditGraph[], future: [] as EditGraph[], mergeKey: '', time: 0 });
  const publish = useCallback(() => { setGraph(data.current.graph); setCounts({ past: data.current.past.length, future: data.current.future.length }); }, []);
  const replace = useCallback((next: EditGraph) => { const state = data.current; if (state.graph.sourceAssetId !== next.sourceAssetId && state.graph.source?.name === next.source?.name) { state.past = state.past.map((item) => ({ ...item, sourceAssetId: next.sourceAssetId })); state.future = state.future.map((item) => ({ ...item, sourceAssetId: next.sourceAssetId })); } state.graph = next; publish(); }, [publish]);
  const commit = useCallback((update: EditGraph | ((graph: EditGraph) => EditGraph), mergeKey = '') => {
    const state = data.current; const current = state.graph;
    const next = typeof update === 'function' ? update(cloneGraph(current)) : update;
    if (JSON.stringify(next) === JSON.stringify(current)) return;
    const now = Date.now();
    if (!mergeKey || state.mergeKey !== mergeKey || now - state.time > 700) state.past = [...state.past.slice(-79), cloneGraph(current)];
    state.mergeKey = mergeKey; state.time = now; state.future = []; state.graph = next; publish();
  }, [publish]);
  const reset = useCallback((next: EditGraph) => { data.current = { graph: next, past: [], future: [], mergeKey: '', time: 0 }; publish(); }, [publish]);
  const undo = useCallback(() => { const state = data.current; const previous = state.past.pop(); if (!previous) return; state.future.unshift(cloneGraph(state.graph)); state.graph = previous; state.mergeKey = ''; publish(); }, [publish]);
  const redo = useCallback(() => { const state = data.current; const next = state.future.shift(); if (!next) return; state.past.push(cloneGraph(state.graph)); state.graph = next; state.mergeKey = ''; publish(); }, [publish]);
  return { graph, commit, reset, replace, undo, redo, canUndo: counts.past > 0, canRedo: counts.future > 0 };
}

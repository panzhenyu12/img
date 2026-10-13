import type { EditGraph } from '../core/operations';

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8080').replace(/\/$/, '');

export type RemoteAsset = { id: string; filename: string; mimeType: string; width: number; height: number; size: number };

async function expectJSON<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`API ${response.status}: ${message || response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function createSession(): Promise<{ id: string }> {
  const response = await fetch(`${API_BASE}/api/v1/sessions`, { method: 'POST' });
  return expectJSON(response);
}

export async function uploadAsset(sessionId: string, file: File): Promise<RemoteAsset> {
  const body = new FormData();
  body.append('file', file, file.name);
  const response = await fetch(`${API_BASE}/api/v1/sessions/${encodeURIComponent(sessionId)}/assets`, {
    method: 'POST',
    body,
  });
  return expectJSON(response);
}

export async function renderGraph(sessionId: string, graph: EditGraph): Promise<Blob> {
  const { source: _source, ...protocolGraph } = graph;
  const response = await fetch(`${API_BASE}/api/v1/sessions/${encodeURIComponent(sessionId)}/render`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ editGraph: protocolGraph, format: 'png' }),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`API ${response.status}: ${message || response.statusText}`);
  }
  return response.blob();
}

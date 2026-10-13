/** Stable edit protocol consumed by Rust WASM, native render and AI tools. */
export type OperationType =
  | 'identity' | 'crop' | 'rotate' | 'flip' | 'resize' | 'perspective'
  | 'warp.grid' | 'warp.liquify' | 'face.adjust' | 'feature.adjust'
  | 'skin.retouch' | 'body.adjust' | 'clothing.adjust' | 'repair.clone'
  | 'repair.heal' | 'repair.inpaint' | 'paint.brush' | 'color.basic' | 'color.curves' | 'color.hsl' | 'filter.film' | 'filter.lut';

export interface Point { x: number; y: number }
export interface Region extends Point { width: number; height: number }
/** Detector alpha; normalized full-image coordinates, row-major byte data. */
export interface RasterMask {
  shape: 'raster'; width: number; height: number; data: number[];
  from?: Point; to?: Point; region?: Region; points?: Point[];
  strokes?: BrushStroke[]; invert?: boolean; feather?: number;
}
export interface ControlPoint extends Point { inX?: number; inY?: number; outX?: number; outY?: number }
export type SelectionShape = 'rect' | 'ellipse' | 'polygon' | 'brush' | 'linear' | 'radial';
export interface BrushStroke extends Point {
  toX?: number; toY?: number; sourceX?: number; sourceY?: number;
  radius: number; strength: number; mode?: string; direction?: number; erase?: boolean;
}
export interface InlineMask {
  shape: SelectionShape;
  from?: Point; to?: Point;
  region?: Region;
  points?: Point[];
  strokes?: BrushStroke[];
  invert: boolean;
  feather: number;
}
export type EditableMask = InlineMask | RasterMask;
export type BlendMode = 'normal' | 'multiply' | 'screen' | 'overlay' | 'soft-light';
export interface Operation<TParams = Record<string, unknown>> {
  id: string; type: OperationType; version: string; enabled: boolean;
  origin?: 'manual' | 'ai' | 'imported'; label?: string; explanation?: string;
  provenance?: { actorType: 'user' | 'ai' | 'system'; actorId?: string; toolName?: string; model?: string; requestId?: string; confidence?: number } | null;
  maskId: string | null; targetId: string | null; params: TParams;
  controlPoints: ControlPoint[]; engineVersion: string;
}
export interface SourceMetadata { name: string; width: number; height: number }
export interface EditGraph { version: string; sourceAssetId: string; operations: Operation[]; source: SourceMetadata | null }
export const EMPTY_GRAPH: EditGraph = { version: '1.0', sourceAssetId: '', source: null, operations: [] };
export const WHOLE_IMAGE: Region = { x: 0, y: 0, width: 1, height: 1 };
export function createOperation(type: OperationType, params: Record<string, unknown>, label?: string): Operation {
  return { id: crypto.randomUUID(), type, version: '1.0', enabled: true, origin: 'manual', label,
    maskId: null, targetId: null, params: { opacity: 100, blendMode: 'normal', ...params }, controlPoints: [], engineVersion: 'image-core-0.2.0' };
}
export function cloneGraph(graph: EditGraph): EditGraph { return structuredClone(graph); }
export function normalizedRegion(a: Point, b: Point): Region {
  return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), width: Math.abs(a.x - b.x), height: Math.abs(a.y - b.y) };
}
export function clampUnit(value: number): number { return Math.max(0, Math.min(1, value)); }
export function maskForRegion(region: Region, shape: SelectionShape, points?: Point[]): InlineMask {
  return { shape, region, points, strokes: [], invert: false, feather: 0 };
}

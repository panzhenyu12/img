import { createOperation, type ControlPoint, type EditGraph, type Operation, type OperationType } from './operations';

const OPERATION_TYPES: OperationType[] = ['identity', 'crop', 'rotate', 'flip', 'resize', 'perspective', 'warp.grid', 'warp.liquify', 'face.adjust', 'feature.adjust', 'skin.retouch', 'body.adjust', 'clothing.adjust', 'repair.clone', 'repair.heal', 'repair.inpaint', 'paint.brush', 'color.basic', 'color.curves', 'color.hsl', 'filter.film', 'filter.lut'];
const isRecord = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);

/** Import only operation data; the currently loaded original remains authoritative. */
export function normalizeProject(value: unknown, current: EditGraph): EditGraph {
  if (!isRecord(value) || !Array.isArray(value.operations)) throw new Error('操作工程必须包含 operations 数组');
  if (value.version !== undefined && value.version !== '1.0') throw new Error('当前仅支持 1.0 版本的操作工程');
  if (value.operations.length > 256) throw new Error('操作工程最多支持 256 个操作');
  const ids = new Set<string>();
  const operations = value.operations.map((input, index): Operation => {
    const invalid = (message: string): never => { throw new Error(`第 ${index + 1} 个操作：${message}`); };
    if (!isRecord(input)) return invalid('格式无效');
    const type = input.type === 'core.identity' ? 'identity' : input.type;
    if (typeof type !== 'string' || !OPERATION_TYPES.includes(type as OperationType)) return invalid('不支持的工具类型');
    if (input.version !== undefined && input.version !== '1.0') return invalid('仅支持 1.0 版本');
    if (input.enabled !== undefined && typeof input.enabled !== 'boolean') return invalid('enabled 必须为布尔值');
    if (input.params !== undefined && !isRecord(input.params)) return invalid('params 必须为对象');
    const params = structuredClone(input.params ?? {}) as Record<string, unknown>;
    const rawControls = (type === 'warp.grid' ? params.points ?? params.controlPoints ?? input.controlPoints : input.controlPoints) ?? [];
    if (!Array.isArray(rawControls)) return invalid('controlPoints 必须为数组');
    if (rawControls.length > 1024) return invalid('控制点最多为 1024 个');
    const controlPoints = rawControls.map((point): ControlPoint => {
      if (!isRecord(point)) return invalid('控制点格式无效');
      const x = point.x ?? point.inX; const y = point.y ?? point.inY;
      if (typeof x !== 'number' || !Number.isFinite(x) || typeof y !== 'number' || !Number.isFinite(y)) return invalid('控制点必须包含有效坐标');
      const normalized: ControlPoint = { x, y };
      for (const key of ['inX', 'inY', 'outX', 'outY'] as const) {
        const coordinate = point[key];
        if (coordinate !== undefined) {
          if (typeof coordinate !== 'number' || !Number.isFinite(coordinate)) return invalid('控制点坐标无效');
          normalized[key] = coordinate;
        }
      }
      return normalized;
    });
    if (type === 'warp.grid') { delete params.points; delete params.controlPoints; }
    const defaults = createOperation(type as OperationType, params);
    let id = typeof input.id === 'string' && input.id.trim() ? input.id : defaults.id;
    if (ids.has(id)) id = defaults.id;
    ids.add(id);
    return {
      ...defaults, id, enabled: (input.enabled as boolean | undefined) ?? true, origin: 'imported',
      label: typeof input.label === 'string' ? input.label : type,
      explanation: typeof input.explanation === 'string' ? input.explanation : undefined,
      maskId: typeof input.maskId === 'string' ? input.maskId : null,
      targetId: typeof input.targetId === 'string' ? input.targetId : null,
      controlPoints,
    };
  });
  return { ...current, version: '1.0', operations };
}

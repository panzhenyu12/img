import { useEffect, useRef, useState } from 'react';
import type { ChangeEvent, DragEvent } from 'react';
import { createSession, uploadAsset } from './api/client';
import { Sidebar } from './components/Sidebar';
import { CanvasStage, type CanvasGesture, type InteractionMode } from './components/CanvasStage';
import { ToolPanel, type BrushSettings, type DetectedRegion } from './components/ToolPanel';
import { AIPanel } from './components/AIPanel';
import { useHistory } from './hooks/useHistory';
import { decodeImage, pixelsToCanvas, renderImage, type ImagePixels } from './core/engine';
import { FILM_PRESETS, TOOL_BY_ID, TOOLS, defaultsFor, portraitParametersFor, type PortraitCapabilities, type ToolId } from './core/registry';
import { createOperation, maskForRegion, WHOLE_IMAGE, type BrushStroke, type ControlPoint, type EditGraph, type InlineMask, type Operation, type OperationType, type Point, type Region, type SelectionShape, type RasterMask } from './core/operations';
import { parseCube } from './core/lut';
import { detectPeople, type SemanticMask } from './core/landmarks';
import { normalizeProject } from './core/project';

const TOOL_TYPES = new Map(TOOLS.filter((tool) => tool.type).map((tool) => [tool.type!, tool.id]));
const MODES: Partial<Record<ToolId, InteractionMode>> = { browse: 'pan', face: 'rect', feature: 'rect', skin: 'rect', body: 'rect', clothing: 'rect', liquify: 'liquify', repair: 'heal', grid: 'grid', mask: 'brush' };
const GRID: ControlPoint[] = Array.from({ length: 9 }, (_, index) => { const x = (index % 3) / 2; const y = Math.floor(index / 3) / 2; return { x, y, inX: x, inY: y, outX: x, outY: y }; });

function App() {
  const history = useHistory(); const { graph, commit, reset, replace, undo, redo, canUndo, canRedo } = history;
  const [source, setSource] = useState<ImagePixels | null>(null); const [preview, setPreview] = useState<ImagePixels | null>(null);
  const [filmThumbnails, setFilmThumbnails] = useState<Record<string, string>>({});
  const [filename, setFilename] = useState(''); const [notice, setNotice] = useState('导入照片开始编辑'); const [engineError, setEngineError] = useState('');
  const [tool, setTool] = useState<ToolId>('browse'); const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mode, setMode] = useState<InteractionMode>('pan'); const [region, setRegion] = useState<Region>(WHOLE_IMAGE); const [regionShape, setRegionShape] = useState<SelectionShape>('rect');
  const [selectionPoints, setSelectionPoints] = useState<Point[]>([]); const [regions, setRegions] = useState<Partial<Record<ToolId, Region>>>({});
  const [brush, setBrush] = useState<BrushSettings>({ radius: .035, strength: 60, liquidMode: 'push', twirlDirection: 1, symmetric: false, repairMode: 'heal' });
  const [paintColor, setPaintColor] = useState('#c56a74');
  const [sample, setSample] = useState<Point | null>(null); const [detected, setDetected] = useState<DetectedRegion[]>([]); const [detecting, setDetecting] = useState(false);
  const [regionProtection, setRegionProtection] = useState<InlineMask | undefined>();
  const [regionSubjectMask, setRegionSubjectMask] = useState<RasterMask | undefined>();
  const [regionLandmarks, setRegionLandmarks] = useState<Point[]>([]); const [regionPart, setRegionPart] = useState<string | null>(null); const [regionSemanticMasks, setRegionSemanticMasks] = useState<Record<string, SemanticMask>>({});
  const [proposalPreview, setProposalPreview] = useState<Operation[]>([]); const [exportFormat, setExportFormat] = useState<'png' | 'jpeg'>('png'); const [quality, setQuality] = useState(92);
  const [busy, setBusy] = useState(false); const [zoom, setZoom] = useState(1); const [compare, setCompare] = useState(false); const [compareSplit, setCompareSplit] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null); const [exporting, setExporting] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null); const graphInput = useRef<HTMLInputElement>(null); const request = useRef(0); const projectRequest = useRef(0); const sourceToken = useRef(''); const graphRef = useRef(graph); graphRef.current = graph;
  const selected = graph.operations.find((operation) => operation.id === selectedId);
  const definition = TOOL_BY_ID[tool];
  const currentOperation = selected && (selected.type === definition.type || (selected.type === 'paint.brush' && ['skin', 'feature'].includes(tool)) || ['mask', 'browse', 'transform', 'grid', 'repair', 'film'].includes(tool)) ? selected : [...graph.operations].reverse().find((operation) => operation.type === definition.type);
  const operation = currentOperation;
  const activeFaceMasks = (operation?.params.semanticMasks as Record<string, SemanticMask> | undefined) ?? regionSemanticMasks;
  const hasFaceSemantics = ['leftEye', 'rightEye', 'leftBrow', 'rightBrow', 'lips'].every(name => !!activeFaceMasks[name]);
  const effectiveRegion = (['face', 'feature', 'skin', 'body', 'clothing'].includes(tool) ? operation?.params.region as Region | undefined : undefined) ?? regions[tool] ?? region;
  const portraitCapabilities: PortraitCapabilities = { region: effectiveRegion, shape: (operation?.params.mask as InlineMask | undefined)?.shape ?? regionShape, points: (operation?.params.mask as InlineMask | undefined)?.points ?? selectionPoints, semanticMasks: activeFaceMasks, bodyPart: typeof operation?.params.bodyPart === 'string' ? operation.params.bodyPart : regionPart ?? undefined, subjectMask: (operation?.params.subjectMask as RasterMask | undefined) ?? regionSubjectMask };

  useEffect(() => {
    if (!source) return;
    const id = ++request.current;
    const controller = new AbortController();
    setBusy(true);
    const timer = window.setTimeout(() => {
      renderImage(source, proposalPreview.length ? { ...graph, operations: [...graph.operations, ...proposalPreview] } : graph, { signal: controller.signal, kind: 'preview' }).then((image) => { if (id === request.current && !controller.signal.aborted) { setPreview(image); setEngineError(''); setBusy(false); } }).catch((error) => { if (id === request.current && !controller.signal.aborted) { if (!(error instanceof Error && error.name === 'AbortError')) setEngineError(error instanceof Error ? error.message : 'Rust 渲染失败'); setBusy(false); } });
    }, 45);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [source, graph, proposalPreview]);

  useEffect(() => { setProposalPreview([]); }, [graph]);
  useEffect(() => { if (!source || tool !== 'film' || Object.keys(filmThumbnails).length) return; const controller = new AbortController(); const original = pixelsToCanvas(source); const canvas = document.createElement('canvas'); const ratio = Math.min(120 / source.width, 120 / source.height); canvas.width = Math.max(1, Math.round(source.width * ratio)); canvas.height = Math.max(1, Math.round(source.height * ratio)); const context = canvas.getContext('2d'); if (!context) return; context.drawImage(original, 0, 0, canvas.width, canvas.height); const thumbnail: ImagePixels = { width: canvas.width, height: canvas.height, pixels: new Uint8Array(context.getImageData(0, 0, canvas.width, canvas.height).data) }; void (async () => { const images: Record<string, string> = {}; for (const preset of FILM_PRESETS) { try { const image = await renderImage(thumbnail, { ...graphRef.current, operations: [createOperation('filter.film', { preset: preset.id, intensity: 75, seed: 42 }, preset.name)] }, { signal: controller.signal, kind: 'background' }); images[preset.id] = pixelsToCanvas(image).toDataURL('image/jpeg', .8); } catch { return; } if (controller.signal.aborted) return; } if (!controller.signal.aborted) setFilmThumbnails(images); })(); return () => controller.abort(); }, [source, tool, filmThumbnails]);
  const importImage = async (file: File) => {
    if (!file.type.startsWith('image/')) { setNotice('请选择 JPG、PNG 或 WebP 图片'); return; }
    try {
      const token = crypto.randomUUID(); sourceToken.current = token;
      const image = await decodeImage(file); if (sourceToken.current !== token) return;
      setSource(image); setPreview(image); setFilmThumbnails({}); setFilename(file.name); setSessionId(null); setSelectedId(null); setTool('browse'); setMode('pan'); setRegion(WHOLE_IMAGE); setRegions({}); setRegionLandmarks([]); setRegionPart(null); setRegionSemanticMasks({}); setRegionSubjectMask(undefined); setRegionProtection(undefined); setSample(null); setDetected([]); setZoom(1); setCompare(false); setCompareSplit(false); setProposalPreview([]);
      const next: EditGraph = { version: '1.0', sourceAssetId: `local-${token}`, source: { name: file.name, width: image.width, height: image.height }, operations: [] };
      reset(next); setNotice(`已导入 ${image.width} × ${image.height}`);
      try { const session = await createSession(); const asset = await uploadAsset(session.id, file); if (sourceToken.current !== token) return; setSessionId(session.id); replace({ ...graphRef.current, sourceAssetId: asset.id }); setNotice('图片已就绪 · Rust 图像核心'); }
      catch { if (sourceToken.current === token) setNotice('本地修图模式 · 可正常编辑与导出'); }
    } catch (error) { setNotice(error instanceof Error ? error.message : '导入失败'); }
  };
  const selectTool = (id: ToolId) => {
    const family = (value: ToolId) => ['face', 'feature', 'skin'].includes(value) ? 'face' : value === 'body' ? 'body' : value === 'clothing' ? 'clothing' : 'other';
    if (family(id) !== family(tool)) { setRegionLandmarks([]); setRegionPart(null); setRegionSemanticMasks({}); setRegionSubjectMask(undefined); setRegionProtection(undefined); setRegion(regions[id] ?? WHOLE_IMAGE); setRegionShape('rect'); setSelectionPoints([]); }
    setTool(id); setMode(MODES[id] ?? 'pan');
    if (!['mask', 'browse'].includes(id)) { const existing = [...graph.operations].reverse().find((item) => item.type === TOOL_BY_ID[id].type); setSelectedId(existing?.id ?? null); }
    if (regions[id]) setRegion(regions[id]!);
  };
  const selectOperation = (item: Operation) => {
    setSelectedId(item.id); const id = TOOL_TYPES.get(item.type) ?? (['crop', 'rotate', 'flip', 'resize'].includes(item.type) ? 'transform' : item.type === 'paint.brush' ? 'feature' : item.type === 'filter.lut' ? 'film' : item.type === 'repair.clone' || item.type === 'repair.inpaint' ? 'repair' : 'browse');
    setTool(id); setMode(MODES[id] ?? 'pan'); if (item.type === 'perspective') { setTool('grid'); setMode('perspective'); }
    if (item.type === 'repair.clone') { setTool('repair'); setMode('clone'); }
    if (item.params.region) setRegion(item.params.region as Region);
    setRegionLandmarks(Array.isArray(item.params.landmarks) ? (item.params.landmarks as Point[]) : []);
    setRegionSemanticMasks((item.params.semanticMasks as Record<string, SemanticMask> | undefined) ?? {});
    setRegionSubjectMask(item.params.subjectMask as RasterMask | undefined);
    setRegionProtection(item.params.protectMask as InlineMask | undefined);
    setRegionPart(typeof item.params.bodyPart === 'string' ? item.params.bodyPart : null);
    const itemMask = item.params.mask as InlineMask | undefined; if (itemMask) { setRegionShape(itemMask.shape); setSelectionPoints(itemMask.points ?? []); } else { setRegionShape('rect'); setSelectionPoints([]); }
  };
  const append = (type: OperationType, params: Record<string, unknown>, label?: string) => {
    if (!source) { setNotice('请先导入图片'); return null; }
    if (graphRef.current.operations.length >= 256) { setNotice('工程最多支持 256 个操作，请删除不需要的图层'); return null; }
    if (Array.isArray(params.strokes) && params.strokes.length > 4096) { setNotice('单个图层最多支持 4096 个笔画，请缩小选区或增大笔刷'); return null; }
    if ((params.mask as InlineMask | undefined)?.shape === 'polygon' && !(params.mask as InlineMask).points?.length) params = { ...params, mask: { ...(params.mask as InlineMask), points: selectionPoints } };
    if (type === 'warp.liquify' && operation?.type === 'warp.liquify') { const inherited: Record<string, unknown> = {}; for (const key of ['mask', 'protectMask'] as const) if (operation.params[key]) inherited[key] = structuredClone(operation.params[key]); params = { ...inherited, ...params }; }
    const localType = ['face.adjust', 'feature.adjust', 'skin.retouch', 'body.adjust', 'clothing.adjust'].includes(type);
    const landmarkType = ['face.adjust', 'feature.adjust', 'body.adjust', 'clothing.adjust'].includes(type);
    const semanticParams = ['face.adjust', 'feature.adjust', 'skin.retouch'].includes(type) && Object.keys(regionSemanticMasks).length ? { semanticMasks: structuredClone(regionSemanticMasks) } : {};
    const subjectParams = ['face.adjust', 'body.adjust'].includes(type) && regionSubjectMask ? { subjectMask: structuredClone(regionSubjectMask) } : {};
    params = { ...params, ...subjectParams, ...(type === 'body.adjust' && regionProtection ? { protectMask: structuredClone(regionProtection) } : {}) };
    const withLandmarks = landmarkType && regionLandmarks.length
      ? { ...params, landmarks: regionLandmarks.map(({ x, y }) => ({ x, y })), ...semanticParams, ...(type === 'body.adjust' && regionPart ? { bodyPart: regionPart } : {}) }
      : (type === 'body.adjust' && regionPart ? { ...params, ...semanticParams, bodyPart: regionPart } : { ...params, ...semanticParams });
    const prepared: Record<string, unknown> = localType && regionShape !== 'rect'
      ? { ...withLandmarks, mask: { ...maskForRegion(effectiveRegion, regionShape, selectionPoints), feather: regionShape === 'polygon' ? 8 : 0 } }
      : withLandmarks;
    if (localType) {
      const targetTool = TOOL_TYPES.get(type)!;
      const capabilities = { ...portraitCapabilities, region: prepared.region as Region | undefined, semanticMasks: prepared.semanticMasks as PortraitCapabilities['semanticMasks'], mask: prepared.mask as InlineMask | undefined, bodyPart: prepared.bodyPart as string | undefined, subjectMask: prepared.subjectMask as RasterMask | undefined };
      if (!portraitParametersFor(targetTool, capabilities).length) { setNotice(targetTool === 'face' ? '请先检测人脸，或用套索 / 多边形标记真实脸部轮廓' : targetTool === 'body' ? '请先检测身体并选择部位；手工塑形可使用液化画笔' : '请先检测对应五官或标记皮肤 / 衣物局部区域'); return null; }
    }
    const item = createOperation(type, prepared, label ?? definition.label);
    if (type === 'warp.grid') item.controlPoints = structuredClone(GRID);
    commit((current) => ({ ...current, operations: [...current.operations, item] })); setSelectedId(item.id); setNotice(`已添加${item.label}`); return item;
  };
  const patchOperation = (id: string, update: Partial<Operation> | ((item: Operation) => Operation), mergeKey = '') => commit((current) => ({ ...current, operations: current.operations.map((item) => item.id === id ? typeof update === 'function' ? update(item) : { ...item, ...update } : item) }), mergeKey);
  const changeParam = (key: string, value: unknown) => {
    if (!source) return;
    if (definition.region && !['opacity', 'blendMode', 'mask', 'protectMask'].includes(key)) {
      const allowed = portraitParametersFor(tool, portraitCapabilities).map(item => item.key);
      if (!allowed.length) { setNotice(tool === 'face' ? '请先检测人脸，或用套索 / 多边形标记真实脸部轮廓' : tool === 'body' ? '请先检测身体并选择部位；手工塑形可使用液化画笔' : '请先检测对应五官或标记局部区域'); return; }
      if (key === 'adjustments') value = Object.fromEntries(Object.entries(value as Record<string, unknown>).filter(([name]) => allowed.includes(name)));
      else if (!allowed.includes(key)) return;
    }
    if (tool === 'color' && key === 'blackPoint') value = Math.max(0, Math.min(Number(operation?.params.whitePoint ?? 100) - .1, Number(value)));
    if (tool === 'color' && key === 'whitePoint') value = Math.min(100, Math.max(Number(operation?.params.blackPoint ?? 0) + .1, Number(value)));
    let target = operation;
    const common = ['opacity', 'blendMode', 'mask', 'protectMask'].includes(key); const matchesType = target?.type === definition.type || (target?.type === 'filter.lut' && key === 'amount');
    if (!target || (!common && !matchesType)) {
      if (!definition.type) return;
      if (graphRef.current.operations.length >= 256) { setNotice('工程最多支持 256 个操作，请删除不需要的图层'); return; }
      const defaults = defaultsFor(definition, effectiveRegion);
      const selectionMask = definition.region && regionShape !== 'rect'
        ? { mask: { ...maskForRegion(effectiveRegion, regionShape, selectionPoints), feather: regionShape === 'polygon' ? 8 : 0 } }
        : {};
      const landmarkType = ['face.adjust', 'feature.adjust', 'body.adjust', 'clothing.adjust'].includes(definition.type ?? '');
      const landmarks = landmarkType && regionLandmarks.length ? { landmarks: regionLandmarks.map(({ x, y }) => ({ x, y })) } : {};
      const semanticMasks = ['face.adjust', 'feature.adjust', 'skin.retouch'].includes(definition.type ?? '') && Object.keys(regionSemanticMasks).length ? { semanticMasks: structuredClone(regionSemanticMasks) } : {};
      const bodyPart = definition.type === 'body.adjust' && regionPart ? { bodyPart: regionPart } : {};
      const subjectMask = ['face.adjust', 'body.adjust'].includes(definition.type ?? '') && regionSubjectMask ? { subjectMask: structuredClone(regionSubjectMask) } : {};
      target = createOperation(definition.type, { ...defaults, [key]: value, ...selectionMask, ...landmarks, ...semanticMasks, ...bodyPart, ...subjectMask, ...(definition.type === 'body.adjust' && regionProtection ? { protectMask: structuredClone(regionProtection) } : {}) }, definition.label);
      commit((current) => ({ ...current, operations: [...current.operations, target!] }), `${target.id}:${key}`); setSelectedId(target.id);
    } else patchOperation(target.id, (item) => ({ ...item, params: { ...item.params, [key]: value } }), `${target.id}:${key}`);
  };
  const setTargetRegion = (next: Region, shape: SelectionShape = regionShape, points = selectionPoints, landmarks?: Point[], part?: string, semanticMasks?: Record<string, SemanticMask>, subjectMask?: RasterMask, protectMask?: InlineMask) => {
    setRegion(next); setRegionShape(shape); setSelectionPoints(points); setRegionLandmarks(landmarks ?? []); setRegionPart(part ?? null); setRegionSemanticMasks(semanticMasks ?? {}); setRegionSubjectMask(subjectMask); setRegionProtection(protectMask); setRegions((current) => ({ ...current, [tool]: next }));
    if (operation && ['face', 'feature', 'skin', 'body', 'clothing'].includes(tool)) {
      patchOperation(operation.id, (item) => { const params: Record<string, unknown> = { ...item.params, region: next }; if (landmarks?.length) params.landmarks = landmarks.map(({ x, y }) => ({ x, y })); else delete params.landmarks; if (semanticMasks && Object.keys(semanticMasks).length) params.semanticMasks = structuredClone(semanticMasks); else delete params.semanticMasks; if (tool === 'body' && protectMask) params.protectMask = structuredClone(protectMask); if (['face', 'body'].includes(tool) && subjectMask) params.subjectMask = structuredClone(subjectMask); else delete params.subjectMask; if (tool === 'body' && part) params.bodyPart = part; else if (tool === 'body') delete params.bodyPart; if (shape !== 'rect') Object.assign(params, { mask: { ...maskForRegion(next, shape, points), feather: shape === 'polygon' ? 8 : 0 } }); else delete params.mask; return { ...item, params }; });
    }
  };
  const setMask = (key: 'mask' | 'protectMask', mask: InlineMask | null) => {
    if (!operation) { setNotice('请先选择一个操作图层'); return; }
    if ((mask?.strokes?.length ?? 0) > 4096) { setNotice('单个蒙版最多支持 4096 个笔画，请清除蒙版后重新标记'); return; }
    const next = mask?.shape === 'polygon' ? { ...mask, points: mask.points ?? selectionPoints } : mask;
    patchOperation(operation.id, (item) => { const params = { ...item.params }; if (next) params[key] = next; else delete params[key]; return { ...item, params }; }, `${operation.id}:${key}`);
  };
  const gesture = (event: CanvasGesture) => {
    if (!source || !event.points.length) return;
    if (event.mode === 'linear' || event.mode === 'radial') { setMask('mask', { shape: event.mode, region: event.region, from: event.points[0], to: event.points.at(-1), invert: false, feather: 0 }); return; }
    if (['rect', 'ellipse', 'lasso', 'polygon'].includes(event.mode)) { if (event.region.width < .003 || event.region.height < .003) return; setTargetRegion(event.region, event.shape, event.points); setNotice('区域已标记'); return; }
    if (event.mode === 'paint' || event.mode === 'paintErase') {
      const existing = selected?.type === 'paint.brush' ? selected : [...graph.operations].reverse().find((item) => item.type === 'paint.brush');
      if (event.mode === 'paintErase' && !existing) { setNotice('先建立绘色图层，再使用橡皮'); return; }
      const strokes = event.points.map((point) => ({ ...point, radius: brush.radius, strength: brush.strength, erase: event.mode === 'paintErase' }));
      if (((existing?.params.strokes as BrushStroke[] | undefined)?.length ?? 0) + strokes.length > 4096) { setNotice('绘色图层最多支持 4096 个笔画，请新增绘色图层'); return; }
      if (existing) { patchOperation(existing.id, (item) => ({ ...item, params: { ...item.params, strokes: [...(item.params.strokes as BrushStroke[] ?? []), ...strokes] } })); setSelectedId(existing.id); } else if (!append('paint.brush', { color: [1, 3, 5].map((index) => parseInt(paintColor.slice(index, index + 2), 16)), strokes, opacity: 45, blendMode: 'soft-light' }, '手工绘色')) return;
      setNotice(event.mode === 'paint' ? '绘色笔画已记录' : '已擦除当前绘色层'); return;
    }
    if (['brush', 'erase', 'protect', 'unprotect'].includes(event.mode)) {
      if (!operation) { setNotice('先选择需要蒙版的操作图层'); return; }
      const key = event.mode === 'protect' || event.mode === 'unprotect' ? 'protectMask' : 'mask';
      const previous = operation.params[key] as InlineMask | undefined;
      const strokes: BrushStroke[] = event.points.map((point) => ({ ...point, radius: brush.radius, strength: brush.strength, erase: ['erase', 'unprotect'].includes(event.mode) }));
      if ((previous?.strokes?.length ?? 0) + strokes.length > 4096) { setNotice('单个蒙版最多支持 4096 个笔画，请清除蒙版后重新标记'); return; }
      setMask(key, { ...(previous ?? { shape: 'brush', invert: false, feather: 0 }), strokes: [...(previous?.strokes ?? []), ...strokes] }); setNotice(key === 'mask' ? '蒙版已更新' : '冻结保护已更新'); return;
    }
    const type: OperationType = event.mode === 'liquify' ? 'warp.liquify' : event.mode === 'clone' ? 'repair.clone' : 'repair.heal';
    if (event.mode === 'clone' && !sample) { setNotice('仿制图章需按住 Alt 点击取样'); return; }
    const start = event.points[0];
    const strokes: BrushStroke[] = event.points.map((point, index) => ({ ...point, radius: brush.radius, strength: brush.strength,
      ...(type === 'warp.liquify' ? { mode: brush.liquidMode, ...(brush.liquidMode === 'twirl' ? { direction: brush.twirlDirection } : {}), toX: event.points[index + 1]?.x ?? point.x, toY: event.points[index + 1]?.y ?? point.y } : sample ? { sourceX: Math.max(0, Math.min(1, sample.x + point.x - start.x)), sourceY: Math.max(0, Math.min(1, sample.y + point.y - start.y)) } : {}) }));
    if (type === 'warp.liquify' && brush.symmetric) strokes.push(...strokes.map((stroke) => ({ ...stroke, x: 1 - stroke.x, toX: stroke.toX === undefined ? undefined : 1 - stroke.toX, ...(stroke.direction === undefined ? {} : { direction: -stroke.direction }) })));
    const existing = operation?.type === type ? operation : [...graph.operations].reverse().find((item) => item.type === type);
    if (((existing?.params.strokes as BrushStroke[] | undefined)?.length ?? 0) + strokes.length > 4096) { setNotice('当前图层最多支持 4096 个笔画，请新增笔刷图层'); return; }
    if (existing) patchOperation(existing.id, (item) => ({ ...item, params: { ...item.params, strokes: [...(item.params.strokes as BrushStroke[] ?? []), ...strokes] } }));
    else append(type, { strokes, ...(type === 'warp.liquify' ? {} : { region: WHOLE_IMAGE }) }, type === 'warp.liquify' ? '液化笔刷' : type === 'repair.clone' ? '仿制图章' : '局部修复');
  };
  const controlPoints: ControlPoint[] = operation?.type === 'perspective' ? ((operation.params.corners ?? []) as Point[]).map((point) => ({ ...point, outX: point.x, outY: point.y })) : operation?.type === 'warp.grid' ? operation.controlPoints ?? [] : [];
  const moveControl = (index: number, point: Point) => {
    if (!operation) return;
    if (operation.type === 'perspective') { const corners = [...operation.params.corners as Point[]]; corners[index] = point; patchOperation(operation.id, (item) => ({ ...item, params: { ...item.params, corners } }), `${operation.id}:corners`); }
    else { const controls = structuredClone(operation.controlPoints); controls[index] = { ...controls[index], outX: point.x, outY: point.y }; patchOperation(operation.id, { controlPoints: controls }, `${operation.id}:grid`); }
  };
  const remove = (id: string) => { commit((current) => ({ ...current, operations: current.operations.filter((item) => item.id !== id) })); if (id === selectedId) setSelectedId(null); };
  const resetOperation = () => { if (operation) remove(operation.id); };
  const move = (id: string, direction: number) => commit((current) => { const index = current.operations.findIndex((item) => item.id === id); const destination = index + direction; if (destination < 0 || destination >= current.operations.length) return current; const operations = [...current.operations]; [operations[index], operations[destination]] = [operations[destination], operations[index]]; return { ...current, operations }; });
  const download = (blob: Blob, name: string) => { const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = name; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000); };
  const saveGraph = () => { download(new Blob([JSON.stringify(graph, null, 2)], { type: 'application/json' }), `${filename.replace(/\.[^.]+$/, '') || 'project'}.edit.json`); setNotice('操作工程已保存'); };
  const importGraph = async (file: File) => {
    if (!source) { setNotice('先导入原图，再载入操作工程'); return; }
    const id = ++projectRequest.current; const token = sourceToken.current;
    try {
      const value: unknown = JSON.parse(await file.text()); const base = graphRef.current;
      const candidate = normalizeProject(value, base); setNotice('正在校验操作工程…');
      const image = await renderImage(source, candidate, { kind: 'export' });
      if (id !== projectRequest.current || token !== sourceToken.current) return;
      if (graphRef.current !== base) { setNotice('校验期间已有新的编辑，请重新载入工程'); return; }
      commit(candidate); setPreview(image); setEngineError(''); setSelectedId(candidate.operations.at(-1)?.id ?? null); setNotice('操作工程已载入，可撤销');
    } catch (error) { if (id === projectRequest.current && token === sourceToken.current) setNotice(error instanceof Error ? error.message : '工程载入失败'); }
  };
  const exportImage = async () => {
    if (!source) return; setExporting(true);
    try { const result = await renderImage(source, graph, { kind: 'export' }); let canvas = pixelsToCanvas(result); if (exportFormat === 'jpeg') { const flattened = document.createElement('canvas'); flattened.width = result.width; flattened.height = result.height; const context = flattened.getContext('2d'); if (context) { context.fillStyle = '#ffffff'; context.fillRect(0, 0, result.width, result.height); context.drawImage(canvas, 0, 0); canvas = flattened; } } const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob((value) => value ? resolve(value) : reject(new Error('导出失败')), `image/${exportFormat}`, quality / 100)); download(blob, `${filename.replace(/\.[^.]+$/, '')}-edited.${exportFormat === 'jpeg' ? 'jpg' : 'png'}`); setNotice(`已导出 ${result.width} × ${result.height} ${exportFormat.toUpperCase()}`); }
    catch (error) { setNotice(error instanceof Error ? error.message : '导出失败'); }
    finally { setExporting(false); }
  };
  const importLUT = async (file: File) => { try { const lut = parseCube(await file.text()); const { title, ...params } = lut; append('filter.lut', { ...params, amount: 100 }, title || file.name); setNotice(`已导入 ${lut.size}³ LUT`); } catch (error) { setNotice(error instanceof Error ? error.message : 'LUT 读取失败'); } };
  const detect = async () => {
    if (!source) return; setDetecting(true);
    try {
      // Coordinates belong to this layer's input. Exclude the layer itself so
      // repeated detection never accumulates its current deformation.
      const detectionGraph = graphRef.current;
      const detectionSourceToken = sourceToken.current;
      const layerIndex = operation ? detectionGraph.operations.findIndex(item => item.id === operation.id) : -1;
      const inputGraph = { ...detectionGraph, operations: layerIndex >= 0 ? detectionGraph.operations.slice(0, layerIndex) : detectionGraph.operations };
      const input = await renderImage(source, inputGraph, { kind: 'export' });
      const results = await detectPeople(pixelsToCanvas(input), tool === 'body' ? 'body' : 'face');
      if (graphRef.current.operations !== detectionGraph.operations || sourceToken.current !== detectionSourceToken) { setNotice('编辑已变化，请重新检测当前图层'); return; }
      const detectedRegions: DetectedRegion[] = results.flatMap((result): DetectedRegion[] => {
        // Face detection is a single semantic region. Keep a bounded point
        // set in the graph; MediaPipe's mesh is dense and the Rust preview is
        // intentionally deterministic.
        if (result.kind === 'face') {
          const landmarks = result.points.map(({ x, y }) => ({ x, y }));
          // A support rectangle lets the silhouette move. The ordered source
          // oval is stored separately in semanticMasks.face, not reused as a
          // fixed compositing selection.
          const r = result.region;
          const points = [{ x: r.x, y: r.y }, { x: r.x + r.width, y: r.y }, { x: r.x + r.width, y: r.y + r.height }, { x: r.x, y: r.y + r.height }];
          return [{ id: result.id, kind: result.kind, label: '人脸', region: result.region, points, landmarks, semanticMasks: result.semanticMasks, subjectMask: result.subjectMask }];
        }
        // Pose has no single useful rectangle: expose torso, waist, hips,
        // limbs and the complete silhouette as separate polygon masks. Every
        // operation therefore edits only its selected body part and carries
        // the source pose landmarks for native/WASM replay.
        return (result.parts ?? []).map((part) => ({
          id: `${result.id}-${part.id}`,
          kind: 'body',
          part: part.id,
          label: `${result.id.replace('body-', '人物 ')} · ${part.label}`,
          region: part.region,
          points: part.points.map(({ x, y }) => ({ x, y })),
          landmarks: part.landmarks.map(({ x, y }) => ({ x, y })),
          subjectMask: result.subjectMask,
          protectMask: part.protectMask,
        }));
      });
      setDetected(detectedRegions);
      if (detectedRegions[0]) setTargetRegion(detectedRegions[0].region, detectedRegions[0].points && detectedRegions[0].points.length >= 3 ? 'polygon' : 'rect', detectedRegions[0].points ?? [], detectedRegions[0].landmarks, detectedRegions[0].part, detectedRegions[0].semanticMasks, detectedRegions[0].subjectMask, detectedRegions[0].protectMask);
      const unavailable = results.flatMap(person => person.unavailableParts ?? []);
      setNotice(detectedRegions.length ? `检测到 ${detectedRegions.length} 个语义区域，可切换${unavailable.length ? `；${unavailable.length} 个被遮挡部位轮廓不可见，未开放调整` : ''}` : '未检测到可编辑目标，可以手动标记');
    }
    catch (error) { setNotice(`${error instanceof Error ? error.message : '区域检测失败'}，可手动标记`); }
    finally { setDetecting(false); }
  };
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement).closest('input,textarea,select')) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); if (event.shiftKey) redo(); else undo(); }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') { event.preventDefault(); redo(); }
      if (event.key === 'Delete' || event.key === 'Backspace') { if (selectedId) { event.preventDefault(); remove(selectedId); } }
      if (event.code === 'Space') { event.preventDefault(); setCompare(true); }
      if (event.key === 'Escape') setMode('pan');
      if (event.key === '[') setBrush((current) => ({ ...current, radius: Math.max(.01, current.radius - .005) }));
      if (event.key === ']') setBrush((current) => ({ ...current, radius: Math.min(.25, current.radius + .005) }));
    };
    const release = (event: KeyboardEvent) => { if (event.code === 'Space') setCompare(false); };
    window.addEventListener('keydown', listener); window.addEventListener('keyup', release);
    return () => { window.removeEventListener('keydown', listener); window.removeEventListener('keyup', release); };
  });
  const drop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); const file = event.dataTransfer.files[0]; if (file) void importImage(file); };
  const imageChange = (event: ChangeEvent<HTMLInputElement>) => { const file = event.target.files?.[0]; if (file) void importImage(file); event.target.value = ''; };
  const graphChange = (event: ChangeEvent<HTMLInputElement>) => { const file = event.target.files?.[0]; if (file) void importGraph(file); event.target.value = ''; };

  return <div className="app-shell" onDragOver={(event) => event.preventDefault()} onDrop={drop}>
    <header className="topbar"><div className="brand"><span className="brand-mark">✦</span><span>像素修图</span><span className="brand-badge">WEB DEMO</span></div><div className="topbar-actions"><button className="icon-button" disabled={!canUndo} onClick={undo} aria-label="撤销" title="Ctrl / Cmd + Z">↶</button><button className="icon-button" disabled={!canRedo} onClick={redo} aria-label="重做" title="Ctrl / Cmd + Shift + Z">↷</button><button className={`button button-secondary ${compare ? 'active' : ''}`} disabled={!source} onClick={() => setCompare(!compare)}>原图对比</button><button className="button button-secondary" onClick={() => fileInput.current?.click()}>导入图片</button><select className="export-format" aria-label="导出格式" value={exportFormat} onChange={(event) => setExportFormat(event.target.value as 'png' | 'jpeg')}><option value="png">PNG</option><option value="jpeg">JPEG</option></select>{exportFormat === 'jpeg' && <input className="export-quality" aria-label="JPEG 质量" title="JPEG 质量 1–100" type="number" min={1} max={100} value={quality} onChange={(event) => setQuality(Math.max(1, Math.min(100, Number(event.target.value))))} />}<button className="button button-primary" disabled={!source || exporting} onClick={exportImage}>{exporting ? '导出中…' : `导出 ${exportFormat.toUpperCase()}`}</button></div></header>
    <div className="statusbar"><span className={engineError ? 'status-error' : ''}>{engineError ? `图像核心：${engineError}` : notice}</span><div><button disabled={!source} onClick={saveGraph}>保存工程</button><button disabled={!source} onClick={() => { void navigator.clipboard.writeText(JSON.stringify(graph, null, 2)).then(() => setNotice('操作参数已复制')).catch(() => setNotice('复制失败，请保存工程文件')); }}>复制参数</button><button disabled={!source} onClick={() => { const text = window.prompt('粘贴操作工程 JSON'); if (text) void importGraph(new File([text], 'paste.edit.json', { type: 'application/json' })); }}>粘贴参数</button><button disabled={!source} onClick={() => graphInput.current?.click()}>载入工程</button><button disabled={!graph.operations.length} onClick={() => { commit({ ...graph, operations: [] }); setSelectedId(null); setNotice('操作已清空，原图保持不变'); }}>清空操作</button></div></div>
    {proposalPreview.length > 0 && <div className="proposal-preview-banner"><span>AI 提案预览 · 尚未应用到操作历史</span><button onClick={() => setProposalPreview([])}>取消预览</button></div>}<main className="workspace"><Sidebar tool={tool} onTool={selectTool} operations={graph.operations} selectedId={selectedId} onSelect={selectOperation} onToggle={(id) => patchOperation(id, (item) => ({ ...item, enabled: !item.enabled }))} onDelete={remove} onMove={move} />
      <CanvasStage image={preview} original={source} compare={compare} compareSplit={compareSplit} onCompareSplit={setCompareSplit} busy={busy} zoom={zoom} onZoom={setZoom} mode={mode} region={['browse', 'face', 'feature', 'skin', 'body', 'clothing', 'repair'].includes(tool) ? effectiveRegion : null} mask={operation?.params.mask as InlineMask | undefined} protectMask={operation?.params.protectMask as InlineMask | undefined} controls={controlPoints} sample={sample} radius={brush.radius} onGesture={gesture} onSample={(point) => { setSample(point); setNotice('取样点已设置'); }} onControl={moveControl} onChoose={() => fileInput.current?.click()} />
      <aside className="right-panel">{source ? <><div className="image-info"><strong title={filename}>{filename}</strong><span>{source.width} × {source.height} · {graph.operations.length} 项编辑</span></div><div className="mobile-layer-select"><select aria-label="选择操作图层" value={selectedId ?? ''} onChange={(event) => { const item = graph.operations.find((entry) => entry.id === event.target.value); if (item) selectOperation(item); }}><option value="">选择操作图层</option>{graph.operations.map((item) => <option key={item.id} value={item.id}>{item.label ?? item.type}</option>)}</select>{selected && <div className="button-grid"><button onClick={() => patchOperation(selected.id, { enabled: !selected.enabled })}>{selected.enabled ? '关闭图层' : '启用图层'}</button><button onClick={() => remove(selected.id)}>删除图层</button></div>}</div><ToolPanel key={`${tool}-${selectedId ?? 'new'}`} tool={tool} operation={operation} hasFaceSemantics={hasFaceSemantics} portraitCapabilities={portraitCapabilities} bodyPart={typeof operation?.params.bodyPart === 'string' ? operation.params.bodyPart : regionPart ?? undefined} imageSize={{ width: preview?.width ?? source.width, height: preview?.height ?? source.height }} region={effectiveRegion} regionShape={regionShape} mode={mode} brush={brush} sample={sample} detected={detected} detecting={detecting} onMode={setMode} onBrush={(settings) => setBrush((current) => ({ ...current, ...settings }))} onParam={changeParam} onCreate={append} onReset={resetOperation} onRegion={(next) => setTargetRegion(next, regionShape, selectionPoints, regionLandmarks, regionPart ?? undefined, regionSemanticMasks, regionSubjectMask, (operation?.params.protectMask as InlineMask | undefined) ?? regionProtection)} onDetect={detect} onChooseRegion={(item) => setTargetRegion(item.region, item.points && item.points.length >= 3 ? 'polygon' : 'rect', item.points ?? [], item.landmarks, item.part, item.semanticMasks, item.subjectMask, item.protectMask)} onMask={setMask} paintColor={paintColor} onPaintColor={(color) => { setPaintColor(color); if (selected?.type === 'paint.brush') patchOperation(selected.id, (item) => ({ ...item, params: { ...item.params, color: [1, 3, 5].map((index) => parseInt(color.slice(index, index + 2), 16)) } })); }} onLUT={importLUT} filmThumbnails={filmThumbnails} /><AIPanel sessionId={sessionId} graph={graph} onNotice={setNotice} onPreview={setProposalPreview} onApply={(next) => { commit(next); setSelectedId(next.operations.at(-1)?.id ?? null); }} /></> : <div className="right-empty">照片导入后<br />在这里调整参数<br /><small>手工功能可离线运行</small></div>}</aside>
    </main><input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" className="hidden-input" onChange={imageChange} /><input ref={graphInput} type="file" accept=".json" className="hidden-input" onChange={graphChange} />
  </div>;
}
export default App;

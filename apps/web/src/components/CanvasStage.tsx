import { useEffect, useId, useRef, useState } from 'react';
import type { PointerEvent, WheelEvent } from 'react';
import { clampUnit, normalizedRegion, type ControlPoint, type InlineMask, type Point, type Region, type SelectionShape } from '../core/operations';
import type { ImagePixels } from '../core/engine';
export type InteractionMode = 'pan' | 'rect' | 'ellipse' | 'lasso' | 'polygon' | 'brush' | 'erase' | 'protect' | 'unprotect' | 'liquify' | 'clone' | 'heal' | 'grid' | 'perspective' | 'linear' | 'radial' | 'paint' | 'paintErase';
export interface CanvasGesture { mode: InteractionMode; points: Point[]; region: Region; shape: SelectionShape }
export function CanvasStage({ image, original, compare, compareSplit, onCompareSplit, busy, zoom, onZoom, mode, region, mask, protectMask, controls, sample, radius, onGesture, onSample, onControl, onChoose }: {
  image: ImagePixels | null; original: ImagePixels | null; compare: boolean; compareSplit: boolean; onCompareSplit: (value: boolean) => void;
  busy: boolean; zoom: number; onZoom: (zoom: number) => void; mode: InteractionMode; region: Region | null;
  mask?: InlineMask; protectMask?: InlineMask; controls: ControlPoint[]; sample: Point | null; radius: number;
  onGesture: (gesture: CanvasGesture) => void; onSample: (point: Point) => void; onControl: (index: number, point: Point) => void; onChoose: () => void;
}) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const beforeRef = useRef<HTMLCanvasElement>(null);
  const [viewport, setViewport] = useState({ width: 800, height: 600 });
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [split, setSplit] = useState(50);
  const [cursor, setCursor] = useState<Point | null>(null);
  const [draft, setDraft] = useState<Point[]>([]);
  const pointer = useRef<{ points: Point[]; start: Point; control: number; pan: Point } | null>(null);
  const polygon = useRef<Point[]>([]);
  const current = compare ? original : image;
  useEffect(() => {
    const viewportNode = viewportRef.current; if (!viewportNode) return;
    const observer = new ResizeObserver(([entry]) => setViewport({ width: entry.contentRect.width, height: entry.contentRect.height }));
    observer.observe(viewportNode); return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!current || !canvasRef.current) return;
    const canvas = canvasRef.current; canvas.width = current.width; canvas.height = current.height;
    canvas.getContext('2d')?.putImageData(new ImageData(new Uint8ClampedArray(current.pixels), current.width, current.height), 0, 0);
    if (original && beforeRef.current) {
      const before = beforeRef.current; before.width = original.width; before.height = original.height;
      before.getContext('2d')?.putImageData(new ImageData(new Uint8ClampedArray(original.pixels), original.width, original.height), 0, 0);
    }
  }, [current, original, compareSplit]);
  useEffect(() => { setDraft([]); polygon.current = []; pointer.current = null; }, [mode]);
  useEffect(() => { setPan({ x: 0, y: 0 }); }, [original]);
  const fit = current ? Math.max(.01, Math.min((viewport.width - 68) / current.width, (viewport.height - 68) / current.height, 1)) : 1;
  const displayWidth = Math.max(1, (current?.width ?? 1) * fit * zoom);
  const displayHeight = Math.max(1, (current?.height ?? 1) * fit * zoom);
  const position = (event: PointerEvent<SVGSVGElement>): Point => { const bounds = event.currentTarget.getBoundingClientRect(); return { x: clampUnit((event.clientX - bounds.left) / bounds.width), y: clampUnit((event.clientY - bounds.top) / bounds.height) }; };
  const complete = (points: Point[]) => {
    if (!points.length) return;
    const xs = points.map((point) => point.x); const ys = points.map((point) => point.y);
    const selectionRegion = normalizedRegion({ x: Math.min(...xs), y: Math.min(...ys) }, { x: Math.max(...xs), y: Math.max(...ys) });
    onGesture({ mode, points, region: selectionRegion, shape: mode === 'linear' ? 'linear' : mode === 'radial' ? 'radial' : mode === 'ellipse' ? 'ellipse' : ['lasso', 'polygon'].includes(mode) ? 'polygon' : ['rect', 'pan'].includes(mode) ? 'rect' : 'brush' });
    setDraft([]);
  };
  const pointerDown = (event: PointerEvent<SVGSVGElement>) => {
    if (!current) return;
    const point = position(event);
    if (event.altKey && ['clone', 'heal', 'paint', 'paintErase', 'rect', 'ellipse', 'lasso', 'polygon'].includes(mode)) { onSample(point); return; }
    if (mode === 'polygon') { polygon.current = [...polygon.current, point]; setDraft(polygon.current); return; }
    let control = -1;
    if (mode === 'grid' || mode === 'perspective') {
      let distance = 0.045;
      controls.forEach((candidate, index) => { const delta = Math.hypot((candidate.outX ?? candidate.x) - point.x, (candidate.outY ?? candidate.y) - point.y); if (delta < distance) { distance = delta; control = index; } });
      if (control < 0) return;
    }
    pointer.current = { points: [point], start: { x: event.clientX, y: event.clientY }, control, pan };
    setDraft([point]); event.currentTarget.setPointerCapture(event.pointerId);
  };
  const pointerMove = (event: PointerEvent<SVGSVGElement>) => {
    const point = position(event); setCursor(point);
    const state = pointer.current; if (!state) return;
    if (mode === 'pan') { setPan({ x: state.pan.x + event.clientX - state.start.x, y: state.pan.y + event.clientY - state.start.y }); return; }
    if (state.control >= 0) { onControl(state.control, point); return; }
    if (['rect', 'ellipse', 'linear', 'radial'].includes(mode)) state.points = [state.points[0], point];
    else if (Math.hypot(point.x - state.points.at(-1)!.x, point.y - state.points.at(-1)!.y) > 0.002) state.points.push(point);
    setDraft([...state.points]);
  };
  const pointerUp = (event: PointerEvent<SVGSVGElement>) => {
    const state = pointer.current; pointer.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (state && mode !== 'pan' && state.control < 0) complete(state.points);
    else setDraft([]);
  };
  const finishPolygon = () => { if (polygon.current.length > 2) complete(polygon.current); polygon.current = []; setDraft([]); };
  const wheel = (event: WheelEvent<HTMLDivElement>) => { if (event.ctrlKey || event.metaKey) { event.preventDefault(); onZoom(Math.max(0.1, Math.min(8, zoom * (event.deltaY > 0 ? 0.9 : 1.1)))); } };
  const draftRegion = draft.length > 1 ? normalizedRegion(draft[0], draft.at(-1)!) : null;
  const brushMode = ['brush', 'erase', 'protect', 'unprotect', 'liquify', 'clone', 'heal', 'paint', 'paintErase'].includes(mode);
  return <section className="canvas-stage"><div ref={viewportRef} className={`canvas-viewport mode-${mode}`} onWheel={wheel}>
    {!current ? <div className="empty-state"><div className="upload-icon">↑</div><h2>开始一张新图片</h2><p>导入照片，手工修图始终可用</p><button className="button button-primary" onClick={onChoose}>选择图片</button><span className="file-hint">JPG · PNG · WebP</span></div> : <div className="canvas-frame" style={{ width: displayWidth, height: displayHeight, transform: `translate(${pan.x}px, ${pan.y}px)` }}>
      <canvas ref={canvasRef} aria-label="图片预览" />
      {compareSplit && !compare && original && <div className="compare-before" style={{ width: `${split}%` }}><canvas ref={beforeRef} style={{ width: displayWidth, height: displayHeight }} /></div>}
      <svg className="canvas-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={() => { pointer.current = null; setDraft([]); }} onPointerLeave={() => setCursor(null)} onDoubleClick={finishPolygon} role="img" aria-label={`画布交互：${mode}`}>
        {!compare && <>
          {region && <rect x={region.x} y={region.y} width={region.width} height={region.height} className="region-outline" />}
          {mask && <MaskOverlay width={current.width} height={current.height} mask={mask} className="mask-overlay" />}
          {protectMask && <MaskOverlay width={current.width} height={current.height} mask={protectMask} className="protect-overlay" />}
          {mode === 'rect' && draftRegion && <rect {...draftRegion} className="draft-outline" />}
          {(mode === 'ellipse' || mode === 'radial') && draftRegion && <ellipse cx={draftRegion.x + draftRegion.width / 2} cy={draftRegion.y + draftRegion.height / 2} rx={draftRegion.width / 2} ry={draftRegion.height / 2} className="draft-outline" />}
          {mode === 'linear' && draft.length > 1 && <line x1={draft[0].x} y1={draft[0].y} x2={draft.at(-1)!.x} y2={draft.at(-1)!.y} className="draft-outline" />}
          {['polygon', 'lasso'].includes(mode) && draft.length > 1 && <polyline points={draft.map((point) => `${point.x},${point.y}`).join(' ')} className="draft-outline" />}
          {brushMode && draft.map((point, index) => <ellipse key={index} cx={point.x} cy={point.y} rx={radius * Math.min(current.width, current.height) / current.width} ry={radius * Math.min(current.width, current.height) / current.height} className="brush-draft" />)}
          {brushMode && cursor && <ellipse cx={cursor.x} cy={cursor.y} rx={radius * Math.min(current.width, current.height) / current.width} ry={radius * Math.min(current.width, current.height) / current.height} className="brush-cursor" />}
          {sample && ['clone', 'heal', 'paint', 'paintErase', 'rect', 'ellipse', 'lasso', 'polygon'].includes(mode) && <g className="sample-marker"><line x1={sample.x - .015} x2={sample.x + .015} y1={sample.y} y2={sample.y} /><line x1={sample.x} x2={sample.x} y1={sample.y - .015} y2={sample.y + .015} /></g>}
          {(mode === 'grid' || mode === 'perspective') && controls.map((point, index) => <g key={index}><line x1={point.inX ?? point.x} y1={point.inY ?? point.y} x2={point.outX ?? point.x} y2={point.outY ?? point.y} className="control-guide" /><circle cx={point.outX ?? point.x} cy={point.outY ?? point.y} r={.012} className="control-dot" /></g>)}
        </>}
      </svg>
      {compareSplit && !compare && <><div className="compare-divider" style={{ left: `${split}%` }} /><input className="compare-range" aria-label="前后对比位置" type="range" min={0} max={100} value={split} onChange={(event) => setSplit(Number(event.target.value))} /></>}
    </div>}
    {busy && <span className="render-indicator">正在渲染…</span>}
  </div><div className="canvas-toolbar"><span>{current ? `${current.width} × ${current.height} px` : '未导入图片'}</span><div className="zoom-controls"><button onClick={() => onZoom(Math.max(.1, zoom / 1.15))} aria-label="缩小">−</button><span>{Math.round(zoom * fit * 100)}%</span><button onClick={() => onZoom(Math.min(8, zoom * 1.15))} aria-label="放大">＋</button><button onClick={() => { onZoom(1); setPan({ x: 0, y: 0 }); }} aria-label="适应窗口">适应</button><button onClick={() => onZoom(1 / fit)} aria-label="原始像素100%">100%</button><button className={compareSplit ? 'active' : ''} onClick={() => onCompareSplit(!compareSplit)}>分屏对比</button></div></div></section>;
}
function MaskOverlay({ mask, className, width, height }: { mask: InlineMask; className: string; width: number; height: number }) {
  const id = useId().replace(/:/g, ''); const region = mask.region ?? { x: 0, y: 0, width: 1, height: 1 };
  const from = mask.from ?? { x: region.x, y: region.y }; const to = mask.to ?? { x: region.x + region.width, y: region.y + region.height };
  const feather = Math.max(0, Math.min(100, mask.feather ?? 0)) / 100 * .2;
  const strokes = mask.strokes ?? [];
  const ellipse = { cx: region.x + region.width / 2, cy: region.y + region.height / 2, rx: region.width / 2, ry: region.height / 2 };
  return <>
    <defs>
      <linearGradient id={`${id}-linear`} gradientUnits="userSpaceOnUse" x1={from.x} y1={from.y} x2={to.x} y2={to.y}><stop offset="0" stopColor="black" /><stop offset=".25" stopColor="#282828" /><stop offset=".5" stopColor="#808080" /><stop offset=".75" stopColor="#d7d7d7" /><stop offset="1" stopColor="white" /></linearGradient>
      <radialGradient id={`${id}-radial`}><stop offset="0" stopColor="white" /><stop offset=".25" stopColor="#d7d7d7" /><stop offset=".5" stopColor="#808080" /><stop offset=".75" stopColor="#282828" /><stop offset="1" stopColor="black" /></radialGradient>
      <filter id={`${id}-feather`} x="-1" y="-1" width="3" height="3"><feGaussianBlur stdDeviation={feather / 5} /></filter>
      <filter id={`${id}-inverse`} filterUnits="userSpaceOnUse" x="0" y="0" width="1" height="1" colorInterpolationFilters="sRGB"><feComponentTransfer><feFuncR type="table" tableValues="1 0" /><feFuncG type="table" tableValues="1 0" /><feFuncB type="table" tableValues="1 0" /><feFuncA type="identity" /></feComponentTransfer></filter>
      {strokes.map((stroke, index) => { const inner = Math.max(0, 1 - Math.min(stroke.radius, Math.max(feather, stroke.radius * .12)) / stroke.radius); return <radialGradient id={`${id}-stroke-${index}`} key={index}><stop offset={inner} stopColor={stroke.erase ? 'black' : 'white'} stopOpacity="1" /><stop offset="1" stopColor={stroke.erase ? 'black' : 'white'} stopOpacity="0" /></radialGradient>; })}
      <mask id={`${id}-coverage`} maskUnits="userSpaceOnUse" x="0" y="0" width="1" height="1" style={{ maskType: 'luminance' }}>
        <g filter={mask.invert ? `url(#${id}-inverse)` : undefined}>
          <rect width="1" height="1" fill="black" />
          <g fill="white" filter={feather > 0 && ['rect', 'ellipse', 'polygon'].includes(mask.shape) ? `url(#${id}-feather)` : undefined}>
            {mask.shape === 'rect' && <rect {...region} />}
            {mask.shape === 'ellipse' && <ellipse {...ellipse} />}
            {mask.shape === 'polygon' && <polygon points={(mask.points ?? []).map((point) => `${point.x},${point.y}`).join(' ')} />}
            {mask.shape === 'linear' && <rect width="1" height="1" fill={`url(#${id}-linear)`} />}
            {mask.shape === 'radial' && <ellipse {...ellipse} fill={`url(#${id}-radial)`} />}
          </g>
          {strokes.map((stroke, index) => <ellipse key={index} cx={stroke.x} cy={stroke.y} rx={stroke.radius * Math.min(width, height) / width} ry={stroke.radius * Math.min(width, height) / height} fill={`url(#${id}-stroke-${index})`} opacity={Math.max(0, Math.min(100, stroke.strength)) / 100} />)}
        </g>
      </mask>
    </defs>
    <rect width="1" height="1" mask={`url(#${id}-coverage)`} className={className} style={{ stroke: 'none' }} />
  </>;
}

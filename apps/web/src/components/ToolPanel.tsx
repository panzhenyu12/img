import { useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import { FILM_PRESETS, TOOL_BY_ID, portraitParametersFor, type PortraitCapabilities, type ToolId } from '../core/registry';
import { maskForRegion, WHOLE_IMAGE, type BrushStroke, type EditableMask, type InlineMask, type Operation, type OperationType, type Point, type Region, type SelectionShape, type RasterMask } from '../core/operations';
import type { InteractionMode } from './CanvasStage';
import { NumberField, ParameterGroups, Slider } from './Controls';
export interface BrushSettings { radius: number; strength: number; liquidMode: 'push' | 'pinch' | 'bloat' | 'twirl' | 'reconstruct'; twirlDirection: 1 | -1; symmetric: boolean; repairMode: 'clone' | 'heal' }
/** A detected region may carry an ordered outline so local edits can be
 * composited against the subject contour instead of the full bounding box. */
export interface DetectedRegion {
  id: string;
  kind: string;
  /** Human-readable semantic body area, e.g. 腰部 or 左小腿. */
  label?: string;
  /** Stable semantic id used to identify a body mask. */
  part?: string;
  region: Region;
  /** Ordered perimeter used for the compositing mask. */
  points?: Point[];
  /** Semantic detector landmarks used by the Rust local warp. */
  landmarks?: Point[];
  semanticMasks?: Record<string, { shape: 'polygon'; points: Point[]; feather?: number; invert?: boolean } | RasterMask>;
  subjectMask?: RasterMask;
  partMask?: InlineMask | RasterMask;
  protectMask?: EditableMask;
}
interface ToolPanelProps {
  tool: ToolId; operation?: Operation; imageSize: { width: number; height: number }; region: Region; regionShape: SelectionShape;
  bodyPart?: string;
  hasFaceSemantics?: boolean;
  portraitCapabilities?: PortraitCapabilities;
  mode: InteractionMode; brush: BrushSettings; sample: Point | null; detected: DetectedRegion[]; detecting: boolean;
  onMode: (mode: InteractionMode) => void; onBrush: (settings: Partial<BrushSettings>) => void;
  onParam: (key: string, value: unknown) => void; onCreate: (type: OperationType, params: Record<string, unknown>, label?: string) => void;
  onReset: () => void; onRegion: (region: Region) => void; onDetect: () => void; onChooseRegion: (item: DetectedRegion) => void;
  onMask: (key: 'mask' | 'protectMask', mask: EditableMask | null) => void;
  paintColor: string; onPaintColor: (color: string) => void;
  onLUT: (file: File) => void; filmThumbnails: Record<string, string>;
}
export function ToolPanel(props: ToolPanelProps) {
  const { tool, operation, region, onParam } = props;
  const definition = TOOL_BY_ID[tool];
  const params = operation?.params ?? {};
  const adjustments = (params.adjustments ?? params) as Record<string, unknown>;
  const parameters = portraitParametersFor(tool, props.portraitCapabilities ?? { region, bodyPart: props.bodyPart });
  return <div className="tool-inspector" data-tool={tool}><div className="inspector-heading"><div className="inspector-heading-main"><span className="inspector-kicker">当前工具</span><span className="inspector-title">{definition.label}</span></div>{operation && <button onClick={props.onReset}>重置该项</button>}</div>
    {['face', 'feature', 'skin', 'body', 'clothing', 'browse', 'repair'].includes(tool) && <RegionPanel {...props} />}
    {definition.region && !parameters.length && <p className="panel-tip">{tool === 'face' ? '先自动检测人脸，或用套索 / 多边形描出脸部轮廓。矩形框只标记位置，不能确定真实轮廓。' : tool === 'feature' ? '先检测五官位置，再调整对应部位。手工妆容可使用绘色画笔。' : tool === 'body' ? '先检测身体并选择具体部位。手工塑形可使用液化画笔。' : `先手工标记${tool === 'skin' ? '皮肤' : '衣物'}局部区域，再调参数。`}</p>}
    {tool === 'body' && parameters.length > 0 && <p className="panel-tip">按身体部位独立调整前景轮廓。检查选区和人物分割边缘，再小幅调整。</p>}
    {parameters.length > 0 && tool !== 'film' && operation?.type !== 'paint.brush' && <ParameterGroups parameters={parameters} values={adjustments} onChange={(key, value) => {
      if (['face', 'feature', 'body', 'clothing'].includes(tool)) onParam('adjustments', { ...adjustments, [key]: value }); else onParam(key, value);
    }} />}
    {['skin', 'feature'].includes(tool) && <PaintPanel {...props} />}
    {tool === 'skin' && <div className="panel-block"><h4>局部修复与头发</h4><p className="panel-tip">切换修复笔刷，在皱纹、黑眼圈、斑点处涂抹；碎发可用修复或仿制笔刷，发色和提亮可建立局部调色或绘色层。</p><div className="button-grid"><button onClick={() => props.onMode('heal')}>局部修复笔刷</button><button onClick={() => props.onCreate('color.basic', { region, exposure: 20, saturation: 10 }, '头发局部提亮')}>头发提亮图层</button><button onClick={() => props.onCreate('repair.heal', { region, strokes: [] }, '皮肤局部修复')}>建立修复图层</button></div></div>}
    {tool === 'transform' && <TransformPanel {...props} />}
    {tool === 'liquify' && <BrushPanel {...props} />}
    {tool === 'repair' && <><BrushPanel {...props} /><RepairPanel {...props} /></>}
    {tool === 'grid' && <GridPanel {...props} />}
    {tool === 'mask' && <MaskPanel {...props} />}
    {tool === 'curves' && <CurvesPanel params={params} onParam={onParam} />}
    {tool === 'hsl' && <HSLPanel params={params} onParam={onParam} />}
    {tool === 'film' && <FilmPanel {...props} />}
    {operation && !['crop', 'resize', 'rotate'].includes(operation.type) && <LayerPanel operation={operation} onParam={onParam} onMask={props.onMask} region={region} shape={props.regionShape} />}
  </div>;
}
function PaintPanel({ mode, brush, onBrush, onMode, onCreate, paintColor, onPaintColor }: ToolPanelProps) {
  const color = [1, 3, 5].map((index) => parseInt(paintColor.slice(index, index + 2), 16));
  return <div className="panel-block paint-panel"><h4><span className="panel-step">02</span>手工绘色与妆容</h4><label className="color-field"><span>画笔颜色</span><input aria-label="画笔颜色" type="color" value={paintColor} onChange={(event) => onPaintColor(event.target.value)} /></label><div className="button-grid"><button className={mode === 'paint' ? 'active' : ''} onClick={() => onMode('paint')}>绘色画笔</button><button className={mode === 'paintErase' ? 'active' : ''} onClick={() => onMode('paintErase')}>绘色橡皮</button></div><Slider label="笔刷大小 %" min={1} max={25} value={brush.radius * 100} onChange={(value) => onBrush({ radius: value / 100 })} /><Slider label="笔刷强度" min={1} max={100} value={brush.strength} onChange={(strength) => onBrush({ strength })} /><button className="wide-button" onClick={() => { onCreate('paint.brush', { color, strokes: [], opacity: 45, blendMode: 'soft-light' }, '手工绘色'); onMode('paint'); }}>新增绘色图层</button><p className="panel-tip">用于口红、腮红、高光、发色补色。橡皮只擦当前绘色层，原图保持不变。</p></div>;
}
function RegionPanel({ region, mode, onMode, onRegion, onDetect, detecting, detected, onChooseRegion, tool, onCreate }: ToolPanelProps) {
  const shapeButtons: Array<{ mode: InteractionMode; label: string }> = [{ mode: 'rect', label: '矩形' }, { mode: 'ellipse', label: '椭圆' }, { mode: 'lasso', label: '套索' }, { mode: 'polygon', label: '多边形' }];
  return <div className="panel-block region-panel"><h4><span className="panel-step">01</span>{tool === 'face' || tool === 'feature' || tool === 'skin' ? '脸部区域' : tool === 'body' ? '身体区域' : tool === 'clothing' ? '衣物区域' : '手工选区'}<span className="panel-state">{detected.length ? `已识别 ${detected.length}` : '待标记'}</span></h4><div className="segmented">{shapeButtons.map((item) => <button key={item.mode} className={mode === item.mode ? 'active' : ''} onClick={() => onMode(item.mode)}>{item.label}</button>)}</div><p className="panel-tip">在照片上拖动标记。多边形双击完成。</p><div className="number-grid">{(['x', 'y', 'width', 'height'] as const).map((key) => <NumberField key={key} label={key === 'x' ? '区域 X %' : key === 'y' ? '区域 Y %' : key === 'width' ? '区域宽 %' : '区域高 %'} min={0} max={100} step={.1} value={Math.round(region[key] * 1000) / 10} onChange={(value) => onRegion({ ...region, [key]: Math.max(0, Math.min(1, value / 100)) })} />)}</div><div className="button-grid"><button onClick={() => { const amount = .02; onRegion({ x: Math.max(0, region.x - amount), y: Math.max(0, region.y - amount), width: Math.min(1, region.width + amount * 2), height: Math.min(1, region.height + amount * 2) }); }}>扩大选区</button><button onClick={() => onRegion({ x: region.x + Math.min(.02, region.width / 4), y: region.y + Math.min(.02, region.height / 4), width: Math.max(.001, region.width - .04), height: Math.max(.001, region.height - .04) })}>收缩选区</button><button onClick={() => onRegion(WHOLE_IMAGE)}>使用全图</button>{['face', 'feature', 'skin', 'body'].includes(tool) && <button disabled={detecting} onClick={onDetect}>{detecting ? '检测中…' : '自动检测区域'}</button>}</div>{TOOL_BY_ID[tool].type && ['face', 'feature', 'skin', 'body', 'clothing'].includes(tool) && <button className="wide-button" onClick={() => onCreate(TOOL_BY_ID[tool].type!, { region, ...(tool === 'skin' ? { smooth: 0, texture: 50, whiten: 0, warmth: 0, redness: 0 } : { adjustments: {} }) }, `${TOOL_BY_ID[tool].label}选区图层`)}>为选区新增调整层</button>}{detected.length > 0 && <div className="detected-list">{detected.map((item, index) => <button key={item.id} onClick={() => onChooseRegion(item)}>{item.label ?? (item.kind === 'face' ? `人脸 ${index + 1}` : `身体 ${index + 1}`)}</button>)}</div>}</div>;
}
function TransformPanel({ imageSize, onCreate }: ToolPanelProps) {
  const [crop, setCrop] = useState({ x: 0, y: 0, width: imageSize.width, height: imageSize.height });
  const [angle, setAngle] = useState(0);
  const [size, setSize] = useState(imageSize);
  const [locked, setLocked] = useState(true);
  const validCrop = crop.width > 0 && crop.height > 0 && crop.x >= 0 && crop.y >= 0 && crop.x + crop.width <= imageSize.width && crop.y + crop.height <= imageSize.height;
  return <>
    <div className="panel-block"><h4>旋转与翻转</h4><div className="button-grid"><button onClick={() => onCreate('rotate', { degrees: -90, expand: true }, '逆时针 90°')}>↺ 90°</button><button onClick={() => onCreate('rotate', { degrees: 90, expand: true }, '顺时针 90°')}>↻ 90°</button><button onClick={() => onCreate('flip', { axis: 'horizontal' }, '水平翻转')}>↔ 水平</button><button onClick={() => onCreate('flip', { axis: 'vertical' }, '垂直翻转')}>↕ 垂直</button></div><NumberField label="旋转角度 °" min={-180} max={180} step={.5} value={angle} onChange={setAngle} /><button className="wide-button" onClick={() => onCreate('rotate', { degrees: angle, expand: true }, `旋转 ${angle}°`)}>应用任意旋转</button></div>
    <div className="panel-block"><h4>自由裁剪（像素）</h4><div className="number-grid">{(['x', 'y', 'width', 'height'] as const).map((key) => <NumberField key={key} label={key === 'width' ? '裁剪宽' : key === 'height' ? '裁剪高' : `裁剪 ${key.toUpperCase()}`} value={crop[key]} onChange={(value) => setCrop({ ...crop, [key]: Math.round(value) })} />)}</div><div className="button-grid"><button disabled={!validCrop} onClick={() => onCreate('crop', crop, '自由裁剪')}>应用裁剪</button><button onClick={() => { const width = Math.max(1, Math.round(imageSize.width * .8)); const height = Math.max(1, Math.round(imageSize.height * .8)); onCreate('crop', { x: Math.floor((imageSize.width - width) / 2), y: Math.floor((imageSize.height - height) / 2), width, height }, '中心裁剪 80%'); }}>中心 80%</button></div>{!validCrop && <p className="panel-error">裁剪范围超出当前画布</p>}</div>
    <div className="panel-block"><h4>调整尺寸</h4><label className="check-field"><input type="checkbox" checked={locked} onChange={(event) => setLocked(event.target.checked)} />锁定长宽比</label><div className="number-grid"><NumberField label="输出宽" min={1} max={12000} value={size.width} onChange={(width) => setSize({ width, height: locked ? Math.round(width * imageSize.height / imageSize.width) : size.height })} /><NumberField label="输出高" min={1} max={12000} value={size.height} onChange={(height) => setSize({ height, width: locked ? Math.round(height * imageSize.width / imageSize.height) : size.width })} /></div><button className="wide-button" disabled={size.width < 1 || size.height < 1} onClick={() => onCreate('resize', { width: Math.round(size.width), height: Math.round(size.height) }, '调整尺寸')}>应用尺寸</button></div>
  </>;
}
function BrushPanel({ tool, mode, brush, onBrush, onMode, onCreate, region }: ToolPanelProps) {
  return <div className={`panel-block brush-panel ${tool === 'liquify' ? 'liquify-panel' : 'repair-brush-panel'}`}><h4><span className="panel-step">02</span>{tool === 'liquify' ? '液化笔刷' : '修复笔刷'}</h4>{tool === 'liquify' ? <><div className="button-grid">{(['push', 'pinch', 'bloat', 'twirl', 'reconstruct'] as const).map((value) => <button key={value} className={brush.liquidMode === value && mode === 'liquify' ? 'active' : ''} onClick={() => { onBrush({ liquidMode: value }); onMode('liquify'); }}>{({ push: '推移', pinch: '收缩', bloat: '膨胀', twirl: '旋转', reconstruct: '重建笔刷' })[value]}</button>)}</div>{brush.liquidMode === 'twirl' && <label className="select-field"><span>旋转方向</span><select aria-label="液化旋转方向" value={brush.twirlDirection} onChange={(event) => onBrush({ twirlDirection: Number(event.target.value) as 1 | -1 })}><option value="1">顺时针</option><option value="-1">逆时针</option></select></label>}<label className="check-field"><input aria-label="对称液化" type="checkbox" checked={brush.symmetric} onChange={(event) => onBrush({ symmetric: event.target.checked })} />对称液化</label></> : <div className="button-grid"><button className={mode === 'clone' ? 'active' : ''} onClick={() => { onBrush({ repairMode: 'clone' }); onMode('clone'); }}>仿制图章</button><button className={mode === 'heal' ? 'active' : ''} onClick={() => { onBrush({ repairMode: 'heal' }); onMode('heal'); }}>修复 / 去斑</button></div>}<Slider label="笔刷大小 %" min={1} max={25} value={brush.radius * 100} onChange={(value) => onBrush({ radius: value / 100 })} /><Slider label="笔刷强度" min={1} max={100} value={brush.strength} onChange={(strength) => onBrush({ strength })} /><p className="panel-tip">{tool === 'liquify' ? '在照片上拖动。重建笔刷恢复当前液化层；冻结保护继续有效。对称液化以画布竖直中线镜像笔画。' : '按住 Alt 点击取样，然后涂抹目标。修复笔刷可不取样，自动使用附近纹理。'}</p><button className="wide-button" onClick={() => onCreate(tool === 'liquify' ? 'warp.liquify' : brush.repairMode === 'clone' ? 'repair.clone' : 'repair.heal', { strokes: [], region }, tool === 'liquify' ? '液化笔刷' : '局部修复')}>新增笔刷图层</button></div>;
}
function RepairPanel({ operation, region, regionShape, brush, sample, imageSize, onMode, onCreate, onParam }: ToolPanelProps) {
  const wholeImage = region.width > .999 && region.height > .999;
  const shortEdge = Math.min(imageSize.width, imageSize.height);
  const rx = brush.radius * shortEdge / imageSize.width; const ry = brush.radius * shortEdge / imageSize.height;
  const columns = Math.ceil(region.width / (rx * .8)) + 1; const rows = Math.ceil(region.height / (ry * .8)) + 1;
  const tooManyStamps = columns * rows > 4096;
  const inpaintMask = { ...maskForRegion(region, regionShape), feather: regionShape === 'polygon' ? 10 : 8 };
  const patch = () => {
    if (!sample || tooManyStamps) return;
    const strokes: BrushStroke[] = []; const center = { x: region.x + region.width / 2, y: region.y + region.height / 2 };
    for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
      const x = Math.min(1, region.x + column * region.width / Math.max(1, columns - 1));
      const y = Math.min(1, region.y + row * region.height / Math.max(1, rows - 1));
      strokes.push({ x, y, sourceX: Math.max(0, Math.min(1, sample.x + x - center.x)), sourceY: Math.max(0, Math.min(1, sample.y + y - center.y)), radius: brush.radius, strength: brush.strength });
    }
    onCreate('repair.clone', { region, mask: { ...maskForRegion(region, regionShape), feather: 10 }, strokes }, '取样修补选区');
  };
  const removeBackground = (method: 'exemplar' | 'lama', execution: 'client' | 'server' = 'client') => onCreate('repair.inpaint', { region, mask: inpaintMask, radius: brush.radius, iterations: 16, method, ...(method === 'lama' ? {execution, maxSide: execution === 'client' ? 512 : 1024} : {}) }, method === 'lama' ? `${execution === 'client' ? '客户端' : '服务端'}模型去人 / 去物` : '去除多余人物 / 背景杂物');
  const canProtect = operation?.type === 'repair.inpaint';
  return <>
    <div className="panel-block repair-step"><h4><span className="step-index">1</span>圈选要移除的区域</h4><p className="panel-tip">先完整圈出要移除的人物、路人或杂物。套索和多边形适合沿主体边缘描绘，矩形适合快速处理小物件。</p><div className="segmented"><button onClick={() => onMode('rect')}>矩形</button><button onClick={() => onMode('lasso')}>套索</button><button onClick={() => onMode('polygon')}>多边形</button></div></div>
    <div className="panel-block repair-step"><h4><span className="step-index">2</span>自动去人 / 去物</h4><button className="wide-button" disabled={wholeImage} onClick={() => removeBackground('lama', 'client')}>客户端去人 / 去物</button><button className="wide-button" disabled={wholeImage} onClick={() => removeBackground('lama', 'server')}>服务端去人 / 去物</button><p className="small-note">客户端在浏览器处理，首次需缓存约 208 MB 模型；服务端适合更高分辨率。两种方式都保留为可编辑图层。</p></div>
    <div className="panel-block repair-step"><h4><span className="step-index">3</span>手工填补与主体保护</h4><button className="wide-button" disabled={wholeImage} onClick={() => removeBackground('exemplar')}>去除多余人物 / 背景杂物</button>{canProtect && <><label className="select-field"><span>填补方式</span><select aria-label="填补方式" value={String(operation?.params.method ?? 'exemplar')} onChange={(event) => onParam('method', event.target.value)}><option value="exemplar">手工内容感知填补</option><option value="lama">模型填补</option></select></label>{operation?.params.method === 'lama' && <label className="select-field"><span>处理位置</span><select aria-label="处理位置" value={String(operation.params.execution ?? 'server')} onChange={(event) => onParam('execution', event.target.value)}><option value="client">客户端（浏览器本地）</option><option value="server">服务端</option></select></label>}<button className="wide-button" onClick={() => onMode('protect')}>保护主体（冻结画笔）</button><p className="small-note">在主体脸部、身体和衣服上涂抹蓝色保护区，避免填补传播到主体。</p></>}<button className="wide-button" disabled={wholeImage} onClick={() => onCreate('repair.inpaint', { region, mask: inpaintMask, radius: brush.radius, iterations: 16 }, '内容感知填补')}>填补选区</button><button className="wide-button" disabled={!sample || wholeImage || tooManyStamps} onClick={patch}>从取样点修补选区</button>{sample && <p className="small-note">取样中心 {Math.round(sample.x * 100)}% / {Math.round(sample.y * 100)}%</p>}{tooManyStamps && <p className="panel-error">修补笔画超过 4096，请增大笔刷或缩小选区。</p>}</div>
  </>;
}
function GridPanel({ operation, mode, onMode, onCreate }: ToolPanelProps) {
  const controls = operation?.controlPoints ?? [];
  const makeGrid = () => {
    onCreate('warp.grid', {}, '手动网格'); onMode('grid');
  };
  return <div className="panel-block"><h4>网格与四角透视</h4><div className="button-grid"><button className={mode === 'grid' ? 'active' : ''} onClick={makeGrid}>建立 3×3 网格</button><button className={mode === 'perspective' ? 'active' : ''} onClick={() => { onCreate('perspective', { corners: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }] }, '四角透视'); onMode('perspective'); }}>四角透视</button></div><p className="panel-tip">拖动画布上的控制点改变网格或透视。使用“重置该项”恢复该操作。</p>{controls.length > 0 && <span className="small-note">{controls.length} 个可拖拽控制点</span>}</div>;
}
function MaskPanel({ operation, mode, brush, onMode, onBrush, onMask }: ToolPanelProps) {
  const mask = operation?.params.mask as EditableMask | undefined;
  const protect = operation?.params.protectMask as EditableMask | undefined;
  return <div className="panel-block"><h4>蒙版笔刷</h4><p className="panel-tip">先从左侧选择要限制的操作图层。绿色是生效区域，蓝色是冻结保护。</p><div className="button-grid"><button className={mode === 'linear' ? 'active' : ''} onClick={() => onMode('linear')}>线性渐变</button><button className={mode === 'radial' ? 'active' : ''} onClick={() => onMode('radial')}>径向渐变</button><button className={mode === 'brush' ? 'active' : ''} onClick={() => onMode('brush')}>涂抹蒙版</button><button className={mode === 'erase' ? 'active' : ''} onClick={() => onMode('erase')}>擦除蒙版</button><button className={mode === 'protect' ? 'active' : ''} onClick={() => onMode('protect')}>冻结保护</button><button className={mode === 'unprotect' ? 'active' : ''} onClick={() => onMode('unprotect')}>擦除保护</button></div><Slider label="笔刷大小 %" min={1} max={25} value={brush.radius * 100} onChange={(value) => onBrush({ radius: value / 100 })} /><Slider label="笔刷强度" min={1} max={100} value={brush.strength} onChange={(strength) => onBrush({ strength })} />{mask && <><Slider label="蒙版羽化" min={0} max={100} value={mask.feather ?? 0} onChange={(feather) => onMask('mask', { ...mask, feather })} /><button className="wide-button" onClick={() => onMask('mask', { ...mask, invert: !mask.invert })}>{mask.invert ? '取消蒙版反选' : '反选蒙版'}</button></>}<div className="button-grid"><button disabled={!mask} onClick={() => onMask('mask', null)}>清除蒙版</button><button disabled={!protect} onClick={() => onMask('protectMask', null)}>清除保护</button></div></div>;
}
function CurvesPanel({ params, onParam }: { params: Record<string, unknown>; onParam: (key: string, value: unknown) => void }) {
  const points = (params.points ?? [{ x: 0, y: 0 }, { x: .5, y: .5 }, { x: 1, y: 1 }]) as Point[];
  const change = (index: number, point: Point) => { const next = [...points]; next[index] = { x: Math.max(0, Math.min(1, point.x)), y: Math.max(0, Math.min(1, point.y)) }; onParam('points', next.sort((a, b) => a.x - b.x)); };
  return <div className="panel-block"><h4>色调曲线</h4><label className="select-field"><span>通道</span><select aria-label="曲线通道" value={String(params.channel ?? 'rgb')} onChange={(event) => onParam('channel', event.target.value)}><option value="rgb">RGB</option><option value="red">红</option><option value="green">绿</option><option value="blue">蓝</option></select></label><svg viewBox="0 0 100 100" className="curve-preview" onClick={(event) => { if ((event.target as Element).tagName === 'circle') return; const box = event.currentTarget.getBoundingClientRect(); onParam('points', [...points, { x: (event.clientX - box.left) / box.width, y: 1 - (event.clientY - box.top) / box.height }].sort((a, b) => a.x - b.x)); }}><path d="M0 100 L100 0" className="curve-guide" /><polyline points={points.map((point) => `${point.x * 100},${(1 - point.y) * 100}`).join(' ')} className="curve-line" />{points.map((point, index) => <circle key={index} cx={point.x * 100} cy={(1 - point.y) * 100} r="2" className="curve-dot" />)}</svg><p className="panel-tip">点击曲线添加节点，数值精调输入 / 输出。</p>{points.map((point, index) => <div className="curve-point" key={index}><NumberField label={`输入 ${index + 1}`} min={0} max={100} value={Math.round(point.x * 100)} onChange={(value) => change(index, { ...point, x: value / 100 })} /><NumberField label={`输出 ${index + 1}`} min={0} max={100} value={Math.round(point.y * 100)} onChange={(value) => change(index, { ...point, y: value / 100 })} /><button aria-label="删除曲线节点" disabled={points.length < 3} onClick={() => onParam('points', points.filter((_, item) => item !== index))}>×</button></div>)}</div>;
}
function HSLPanel({ params, onParam }: { params: Record<string, unknown>; onParam: (key: string, value: unknown) => void }) {
  const [hueIndex, setHueIndex] = useState(0);
  const hues = [0, 30, 60, 120, 180, 240, 270, 330]; const names = ['红', '橙', '黄', '绿', '青', '蓝', '紫', '洋红'];
  const hsl = (params.hsl ?? hues.map((hue) => ({ hue, range: 30, shift: 0, saturation: 0, lightness: 0 }))) as Array<Record<string, number>>;
  const current = hsl[hueIndex] ?? { hue: hues[hueIndex], range: 30, shift: 0, saturation: 0, lightness: 0 };
  const change = (key: string, value: number) => { const next = [...hsl]; next[hueIndex] = { ...current, [key]: value }; onParam('hsl', next); };
  return <div className="panel-block"><h4>HSL 颜色分区</h4><div className="hue-buttons">{names.map((name, index) => <button key={name} className={hueIndex === index ? 'active' : ''} style={{ borderColor: `hsl(${hues[index]} 70% 50%)` }} onClick={() => setHueIndex(index)}>{name}</button>)}</div><Slider label="色相偏移" value={current.shift} onChange={(value) => change('shift', value)} /><Slider label="饱和度" value={current.saturation} onChange={(value) => change('saturation', value)} /><Slider label="明度" value={current.lightness} onChange={(value) => change('lightness', value)} /><Slider label="色相范围 °" min={1} max={180} value={current.range} onChange={(value) => change('range', value)} /></div>;
}
function FilmPanel({ operation, onParam, onCreate, onLUT, filmThumbnails }: ToolPanelProps) {
  const [search, setSearch] = useState(''); const [category, setCategory] = useState('全部');
  const [custom, setCustom] = useState<Array<{ name: string; params: Record<string, unknown> }>>(() => { try { return JSON.parse(localStorage.getItem('img-film-presets') ?? '[]'); } catch { return []; } });
  const inputRef = useRef<HTMLInputElement>(null); const params = operation?.params ?? {};
  if (operation?.type === 'filter.lut') return <div className="panel-block"><h4>LUT：{operation.label}</h4><Slider label="LUT 强度" min={0} max={100} value={Number(params.amount ?? 100)} onChange={(amount) => onParam('amount', amount)} /><button className="wide-button" onClick={() => onCreate('filter.film', { preset: 'classic', intensity: 70, grain: 0, vignette: 0, glow: 0, halation: 0, fade: 0, seed: 42 }, '胶片模拟')}>叠加胶片模拟</button></div>;
  const visible = FILM_PRESETS.filter((preset) => (category === '全部' || preset.category === category) && (preset.name + preset.description).includes(search));
  const save = () => { const name = window.prompt('自定义预设名称'); if (!name?.trim()) return; const next = [...custom, { name: name.trim(), params }]; localStorage.setItem('img-film-presets', JSON.stringify(next)); setCustom(next); };
  const file = (event: ChangeEvent<HTMLInputElement>) => { const selected = event.target.files?.[0]; if (selected) onLUT(selected); event.target.value = ''; };
  return <><div className="panel-block"><h4>胶片风格</h4><input className="search-field" aria-label="搜索滤镜" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索风格" /><div className="category-buttons">{['全部', ...new Set(FILM_PRESETS.map((preset) => preset.category))].map((value) => <button key={value} className={category === value ? 'active' : ''} onClick={() => setCategory(value)}>{value}</button>)}</div><div className="film-presets">{visible.map((preset) => <button key={preset.id} className={params.preset === preset.id ? 'active' : ''} onClick={() => onParam('preset', preset.id)}>{filmThumbnails[preset.id] ? <img className="preset-thumbnail" src={filmThumbnails[preset.id]} alt={`${preset.name}效果`} /> : <span className={`preset-swatch preset-${preset.id}`} />}<strong>{preset.name}</strong><small>{preset.description}</small></button>)}</div></div><ParameterGroups parameters={TOOL_BY_ID.film.parameters} values={params} onChange={onParam} /><div className="panel-block"><div className="button-grid"><button onClick={() => onCreate('filter.film', { ...params, intensity: params.intensity ?? 70, preset: params.preset ?? 'classic' }, '叠加胶片')}>叠加滤镜</button><button onClick={save} disabled={!operation}>保存预设</button><button onClick={() => { const next = [...custom, { name: `预设副本 ${custom.length + 1}`, params: structuredClone(params) }]; localStorage.setItem('img-film-presets', JSON.stringify(next)); setCustom(next); }} disabled={!operation}>复制预设</button><button onClick={() => inputRef.current?.click()}>导入 .cube LUT</button></div><input ref={inputRef} className="hidden-input" type="file" accept=".cube" onChange={file} />{custom.map((preset, index) => <div className="custom-preset" key={`${preset.name}-${index}`}><button onClick={() => onCreate('filter.film', preset.params, preset.name)}>{preset.name}</button><button aria-label="删除自定义预设" onClick={() => { const next = custom.filter((_, item) => item !== index); setCustom(next); localStorage.setItem('img-film-presets', JSON.stringify(next)); }}>×</button></div>)}</div></>;
}
function LayerPanel({ operation, onParam, onMask, region, shape }: { operation: Operation; onParam: (key: string, value: unknown) => void; onMask: (key: 'mask' | 'protectMask', mask: EditableMask | null) => void; region: Region; shape: SelectionShape }) {
  const mask = operation.params.mask as EditableMask | undefined;
  return <div className="panel-block layer-panel"><h4>当前操作图层</h4><div className="small-note">{operation.label ?? operation.type}</div><Slider label="不透明度" min={0} max={100} value={Number(operation.params.opacity ?? 100)} onChange={(value) => onParam('opacity', value)} /><label className="select-field"><span>混合模式</span><select aria-label="混合模式" value={String(operation.params.blendMode ?? 'normal')} onChange={(event) => onParam('blendMode', event.target.value)}><option value="normal">正常</option><option value="multiply">正片叠底</option><option value="screen">滤色</option><option value="overlay">叠加</option><option value="soft-light">柔光</option></select></label><div className="button-grid"><button onClick={() => onMask('mask', { shape, region, invert: false, feather: 0 })}>选区转蒙版</button><button disabled={!mask} onClick={() => onMask('mask', mask ? { ...mask, invert: !mask.invert } : null)}>蒙版反选</button></div>{mask && <Slider label="蒙版羽化" min={0} max={100} value={mask.feather ?? 0} onChange={(feather) => onMask('mask', { ...mask, feather })} />}</div>;
}

import type { OperationType, Region, InlineMask, RasterMask, Point } from './operations';

export type ToolId = 'browse' | 'transform' | 'face' | 'feature' | 'skin' | 'body' | 'clothing' | 'liquify' | 'grid' | 'repair' | 'color' | 'curves' | 'hsl' | 'film' | 'mask';
export interface Parameter { key: string; label: string; min: number; max: number; default: number; group: string }
export interface ToolDefinition { id: ToolId; label: string; icon: string; type?: OperationType; region?: 'face' | 'body' | 'clothing'; parameters: Parameter[] }
const p = (key: string, label: string, group: string, min = -100, max = 100, value = 0): Parameter => ({ key, label, group, min, max, default: value });
export const TOOLS: ToolDefinition[] = [
  { id: 'browse', label: '浏览与选区', icon: '⌖', parameters: [] },
  { id: 'transform', label: '裁剪与几何', icon: '◈', parameters: [] },
  { id: 'face', label: '脸型', icon: '◒', type: 'face.adjust', region: 'face', parameters: [
    p('forehead', '额头高度', '轮廓'), p('foreheadWidth', '额头宽度', '轮廓'), p('temples', '太阳穴', '轮廓'), p('cheekbones', '颧骨', '轮廓'), p('jaw', '下颌', '轮廓'), p('chin', '下巴长度', '轮廓'), p('chinWidth', '下巴宽度', '轮廓'), p('chinPoint', '下巴尖度', '轮廓'),
    p('faceWidth', '脸宽 / 瘦脸', '比例'), p('faceHeight', '脸高', '比例'), p('midface', '面中长度', '比例'), p('doubleChin', '双下巴', '比例', 0), p('symmetry', '五官对称校正', '比例', 0),
  ] },
  { id: 'feature', label: '五官与妆容', icon: '◉', type: 'feature.adjust', region: 'face', parameters: [
    p('eyeSize', '眼睛大小', '眼睛'), p('eyeWidth', '眼睛宽度', '眼睛'), p('eyeHeight', '眼睛高度', '眼睛'), p('eyeSpacing', '眼间距', '眼睛'), p('eyeTilt', '眼睛倾斜', '眼睛'), p('eyeCorner', '眼角', '眼睛'), p('eyelid', '眼睑', '眼睛'), p('eyeInnerCorner', '内眼角', '眼睛'), p('eyeOuterCorner', '外眼角', '眼睛'), p('eyeBag', '眼袋', '眼睛', 0), p('darkCircle', '黑眼圈', '眼睛', 0), p('tearTrough', '泪沟', '眼睛', 0),
    p('browLength', '眉长', '眉毛'), p('browWidth', '眉宽', '眉毛'), p('browThickness', '眉毛粗细', '眉毛'), p('browPeak', '眉峰', '眉毛'), p('browTail', '眉尾', '眉毛'), p('browColor', '眉毛深浅', '眉毛'), p('browSpacing', '眉间距', '眉毛'),
    p('noseBridge', '鼻梁宽度', '鼻子'), p('noseWings', '鼻翼', '鼻子'), p('noseLength', '鼻长', '鼻子'), p('noseTip', '鼻尖', '鼻子'), p('noseHeight', '鼻子高度', '鼻子'), p('noseStraight', '鼻子矫正', '鼻子'), p('nostril', '鼻孔', '鼻子'), p('noseRoot', '山根', '鼻子'), p('noseBase', '鼻基底', '鼻子'), p('noseLipAngle', '鼻唇角', '鼻子'),
    p('mouthWidth', '嘴宽', '嘴唇'), p('mouthHeight', '嘴高', '嘴唇'), p('upperLip', '上唇厚度', '嘴唇'), p('lowerLip', '下唇厚度', '嘴唇'), p('lipColor', '唇色', '嘴唇'), p('philtrum', '人中', '嘴唇'), p('teethWhiten', '牙齿美白', '嘴唇', 0), p('lipPeak', '唇峰', '嘴唇'), p('lipCorner', '嘴角', '嘴唇'), p('lipBrightness', '唇部明亮', '嘴唇'), p('lipSaturation', '唇部饱和', '嘴唇'), p('symmetry', '五官对称校正', '比例', 0),
  ] },
  { id: 'skin', label: '皮肤与质感', icon: '✧', type: 'skin.retouch', region: 'face', parameters: [
    p('smooth', '磨皮', '皮肤', 0), p('texture', '纹理保留', '皮肤', 0, 100, 50), p('whiten', '美白', '皮肤', 0), p('warmth', '肤色冷暖', '皮肤'), p('redness', '红润 / 去红', '皮肤'),
  ] },
  { id: 'body', label: '身形', icon: '♧', type: 'body.adjust', region: 'body', parameters: [
    p('neck', '脖颈', '上身'), p('shoulder', '肩宽', '上身'), p('waist', '腰围', '上身'), p('belly', '腹部', '上身'), p('arms', '手臂', '上身'), p('neckLength', '脖颈长度', '上身'), p('trapezius', '斜方肌', '上身'), p('shoulderLevel', '肩膀水平', '上身'), p('collarbone', '锁骨', '上身'), p('forearms', '前臂', '上身'),
    p('legs', '腿部粗细', '下身'), p('hips', '胯宽', '下身'), p('height', '身高比例', '下身'), p('legLength', '腿长', '下身'), p('thighs', '大腿', '下身'), p('calves', '小腿', '下身'), p('ankles', '脚踝', '下身'), p('posture', '体态', '比例'), p('headRatio', '头身比例', '比例'),
  ] },
  { id: 'clothing', label: '衣物', icon: '♜', type: 'clothing.adjust', region: 'clothing', parameters: [
    p('smooth', '褶皱平整', '质感', 0), p('texture', '纹理保护', '质感', 0, 100, 70), p('hue', '颜色色相', '质感'), p('saturation', '颜色饱和', '质感'), p('brightness', '衣物亮度', '质感'),
    p('fit', '版型收放', '版型'), p('edges', '边缘修整', '版型'), p('sleeves', '袖子宽度', '版型'), p('skirt', '裙摆宽度', '版型'), p('collar', '衣领', '版型'), p('cuffs', '袖口', '版型'), p('sleeveLength', '袖长', '版型'),
  ] },
  { id: 'liquify', label: '液化', icon: '≈', type: 'warp.liquify', parameters: [] },
  { id: 'grid', label: '网格与透视', icon: '▦', type: 'warp.grid', parameters: [] },
  { id: 'repair', label: '修复与仿制', icon: '✎', type: 'repair.heal', parameters: [] },
  { id: 'color', label: '光影与调色', icon: '◌', type: 'color.basic', parameters: [
    p('exposure', '曝光', '光影'), p('brightness', '亮度', '光影'), p('contrast', '对比度', '光影'), p('highlights', '高光', '光影'), p('shadows', '阴影', '光影'), p('whites', '白色色阶', '光影'), p('blacks', '黑色色阶', '光影'),
    p('temperature', '色温', '颜色'), p('tint', '色调', '颜色'), p('saturation', '饱和度', '颜色'), p('vibrance', '自然饱和度', '颜色'), p('clarity', '清晰度', '质感'), p('sharpness', '锐化', '质感', 0), p('texture', '纹理', '质感'), p('dehaze', '去雾', '质感'), p('noise', '降噪', '质感', 0), p('blackWhite', '黑白', '颜色', 0), p('blackPoint', '黑色色阶端点', '色阶', 0, 99.9), p('whitePoint', '白色色阶端点', '色阶', .1, 100, 100), p('gamma', '中间调 Gamma', '色阶', .1, 5, 1),
  ] },
  { id: 'curves', label: '色调曲线', icon: '⌁', type: 'color.curves', parameters: [] },
  { id: 'hsl', label: 'HSL 色彩', icon: '◑', type: 'color.hsl', parameters: [] },
  { id: 'film', label: '胶片模拟', icon: '◇', type: 'filter.film', parameters: [
    p('intensity', '滤镜强度', '效果', 0, 100, 70), p('grain', '颗粒', '效果', 0), p('vignette', '暗角', '效果', 0), p('glow', '辉光', '效果', 0), p('halation', '光晕', '效果', 0), p('fade', '黑位褪色', '效果', 0),
  ] },
  { id: 'mask', label: '蒙版与冻结', icon: '◍', parameters: [] },
];
export const FILM_PRESETS = [
  { id: 'neutral', name: '原片', category: '自然', description: '保留原有色彩' },
  { id: 'warm', name: '暖光人像', category: '人像', description: '温暖、柔和肤色' },
  { id: 'cool', name: '冷调影院', category: '电影', description: '冷静蓝调与柔和反差' },
  { id: 'classic', name: '经典纪实', category: '纪实', description: '温暖低饱和纪实风格' },
  { id: 'fade', name: '复古褪色', category: '复古', description: '提升黑位、轻柔反差' },
  { id: 'vivid', name: '鲜艳风景', category: '风景', description: '鲜艳通透色彩' },
  { id: 'mono', name: '银盐黑白', category: '黑白', description: '黑白高反差' },
  { id: 'cinema', name: '青橙电影', category: '电影', description: '阴影青调、高光暖调' },
  { id: 'provia', name: 'Provia 标准', category: '富士风格', description: '自然反差、干净肤色和细腻高光' },
  { id: 'astia', name: 'Astia 柔和', category: '富士风格', description: '柔和反差、轻微暖高光，适合人像' },
  { id: 'velvia', name: 'Velvia 鲜艳', category: '富士风格', description: '高饱和风景色彩，保留亮部层次' },
  { id: 'classicNegative', name: 'Classic Negative', category: '富士风格', description: '低饱和复古色，青冷阴影和暖高光' },
  { id: 'eterna', name: 'Eterna 电影', category: '富士风格', description: '低反差电影色，压住高光不过曝' },
  { id: 'acros', name: 'Acros 黑白', category: '黑白', description: '线性光银盐响应，灰阶和高光更平顺' },
];
export const TOOL_BY_ID = Object.fromEntries(TOOLS.map((tool) => [tool.id, tool])) as Record<ToolId, ToolDefinition>;
export interface PortraitCapabilities {
  region?: Region;
  shape?: string;
  points?: Point[];
  semanticMasks?: Record<string, { shape?: string; points?: Point[]; width?: number; height?: number; data?: number[] }>;
  bodyPart?: string;
  subjectMask?: RasterMask;
  mask?: InlineMask;
}
const FACE_CONTOUR_PARAMETERS = ['forehead', 'foreheadWidth', 'temples', 'cheekbones', 'jaw', 'chin', 'chinWidth', 'chinPoint', 'faceWidth'];
const PENDING_FEATURE_PARAMETERS = ['eyeBag', 'tearTrough', 'noseLipAngle', 'noseHeight'];
const CLOTHING_PIXEL_PARAMETERS = ['smooth', 'texture', 'hue', 'saturation', 'brightness'];
export const DEFERRED_PORTRAIT_PARAMETERS: Partial<Record<ToolId, readonly string[]>> = {
  face: ['faceHeight', 'midface', 'doubleChin'],
  feature: PENDING_FEATURE_PARAMETERS,
  body: ['neck', 'neckLength', 'trapezius', 'shoulderLevel', 'collarbone', 'height', 'legLength', 'ankles', 'posture', 'headRatio'],
  clothing: ['fit', 'edges', 'sleeves', 'skirt', 'collar', 'cuffs', 'sleeveLength'],
};
export function implementedParametersFor(tool: ToolId): Parameter[] {
  return TOOL_BY_ID[tool].parameters.filter(item => !DEFERRED_PORTRAIT_PARAMETERS[tool]?.includes(item.key));
}
/** A whole-image default is not a face/body location. Keep manual editing
 * available through explicit skin/clothing selections, face outlines and
 * freehand liquify; do not guess named anatomical regions from the canvas. */
export function hasExplicitPortraitRegion(region?: Region): boolean {
  return !!region && region.width > .003 && region.height > .003 && !(region.x < .001 && region.y < .001 && region.width > .998 && region.height > .998);
}
export function portraitParametersFor(tool: ToolId, context: PortraitCapabilities): Parameter[] {
  const definition = TOOL_BY_ID[tool];
  const masks = context.semanticMasks ?? {};
  const hasMask = (name: string) => (masks[name]?.points?.length ?? 0) >= 3 || (masks[name]?.shape === 'raster' && (masks[name]?.data?.length ?? 0) > 0);
  const explicit = hasExplicitPortraitRegion(context.region);
  if (tool === 'face') {
    const outline = hasMask('face') || hasMask('faceOutline') || ((context.shape ?? context.mask?.shape) === 'polygon' && (context.points ?? context.mask?.points ?? []).length >= 3);
    if (!outline || !explicit) return [];
    return definition.parameters.filter(item => FACE_CONTOUR_PARAMETERS.includes(item.key) || (item.key === 'symmetry' && ['leftEye', 'rightEye', 'leftBrow', 'rightBrow', 'lips'].every(hasMask)));
  }
  if (tool === 'feature') return definition.parameters.filter(item => {
    if (!explicit || PENDING_FEATURE_PARAMETERS.includes(item.key)) return false;
    if (item.key === 'symmetry') return ['leftEye', 'rightEye', 'leftBrow', 'rightBrow', 'lips'].every(hasMask);
    if (item.key.startsWith('eye') || item.key === 'eyelid' || item.key === 'darkCircle') return hasMask('leftEye') && hasMask('rightEye');
    if (item.key.startsWith('brow')) return hasMask('leftBrow') && hasMask('rightBrow');
    if (item.key.startsWith('nose') || item.key === 'nostril') return hasMask('nose');
    return hasMask('lips');
  });
  if (tool === 'body') return explicit && context.bodyPart && context.subjectMask ? bodyParametersFor(context.bodyPart) : [];
  if (tool === 'skin') return explicit ? definition.parameters : [];
  if (tool === 'clothing') return explicit ? definition.parameters.filter(item => CLOTHING_PIXEL_PARAMETERS.includes(item.key)).map(item => item.key === 'smooth' ? { ...item, label: '褶皱细纹弱化' } : item) : [];
  return definition.parameters;
}
/** Semantic Pose operations expose only controls implemented for that area.
 * A subject silhouette is required for nonzero reshaping. Height,
 * legLength and posture need a distinct axial/structural algorithm and must
 * never be advertised as contour width controls on a detected part. */
export function bodyParameterKeys(part?: string): readonly string[] | null {
  if (!part) return null;
  if (part === 'waist') return ['waist', 'belly'];
  if (part === 'hips') return ['hips'];
  if (part === 'torso') return ['shoulder', 'waist', 'belly', 'hips'];
  if (part.endsWith('upper-arm')) return ['arms'];
  if (part.endsWith('forearm')) return ['arms', 'forearms'];
  if (part.endsWith('thigh')) return ['legs', 'thighs'];
  if (part.endsWith('calf')) return ['legs', 'calves'];
  if (part === 'full-body') return ['shoulder', 'waist', 'belly', 'hips', 'arms', 'forearms', 'legs', 'thighs', 'calves'];
  return [];
}
export function bodyParametersFor(part?: string): Parameter[] {
  const keys = bodyParameterKeys(part);
  return keys ? TOOL_BY_ID.body.parameters.filter((parameter) => keys.includes(parameter.key)) : TOOL_BY_ID.body.parameters;
}
export function bodyPartDescription(part?: string): string {
  if (part === 'full-body') return '全身粗细参数按 Pose 部位分别调整。建议选择腰部、胯部或单侧肢体，检查并修正自动选区；身高、腿长与体态暂未接入语义重塑。';
  if (part) return '当前只显示该部位支持的粗细参数。中轴和关节端点保持，边缘逐渐归零；自动选区可用套索修正。';
  return '身体重塑需要真实主体轮廓与关键点；手工自由推拉请使用液化。自动检测并选择具体身体部位，可使用 Pose 语义调整。';
}
/** The AI gateway must use this allow-list, rather than invent parameter names. */
export const PARAMETER_REGISTRY = Object.fromEntries(TOOLS.filter((tool) => tool.type).map((tool) => [tool.type, {
  operation: tool.type, region: tool.region, keys: Object.fromEntries(implementedParametersFor(tool.id).map((item) => [item.key, { min: item.min, max: item.max, default: item.default, label: item.label }]))
}]));
export function defaultsFor(tool: ToolDefinition, region?: Region): Record<string, unknown> {
  const values = Object.fromEntries(tool.parameters.map((item) => [item.key, item.default]));
  if (['face', 'feature', 'body', 'clothing'].includes(tool.id)) return { region, adjustments: values };
  if (tool.id === 'skin') return { region, ...values };
  if (tool.id === 'film') return { ...values, preset: 'classic', seed: 42 };
  return values;
}

export const COMMON_PARAMETER_KEYS = ['region', 'mask', 'protectMask', 'opacity', 'blendMode'] as const;
/** Structural parameters for canvas tools, kept explicit for AI tool discovery. */
export const OPERATION_PARAMETER_KEYS: Record<OperationType, readonly string[]> = {
  identity: [], crop: ['x', 'y', 'width', 'height'], rotate: ['degrees', 'expand'], flip: ['axis'], resize: ['width', 'height'],
  perspective: ['corners'], 'warp.grid': ['points', 'controlPoints'], 'warp.liquify': ['strokes'],
  'face.adjust': ['adjustments', 'landmarks', 'semanticMasks', 'subjectMask'], 'feature.adjust': ['adjustments', 'landmarks', 'semanticMasks'], 'body.adjust': ['adjustments', 'landmarks', 'bodyPart', 'subjectMask', 'partMask'], 'clothing.adjust': ['adjustments', 'landmarks'],
  'skin.retouch': [...TOOL_BY_ID.skin.parameters.map((item) => item.key), 'landmarks', 'semanticMasks'],
  'repair.clone': ['strokes'], 'repair.heal': ['strokes'], 'repair.inpaint': ['radius', 'iterations', 'method', 'execution', 'maxSide', 'contextPadding', 'maskDilation', 'seamBlend', 'contextMode'], 'paint.brush': ['color', 'strokes'],
  'color.basic': TOOL_BY_ID.color.parameters.map((item) => item.key), 'color.curves': ['channel', 'points'], 'color.hsl': ['hsl', 'channels', 'hue', 'saturation', 'lightness'],
  'filter.film': ['preset', 'seed', ...TOOL_BY_ID.film.parameters.map((item) => item.key)], 'filter.lut': ['size', 'data', 'amount', 'domainMin', 'domainMax'],
};
export const TOOL_PROTOCOL = {
  units: { slider: 'percentage', region: 'normalized', strokeCoordinates: 'normalized', strokeRadius: 'normalized-to-short-edge', crop: 'pixels', curve: 'normalized', opacity: 'percentage' },
  operations: OPERATION_PARAMETER_KEYS, common: COMMON_PARAMETER_KEYS,
  masks: ['rect', 'ellipse', 'polygon', 'brush', 'linear', 'radial'],
  blends: ['normal', 'multiply', 'screen', 'overlay', 'soft-light'],
  liquifyModes: ['push', 'pinch', 'bloat', 'twirl', 'reconstruct'],
  liquifyTwirlDirections: [-1, 1],
};

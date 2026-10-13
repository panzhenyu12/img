import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const registry = await import(pathToFileURL(path.join(root, 'apps/web/src/core/registry.ts')).href);
const engineVersion = fs.readFileSync(path.join(root, 'core/image-core/Cargo.toml'), 'utf8').match(/^version\s*=\s*"([^"]+)"/m)?.[1];
const registered = structuredClone(registry.PARAMETER_REGISTRY);
const semanticMaskNames = ['face', 'faceOutline', 'faceContour', 'faceContourMask', 'skin', 'neck', 'accessories', 'ears', 'skinProtection', 'eyes', 'leftEye', 'rightEye', 'brows', 'leftBrow', 'rightBrow', 'underEyes', 'leftUnderEye', 'rightUnderEye', 'lips', 'lip', 'teeth', 'hair', 'nose', 'noseBridge', 'noseWings', 'noseTip', 'noseRoot', 'noseBase', 'leftNostril', 'rightNostril'];
const semanticMasks = {
  type: 'object', maxProperties: semanticMaskNames.length,
  properties: Object.fromEntries(semanticMaskNames.map(name => [name, { oneOf: [{ $ref: 'mask.schema.json' }, { type: 'array', minItems: 3, maxItems: 256, items: { $ref: 'mask.schema.json#/$defs/point' } }] }])),
  additionalProperties: false,
  description: 'Named face feature masks; skin excludes eyes/brows/mouth, makeup uses its feature. Coordinates are normalized. Optional and independent of AI.',
};
const landmarks = {
  type: 'array',
  maxItems: 128,
  description: 'Canonical normalized anatomical landmarks (face:36; body:segment endpoints). Contours are separate semantic masks.',
  items: {
    type: 'object',
    required: ['x', 'y'],
    properties: {
      x: { min: 0, max: 1 }, y: { min: 0, max: 1 },
      toX: { min: 0, max: 1 }, toY: { min: 0, max: 1 },
      weight: { min: 0.01, max: 10, default: 1 },
    },
  },
};
for (const type of ['face.adjust', 'feature.adjust', 'body.adjust', 'clothing.adjust']) {
  const entry = registered[type];
  entry.keys = { adjustments: { type: 'object', properties: entry.keys }, landmarks };
}
for (const type of ['face.adjust', 'feature.adjust', 'skin.retouch']) registered[type].keys.semanticMasks = semanticMasks;
registered['skin.retouch'].keys.landmarks = landmarks;
registered['body.adjust'].keys.subjectMask = { $ref: 'mask.schema.json', description: 'Person segmentation alpha; raster row-major bytes, 1..512 dimensions, alpha moves with foreground. Required for body contour edits.' };
registered['body.adjust'].keys.partMask = { $ref: 'mask.schema.json', description: 'Target body part alpha, separate from the whole-person foreground; neighboring body parts remain fixed.' };
registered['face.adjust'].keys.subjectMask = { $ref: 'mask.schema.json', description: 'Full person alpha excluded from background donors during face reshaping' };
registered['body.adjust'].keys.bodyPart = {
  enum: ['full-body', 'torso', 'waist', 'hips', 'left-upper-arm', 'right-upper-arm', 'left-forearm', 'right-forearm', 'left-thigh', 'right-thigh', 'left-calf', 'right-calf'],
  description: 'Semantic Pose mask selected by the browser detector',
};
const point = { type: 'object', properties: { x: { min: 0, max: 1 }, y: { min: 0, max: 1 } }, required: ['x', 'y'] };
const controls = { type: 'array', maxItems: 1024, items: { inX: 'normalized source x', inY: 'normalized source y', outX: 'normalized destination x', outY: 'normalized destination y' } };
const stroke = { x: { min: 0, max: 1 }, y: { min: 0, max: 1 }, radius: { min: 0.00001, max: 1 }, strength: { min: 0, max: 100 } };
const catalog = {
  version: '1.0',
  engineVersion: `image-core-${engineVersion}`,
  coordinates: 'region/mask/controlPoints/strokes: normalized 0..1; crop/resize: pixels',
  sliderUnit: 'percent',
  operations: {
    identity: { keys: {} },
    crop: { keys: { x: { unit: 'px' }, y: { unit: 'px' }, width: { min: 1, unit: 'px' }, height: { min: 1, unit: 'px' } } },
    rotate: { keys: { degrees: { min: -360, max: 360 }, expand: { type: 'boolean', default: true } } },
    flip: { keys: { axis: { enum: ['horizontal', 'vertical'] } } },
    resize: { keys: { width: { min: 1, max: 16384 }, height: { min: 1, max: 16384 } } },
    perspective: { keys: { corners: { type: 'array', length: 4, order: ['TL', 'TR', 'BR', 'BL'] } } },
    ...registered,
    'warp.grid': { keys: { points: controls }, operationControlPoints: controls },
    'warp.liquify': { keys: { strokes: { type: 'array', maxItems: 4096, items: { ...stroke, toX: { min: -1, max: 2 }, toY: { min: -1, max: 2 }, mode: { enum: ['push', 'pinch', 'bloat', 'twirl', 'reconstruct'] }, direction: { enum: [-1, 1], default: 1 } } } }, description: 'Reconstruct restores this layer input; mirrored twirl strokes reverse direction' },
    'repair.clone': { keys: { strokes: { type: 'array', maxItems: 4096, items: { ...stroke, sourceX: { min: 0, max: 1 }, sourceY: { min: 0, max: 1 } }, required: ['x', 'y', 'radius', 'sourceX', 'sourceY'] } } },
    'repair.heal': { keys: { strokes: { type: 'array', maxItems: 4096, items: { ...stroke, sourceX: { min: 0, max: 1 }, sourceY: { min: 0, max: 1 } }, required: ['x', 'y', 'radius'] } } },
    'repair.inpaint': { keys: { radius: { min: 0.00001, max: 1, default: 0.025 }, iterations: { min: 1, max: 64, default: 6, type: 'integer' }, method: { enum: ['exemplar', 'lama'], default: 'exemplar' }, execution: {enum:['client','server'],default:'server'}, maxSide: {min:256,max:2048,default:1024,type:'integer'}, contextPadding: {min:64,max:1024,default:256,type:'integer'}, maskDilation: {min:0,max:32,default:4,type:'integer'}, seamBlend: {enum:['none','poisson'],default:'none'}, contextMode: {enum:['full','crop'],default:'full'} }, description: 'Offline Rust exemplar fill, browser ONNX LaMa (execution=client; maxSide<=512; seamBlend=none), or server LaMa model; the shared Rust mask compositor preserves unselected pixels, protection and alpha. LaMa is independent of the AI agent.' },
    'paint.brush': { keys: { color: { type: 'array', length: 3, min: 0, max: 255 }, strokes: { type: 'array', maxItems: 4096, items: { ...stroke, erase: { type: 'boolean' } } } }, description: 'Paint color onto a reversible layer; erase restores this layer to the source' },
    'color.curves': { keys: { channel: { enum: ['rgb', 'red', 'green', 'blue'] }, points: { type: 'array', maxItems: 1024, minItems: 2, items: point } } },
    'color.hsl': { keys: { hue: { min: -100, max: 100 }, saturation: { min: -100, max: 100 }, lightness: { min: -100, max: 100 }, channels: { type: 'array', items: { hue: { min: 0, max: 360 }, range: { min: 1, max: 180 }, shift: { min: -100, max: 100 }, saturation: { min: -100, max: 100 }, lightness: { min: -100, max: 100 } } } } },
    'filter.film': { ...registered['filter.film'], keys: { ...registered['filter.film'].keys, preset: { enum: registry.FILM_PRESETS.map(preset => preset.id) }, seed: { min: 0, max: 4294967295, type: 'integer', default: 42 } } },
    'filter.lut': { keys: { size: { min: 2, max: 65, type: 'integer' }, data: { type: 'array', length: 'size^3', order: 'red axis fastest', items: 'RGB tuple [r,g,b] with values 0..1' }, domainMin: { type: 'array', length: 3 }, domainMax: { type: 'array', length: 3 }, amount: { min: 0, max: 100 } } },
  },
  filmPresets: registry.FILM_PRESETS,
  commonParams: {
    region: 'normalized rectangle {x,y,width,height}',
    mask: 'inline {shape: rect|ellipse|polygon|brush|linear|radial|raster,width,height,data,region,points,strokes,from,to,invert,feather}',
    protectMask: 'inline mask; selected area stays unmodified',
    opacity: '0..100',
    blendMode: ['normal', 'multiply', 'screen', 'overlay', 'soft-light', 'darken', 'lighten'],
  },
};
for (const target of ['schemas/operation-catalog.json', 'services/api/internal/api/operation-catalog.json']) {
  fs.writeFileSync(path.join(root, target), JSON.stringify(catalog, null, 2) + '\n');
}
const parameterTools = registry.TOOLS.filter(tool => tool.type && tool.parameters.length).map(tool => ({...tool,parameters:registry.implementedParametersFor(tool.id)}));
const parameterCount = parameterTools.reduce((sum, tool) => sum + tool.parameters.length, 0);
const ledger = [
  '# FUNC-010 第一期参数逐项对账',
  '',
  '由 `scripts/generate_catalog.mjs` 从 Web registry 生成；修改参数后运行 `make catalog`，不要手工修改表格。',
  '',
  `共 ${parameterCount} 项注册滑杆，${Object.keys(catalog.operations).length} 类 Operation。名称、范围和默认值对应实际 UI；算法与交互实现分别见 [FUNC-007](FUNC-007-shared-rust-render-engine.md) 和 [FUNC-008](FUNC-008-web-editor-tools.md)。`,
  '',
  '用户标记区域后拉动参数；对称校正须具备可靠眼/眉/唇语义，Pose语义身形只显示已接入的粗细参数；Web WASM 与 Go native 使用同一 Rust 算法。脸型/五官/身形/衣物写入 `params.adjustments`；其他分类直接写入 `params`。区域、蒙版、笔画和图层透明度另见生成目录及 FUNC-008。',
  '',
  '验收依据：Playwright 逐注册参数执行真实 WASM，比较默认值与修改值的像素；皮肤/衣物纹理参数结合磨皮测试。像素变化仅表示执行成功。算法需分别验证参数方向、实际轮廓/面积、冻结区、透明度和取样隔离，真实照片另作效果核对；自然度未达专业标准的失败保留。见 [算法验证](../qa/algorithm-validation.md)、[多图人像](../qa/portrait-multisample-review.md) 与 [多图身形](../qa/body-multisample-review.md)。',
  '',
];
for (const tool of parameterTools) {
  ledger.push(`## ${tool.label}（${tool.parameters.length}项 / ${tool.type}）`, '', '| 分组 | 功能 | 参数 key | 最小 | 最大 | 默认 |', '|---|---|---|---:|---:|---:|');
  for (const parameter of tool.parameters) ledger.push(`| ${parameter.group} | ${parameter.label} | ${parameter.key} | ${parameter.min} | ${parameter.max} | ${parameter.default} |`);
  ledger.push('');
}
ledger.push('## 变更记录', '', '| 日期 | 变更 |', '|---|---|', '| 2026-10-12 | 建立一期注册参数逐项对账与统一目录自动生成。 |', '');
fs.writeFileSync(path.join(root, 'doc/func/FUNC-010-parameter-ledger.md'), ledger.join('\n'));
console.log('Generated operation catalog for AI and Go from Web parameter registry');

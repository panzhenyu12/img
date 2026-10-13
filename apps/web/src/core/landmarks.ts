import { FaceLandmarker, PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { RasterMask, InlineMask } from './operations';

export interface DetectionPoint { x: number; y: number; z?: number; visibility?: number }
export interface DetectionRegion { x: number; y: number; width: number; height: number }
export interface PolygonSemanticMask { shape: 'polygon'; points: DetectionPoint[]; feather?: number; invert?: boolean }
export type SemanticMask = PolygonSemanticMask | RasterMask;
/** A semantic body area derived from MediaPipe Pose landmarks. Points are an
 * ordered polygon in normalized image coordinates and landmarks are the
 * source control points used by the edit operation. */
export interface DetectedPart {
  id: string;
  label: string;
  region: DetectionRegion;
  points: DetectionPoint[];
  landmarks: DetectionPoint[];
  protectMask?: InlineMask;
}
export interface DetectedPerson {
  id: string;
  kind: 'face' | 'body';
  region: DetectionRegion;
  points: DetectionPoint[];
  parts?: DetectedPart[];
  semanticMasks?: Record<string, SemanticMask>;
  subjectMask?: RasterMask;
  unavailableParts?: Array<{ id: string; label: string; reason: string }>;
}

let face: Promise<FaceLandmarker> | undefined;
let pose: Promise<PoseLandmarker> | undefined;

function validPoint(point: DetectionPoint | undefined): point is DetectionPoint {
  return !!point && Number.isFinite(point.x) && Number.isFinite(point.y);
}
function clamp(value: number): number { return Math.max(0, Math.min(1, value)); }
function boxFromPoints(points: DetectionPoint[]): DetectionRegion {
  const valid = points.filter(validPoint);
  if (!valid.length) return { x: 0, y: 0, width: 1, height: 1 };
  const left = clamp(Math.min(...valid.map(p => p.x)));
  const top = clamp(Math.min(...valid.map(p => p.y)));
  const right = clamp(Math.max(...valid.map(p => p.x)));
  const bottom = clamp(Math.max(...valid.map(p => p.y)));
  return { x: left, y: top, width: Math.max(0.001, right - left), height: Math.max(0.001, bottom - top) };
}
function expandBox(region: DetectionRegion, pad = 0.012): DetectionRegion {
  const x = Math.max(0, region.x - pad); const y = Math.max(0, region.y - pad);
  const right = Math.min(1, region.x + region.width + pad); const bottom = Math.min(1, region.y + region.height + pad);
  return { x, y, width: Math.max(0.001, right - x), height: Math.max(0.001, bottom - y) };
}
function regionForPolygon(points: DetectionPoint[]): DetectionRegion { return expandBox(boxFromPoints(points), 0.008); }
function point(points: DetectionPoint[], index: number): DetectionPoint | undefined { return validPoint(points[index]) ? points[index] : undefined; }
function midpoint(a: DetectionPoint, b: DetectionPoint): DetectionPoint { return { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 }; }
function offset(point: DetectionPoint, dx: number, dy: number): DetectionPoint { return { x: clamp(point.x + dx), y: clamp(point.y + dy) }; }
/** A padded segment polygon. Width is proportional to the segment length, so
 * the mask follows the limb instead of covering a rectangular background. */
function segmentPolygon(a: DetectionPoint, b: DetectionPoint, widthRatio = 0.24, imageWidth = 1, imageHeight = 1): DetectionPoint[] {
  const dx = (b.x - a.x) * imageWidth; const dy = (b.y - a.y) * imageHeight; const length = Math.max(Math.hypot(dx, dy), .001);
  const half = length * widthRatio * 0.5; const nx = -dy / length * half / imageWidth; const ny = dx / length * half / imageHeight;
  return [offset(a, nx, ny), offset(b, nx, ny), offset(b, -nx, -ny), offset(a, -nx, -ny)];
}
function rasterCoverage(mask: RasterMask, x: number, y: number): number {
  if (x < 0 || y < 0 || x > 1 || y > 1) return 0;
  const px = Math.max(0, Math.min(mask.width - 1, x * mask.width - .5)); const py = Math.max(0, Math.min(mask.height - 1, y * mask.height - .5));
  const x0 = Math.floor(px), y0 = Math.floor(py), x1 = Math.min(mask.width - 1, x0 + 1), y1 = Math.min(mask.height - 1, y0 + 1), tx = px - x0, ty = py - y0;
  return (mask.data[y0 * mask.width + x0] * (1 - tx) * (1 - ty) + mask.data[y0 * mask.width + x1] * tx * (1 - ty) + mask.data[y1 * mask.width + x0] * (1 - tx) * ty + mask.data[y1 * mask.width + x1] * tx * ty) / 255;
}
/** A visible limb needs a nearby exterior edge. A bone hidden inside a skirt
 * or torso is not an independently editable thigh/arm silhouette. */
export function measuredLimbSupport(part: DetectedPart, mask: RasterMask, imageWidth: number, imageHeight: number, pairedThigh?: DetectedPart): DetectionPoint[] | undefined {
  const [a, b] = part.landmarks; if (!a || !b) return undefined;
  const dx = (b.x - a.x) * imageWidth, dy = (b.y - a.y) * imageHeight, length = Math.hypot(dx, dy);
  if (length < 4) return undefined;
  if (part.id.endsWith('thigh') && pairedThigh) {
    // Pose predicts bones under skirts and merged/crossed legs. A nearby
    // *outer* edge alone would incorrectly label a skirt as a thigh. Require
    // two distal cross-sections with a real background gap between the legs.
    const [otherA, otherB] = pairedThigh.landmarks;
    let separated = 0;
    for (const t of [.5, .6, .7, .8]) {
      const x = a.x + (b.x-a.x)*t, y = a.y + (b.y-a.y)*t;
      const ox = otherA.x + (otherB.x-otherA.x)*t, oy = otherA.y + (otherB.y-otherA.y)*t;
      const span = Math.hypot((ox-x)*imageWidth,(oy-y)*imageHeight);
      if (span < 4) continue;
      const count = Math.ceil(span);
      let gap = 0;
      for (let i = 1; i < count; i++) {
        if (rasterCoverage(mask,x+(ox-x)*i/count,y+(oy-y)*i/count)<.5) gap++;
        else gap = 0;
        if (gap >= 2) { separated++; break; }
      }
    }
    if (separated < 2) return undefined;
  }
  const nx = -dy / length, ny = dx / length;
  const spans: Array<{t:number;radius:number}> = [];
  for (const t of [.2, .3, .4, .5, .6, .7, .8]) {
    const x = a.x * imageWidth + dx * t, y = a.y * imageHeight + dy * t;
    if (rasterCoverage(mask, x / imageWidth, y / imageHeight) < .5) continue;
    const sides = [-1, 1].map(sign => {
      for (let d = 1; d <= length * .7; d++) if (rasterCoverage(mask, (x + nx * sign * d) / imageWidth, (y + ny * sign * d) / imageHeight) < .5) return d;
      return Infinity;
    });
    const near = Math.min(...sides); if (Number.isFinite(near)) spans.push({t,radius:near});
  }
  if (spans.length < 2) return undefined;
  const median = (items: number[]) => [...items].sort((x,y)=>x-y)[Math.floor(items.length/2)];
  const radius = median(spans.map(span=>span.radius));
  if (part.id.endsWith('thigh') && radius > length * .38) return undefined;
  if (part.id.endsWith('calf')) {
    const proximal = spans.filter(span=>span.t<=.3).map(span=>span.radius);
    const distal = spans.filter(span=>span.t>=.7).map(span=>span.radius);
    // A broad sheet around the predicted knee followed by an abrupt thin
    // visible lower leg indicates a skirt/occlusion, not a complete calf.
    if (radius > length*.42 || proximal.length>=2 && distal.length>=2 && median(proximal)>length*.25 && median(proximal)>median(distal)*2.3) return undefined;
  }
  const half = Math.max(3, radius * 1.45 + 2);
  return [offset(a, nx * half / imageWidth, ny * half / imageHeight), offset(b, nx * half / imageWidth, ny * half / imageHeight), offset(b, -nx * half / imageWidth, -ny * half / imageHeight), offset(a, -nx * half / imageWidth, -ny * half / imageHeight)];
}
function torsoPolygon(points: DetectionPoint[], shoulderPad = 0.08): DetectionPoint[] {
  const leftShoulder = point(points, 11); const rightShoulder = point(points, 12);
  const leftHip = point(points, 23); const rightHip = point(points, 24);
  if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return [];
  const shoulderDx = rightShoulder.x - leftShoulder.x; const shoulderDy = rightShoulder.y - leftShoulder.y;
  const length = Math.max(Math.hypot(shoulderDx, shoulderDy), 0.01); const nx = -shoulderDy / length * length * shoulderPad; const ny = shoulderDx / length * length * shoulderPad;
  return [offset(leftShoulder, nx, ny), offset(rightShoulder, nx, ny), offset(rightHip, -nx * 0.7, -ny * 0.7), offset(leftHip, -nx * 0.7, -ny * 0.7)];
}
function convexHull(points: DetectionPoint[]): DetectionPoint[] {
  const valid = points.filter(validPoint).map(p => ({ x: clamp(p.x), y: clamp(p.y) })).sort((a, b) => a.x - b.x || a.y - b.y);
  const unique: DetectionPoint[] = [];
  for (const p of valid) { const previous = unique.at(-1); if (!previous || Math.abs(previous.x - p.x) > 1e-6 || Math.abs(previous.y - p.y) > 1e-6) unique.push(p); }
  if (unique.length < 3) return unique;
  const cross = (o: DetectionPoint, a: DetectionPoint, b: DetectionPoint) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower: DetectionPoint[] = []; for (const p of unique) { while (lower.length >= 2 && cross(lower.at(-2)!, lower.at(-1)!, p) <= 0) lower.pop(); lower.push(p); }
  const upper: DetectionPoint[] = []; for (const p of [...unique].reverse()) { while (upper.length >= 2 && cross(upper.at(-2)!, upper.at(-1)!, p) <= 0) upper.pop(); upper.push(p); }
  lower.pop(); upper.pop(); return [...lower, ...upper];
}
const LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
const LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];
const LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
const RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];
const LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384];
/** Canonical face oval, in perimeter order. Never substitute a convex hull or
 * an index-strided sample of the dense mesh for the silhouette. */
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const NOSE = [168, 193, 122, 196, 3, 51, 45, 220, 115, 48, 64, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 281, 248, 419, 351, 417];
const NOSE_BRIDGE = [168, 193, 122, 196, 3, 4, 248, 419, 351, 417];
const NOSE_WINGS = [49, 48, 64, 98, 97, 2, 326, 327, 294, 278, 279, 5];
const NOSE_TIP = [5, 45, 220, 115, 48, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275];
const NOSE_ROOT = [168, 193, 122, 6, 351, 417];
const NOSE_BASE = [64, 98, 97, 2, 326, 327, 294, 19];
const LEFT_NOSTRIL = [64, 98, 97, 2, 19];
const RIGHT_NOSTRIL = [294, 327, 326, 2, 19];

function semanticPolygon(points: DetectionPoint[], indices: number[]): DetectionPoint[] {
  const selected = indices.map((index) => point(points, index)).filter(validPoint).map(({x,y}) => ({ x:clamp(x), y:clamp(y) }));
  // Canonical eye/brow/lip indices follow the actual perimeter. A convex hull
  // would fill the cupid's bow and bend of the brow with makeup.
  return selected;
}
/** Derive stable face feature masks from MediaPipe's canonical 468-point mesh.
 * These polygons are serialized with an operation, keeping preview/export and
 * future AI callers on the same semantic boundaries. */
export function faceSemanticMasks(points: DetectionPoint[], imageWidth = 1, imageHeight = 1): Record<string, PolygonSemanticMask> {
  const mask = (indices: number[]): PolygonSemanticMask | undefined => {
    const polygon = semanticPolygon(points, indices);
    return polygon.length >= 3 ? { shape: 'polygon', points: polygon, feather: 2, invert: false } : undefined;
  };
  const outline = semanticPolygon(points, FACE_OVAL);
  const masks: Record<string, PolygonSemanticMask> = {};
  if (outline.length >= 3) {
    masks.face = { shape: 'polygon', points: outline, feather: 2, invert: false };
    masks.faceOutline = { shape: 'polygon', points: outline, feather: 2, invert: false };
    masks.skin = { shape: 'polygon', points: outline, feather: 2, invert: false };
  }
  const lips = mask(LIP_OUTER); if (lips) masks.lips = lips;
  const teeth = mask(LIP_INNER); if (teeth) masks.teeth = teeth;
  const leftBrow = mask(LEFT_BROW); const rightBrow = mask(RIGHT_BROW);
  if (leftBrow) masks.leftBrow = leftBrow; if (rightBrow) masks.rightBrow = rightBrow;
  const leftEye = mask(LEFT_EYE); const rightEye = mask(RIGHT_EYE);
  if (leftEye) masks.leftEye = leftEye; if (rightEye) masks.rightEye = rightEye;
  for (const [name, indices] of Object.entries({ nose: NOSE, noseBridge: NOSE_BRIDGE, noseWings: NOSE_WINGS, noseTip: NOSE_TIP, noseRoot: NOSE_ROOT, noseBase: NOSE_BASE, leftNostril: LEFT_NOSTRIL, rightNostril: RIGHT_NOSTRIL })) {
    const polygon = mask(indices); if (polygon) masks[name] = polygon;
  }
  const underEye = (indices: number[]): PolygonSemanticMask | undefined => {
    const eye = semanticPolygon(points, indices); if (eye.length < 3 || !leftEye || !rightEye || !lips) return undefined;
    const center = (items: DetectionPoint[]) => ({ x: items.reduce((s,p) => s+p.x,0)/items.length, y: items.reduce((s,p) => s+p.y,0)/items.length });
    const l = center(leftEye.points); const r = center(rightEye.points); const mouth = center(lips.points);
    const dx = (r.x-l.x)*imageWidth; const dy = (r.y-l.y)*imageHeight; const length = Math.max(Math.hypot(dx,dy), 1e-6);
    let nx = -dy/length; let ny = dx/length;
    if (nx*(mouth.x-(l.x+r.x)/2)*imageWidth+ny*(mouth.y-(l.y+r.y)/2)*imageHeight<0) { nx=-nx; ny=-ny; }
    const projections = eye.map(p => p.x*imageWidth*dx/length+p.y*imageHeight*dy/length);
    const depth = (Math.max(...projections)-Math.min(...projections))*0.45;
    const polygon = convexHull([...eye, ...eye.map(p => offset(p, nx*depth/imageWidth, ny*depth/imageHeight))]);
    return { shape: 'polygon', points: polygon, feather: 2, invert: false };
  };
  const underLeft = underEye(LEFT_EYE); const underRight = underEye(RIGHT_EYE);
  if (underLeft) masks.leftUnderEye = underLeft; if (underRight) masks.rightUnderEye = underRight;
  return masks;
}

/** Build masks for torso, waist, hips and each arm/leg. Missing landmarks are
 * skipped; the whole-person hull remains available as a safe fallback. */
export function bodyParts(points: DetectionPoint[], imageWidth = 1, imageHeight = 1): DetectedPart[] {
  const originalPoints = points;
  // Face landmarks do not carry useful visibility. Apply this confidence
  // rule only to Pose joints, retaining canonical indices for full-body.
  points = points.map(p => (p.visibility !== undefined && p.visibility < 0.5) || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1 ? { ...p, x: NaN } : p);
  const definitions: Array<{ id: string; label: string; polygon: DetectionPoint[]; landmarks: DetectionPoint[] }> = [];
  const add = (id: string, label: string, polygon: DetectionPoint[], landmarks: DetectionPoint[] = polygon) => {
    if (polygon.length >= 3 && landmarks.filter(validPoint).length >= 2) definitions.push({ id, label, polygon, landmarks: id === 'full-body' ? landmarks : landmarks.filter(validPoint) });
  };
  const shoulders = [point(points, 11), point(points, 12)].filter(validPoint);
  const hips = [point(points, 23), point(points, 24)].filter(validPoint);
  const torso = torsoPolygon(points);
  add('torso', '躯干', torso, [...shoulders, ...hips]);
  if (torso.length >= 4) {
    const leftShoulder = point(points, 11)!; const rightShoulder = point(points, 12)!; const leftHip = point(points, 23)!; const rightHip = point(points, 24)!;
    const shoulderCenter = midpoint(leftShoulder, rightShoulder); const hipCenter = midpoint(leftHip, rightHip);
    const waistCenter = { x: shoulderCenter.x * 0.35 + hipCenter.x * 0.65, y: shoulderCenter.y * 0.35 + hipCenter.y * 0.65 };
    const half = Math.max(Math.abs(rightHip.x - leftHip.x) * 0.34, 0.015);
    const h = Math.max(Math.abs(hipCenter.y - shoulderCenter.y) * 0.26, 0.02);
    add('waist', '腰部', [offset(waistCenter, -half, -h), offset(waistCenter, half, -h), offset(waistCenter, half, h), offset(waistCenter, -half, h)], [shoulderCenter, waistCenter, hipCenter]);
    const hipHalf = Math.max(Math.abs(rightHip.x - leftHip.x) * 0.62, 0.02); const hipH = Math.max(Math.abs(hipCenter.y - shoulderCenter.y) * 0.2, 0.025);
    add('hips', '胯部', [offset(hipCenter, -hipHalf, -hipH), offset(hipCenter, hipHalf, -hipH), offset(hipCenter, hipHalf, hipH), offset(hipCenter, -hipHalf, hipH)], [leftHip, rightHip, hipCenter]);
  }
  const limb = (side: 'left' | 'right', shoulder: number, elbow: number, wrist: number, hip: number, knee: number, ankle: number) => {
    const s = point(points, shoulder); const e = point(points, elbow); const w = point(points, wrist); const h = point(points, hip); const k = point(points, knee); const a = point(points, ankle);
    if (s && e) add(`${side}-upper-arm`, `${side === 'left' ? '左' : '右'}上臂`, segmentPolygon(s, e, 0.34, imageWidth, imageHeight), [s, e]);
    if (e && w) add(`${side}-forearm`, `${side === 'left' ? '左' : '右'}前臂`, segmentPolygon(e, w, 0.28, imageWidth, imageHeight), [e, w]);
    if (h && k) add(`${side}-thigh`, `${side === 'left' ? '左' : '右'}大腿`, segmentPolygon(h, k, 0.42, imageWidth, imageHeight), [h, k]);
    if (k && a) add(`${side}-calf`, `${side === 'left' ? '左' : '右'}小腿`, segmentPolygon(k, a, 0.34, imageWidth, imageHeight), [k, a]);
  };
  limb('left', 11, 13, 15, 23, 25, 27); limb('right', 12, 14, 16, 24, 26, 28);
  const all = convexHull(points);
  if ([11,12,13,14,15,16,23,24,25,26,27,28].every(index => validPoint(points[index]))) add('full-body', '全身', all, originalPoints.map(p => ({ x: clamp(p.x), y: clamp(p.y) })));
  // Put the complete silhouette first so the normal “自动检测” action has a
  // safe default. The remaining entries are independently selectable masks.
  definitions.sort((a, b) => (a.id === 'full-body' ? -1 : b.id === 'full-body' ? 1 : 0));
  return definitions.map((part, index) => ({ id: part.id || `part-${index + 1}`, label: part.label, region: regionForPolygon(part.polygon), points: part.polygon, landmarks: part.landmarks }));
}

function occludingArmProtection(points: DetectionPoint[], imageWidth: number, imageHeight: number, includeUpperArms: boolean): InlineMask {
  const shortEdge = Math.min(imageWidth,imageHeight);
  const strokes: NonNullable<InlineMask['strokes']> = [];
  const posePoint = (index: number) => {
    const p=points[index]; return validPoint(p) && (p.visibility === undefined || p.visibility>=0.5) ? p : undefined;
  };
  const segment = (a: DetectionPoint,b: DetectionPoint,ratio: number) => {
    const length=Math.hypot((b.x-a.x)*imageWidth,(b.y-a.y)*imageHeight);
    const radius=Math.max(.012,length/shortEdge*ratio);
    const count=Math.max(1,Math.ceil(length/(radius*shortEdge*.65)));
    for(let i=0;i<=count;i++) strokes.push({x:clamp(a.x+(b.x-a.x)*i/count),y:clamp(a.y+(b.y-a.y)*i/count),radius,strength:100});
  };
  for(const [shoulder,elbow,wrist,fingers] of [[11,13,15,[17,19,21]],[12,14,16,[18,20,22]]] as const) {
    const s=posePoint(shoulder); const e=posePoint(elbow); const w=posePoint(wrist);
    if(includeUpperArms&&s&&e) segment(s,e,.20);
    if(e&&w) segment(e,w,.19);
    if(w) {
      const fingertips=fingers.map(posePoint).filter(validPoint);
      const radius=Math.max(.025,...fingertips.map(p=>Math.hypot((p.x-w.x)*imageWidth,(p.y-w.y)*imageHeight)/shortEdge*1.15));
      strokes.push({x:clamp(w.x),y:clamp(w.y),radius,strength:100});
    }
  }
  return {shape:'brush',strokes,invert:false,feather:0};
}

/** Optional, local model inference; manual regions need no models or network. */
export async function detectPeople(source: HTMLImageElement | HTMLCanvasElement, kind: 'face' | 'body' = 'face'): Promise<DetectedPerson[]> {
  const vision = await FilesetResolver.forVisionTasks('/vision');
  if (kind === 'face') {
    face ??= FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate: 'CPU' },
      runningMode: 'IMAGE', numFaces: 8,
    }).catch(error => { face = undefined; throw error; });
    const result = (await face).detect(source);
    const imageWidth = source instanceof HTMLImageElement ? source.naturalWidth : source.width;
    const imageHeight = source instanceof HTMLImageElement ? source.naturalHeight : source.height;
    return Promise.all(result.faceLandmarks.map(async (mesh, i) => {
      const points = semanticPolygon(mesh, FACE_OVAL);
      const bounds = boxFromPoints(points);
      // The edit support extends beyond the original silhouette. The face
      // semantic polygon remains the source foreground alpha.
      const padx = bounds.width * .2; const pady = bounds.height * .2;
      const x = Math.max(0, bounds.x - padx); const y = Math.max(0, bounds.y - pady);
      const region = { x, y, width: Math.min(1, bounds.x + bounds.width + padx) - x, height: Math.min(1, bounds.y + bounds.height + pady) - y };
      const person: DetectedPerson = { id: `face-${i + 1}`, kind, region, points, semanticMasks: faceSemanticMasks(mesh, imageWidth, imageHeight) };
      try {
        const { parseFace } = await import('./parsing');
        const parsed = await parseFace(source, person);
        const semanticMasks: Record<string, SemanticMask> = { ...person.semanticMasks, ...parsed.semanticMasks };
        const outline = semanticMasks.faceOutline;
        if (!outline || outline.shape !== 'polygon' || outline.points.length < 3) throw new Error('缺少像素脸部轮廓');
        const silhouette = boxFromPoints(outline.points);
        const sx = Math.max(0, silhouette.x - silhouette.width * .2); const sy = Math.max(0, silhouette.y - silhouette.height * .2);
        const support = { x: sx, y: sy, width: Math.min(1, silhouette.x + silhouette.width * 1.2) - sx, height: Math.min(1, silhouette.y + silhouette.height * 1.2) - sy };
        // Canonical 36 points locate anatomical parameters. Pixel parsing
        // defines foreground membership and the independent support bounds.
        return { ...person, region: support, semanticMasks, subjectMask: parsed.subjectMask };
      } catch (error) {
        throw new Error(`人脸像素解析不可用：${error instanceof Error ? error.message : String(error)}；可用套索手工标记真实轮廓`);
      }
    }));
  }
  pose ??= PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: '/models/pose_landmarker_lite.task', delegate: 'CPU' },
    runningMode: 'IMAGE', numPoses: 4, outputSegmentationMasks: true,
  }).catch(error => { pose = undefined; throw error; });
  const result = (await pose).detect(source);
  const imageWidth = source instanceof HTMLImageElement ? source.naturalWidth : source.width;
  const imageHeight = source instanceof HTMLImageElement ? source.naturalHeight : source.height;
  try {
    return result.landmarks.map((points, i) => {
      const segmentation = result.segmentationMasks?.[i];
      let subjectMask: RasterMask | undefined;
      if (segmentation) {
        const values = segmentation.getAsFloat32Array();
        const scale = Math.min(1, 256/Math.max(segmentation.width,segmentation.height));
        const width = Math.max(1,Math.round(segmentation.width*scale)); const height = Math.max(1,Math.round(segmentation.height*scale));
        const data = Array.from({ length: width*height }, (_,index) => {
          const x = Math.min(segmentation.width-1, Math.floor((index%width+0.5)*segmentation.width/width));
          const y = Math.min(segmentation.height-1, Math.floor((Math.floor(index/width)+0.5)*segmentation.height/height));
          // Suppress weak background probabilities and keep an antialiased edge.
          const alpha = clamp((values[y*segmentation.width+x]-0.35)/0.3);
          return Math.round(alpha*255);
        });
        subjectMask = { shape: 'raster', width, height, data };
      }
      const unavailableParts: Array<{ id: string; label: string; reason: string }> = [];
      const candidates = bodyParts(points, imageWidth, imageHeight);
      const parts = candidates.flatMap(part => {
        if (!subjectMask) return [part];
        // The support polygon is larger than the silhouette so it can move
        // across its original boundary. The raster alpha retains limb gaps.
        const cx = part.region.x+part.region.width/2; const cy = part.region.y+part.region.height/2;
        const limb = /(?:upper-arm|forearm|thigh|calf)$/.test(part.id);
        const pairedThigh = part.id.endsWith('thigh') ? candidates.find(other => other.id === (part.id.startsWith('left') ? 'right-thigh' : 'left-thigh')) : undefined;
        const measured = limb ? measuredLimbSupport(part, subjectMask, imageWidth, imageHeight, pairedThigh) : undefined;
        if (limb && !measured) { unavailableParts.push({ id: part.id, label: part.label, reason: '独立肢体轮廓不可见，可能被衣物或其他身体部位遮挡' }); return []; }
        const support = measured ?? part.points.map(p => ({ x: clamp(cx+(p.x-cx)*1.5), y: clamp(cy+(p.y-cy)*1.5) }));
        const protectMask = ['waist','hips','torso'].includes(part.id) ? occludingArmProtection(points,imageWidth,imageHeight,part.id!=='torso') : undefined;
        return [{ ...part, points: support, region: regionForPolygon(support), protectMask }];
      });
      const available = unavailableParts.length ? parts.filter(part => part.id !== 'full-body') : parts;
      return { id: `body-${i + 1}`, kind, region: boxFromPoints(points), points, parts: available, subjectMask, unavailableParts };
    });
  } finally { result.close(); }
}

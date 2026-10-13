export interface InpaintImage { width: number; height: number; pixels: Uint8Array }
export interface ClientInpaintOptions { maxSide?: number; contextMode?: 'full' | 'crop'; contextPadding?: number; maskDilation?: number; seamBlend?: 'none' | 'poisson' }
export interface PreparedInpaint {
  image: Float32Array; mask: Float32Array; box: [number, number, number, number];
  scaledWidth: number; scaledHeight: number;
}
export const INPAINT_SIDE = 512;

function symmetric(index: number, length: number): number {
  const p = index % (length * 2); return p < length ? p : length * 2 - p - 1;
}

/** Manhattan distance gives scipy's default cross dilation in two O(N)
 * passes, without repeatedly traversing the entire photograph. */
function dilate(target: Uint8Array, width: number, height: number, radius: number): Uint8Array {
  if (!radius) return Uint8Array.from(target, v => v ? 1 : 0);
  const distances = new Uint16Array(target.length).fill(radius + 1);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = y * width + x;
    distances[i] = target[i] ? 0 : Math.min(distances[i], x ? distances[i - 1] + 1 : radius + 1, y ? distances[i - width] + 1 : radius + 1);
  }
  for (let y = height - 1; y >= 0; y--) for (let x = width - 1; x >= 0; x--) {
    const i = y * width + x;
    distances[i] = Math.min(distances[i], x + 1 < width ? distances[i + 1] + 1 : radius + 1, y + 1 < height ? distances[i + width] + 1 : radius + 1);
  }
  return Uint8Array.from(distances, d => d <= radius ? 1 : 0);
}

/** Equal positive area footprints for RGB and masks avoid dark context halos.
 * Unknown/protected/transparent RGB is removed before resampling. Aspect ratio
 * is preserved; symmetric padding supplies the fixed square ONNX input. */
export function prepareClientInpaint(source: InpaintImage, target: Uint8Array, protection: Uint8Array, options: ClientInpaintOptions = {}): PreparedInpaint {
  const {width, height, pixels} = source;
  if (pixels.length !== width * height * 4 || target.length !== width * height || protection.length !== target.length) throw new Error('模型图片与蒙版尺寸不匹配');
  const {maxSide = 512, contextPadding = 256, maskDilation = 4, contextMode = 'full', seamBlend = 'none'} = options;
  if (!Number.isInteger(maxSide) || maxSide < 256 || maxSide > 512 || !Number.isInteger(contextPadding) || contextPadding < 64 || contextPadding > 1024 || !Number.isInteger(maskDilation) || maskDilation < 0 || maskDilation > 32 || !['full', 'crop'].includes(contextMode) || seamBlend !== 'none') throw new Error('客户端模型参数不支持：最长边最多 512，接缝方式为 none');
  let left = width, top = height, right = 0, bottom = 0;
  const selected = new Uint8Array(target.length);
  for (let i = 0; i < target.length; i++) if (target[i] && !protection[i] && pixels[i * 4 + 3]) {
    selected[i] = 1; const x = i % width, y = Math.floor(i / width);
    left = Math.min(left, x); top = Math.min(top, y); right = Math.max(right, x + 1); bottom = Math.max(bottom, y + 1);
  }
  if (left >= right || top >= bottom) throw new Error('没有可填补的选区');
  if (contextMode === 'full') {left = 0; top = 0; right = width; bottom = height;}
  else {left = Math.max(0, left - contextPadding); top = Math.max(0, top - contextPadding); right = Math.min(width, right + contextPadding); bottom = Math.min(height, bottom + contextPadding);}
  const cropWidth = right - left, cropHeight = bottom - top;
  const scale = Math.min(1, maxSide / Math.max(cropWidth, cropHeight));
  const sw = Math.max(1, Math.round(cropWidth * scale)), sh = Math.max(1, Math.round(cropHeight * scale));
  const unknown = dilate(selected, width, height, maskDilation);
  for (let i = 0; i < unknown.length; i++) if (protection[i] || !pixels[i * 4 + 3]) unknown[i] = 1;
  const rgb = new Float32Array(sw * sh * 3), mask = new Uint8Array(sw * sh);
  for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
    const x0 = x * cropWidth / sw, x1 = (x + 1) * cropWidth / sw;
    const y0 = y * cropHeight / sh, y1 = (y + 1) * cropHeight / sh;
    const i = y * sw + x;
    let r = 0, g = 0, b = 0;
    for (let sy = Math.floor(y0); sy < Math.ceil(y1); sy++) for (let sx = Math.floor(x0); sx < Math.ceil(x1); sx++) {
      const weight = (Math.min(x1, sx + 1) - Math.max(x0, sx)) * (Math.min(y1, sy + 1) - Math.max(y0, sy));
      if (weight <= 0) continue;
      const j = (top + sy) * width + left + sx;
      if (unknown[j]) mask[i] = 1;
      else { r += pixels[j * 4] * weight; g += pixels[j * 4 + 1] * weight; b += pixels[j * 4 + 2] * weight; }
    }
    if (!mask[i]) { const norm = 1 / ((x1 - x0) * (y1 - y0) * 255); rgb[i * 3] = r * norm; rgb[i * 3 + 1] = g * norm; rgb[i * 3 + 2] = b * norm; }
  }
  if (mask.every(v => !!v)) throw new Error('选区周围需要保留可用的背景');
  const count = INPAINT_SIDE * INPAINT_SIDE;
  const paddedRgb = new Float32Array(count * 3), paddedMask = new Float32Array(count);
  for (let y = 0; y < INPAINT_SIDE; y++) for (let x = 0; x < INPAINT_SIDE; x++) {
    const i = y * INPAINT_SIDE + x, j = symmetric(y, sh) * sw + symmetric(x, sw);
    paddedMask[i] = mask[j];
    for (let c = 0; c < 3; c++) paddedRgb[c * count + i] = rgb[j * 3 + c];
  }
  return {image: paddedRgb, mask: paddedMask, box: [left, top, right, bottom], scaledWidth: sw, scaledHeight: sh};
}

/** This pinned export returns NCHW float RGB in [0,255]. Restore only the
 * canonical target. Final feather/opacity/blend is applied once by Rust. */
export function restoreClientInpaint(source: InpaintImage, target: Uint8Array, protection: Uint8Array, prepared: PreparedInpaint, prediction: Float32Array): Uint8Array {
  const count = INPAINT_SIDE * INPAINT_SIDE;
  if (prediction.length !== count * 3) throw new Error('客户端模型返回了错误尺寸');
  const out = new Uint8Array(source.pixels);
  const [left, top, right, bottom] = prepared.box, {scaledWidth: sw, scaledHeight: sh} = prepared;
  for (let y = top; y < bottom; y++) for (let x = left; x < right; x++) {
    const i = y * source.width + x;
    if (!target[i] || protection[i] || !source.pixels[i * 4 + 3]) continue;
    const fx = Math.max(0, Math.min(sw - 1, (x - left + .5) * sw / (right - left) - .5));
    const fy = Math.max(0, Math.min(sh - 1, (y - top + .5) * sh / (bottom - top) - .5));
    const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(sw - 1, x0 + 1), y1 = Math.min(sh - 1, y0 + 1), tx = fx - x0, ty = fy - y0;
    for (let c = 0; c < 3; c++) {
      const offset = c * count;
      const value = (prediction[offset + y0 * INPAINT_SIDE + x0] * (1 - tx) + prediction[offset + y0 * INPAINT_SIDE + x1] * tx) * (1 - ty) + (prediction[offset + y1 * INPAINT_SIDE + x0] * (1 - tx) + prediction[offset + y1 * INPAINT_SIDE + x1] * tx) * ty;
      if (!Number.isFinite(value)) throw new Error('客户端模型产生了无效像素');
      out[i * 4 + c] = Math.max(0, Math.min(255, Math.round(value)));
    }
  }
  return out;
}

export function packInpaintRgb(source: InpaintImage, target: Uint8Array, protection: Uint8Array, candidate: Uint8Array, budget: number): Uint8Array {
  let count = 0;
  for (let i = 0; i < target.length; i++) if (target[i] && !protection[i] && source.pixels[i * 4 + 3]) count++;
  // Store only selected RGB; a 12MP photo with a small person should not evict
  // its candidate just because its unchanged full RGBA exceeds 32MiB.
  if (count * 3 > budget) return new Uint8Array();
  const bytes = new Uint8Array(count * 3); let cursor = 0;
  for (let i = 0; i < target.length; i++) if (target[i] && !protection[i] && source.pixels[i * 4 + 3]) {bytes.set(candidate.subarray(i * 4, i * 4 + 3), cursor); cursor += 3;}
  return bytes;
}

export function restoreInpaintRgb(source: InpaintImage, target: Uint8Array, protection: Uint8Array, bytes: Uint8Array): Uint8Array {
  const candidate = new Uint8Array(source.pixels); let cursor = 0;
  for (let i = 0; i < target.length; i++) if (target[i] && !protection[i] && source.pixels[i * 4 + 3]) {candidate.set(bytes.subarray(cursor, cursor + 3), i * 4); cursor += 3;}
  if (cursor !== bytes.length) throw new Error('客户端模型缓存尺寸不匹配');
  return candidate;
}

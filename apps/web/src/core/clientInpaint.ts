import * as ort from 'onnxruntime-web/webgpu';
import ortModuleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs?url';
import ortWasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url';
import { INPAINT_SIDE, prepareClientInpaint, restoreClientInpaint, type ClientInpaintOptions, type InpaintImage, packInpaintRgb, restoreInpaintRgb } from './inpaintPreprocess';

const MODEL_SHA = '1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6';
let session: Promise<ort.InferenceSession> | undefined;
let backend = 'wasm';
const candidates = new Map<string, Uint8Array>();
let candidateBytes = 0;
const CACHE_BYTES = 32 * 1024 * 1024;

/** Integrity is checked before writing disk cache; every bad entry is removed
 * so replacing a broken model file can recover without clearing site data. */
export async function loadClientInpaintModel(): Promise<ArrayBuffer> {
  const url = new URL(`/models/lama-fp32-512.onnx?sha=${MODEL_SHA}`, self.location.origin).href;
  let cache: Cache | undefined, response: Response | undefined;
  try {cache = await caches.open('retouch-inpaint-model-v1'); response = await cache.match(url);} catch { /* Private mode may disallow persistent cache. */ }
  const fromCache = !!response;
  if (!response) response = await fetch(url, {cache: 'reload'});
  const rejectModel = async (message: string): Promise<never> => {if (cache) await cache.delete(url).catch(() => false); throw new Error(message);};
  if (!response.ok || response.headers.get('content-type')?.includes('text/html')) return rejectModel('未安装客户端去人模型，请运行 make client-models');
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== 208044816) return rejectModel('客户端去人模型文件不完整，请重新运行 make client-models');
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', buffer));
  if (Array.from(digest, v => v.toString(16).padStart(2, '0')).join('') !== MODEL_SHA) return rejectModel('客户端去人模型校验失败，请重新运行 make client-models');
  if (cache && !fromCache) try {await cache.put(url, new Response(buffer, {headers:{'Content-Type':'application/octet-stream'}}));} catch { /* Inference can work without disk quota. */ }
  return buffer;
}

async function loadSession(): Promise<ort.InferenceSession> {
  if (!session) session = (async () => {
    ort.env.wasm.wasmPaths = {mjs: new URL(ortModuleUrl, self.location.origin).href, wasm: new URL(ortWasmUrl, self.location.origin).href};
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    ort.env.wasm.proxy = false;
    const buffer = await loadClientInpaintModel();
    if ('gpu' in navigator && await navigator.gpu.requestAdapter().catch(() => null)) {
      try {const result = await ort.InferenceSession.create(buffer, {executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all'}); backend = 'webgpu eligible, wasm fallback allowed'; return result;}
      catch { /* Unsupported GPU/ops keep processing local via WASM. */ }
    }
    backend = 'wasm';
    return ort.InferenceSession.create(buffer, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
  })().catch(error => {session = undefined; throw error;});
  return session;
}

async function candidateKey(source: InpaintImage, target: Uint8Array, protection: Uint8Array, options: ClientInpaintOptions): Promise<string> {
  const hashes = await Promise.all([source.pixels, target, protection].map(async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new Uint8Array(bytes))), v => v.toString(16).padStart(2,'0')).join('')));
  return JSON.stringify([MODEL_SHA, 'area-mask-v1', source.width, source.height, ...hashes, options.maxSide ?? 512, options.contextMode ?? 'full', options.contextPadding ?? 256, options.maskDilation ?? 4, options.seamBlend ?? 'none']);
}

/** Runs only in the browser; failures never send pixels to a remote API. */
export async function clientInpaint(source: InpaintImage, target: Uint8Array, protection: Uint8Array, options: ClientInpaintOptions = {}): Promise<Uint8Array> {
  if (!target.some(v => v)) return new Uint8Array(source.pixels);
  const key = await candidateKey(source, target, protection, options);
  const cached = candidates.get(key);
  if (cached) {candidates.delete(key); candidates.set(key, cached); return restoreInpaintRgb(source,target,protection,cached);}
  const started = performance.now();
  const prepared = prepareClientInpaint(source, target, protection, options);
  const model = await loadSession();
  const image = new ort.Tensor('float32', prepared.image, [1,3,INPAINT_SIDE,INPAINT_SIDE]);
  const mask = new ort.Tensor('float32', prepared.mask, [1,1,INPAINT_SIDE,INPAINT_SIDE]);
  let output: Record<string, ort.Tensor> | undefined;
  try {
    const inferenceStart = performance.now();
    output = await model.run({image, mask});
    const result = output[model.outputNames[0]];
    if (result.type !== 'float32' || result.dims.join(',') !== '1,3,512,512') throw new Error('客户端去人模型输出协议不匹配');
    const candidate = restoreClientInpaint(source, target, protection, prepared, result.data as Float32Array);
    const bytes = packInpaintRgb(source,target,protection,candidate,CACHE_BYTES);
    while (candidateBytes + bytes.byteLength > CACHE_BYTES && candidates.size) {const oldest = candidates.keys().next().value!; candidateBytes -= candidates.get(oldest)!.byteLength; candidates.delete(oldest);}
    if (bytes.byteLength) {candidates.set(key, bytes); candidateBytes += bytes.byteLength;}
    console.info('client-inpaint', JSON.stringify({backend, threads: ort.env.wasm.numThreads, isolated: self.crossOriginIsolated, model: MODEL_SHA, inferenceMs: performance.now() - inferenceStart, totalMs: performance.now() - started, input: [source.width, source.height], modelSize: [512,512], box: prepared.box}));
    return candidate;
  } finally {image.dispose(); mask.dispose(); if (output) for (const tensor of Object.values(output)) tensor.dispose();}
}

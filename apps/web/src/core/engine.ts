import type { EditGraph } from './operations';
export interface ImagePixels { width: number; height: number; pixels: Uint8Array }
interface Request { id: number; width: number; height: number; pixels: Uint8Array; graphJson: string }
interface Response extends ImagePixels { id: number; error?: string; resetWorker?: boolean }
export type RenderKind = 'preview' | 'export' | 'background';
export interface RenderOptions { signal?: AbortSignal; kind?: RenderKind }
interface RenderTask {
  id: number; kind: RenderKind; source: ImagePixels; graphJson: string;
  modelGraph: boolean;
  bodySliderKey?: string;
  resolve: (image: ImagePixels) => void; reject: (error: Error) => void;
  signal?: AbortSignal; onAbort?: () => void; settled: boolean;
}
let requestId = 0;
let worker: Worker | null = null;
let active: RenderTask | null = null;
let latestPreview: RenderTask | null = null;
const exportsQueue: RenderTask[] = [];
const backgroundQueue: RenderTask[] = [];
let workerGenerations = 0;
let cancelledPreviews = 0;

function abortError(message = '图像渲染已取消'): Error { return new DOMException(message, 'AbortError'); }
function cleanup(task: RenderTask) {
  if (task.signal && task.onAbort) task.signal.removeEventListener('abort', task.onAbort);
  task.onAbort = undefined;
}
function rejectTask(task: RenderTask, error: Error) {
  if (task.settled) return;
  task.settled = true; cleanup(task); task.reject(error);
}
function resolveTask(task: RenderTask, image: ImagePixels) {
  if (task.settled) return;
  task.settled = true; cleanup(task); task.resolve(image);
}
function terminateWorker() { worker?.terminate(); worker = null; }
function failAll(error: Error) {
  terminateWorker();
  if (active) rejectTask(active, error);
  active = null;
  if (latestPreview) rejectTask(latestPreview, error);
  latestPreview = null;
  for (const task of [...exportsQueue, ...backgroundQueue]) rejectTask(task, error);
  exportsQueue.length = 0; backgroundQueue.length = 0;
}
function getWorker(): Worker {
  if (worker) return worker;
  const next = new Worker(new URL('./render.worker.ts', import.meta.url), { type: 'module' });
  worker = next; workerGenerations++;
  next.onmessage = (event: MessageEvent<Response>) => {
    // A terminated preview worker can have a queued message in the host event
    // loop. Its generation must never resolve a task on a replacement worker.
    if (worker !== next || !active || event.data.id !== active.id) return;
    const task = active; active = null;
    if (event.data.error) {
      rejectTask(task, new Error(event.data.error));
      if (event.data.resetWorker) terminateWorker();
    }
    else if (!Number.isInteger(event.data.width) || !Number.isInteger(event.data.height)
      || event.data.width < 1 || event.data.height < 1
      || event.data.width * event.data.height > 32_000_000
      || !(event.data.pixels instanceof Uint8Array)
      || event.data.pixels.length !== event.data.width * event.data.height * 4) {
      rejectTask(task, new Error('Rust 渲染返回无效 RGBA 数据'));
      failAll(new Error('Rust 渲染 Worker 返回无效数据')); return;
    } else resolveTask(task, { width: event.data.width, height: event.data.height, pixels: event.data.pixels });
    pump();
  };
  next.onerror = (event) => {
    if (worker !== next) return;
    event.preventDefault();
    failAll(new Error(event.message || 'Rust 渲染 Worker 启动失败'));
  };
  next.onmessageerror = () => { if (worker === next) failAll(new Error('Rust 渲染 Worker 数据传输失败')); };
  return next;
}
function pump() {
  if (active) return;
  // Exports keep their immutable snapshot and are never replaced by sliders.
  const task = exportsQueue.shift() ?? latestPreview ?? backgroundQueue.shift();
  if (!task) return;
  if (task === latestPreview) latestPreview = null;
  if (task.settled || task.signal?.aborted) {
    rejectTask(task, abortError()); pump(); return;
  }
  active = task;
  try {
    const target = getWorker();
    // Only active tasks copy/transfer pixels. Queued previews retain the source
    // reference and do not allocate a full image buffer for every slider event.
    const pixels = new Uint8Array(task.source.pixels);
    const message: Request = { id: task.id, width: task.source.width, height: task.source.height, pixels, graphJson: task.graphJson };
    target.postMessage(message, [pixels.buffer]);
  } catch (error) { failAll(error instanceof Error ? error : new Error(String(error))); }
}
function cancel(task: RenderTask, sourceChanged = false) {
  const wasSettled = task.settled;
  rejectTask(task, abortError());
  if (task === active) {
    if (task.kind === 'preview') {
      if (!wasSettled) cancelledPreviews++;
      const nextBodySlider = latestPreview?.bodySliderKey;
      const reuseBodyLayers = !sourceChanged && task.bodySliderKey !== undefined && (latestPreview === null || task.bodySliderKey === nextBodySlider);
      // Model sessions and successful candidates live inside this Worker.
      // Keep the current model render running while sliders replace its
      // caller, then dispatch the latest preview with the reusable candidate.
      // A new photograph must not wait behind the old model calculation.
      if ((!task.modelGraph && !reuseBodyLayers) || sourceChanged) {
        // Synchronous Rust work can only be interrupted by termination.
        terminateWorker(); active = null; pump();
      }
    }
    // Explicit abort of an export/background task rejects its caller, but the
    // active computation must finish before dispatching another message.
    return;
  }
  if (task === latestPreview) latestPreview = null;
  for (const queue of [exportsQueue, backgroundQueue]) {
    const index = queue.indexOf(task); if (index >= 0) queue.splice(index, 1);
  }
  pump();
}

/** One active render; latest preview only; durable export/background queues. */
export function renderImage(source: ImagePixels, graph: EditGraph, options: RenderOptions = {}): Promise<ImagePixels> {
  if (options.signal?.aborted) return Promise.reject(abortError());
  let graphJson: string;
  try { const { source: _metadata, ...protocol } = graph; graphJson = JSON.stringify(protocol); }
  catch (error) { return Promise.reject(error instanceof Error ? error : new Error(String(error))); }
  return new Promise((resolve, reject) => {
    const modelGraph = graph.operations.some(op => op.type === 'repair.inpaint' && op.params.method === 'lama' && op.enabled !== false && Number(op.params.opacity ?? 100) > 0);
    // Consecutive values on the final body slider share the exact upstream
    // image and segmentation. Let an already running frame populate Rust's
    // bounded F/B cache; its rejected caller cannot display a stale result.
    // Different images, masks or upstream edits still terminate old CPU work.
    const last = graph.operations.at(-1);
    const bodySliderKey = !modelGraph && last?.type === 'body.adjust' && last.enabled !== false
      ? JSON.stringify({...graph,source:undefined,operations:[...graph.operations.slice(0,-1),{...last,params:{...last.params,adjustments:undefined}}]})
      : undefined;
    const task: RenderTask = { id: ++requestId, kind: options.kind ?? 'preview', source, graphJson, modelGraph, bodySliderKey, resolve, reject, signal: options.signal, settled: false };
    if (task.signal) { task.onAbort = () => cancel(task); task.signal.addEventListener('abort', task.onAbort, { once: true }); }
    if (task.kind === 'preview') {
      if (latestPreview) rejectTask(latestPreview, abortError('预览已被更新参数替代'));
      latestPreview = task;
      if (active?.kind === 'preview') cancel(active, active.source !== source);
    } else (task.kind === 'export' ? exportsQueue : backgroundQueue).push(task);
    pump();
  });
}

/** Lightweight diagnostics for concurrency/cancellation acceptance checks. */
export function getRenderSchedulerState() {
  return { active: active ? { id: active.id, kind: active.kind } : null,
    preview: latestPreview?.id ?? null, exports: exportsQueue.length,
    background: backgroundQueue.length, workerGenerations, cancelledPreviews };
}
if (import.meta.hot) import.meta.hot.dispose(() => failAll(abortError('开发服务器已更新渲染模块')));
export function decodeImage(file: File): Promise<ImagePixels> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      try {
        if (image.naturalWidth * image.naturalHeight > 32_000_000) throw new Error('Demo 支持最多 3200 万像素，请先缩小图片');
        const canvas = document.createElement('canvas'); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
        const context = canvas.getContext('2d');
        if (!context) throw new Error('浏览器无法创建 Canvas');
        context.drawImage(image, 0, 0);
        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        resolve({ width: canvas.width, height: canvas.height, pixels: new Uint8Array(imageData.data) });
      } catch (error) {
        reject(error instanceof Error ? error : new Error('浏览器无法读取图片像素，请缩小图片后重试'));
      } finally { URL.revokeObjectURL(url); }
    };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error('图片解码失败，请选择 JPG、PNG 或 WebP')); };
    image.src = url;
  });
}
export function pixelsToCanvas(image: ImagePixels): HTMLCanvasElement {
  const canvas = document.createElement('canvas'); canvas.width = image.width; canvas.height = image.height;
  canvas.getContext('2d')?.putImageData(new ImageData(new Uint8ClampedArray(image.pixels), image.width, image.height), 0, 0);
  return canvas;
}

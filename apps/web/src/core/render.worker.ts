/// <reference lib="webworker" />
interface RustImage { width: number; height: number; pixels: Uint8Array }
interface RustModule { default: () => Promise<unknown>; render_rgba: (width: number, height: number, pixels: Uint8Array, graph: string) => RustImage }
let runtime: Promise<RustModule> | undefined;
let rendering = false;
async function loadRuntime(): Promise<RustModule> {
  if (!runtime) {
    // Vite adds ?import to relative dynamic imports, which cannot address
    // generated assets in public/. An absolute same-origin URL works in
    // both the dev server and the production bundle.
    const moduleURL = new URL('/wasm/image_core.js', self.location.origin).href;
    runtime = import(/* @vite-ignore */ moduleURL).then(async (module: RustModule) => { await module.default(); return module; }).catch((error: unknown) => { runtime = undefined; throw error; });
  }
  return runtime;
}
self.onmessage = async (event: MessageEvent<{ id: number; width: number; height: number; pixels: Uint8Array; graphJson: string }>) => {
  const { id, width, height, pixels, graphJson } = event.data;
  if (rendering) { self.postMessage({ id, error: '渲染 Worker 同时只能执行一个任务' }); return; }
  rendering = true;
  let runtimeLoaded = false;
  try {
    const wasm = await loadRuntime();
    runtimeLoaded = true;
    const result = wasm.render_rgba(width, height, pixels, graphJson);
    const output = { id, width: result.width, height: result.height, pixels: new Uint8Array(result.pixels) };
    self.postMessage(output, { transfer: [output.pixels.buffer] });
  } catch (error) {
    // Failed ES-module imports remain cached inside a Worker. The host must
    // replace that Worker to recover after a temporary module/WASM fetch error.
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error), resetWorker: !runtimeLoaded });
  } finally { rendering = false; }
};
export {};

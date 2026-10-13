/// <reference lib="webworker" />
interface RustImage { width: number; height: number; pixels: Uint8Array }
interface GraphNode { type: string; enabled?: boolean; params?: Record<string,unknown> }
interface RustModule {
  default: () => Promise<unknown>;
  render_rgba: (width: number, height: number, pixels: Uint8Array, graph: string) => RustImage;
  inpaint_masks: (width: number, height: number, pixels: Uint8Array, operation: string) => { target: Uint8Array; protection: Uint8Array };
  composite_inpaint: (width: number, height: number, pixels: Uint8Array, candidate: Uint8Array, operation: string) => Uint8Array;
}
let runtime: Promise<RustModule> | undefined;
let rendering = false;
class ClientModuleLoadError extends Error {
  constructor(error: unknown) {
    super(`客户端去人模块加载失败：${error instanceof Error ? error.message : String(error)}。请重试。`);
  }
}
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

async function png(image: RustImage): Promise<Blob> {
  const canvas=new OffscreenCanvas(image.width,image.height);
  canvas.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(image.pixels),image.width,image.height),0,0);
  return canvas.convertToBlob({type:'image/png'});
}
async function maskPng(width:number,height:number,mask:Uint8Array):Promise<Blob> {
  const pixels=new Uint8Array(width*height*4);
  for(let i=0;i<mask.length;i++)pixels.set([mask[i],mask[i],mask[i],255],i*4);
  return png({width,height,pixels});
}
async function modelFill(wasm:RustModule,source:RustImage,op:GraphNode):Promise<RustImage> {
  const operation=JSON.stringify(op);
  const masks=wasm.inpaint_masks(source.width,source.height,source.pixels,operation);
  if(!masks.target.some(v=>v>0))return source;
  if (op.params?.execution === 'client') {
    // A failed ES-module fetch is cached for this Worker's lifetime. Reset
    // only import failures; model/parameter failures retain reusable sessions.
    let clientModule: typeof import('./clientInpaint');
    try {clientModule = await import('./clientInpaint');}
    catch (error) {throw new ClientModuleLoadError(error);}
    try {
      const {clientInpaint} = clientModule;
      const candidate = await clientInpaint(source, masks.target, masks.protection, {
        maxSide: Number(op.params.maxSide ?? 512), contextMode: op.params.contextMode as 'full' | 'crop' | undefined,
        contextPadding: Number(op.params.contextPadding ?? 256), maskDilation: Number(op.params.maskDilation ?? 4), seamBlend: op.params.seamBlend as 'none' | 'poisson' | undefined,
      });
      return {...source, pixels: wasm.composite_inpaint(source.width,source.height,source.pixels,candidate,operation)};
    } catch (error) {throw new Error(`客户端去人失败：${error instanceof Error ? error.message : String(error)}。可切换服务端处理或手工填补。`);}
  }
  const form=new FormData();
  const blobs=await Promise.all([png(source),maskPng(source.width,source.height,masks.target),maskPng(source.width,source.height,masks.protection)]);
  form.append('image',blobs[0],'input.png');form.append('mask',blobs[1],'mask.png');form.append('protect_mask',blobs[2],'protect.png');
  for(const [key,field,value] of [['maxSide','max_side',1024],['contextPadding','context_padding',256],['maskDilation','mask_dilation',4],['seamBlend','seam_blend','none'],['contextMode','context_mode','full']] as const) {
    form.append(field,String(op.params?.[key] ?? value));
  }
  const apiOrigin=(import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8080').replace(/\/$/, '');
  const response=await fetch(`${apiOrigin}/api/inpaint/model`,{method:'POST',body:form});
  if(!response.ok) {
    let detail='';try { const data=await response.json();detail=typeof data.error==='string'?data.error:data.error?.message ?? ''; } catch { /* preserve the HTTP status */ }
    throw new Error(`本地模型填补失败（${response.status}）${detail?`：${detail}`:''}。可选择手工内容感知填补继续修图。`);
  }
  // Browser Canvas decoding uses premultiplied alpha; the Rust compositor
  // restores every source alpha and unselected byte after decoding the model.
  const bitmap=await createImageBitmap(await response.blob());
  try {
    if(bitmap.width!==source.width || bitmap.height!==source.height)throw new Error('模型填补返回了错误尺寸');
    const canvas=new OffscreenCanvas(source.width,source.height);const ctx=canvas.getContext('2d')!;ctx.drawImage(bitmap,0,0);
    const candidate=new Uint8Array(ctx.getImageData(0,0,source.width,source.height).data);
    return {...source,pixels:wasm.composite_inpaint(source.width,source.height,source.pixels,candidate,operation)};
  } finally {bitmap.close();}
}

async function render(wasm:RustModule,source:RustImage,graphJson:string):Promise<RustImage> {
  const graph=JSON.parse(graphJson) as {operations?:GraphNode[]};
  if(!graph.operations?.some(op=>op.type==='repair.inpaint' && op.params?.method==='lama')) {
    return wasm.render_rgba(source.width,source.height,source.pixels,graphJson);
  }
  if(graph.operations.length>256)throw new Error('EditGraph supports at most 256 operations');
  let current=source;let segment:GraphNode[]=[];
  const flush=()=>{
    if(!segment.length)return;
    current=wasm.render_rgba(current.width,current.height,current.pixels,JSON.stringify({...graph,operations:segment}));
    segment=[];
  };
  for(const op of graph.operations) {
    if(op.type==='repair.inpaint' && op.params?.method==='lama') {flush();current=await modelFill(wasm,current,op);}
    else segment.push(op);
  }
  flush();return current;
}
self.onmessage = async (event: MessageEvent<{ id: number; width: number; height: number; pixels: Uint8Array; graphJson: string }>) => {
  const { id, width, height, pixels, graphJson } = event.data;
  if (rendering) { self.postMessage({ id, error: '渲染 Worker 同时只能执行一个任务' }); return; }
  rendering = true;
  let runtimeLoaded = false;
  try {
    const wasm = await loadRuntime();
    runtimeLoaded = true;
    const result = await render(wasm,{width,height,pixels},graphJson);
    const output = { id, width: result.width, height: result.height, pixels: new Uint8Array(result.pixels) };
    self.postMessage(output, { transfer: [output.pixels.buffer] });
  } catch (error) {
    // Failed ES-module imports remain cached inside a Worker. The host must
    // replace that Worker to recover after a temporary module/WASM fetch error.
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error), resetWorker: !runtimeLoaded || error instanceof ClientModuleLoadError });
  } finally { rendering = false; }
};
export {};

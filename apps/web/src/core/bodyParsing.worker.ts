import * as ort from 'onnxruntime-web/wasm';
import ortModuleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import ortWasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import {bodyInput} from './bodyParsing';
const scope=self as unknown as {crossOriginIsolated:boolean;onmessage:(event:MessageEvent)=>void;postMessage:(result:unknown,transfer?:Transferable[])=>void};
const threads=scope.crossOriginIsolated&&typeof SharedArrayBuffer!=='undefined'?4:1;
ort.env.wasm.wasmPaths={mjs:new URL(ortModuleUrl,self.location.href).href,wasm:new URL(ortWasmUrl,self.location.href).href};
ort.env.wasm.numThreads=threads;
let session:Promise<ort.InferenceSession>|undefined,loadMs=0;
// One model invocation at a time; serializing source tensors avoids memory
// spikes when the user asks to detect again during a cold model load.
let queue=Promise.resolve();
scope.onmessage=event=>{queue=queue.then(async()=>{
  const {id,width,height,pixels}=event.data;
  try {
    session??=(async()=>{const start=performance.now();const model=await ort.InferenceSession.create('/models/bodyparser.onnx',{executionProviders:['wasm'],graphOptimizationLevel:'all'});loadMs=performance.now()-start;return model;})().catch(error=>{session=undefined;throw error;});
    const model=await session,input=bodyInput(pixels,width,height),start=performance.now();
    const result=(await model.run({[model.inputNames[0]]:new ort.Tensor('float32',input,[1,3,576,384])}))[model.outputNames[0]];
    const logits=result.data as Float32Array;
    scope.postMessage({id,logits,dims:result.dims,runtime:{loadMs,inferenceMs:performance.now()-start,requestedThreads:threads,crossOriginIsolated:scope.crossOriginIsolated,provider:'wasm'}},[logits.buffer]);
  }catch(error){scope.postMessage({id,error:`人体像素解析模型不可用：${String(error)}。请运行 make body-models，或手工描绘目标部位及保护区。`});}
});};

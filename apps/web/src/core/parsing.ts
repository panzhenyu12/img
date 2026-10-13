import * as ort from 'onnxruntime-web/wasm';
import ortModuleUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import ortWasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import type { DetectedPerson, DetectionPoint, SemanticMask } from './landmarks';
import type { RasterMask } from './operations';

// BiSeNet / CelebAMask-HQ: background, skin, brows, eyes, glasses,
// ears, earrings, nose, inner mouth, lips, neck, necklace, cloth, hair, hat.
const HEAD = new Set([1,2,3,4,5,6,7,8,9,10,11,12,13,17,18]);
const FACE_CONTROL = new Set([1,2,3,4,5,6,7,8,9,10,11,12,13]);
const SKIN = new Set([1,7,8,10,14]);
let session: Promise<ort.InferenceSession> | undefined;
const clamp = (v: number) => Math.max(0,Math.min(1,v));
// Class confidence is not physical opacity. Suppress probability tails on
// certain background, retain a narrow antialias band and the exact .5 edge.
function coverageAlpha(probability: number): number {
  const t=clamp((probability-.4)/.2);return t*t*(3-2*t);
}

function simplify(points: DetectionPoint[], epsilon: number): DetectionPoint[] {
  if(points.length<=3) return points;
  const first=points[0],last=points.at(-1)!;
  const dx=last.x-first.x,dy=last.y-first.y;
  let best=0,index=0;
  for(let i=1;i<points.length-1;i++) {
    const p=points[i];const t=clamp(((p.x-first.x)*dx+(p.y-first.y)*dy)/(dx*dx+dy*dy||1));
    const distance=Math.hypot(p.x-first.x-dx*t,p.y-first.y-dy*t);
    if(distance>best) {best=distance;index=i;}
  }
  return best>epsilon ? [...simplify(points.slice(0,index+1),epsilon).slice(0,-1),...simplify(points.slice(index),epsilon)] : [first,last];
}

/** Marching squares at alpha=.5, preserving concavities rather than a hull.
 * Pixel-center positions agree with Rust's raster bilinear sampling. */
export function alphaOutline(mask: RasterMask): DetectionPoint[] {
  const {width:w,height:h,data}=mask;
  type P=[number,number];const edges=new Map<string,{p:P,neighbors:string[]}>();
  const key=(p:P)=>`${p[0].toFixed(6)},${p[1].toFixed(6)}`;
  const connect=(a:P,b:P)=>{
    const ak=key(a),bk=key(b);if(ak===bk)return;
    const ae=edges.get(ak)??{p:a,neighbors:[]},be=edges.get(bk)??{p:b,neighbors:[]};
    ae.neighbors.push(bk);be.neighbors.push(ak);edges.set(ak,ae);edges.set(bk,be);
  };
  const at=(x:number,y:number)=>x<0||y<0||x>=w||y>=h?0:data[y*w+x];
  for(let y=-1;y<h;y++) for(let x=-1;x<w;x++) {
    const values=[at(x,y),at(x+1,y),at(x+1,y+1),at(x,y+1)];
    const corners:P[]=[[x+.5,y+.5],[x+1.5,y+.5],[x+1.5,y+1.5],[x+.5,y+1.5]];
    const crossing:P[]=[];const sides:number[]=[];
    for(let i=0;i<4;i++) {
      const j=(i+1)%4;if((values[i]>=127.5)===(values[j]>=127.5))continue;
      const t=(127.5-values[i])/(values[j]-values[i]);
      crossing.push([corners[i][0]+(corners[j][0]-corners[i][0])*t,corners[i][1]+(corners[j][1]-corners[i][1])*t]);sides.push(i);
    }
    if(crossing.length===2)connect(crossing[0],crossing[1]);
    else if(crossing.length===4) {
      const centerInside=values.reduce((s,v)=>s+v,0)/4>=127.5;
      const aroundFirst=(values[0]>=127.5)!==centerInside;
      const pairs=aroundFirst?[[0,3],[1,2]]:[[0,1],[2,3]];
      for(const [a,b] of pairs) connect(crossing[sides.indexOf(a)],crossing[sides.indexOf(b)]);
    }
  }
  const visited=new Set<string>();const loops:DetectionPoint[][]=[];
  for(const start of edges.keys()) {
    if(visited.has(start))continue;
    const loop:DetectionPoint[]=[];let current=start,previous='';
    for(let steps=0;steps<=edges.size;steps++) {
      if(visited.has(current))break;
      visited.add(current);const e=edges.get(current)!;loop.push({x:e.p[0]/w,y:e.p[1]/h});
      const next=e.neighbors.find(k=>k!==previous);if(!next)break;
      previous=current;current=next;
      if(current===start) {loops.push(loop);break;}
    }
  }
  const area=(p:DetectionPoint[])=>Math.abs(p.reduce((s,a,i)=>{const b=p[(i+1)%p.length];return s+a.x*b.y-b.x*a.y;},0));
  const largest=loops.sort((a,b)=>area(b)-area(a))[0];
  if(!largest||largest.length<8) throw new Error('人像解析未获得闭合头部轮廓，请手工描绘轮廓。');
  // Split the loop in half so closed RDP does not collapse to a single point.
  let epsilon=0.3/Math.max(w,h),out=largest;
  for(let i=0;i<20;i++) {
    const middle=Math.floor(largest.length/2);
    out=[...simplify(largest.slice(0,middle+1),epsilon).slice(0,-1),...simplify([...largest.slice(middle),largest[0]],epsilon).slice(0,-1)];
    if(out.length<=256)break;epsilon*=1.25;
  }
  if(out.length>256)throw new Error('头部边界过于复杂，请缩小选区后重新检测。');
  return out;
}

function sample(values: Float32Array,w: number,h: number,x: number,y: number): number {
  if(x<-.5||y<-.5||x>w-.5||y>h-.5)return 0;
  x=Math.max(0,Math.min(w-1,x));y=Math.max(0,Math.min(h-1,y));
  const x0=Math.floor(x),y0=Math.floor(y),x1=Math.min(w-1,x0+1),y1=Math.min(h-1,y0+1),tx=x-x0,ty=y-y0;
  return values[y0*w+x0]*(1-tx)*(1-ty)+values[y0*w+x1]*tx*(1-ty)+values[y1*w+x0]*(1-tx)*ty+values[y1*w+x1]*tx*ty;
}

export async function parseFace(source: HTMLImageElement | HTMLCanvasElement, person: DetectedPerson, cropScale=1): Promise<{semanticMasks: Record<string,SemanticMask>;subjectMask:RasterMask}> {
  ort.env.wasm.wasmPaths={mjs:new URL(ortModuleUrl,window.location.href).href,wasm:new URL(ortWasmUrl,window.location.href).href};ort.env.wasm.numThreads=1;
  session??=ort.InferenceSession.create('/models/faceparser.onnx',{executionProviders:['wasm'],graphOptimizationLevel:'all'}).catch(error=>{session=undefined;throw new Error(`像素级人像解析模型不可用：${String(error)}。请运行 make models，或手工描绘轮廓。`);});
  const model=await session;
  const width=source instanceof HTMLImageElement?source.naturalWidth:source.width;
  const height=source instanceof HTMLImageElement?source.naturalHeight:source.height;
  const xs=person.points.map(p=>p.x*width),ys=person.points.map(p=>p.y*height);
  const left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys);
  const size=Math.max((right-left)*2.3,(bottom-top)*1.65)*cropScale;
  const x=(left+right-size)/2,y=(top+bottom-size)/2;
  const canvas=document.createElement('canvas');canvas.width=512;canvas.height=512;
  const ctx=canvas.getContext('2d',{willReadFrequently:true})!;
  ctx.fillStyle='#808080';ctx.fillRect(0,0,512,512);
  ctx.drawImage(source,x,y,size,size,0,0,512,512);
  const pixels=ctx.getImageData(0,0,512,512).data,n=512*512;
  const input=new Float32Array(n*3),mean=[.485,.456,.406],std=[.229,.224,.225];
  for(let c=0;c<3;c++)for(let i=0;i<n;i++)input[c*n+i]=(pixels[i*4+c]/255-mean[c])/std[c];
  const result=await model.run({[model.inputNames[0]]:new ort.Tensor('float32',input,[1,3,512,512])});
  const output=result[model.outputNames[0]],dims=output.dims;
  if(dims.length!==4||dims[1]!==19)throw new Error(`人像解析模型输出格式不匹配：${dims.join('×')}`);
  const [oh,ow]=[dims[2],dims[3]],count=ow*oh,logits=output.data as Float32Array;
  const head=new Float32Array(count),faceControl=new Float32Array(count),skin=new Float32Array(count),neck=new Float32Array(count),hair=new Float32Array(count),personAlpha=new Float32Array(count);
  for(let i=0;i<count;i++) {
    let max=-Infinity;for(let c=0;c<19;c++)max=Math.max(max,logits[c*count+i]);
    let total=0,hw=0,fw=0,sw=0,nw=0,aw=0,pw=0;
    for(let c=0;c<19;c++) {
      const score=Math.exp(logits[c*count+i]-max);total+=score;
      if(HEAD.has(c))hw+=score;if(FACE_CONTROL.has(c))fw+=score;if(SKIN.has(c))sw+=score;if(c===14)nw+=score;if(c===17)aw+=score;if(c!==0)pw+=score;
    }
    head[i]=hw/total;faceControl[i]=fw/total;skin[i]=sw/total;neck[i]=nw/total;hair[i]=aw/total;personAlpha[i]=pw/total;
  }
  // Long hair may extend beyond the initial face crop. Expand inference
  // instead of inventing a vertical foreground cut at an internal crop edge.
  const touches=(side:'left'|'right'|'top'|'bottom')=>{
    for(let j=0;j<Math.max(ow,oh);j++) {
      const i=side==='left'?Math.min(j,oh-1)*ow:side==='right'?Math.min(j,oh-1)*ow+ow-1:side==='top'?Math.min(j,ow-1):(oh-1)*ow+Math.min(j,ow-1);
      if(head[i]>.6)return true;
    }
    return false;
  };
  const clipped=(x>1&&touches('left'))||(x+size<width-1&&touches('right'))||(y>1&&touches('top'))||(y+size<height-1&&touches('bottom'));
  if(clipped&&cropScale<4) return parseFace(source,person,cropScale*1.7);
  if(clipped)throw new Error('头部解析仍触及内部裁切边界，请手工描绘完整轮廓。');
  const scale=Math.min(1,512/Math.max(width,height)),rw=Math.max(1,Math.round(width*scale)),rh=Math.max(1,Math.round(height*scale));
  const raster=(values:Float32Array):RasterMask=>({shape:'raster',width:rw,height:rh,data:Array.from({length:rw*rh},(_,i)=>{
    const px=(i%rw+.5)*width/rw,py=(Math.floor(i/rw)+.5)*height/rh;
    return Math.round(coverageAlpha(sample(values,ow,oh,(px-x)/size*ow-.5,(py-y)/size*oh-.5))*255);
  })});
  const face=raster(head),faceContourMask=raster(faceControl);
  return {semanticMasks:{face,faceOutline:{shape:'polygon',points:alphaOutline(face),feather:0,invert:false},faceContourMask,faceContour:{shape:'polygon',points:alphaOutline(faceContourMask),feather:0,invert:false},skin:raster(skin),neck:raster(neck),hair:raster(hair)},subjectMask:raster(personAlpha)};
}

import type { DetectionPoint } from './landmarks';
import type { RasterMask } from './operations';

/** FASHN SegFormer-B4, official 18-class order. Clothing and body parts are
 * distinct: the model does not predict separate left/right sleeves. Pose
 * seeds split connected garment pixels below; they never replace foreground. */
export const BODY_CLASSES = ['background','face','hair','top','dress','skirt','pants','belt','bag','hat','scarf','glasses','arms','hands','legs','feet','torso','jewelry'] as const;
export type BodyClassMasks = Record<string,RasterMask>;
export interface BodyLayers { subjectMask: RasterMask; partMasks: Record<string,RasterMask>; protectMask: RasterMask; classMasks: BodyClassMasks }
export interface BodyParserRuntime {loadMs:number;inferenceMs:number;requestedThreads:number;crossOriginIsolated:boolean;provider:'wasm'}
let lastRuntime:BodyParserRuntime|undefined;
export const bodyParserRuntime=()=>lastRuntime;
const clamp = (v:number)=>Math.max(0,Math.min(1,v));
const alpha = (v:number)=>{const t=clamp((v-.4)/.2);return t*t*(3-2*t);};
function sample(mask:RasterMask,x:number,y:number):number {
  const px=Math.max(0,Math.min(mask.width-1,x*mask.width-.5)),py=Math.max(0,Math.min(mask.height-1,y*mask.height-.5));
  const x0=Math.floor(px),y0=Math.floor(py),x1=Math.min(mask.width-1,x0+1),y1=Math.min(mask.height-1,y0+1),tx=px-x0,ty=py-y0;
  return (mask.data[y0*mask.width+x0]*(1-tx)*(1-ty)+mask.data[y0*mask.width+x1]*tx*(1-ty)+mask.data[y1*mask.width+x0]*(1-tx)*ty+mask.data[y1*mask.width+x1]*tx*ty)/255;
}

class Heap {
  values:Array<[number,number]>=[];
  push(value:[number,number]) {let i=this.values.length;this.values.push(value);while(i){const p=(i-1)>>1;if(this.values[p][0]<=value[0])break;this.values[i]=this.values[p];i=p;}this.values[i]=value;}
  pop():[number,number]|undefined {const first=this.values[0],last=this.values.pop();if(!this.values.length)return first;let i=0;while(true){let child=i*2+1;if(child>=this.values.length)break;if(child+1<this.values.length&&this.values[child+1][0]<this.values[child][0])child++;if(this.values[child][0]>=last![0])break;this.values[i]=this.values[child];i=child;}this.values[i]=last!;return first;}
}
type Bone={id:string;a:DetectionPoint;b:DetectionPoint;lower:boolean;length:number};
const good=(p:DetectionPoint|undefined):p is DetectionPoint=>!!p&&Number.isFinite(p.x)&&Number.isFinite(p.y)&&(p.visibility===undefined||p.visibility>=.5);
const mid=(a:DetectionPoint,b:DetectionPoint)=>({x:(a.x+b.x)/2,y:(a.y+b.y)/2});

/** Color-sensitive geodesic ownership over real parsing pixels, with separate
 * competing skeleton seeds. Hands/hair/accessories are independent stationary
 * layers, including a held laptop classified by FASHN as a bag. Unknown
 * classes and model errors remain visible in the masks and require correction. */
export function deriveBodyLayers(source:HTMLCanvasElement,points:DetectionPoint[],classes:BodyClassMasks,instanceMask?:RasterMask):BodyLayers {
  const scale=Math.min(1,512/Math.max(source.width,source.height)),w=Math.round(source.width*scale),h=Math.round(source.height*scale),n=w*h;
  const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;canvas.getContext('2d')!.drawImage(source,0,0,w,h);
  const rgb=canvas.getContext('2d',{willReadFrequently:true})!.getImageData(0,0,w,h).data;
  const probabilities=BODY_CLASSES.map(name=>Float32Array.from({length:n},(_,i)=>classes[name]?sample(classes[name],(i%w+.5)/w,(Math.floor(i/w)+.5)/h):0));
  const foreground=new Float32Array(n),upper=new Float32Array(n),lower=new Float32Array(n),blocked=new Uint8Array(n);
  const protectionClasses=[1,2,8,9,10,11,13,15,17];
  for(let i=0;i<n;i++) {
    // Instance masks are deliberately coarse. Recover only their immediate
    // uncertainty band; FASHN still decides actual garment/background alpha.
    // This avoids clipping photographed silhouette pixels while retaining
    // instance separation for erroneous remote garment-class islands.
    const nx=(i%w+.5)/w,ny=(Math.floor(i/w)+.5)/h;
    let instance=instanceMask?sample(instanceMask,nx,ny):1;
    if(instanceMask&&instance<.6) {
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++) {
        instance=Math.max(instance,sample(instanceMask,nx+dx/instanceMask.width,ny+dy/instanceMask.height));
      }
    }
    foreground[i]=alpha(1-probabilities[0][i])*alpha(instance);
    upper[i]=[3,4,12,16].reduce((s,c)=>s+probabilities[c][i],0);
    lower[i]=[5,6,14].reduce((s,c)=>s+probabilities[c][i],0);
    const protectedProbability=protectionClasses.reduce((s,c)=>s+probabilities[c][i],0);
    // Tails are not protection: require a foreground class majority and
    // dilate only its actual boundary below, instead of freezing the ROI.
    blocked[i]=protectedProbability>.45&&foreground[i]>.01?1:0;
  }
  const protectedPixels=blocked.slice();
  for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++) {
    const i=y*w+x;if(!blocked[i])continue;
    for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++) {const j=i+dy*w+dx;if(j>=0&&j<n&&foreground[j]>.01)protectedPixels[j]=1;}
  }
  const bones:Bone[]=[];
  const add=(id:string,a:DetectionPoint|undefined,b:DetectionPoint|undefined,isLower=false)=>{
    if(!good(a)||!good(b))return;const length=Math.hypot((b.x-a.x)*w,(b.y-a.y)*h);if(length>3)bones.push({id,a,b,lower:isLower,length});
  };
  if([11,12,23,24].every(i=>good(points[i])))add('torso',mid(points[11],points[12]),mid(points[23],points[24]));
  for(const [id,a,b,isLower] of [['left-upper-arm',11,13,false],['right-upper-arm',12,14,false],['left-forearm',13,15,false],['right-forearm',14,16,false],['left-thigh',23,25,true],['right-thigh',24,26,true],['left-calf',25,27,true],['right-calf',26,28,true]] as const)add(id,points[a],points[b],isLower);
  // Heap costs are JS doubles. Rounding them into Float32 can round upward;
  // an equal path then compares as "shorter" indefinitely and floods the
  // heap on flat two-colour garments. Preserve the same precision as costs.
  const distance=new Float64Array(n).fill(Infinity),owner=new Int16Array(n).fill(-1),heap=new Heap();
  const allowed=(i:number,bone:Bone)=>{
    // A visible waist can be covered by the top of a skirt/trousers. Pose's
    // hip joint is below the garment waistband; excluding these classes
    // removes the actual waist while retaining only upper-shirt texture.
    const probability=bone.id==='torso'?[3,4,5,6,7,16].reduce((s,c)=>s+probabilities[c][i],0):bone.lower?lower[i]:upper[i];
    return foreground[i]>.01&&!protectedPixels[i]&&probability>.25;
  };
  bones.forEach((bone,label)=>{
    const steps=Math.ceil(bone.length*2);
    for(let step=0;step<=steps;step++) {
      const t=step/steps;if(bone.id!=='torso'&&(t<.12||t>.88))continue;
      const x=Math.max(0,Math.min(w-1,Math.floor((bone.a.x+(bone.b.x-bone.a.x)*t)*w))),y=Math.max(0,Math.min(h-1,Math.floor((bone.a.y+(bone.b.y-bone.a.y)*t)*h))),i=y*w+x;
      if(allowed(i,bone)&&distance[i]>0){owner[i]=label;distance[i]=0;heap.push([0,i]);}
    }
  });
  while(heap.values.length) {
    const [cost,i]=heap.pop()!;if(cost>distance[i]+1e-4)continue;
    const label=owner[i],bone=bones[label],x=i%w,y=Math.floor(i/w);
    const neighbors=[x>0?i-1:-1,x+1<w?i+1:-1,y>0?i-w:-1,y+1<h?i+w:-1];
    for(const j of neighbors) {
      if(j<0||!allowed(j,bone))continue;
      let color=0;for(let c=0;c<3;c++)color+=(rgb[i*4+c]-rgb[j*4+c])**2;
      const candidate=cost+1+Math.sqrt(color)/255*8;
      if(candidate<distance[j]){distance[j]=candidate;owner[j]=label;heap.push([candidate,j]);}
    }
  }
  const raster=(values:ArrayLike<number>):RasterMask=>({shape:'raster',width:w,height:h,data:Array.from(values)});
  const partMasks:Record<string,RasterMask>={};
  bones.forEach((bone,label)=>{
    const values=Array.from({length:n},(_,i)=>owner[i]===label?Math.round(foreground[i]*255):0);
    partMasks[bone.id]=raster(values);
  });
  // Waist/hips select torso pixels; geometric support determines where along
  // that actual garment layer the parameter acts. Sleeves remain separate.
  if(partMasks.torso){partMasks.waist=partMasks.torso;partMasks.hips=partMasks.torso;}
  partMasks['full-body']=raster(Array.from(foreground,(v,i)=>protectedPixels[i]?0:Math.round(v*255)));
  return {subjectMask:raster(Array.from(foreground,v=>Math.round(v*255))),partMasks,protectMask:raster(Array.from(protectedPixels,v=>v?255:0)),classMasks:classes};
}

/** Positive-area resampling uses the official INTER_AREA footprint for this
 * fixed input, avoiding browser-dependent Canvas interpolation. Integer tie
 * rounding can differ from OpenCV by one RGB level; the real parity report
 * records this rather than describing the preprocessing as bit-exact. */
export function bodyInput(pixels:Uint8ClampedArray,width:number,height:number):Float32Array {
  const ow=384,oh=576,n=ow*oh,result=new Float32Array(n*3),mean=[.485,.456,.406],std=[.229,.224,.225];
  for(let y=0;y<oh;y++)for(let x=0;x<ow;x++) {
    const left=x*width/ow,right=(x+1)*width/ow,top=y*height/oh,bottom=(y+1)*height/oh,sum=[0,0,0];let weight=0;
    for(let sy=Math.floor(top);sy<Math.ceil(bottom);sy++)for(let sx=Math.floor(left);sx<Math.ceil(right);sx++) {
      const a=(Math.min(right,sx+1)-Math.max(left,sx))*(Math.min(bottom,sy+1)-Math.max(top,sy));if(a<=0)continue;
      const i=(Math.min(height-1,sy)*width+Math.min(width-1,sx))*4;for(let c=0;c<3;c++)sum[c]+=pixels[i+c]*a;weight+=a;
    }
    for(let c=0;c<3;c++)result[c*n+y*ow+x]=(Math.round(sum[c]/weight)/255-mean[c])/std[c];
  }
  return result;
}

export function decodeBodyLogits(logits:Float32Array,dims:readonly number[],width:number,height:number):BodyClassMasks {
  if(dims.length!==4||dims[1]!==18||dims[0]!==1)throw new Error('人体解析模型输出格式不匹配');
  const [h,w]=[dims[2],dims[3]],n=w*h;
  const scale=Math.min(1,384/Math.max(width,height)),rw=Math.round(width*scale),rh=Math.round(height*scale),rn=rw*rh;
  const classes=BODY_CLASSES.map(()=>new Array<number>(rn));
  for(let i=0;i<rn;i++) {
    const x=Math.max(0,Math.min(w-1,(i%rw+.5)*w/rw-.5)),y=Math.max(0,Math.min(h-1,(Math.floor(i/rw)+.5)*h/rh-.5));
    const x0=Math.floor(x),y0=Math.floor(y),x1=Math.min(w-1,x0+1),y1=Math.min(h-1,y0+1),tx=x-x0,ty=y-y0,scores=[];let max=-Infinity;
    for(let c=0;c<18;c++){const at=(px:number,py:number)=>logits[c*n+py*w+px];const v=at(x0,y0)*(1-tx)*(1-ty)+at(x1,y0)*tx*(1-ty)+at(x0,y1)*(1-tx)*ty+at(x1,y1)*tx*ty;scores.push(v);max=Math.max(max,v);}
    const exp=scores.map(v=>Math.exp(v-max)),total=exp.reduce((s,v)=>s+v,0);for(let c=0;c<18;c++)classes[c][i]=Math.round(exp[c]/total*255);
  }
  return Object.fromEntries(BODY_CLASSES.map((name,c)=>[name,{shape:'raster',width:rw,height:rh,data:classes[c]}]));
}

let modelWorker:Worker|undefined,requestId=0;
const pending=new Map<number,{resolve:(result:{logits:Float32Array;dims:number[];runtime:BodyParserRuntime})=>void;reject:(error:Error)=>void}>();
function bodyWorker():Worker {
  if(modelWorker)return modelWorker;
  const worker=new Worker(new URL('./bodyParsing.worker.ts',import.meta.url),{type:'module'});
  worker.onmessage=event=>{const result=event.data,request=pending.get(result.id);if(!request)return;pending.delete(result.id);if(result.error)request.reject(new Error(result.error));else request.resolve(result);};
  worker.onerror=event=>{const error=new Error(`人体像素解析 Worker 失败：${event.message}。请手工描绘目标部位与保护区。`);for(const request of pending.values())request.reject(error);pending.clear();worker.terminate();modelWorker=undefined;};
  modelWorker=worker;return worker;
}

/** Independent worker/ORT session: a previously initialized face parser's
 * one-thread pool cannot limit this model, and inference never blocks UI. */
export async function parseBodyClasses(source:HTMLCanvasElement):Promise<BodyClassMasks> {
  const width=source.width,height=source.height,pixels=source.getContext('2d',{willReadFrequently:true})!.getImageData(0,0,width,height).data,id=++requestId;
  const result=await new Promise<{logits:Float32Array;dims:number[];runtime:BodyParserRuntime}>((resolve,reject)=>{pending.set(id,{resolve,reject});bodyWorker().postMessage({id,width,height,pixels},[pixels.buffer]);});
  lastRuntime=result.runtime;
  return decodeBodyLogits(result.logits,result.dims,width,height);
}

/** Same photographs and adjustment=60 as V2. Parsing/part masks are explicit
 * graph inputs, so native and WASM use the same pixel layers. Old outputs are
 * retained; locality does not automatically grant a visual quality score. */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const execute=promisify(execFile),root=path.resolve(import.meta.dirname,'..');
const binary=process.env.BODY_RENDER_BINARY??path.join(root,'.run/body-target/release/image-render');
const version=process.env.BODY_REVIEW_VERSION??'v3',previousVersion=process.env.BODY_PREVIOUS_VERSION??'v2';
if(!/^v\d+$/.test(version)||!/^v\d+$/.test(previousVersion))throw new Error('Review version must be v followed by a number');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const rendererSha256=sha(await fs.readFile(binary));
const verifyBinary=async()=>{if(sha(await fs.readFile(binary))!==rendererSha256)throw new Error('Renderer changed during the review; discard this run.');};
const ids=(process.env.BODY_IDS??'fashion-standing,fullbody-skirt,fullbody-side-fashion,tilted-shirt,action-suit').split(',');
const browser=await chromium.launch({headless:true}),results=[];
try {
 const page=await browser.newPage();await page.goto(process.env.RETOUCH_WEB_URL??'http://127.0.0.1:5173/');
 for(const id of ids) {
  const oldDir=path.join(root,'test/results/body-contour-v2',id),previousDir=path.join(root,'test/results',previousVersion==='v2'?'body-contour-v2':`body-parts-${previousVersion}`,id),dir=path.join(root,'test/results',`body-parts-${version}`,id);await fs.mkdir(dir,{recursive:true});
  const source=(await fs.readFile(path.join(oldDir,'source.png'))).toString('base64');
  const old=JSON.parse(await fs.readFile(path.join(oldDir,'quality.json'),'utf8'));
  const classes=JSON.parse(await fs.readFile(path.join(root,'.run/fashn-results',id,'masks.json'),'utf8'));
  const oldGraphs={};for(const check of old.checks)if(check.metrics)oldGraphs[check.part]=JSON.parse(await fs.readFile(path.join(oldDir,check.part+'.json'),'utf8'));
  const prepared=JSON.parse(await page.evaluate(async(encoded)=>{
   const {source,classes,oldGraphs}=JSON.parse(encoded);
   const image=new Image();image.src='data:image/png;base64,'+source;await image.decode();const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;canvas.getContext('2d').drawImage(image,0,0);
   const {detectBodyPose,measuredLimbSupport,measuredTorsoSupport,bodyParts}=await import('/src/core/landmarks.ts');
   const {deriveBodyLayers}=await import('/src/core/bodyParsing.ts');
   const start=performance.now(),person=(await detectBodyPose(canvas))[0];if(!person)throw new Error('No Pose');
   const layers=deriveBodyLayers(canvas,person.points,classes,person.subjectMask);const parts=bodyParts(person.points,canvas.width,canvas.height);
   const checks=Object.entries(oldGraphs).map(([id,oldGraph])=>{
    const graph=structuredClone(oldGraph),p=graph.operations[0].params,part=parts.find(part=>part.id===id),partMask=layers.partMasks[id];
    if(!part||!partMask)return {part:id,status:'UNAVAILABLE-parser-part'};
    const paired=parts.find(part=>part.id===(id.startsWith('left')?'right-thigh':'left-thigh'));
    const limb=/(?:upper-arm|forearm|thigh|calf)$/.test(id);
    const support=limb?measuredLimbSupport({...part,partMask},partMask,canvas.width,canvas.height,paired):measuredTorsoSupport(part,partMask,canvas.width,canvas.height);
    if(!support)return {part:id,status:'UNAVAILABLE-parser-contour'};
    const left=Math.max(0,Math.min(...support.map(v=>v.x))-.008),top=Math.max(0,Math.min(...support.map(v=>v.y))-.008),right=Math.min(1,Math.max(...support.map(v=>v.x))+.008),bottom=Math.min(1,Math.max(...support.map(v=>v.y))+.008);
    Object.assign(p,{subjectMask:layers.subjectMask,partMask,protectMask:layers.protectMask,mask:{shape:'polygon',points:support,feather:8},region:{x:left,y:top,width:right-left,height:bottom-top},landmarks:part.landmarks.map(({x,y})=>({x,y}))});
    return {part:id,graph};
   });
   return JSON.stringify({checks,layerMs:Math.round(performance.now()-start),layers,points:person.points});
  },JSON.stringify({source,classes,oldGraphs})));
  await fs.writeFile(path.join(dir,'source.png'),Buffer.from(source,'base64'));
  await fs.writeFile(path.join(dir,'layers.json'),JSON.stringify({layers:prepared.layers,points:prepared.points}));
  const checks=[];
  for(const check of prepared.checks) {
   if(!check.graph){checks.push(check);continue;}
   const graphPath=path.join(dir,check.part+'.json'),output=path.join(dir,check.part+'-output.png');await fs.writeFile(graphPath,JSON.stringify(check.graph));
   try {
    await verifyBinary();const start=performance.now();await execute(binary,['--input',path.join(dir,'source.png'),'--output',output,'--graph',graphPath]);const elapsedMs=Math.round(performance.now()-start);await verifyBinary();
    const review=await page.evaluate(async(encoded)=>{
     const {source,output,oldOutput,params,version,previousVersion}=JSON.parse(encoded);
     const load=async data=>{const image=new Image();image.src='data:image/png;base64,'+data;await image.decode();return image;};
     const [original,newImage,oldImage]=await Promise.all([load(source),load(output),load(oldOutput)]),w=original.width,h=original.height;
     const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');ctx.drawImage(original,0,0);const before=ctx.getImageData(0,0,w,h);ctx.clearRect(0,0,w,h);ctx.drawImage(newImage,0,0);const after=ctx.getImageData(0,0,w,h),diff=ctx.createImageData(w,h);
     const at=(mask,x,y)=>{const px=Math.max(0,Math.min(mask.width-1,x*mask.width-.5)),py=Math.max(0,Math.min(mask.height-1,y*mask.height-.5)),x0=Math.floor(px),y0=Math.floor(py),x1=Math.min(mask.width-1,x0+1),y1=Math.min(mask.height-1,y0+1),tx=px-x0,ty=py-y0;return (mask.data[y0*mask.width+x0]*(1-tx)*(1-ty)+mask.data[y0*mask.width+x1]*tx*(1-ty)+mask.data[y1*mask.width+x0]*(1-tx)*ty+mask.data[y1*mask.width+x1]*tx*ty)/255;};
     const metrics={changedPixels:0,changedProtectedPixels:0,changedNonTargetPixels:0,changedKnownBackgroundPixels:0,changedOutsideSupportPixels:0};
     for(let y=0;y<h;y++)for(let x=0;x<w;x++) {const i=(y*w+x)*4,nx=(x+.5)/w,ny=(y+.5)/h;let delta=0;for(let c=0;c<4;c++)delta+=Math.abs(before.data[i+c]-after.data[i+c]);if(delta){metrics.changedPixels++;if(at(params.protectMask,nx,ny)>0)metrics.changedProtectedPixels++;if(at(params.partMask,nx,ny)<.01)metrics.changedNonTargetPixels++;if(at(params.subjectMask,nx,ny)<.01)metrics.changedKnownBackgroundPixels++;const r=params.region;if(nx<r.x||nx>r.x+r.width||ny<r.y||ny>r.y+r.height)metrics.changedOutsideSupportPixels++;}diff.data.set([Math.min(255,delta*8),0,0,255],i);}
     const difference=document.createElement('canvas');difference.width=w;difference.height=h;difference.getContext('2d').putImageData(diff,0,0);
     const r=params.region,left=Math.max(0,Math.floor(r.x*w)-16),top=Math.max(0,Math.floor(r.y*h)-16),cw=Math.min(w-left,Math.ceil(r.width*w)+32),ch=Math.min(h-top,Math.ceil(r.height*h)+32);
     const comparison=document.createElement('canvas');comparison.width=cw*4;comparison.height=ch+32;const out=comparison.getContext('2d');out.fillStyle='#161b22';out.fillRect(0,0,comparison.width,comparison.height);out.fillStyle='#fff';out.font='14px sans-serif';[[original,'Original 100%'],[oldImage,previousVersion.toUpperCase()+' / 60'],[newImage,version.toUpperCase()+' / 60'],[difference,'Difference x8']].forEach(([image,label],i)=>{out.fillText(label,i*cw+8,22);out.drawImage(image,left,top,cw,ch,i*cw,32,cw,ch);});
     return {metrics,crop:{left,top,width:cw,height:ch},png:comparison.toDataURL().split(',')[1]};
    },JSON.stringify({source,output:(await fs.readFile(output)).toString('base64'),oldOutput:(await fs.readFile(path.join(previousDir,check.part+'-output.png'))).toString('base64'),params:check.graph.operations[0].params,version,previousVersion}));
    await fs.mkdir(path.join(root,'doc/qa',`body-parts-${version}`),{recursive:true});await fs.writeFile(path.join(root,'doc/qa',`body-parts-${version}`,id+'-'+check.part+'.png'),Buffer.from(review.png,'base64'));
    checks.push({part:check.part,status:review.metrics.changedPixels&&Object.entries(review.metrics).every(([key,value])=>key==='changedPixels'||value===0)?'NEEDS-visual-review':'FAIL-contract',elapsedMs,graphSha256:sha(await fs.readFile(graphPath)),outputSha256:sha(await fs.readFile(output)),metrics:review.metrics,crop:review.crop});
   }catch(error){checks.push({part:check.part,status:'FAIL-render',error:String(error)});}
  }
  const record={id,rendererSha256,sourceSha256:sha(Buffer.from(source,'base64')),parserCacheSha256:sha(await fs.readFile(path.join(root,'.run/fashn-results',id,'masks.json'))),parserOrigin:'official Torch logits -> full-source bilinear -> softmax -> probability raster (cached); not the browser decoder',layerMs:prepared.layerMs,checks};results.push(record);await fs.writeFile(path.join(dir,'quality.json'),JSON.stringify(record,null,2));console.log(JSON.stringify(record));
 }
 await fs.writeFile(path.join(root,'doc/qa',`body-parts-${version}-metrics.json`),JSON.stringify(results,null,2));
 await verifyBinary();await fs.writeFile(path.join(root,'doc/qa',`body-parts-${version}-provenance.json`),JSON.stringify({binary,rendererSha256,version,previousVersion,verifiedBeforeAndAfterEveryRender:true,parserModelSha256:'3b4925715aecbf4996fea5de5e7ce30b173c20f751f21cffe826b5db46ed7729',adjustment:60,checks:results.flatMap(result=>result.checks).length},null,2));
}finally {await browser.close();}

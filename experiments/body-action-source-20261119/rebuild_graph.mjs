/** Rebuild waist geometry from source-bound material ownership, without inference.
 * Canonical Pose landmarks, source/instance/protection and strength stay fixed.
 * Optical alpha is not a substitute for geometric material ownership here.
 */
import fs from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {createServer} from '../../apps/web/node_modules/vite/dist/node/index.js';
import {chromium} from '../../apps/web/node_modules/playwright/index.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const [evidenceArg,maskArg,outputArg]=process.argv.slice(2);
if(!evidenceArg||!maskArg||!outputArg)throw new Error('source-evidence torso-mask-json new-output-directory required');
const base=path.resolve(evidenceArg),maskPath=path.resolve(maskArg),out=path.resolve(outputArg);
const sha=async p=>{const h=createHash('sha256');for await(const b of createReadStream(p))h.update(b);return h.digest('hex');};
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const source=path.join(base,'source.png'),oldPath=path.join(base,'source-graph.json');
const observation=await read(path.join(base,'observation.json')),original=await read(oldPath),record=await read(path.join(base,'report.json'));
const input=await read(maskPath),originalMask=input.mask;
if(record.status!=='completed'||record.historicalPreparedInput||await sha(source)!==record.canonicalSourceSHA256||await sha(oldPath)!==record.graphSHA256)throw new Error('frozen fresh original evidence mismatch');
if(!input.mask||input.sourceSHA256!==record.canonicalSourceSHA256)throw new Error('candidate must contain sourceSHA256 and its bound geometric mask');
if(originalMask.shape!=='raster'||!Number.isSafeInteger(originalMask.width)||!Number.isSafeInteger(originalMask.height)||originalMask.width<1||originalMask.height<1||!Array.isArray(originalMask.data)||originalMask.data.length!==originalMask.width*originalMask.height||originalMask.data.some(x=>!Number.isInteger(x)||x<0||x>255))throw new Error('valid source-coordinate byte RasterMask required');
// Public RasterMask schema permits at most 512 per axis. Match the existing
// deriveBodyLayers grid policy, retaining the full-size source photograph.
// Positive area averaging only changes serialization resolution; no class
// threshold, morphology, protection or requested strength is adjusted.
function publicRaster(source) {
 if(Math.max(source.width,source.height)<=512)return source;
 const scale=512/Math.max(source.width,source.height),width=Math.round(source.width*scale),height=Math.round(source.height*scale),data=new Array(width*height);
 for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
  const left=x*source.width/width,right=(x+1)*source.width/width,top=y*source.height/height,bottom=(y+1)*source.height/height;let sum=0,area=0;
  for(let sy=Math.floor(top);sy<Math.ceil(bottom);sy++)for(let sx=Math.floor(left);sx<Math.ceil(right);sx++) {
   const a=(Math.min(right,sx+1)-Math.max(left,sx))*(Math.min(bottom,sy+1)-Math.max(top,sy));
   if(a>0){sum+=source.data[Math.min(source.height-1,sy)*source.width+Math.min(source.width-1,sx)]*a;area+=a;}
  }
  data[y*width+x]=Math.round(sum/area);
 }
 return {...source,width,height,data};
}
const mask=publicRaster(originalMask);
const op=original.operations?.[0];
if(original.operations.length!==1||op.type!=='body.adjust'||op.params.bodyPart!=='waist'||op.enabled!==true)throw new Error('one original enabled waist operation required');
await fs.mkdir(out,{recursive:false});
const modulePaths=['apps/web/src/core/landmarks.ts','apps/web/src/core/bodyParsing.ts'];
const modules=Object.fromEntries(await Promise.all(modulePaths.map(async p=>[p,await sha(path.join(root,p))])));
const protocol={sourceSHA256:record.canonicalSourceSHA256,originalGraphSHA256:record.graphSHA256,observationSHA256:await sha(path.join(base,'observation.json')),
 candidatePath:maskPath,candidateSHA256:await sha(maskPath),candidateSourceBound:!!input.mask,modules,scriptSHA256:await sha(fileURLToPath(import.meta.url)),
 sourceOnly:true,newInference:false,bodyFormulaChanged:false,canonicalLandmarksPreserved:true,subjectAndProtectionPreserved:true,
 rebuiltFields:['params.partMask','params.mask.points','params.region'],supportMethod:'current bodyParts + measuredTorsoSupport; exact current regionForPolygon margin .008',
 candidateRasterAdaptation:{inputDimensions:[originalMask.width,originalMask.height],publicDimensions:[mask.width,mask.height],method:mask===originalMask?'identity':'positive-area average with byte rounding; max axis 512 public schema',sourcePhotographResized:false},
 status:'starting',sourceMaskMeaning:'geometric source-material ownership; not declared optical alpha truth'};
const write=()=>fs.writeFile(path.join(out,'report.json'),JSON.stringify(protocol,null,2)+'\n');await write();
await fs.writeFile(path.join(out,'public-torso-mask.json'),JSON.stringify({sourceSHA256:record.canonicalSourceSHA256,mask}));
protocol.publicMaskSHA256=await sha(path.join(out,'public-torso-mask.json'));await write();
let server,browser;
try {
 server=await createServer({configFile:false,root:path.join(root,'apps/web'),cacheDir:path.join(out,'module-cache'),appType:'custom',logLevel:'error',
  server:{host:'127.0.0.1',port:0,strictPort:false,open:false,hmr:false},plugins:[{name:'source-geometry-only',configureServer(s){s.middlewares.use('/algorithm-only',(_q,r)=>{r.setHeader('Content-Type','text/html');r.end('<!doctype html><meta charset="utf-8"><title>Body source geometry</title>');});}}]});
 await server.listen();const port=server.httpServer.address().port;
 browser=await chromium.launch({headless:true});const page=await browser.newPage();
 await page.route('**/*',route=>{const u=new URL(route.request().url());return u.hostname==='127.0.0.1'||['data:','blob:'].includes(u.protocol)?route.continue():route.abort();});
 await page.goto(`http://127.0.0.1:${port}/algorithm-only`);
 const result=await page.evaluate(async({points,mask,width,height,encoded,originalLandmarks})=>{
  const {bodyParts,measuredTorsoSupport}=await import('/src/core/landmarks.ts');
  const canonical=bodyParts(points,width,height).find(p=>p.id==='waist');
  if(!canonical)throw new Error('source Pose has no canonical waist');
  const landmarks=canonical.landmarks.map(({x,y})=>({x,y}));
  if(JSON.stringify(landmarks)!==JSON.stringify(originalLandmarks))throw new Error('canonical source landmarks differ from original graph');
  const support=measuredTorsoSupport({...canonical,partMask:mask},mask,width,height);
  if(!support||support.length<3||support.some(p=>!Number.isFinite(p.x)||!Number.isFinite(p.y)))return {available:false,reason:'current source measurement cannot establish waist support'};
  // Exact private regionForPolygon = expandBox(boxFromPoints(support), .008)
  // from landmarks.ts; no hand-expanded region or old support is reused.
  const clamp=x=>Math.max(0,Math.min(1,x));
  const left=clamp(Math.min(...support.map(p=>p.x))),top=clamp(Math.min(...support.map(p=>p.y)));
  const right=clamp(Math.max(...support.map(p=>p.x))),bottom=clamp(Math.max(...support.map(p=>p.y)));
  const x=Math.max(0,left-.008),y=Math.max(0,top-.008),r=Math.min(1,left+Math.max(.001,right-left)+.008),b=Math.min(1,top+Math.max(.001,bottom-top)+.008);
  const region={x,y,width:Math.max(.001,r-x),height:Math.max(.001,b-y)};
  const image=new Image();image.src='data:image/png;base64,'+encoded;await image.decode();
  if(image.naturalWidth!==width||image.naturalHeight!==height)throw new Error('source canvas dimensions mismatch');
  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);
  const pixels=ctx.getImageData(0,0,width,height),rgba=pixels.data;
  for(let py=0;py<height;py++)for(let px=0;px<width;px++){
   const mx=Math.max(0,Math.min(mask.width-1,(px+.5)*mask.width/width-.5)),my=Math.max(0,Math.min(mask.height-1,(py+.5)*mask.height/height-.5));
   const x0=Math.floor(mx),y0=Math.floor(my),x1=Math.min(mask.width-1,x0+1),y1=Math.min(mask.height-1,y0+1),dx=mx-x0,dy=my-y0;
   const a=(mask.data[y0*mask.width+x0]*(1-dx)*(1-dy)+mask.data[y0*mask.width+x1]*dx*(1-dy)+mask.data[y1*mask.width+x0]*(1-dx)*dy+mask.data[y1*mask.width+x1]*dx*dy)/255*.55;
   const i=(py*width+px)*4;[30,230,80].forEach((v,c)=>rgba[i+c]=Math.round(rgba[i+c]*(1-a)+v*a));
  }
  ctx.putImageData(pixels,0,0);ctx.strokeStyle='#1482ff';ctx.lineWidth=2;ctx.beginPath();support.forEach((p,i)=>i?ctx.lineTo(p.x*width,p.y*height):ctx.moveTo(p.x*width,p.y*height));ctx.closePath();ctx.stroke();
  ctx.strokeStyle='#ffdc14';ctx.beginPath();landmarks.forEach((p,i)=>i?ctx.lineTo(p.x*width,p.y*height):ctx.moveTo(p.x*width,p.y*height));ctx.stroke();
  return {available:true,landmarks,support:support.map(({x,y})=>({x,y})),region,overlay:canvas.toDataURL('image/png').split(',')[1]};
 },{points:observation.points,mask,width:record.dimensions.width,height:record.dimensions.height,encoded:(await fs.readFile(source)).toString('base64'),originalLandmarks:op.params.landmarks});
 for(const [p,hash]of Object.entries(modules))if(await sha(path.join(root,p))!==hash)throw new Error('geometry source changed during rebuild');
 if(await sha(maskPath)!==protocol.candidateSHA256||await sha(source)!==protocol.sourceSHA256||await sha(oldPath)!==protocol.originalGraphSHA256)throw new Error('frozen source or candidate changed during rebuild');
 if(!result.available){Object.assign(protocol,{status:'unavailable',reason:result.reason});await write();console.log(JSON.stringify(protocol));process.exitCode=2;}
 else {
  const graph=structuredClone(original);graph.operations[0].params.partMask=mask;graph.operations[0].params.mask={...op.params.mask,points:result.support};graph.operations[0].params.region=result.region;
  await fs.writeFile(path.join(out,'source-graph.json'),JSON.stringify(graph));await fs.writeFile(path.join(out,'source-support-overlay.png'),Buffer.from(result.overlay,'base64'));
  delete result.overlay;await fs.writeFile(path.join(out,'support.json'),JSON.stringify(result,null,2)+'\n');
  Object.assign(protocol,{status:'completed',graphSHA256:await sha(path.join(out,'source-graph.json')),supportSHA256:await sha(path.join(out,'support.json')),
   originalGeometrySame:JSON.stringify(op.params.mask.points)===JSON.stringify(result.support)&&JSON.stringify(op.params.region)===JSON.stringify(result.region)});await write();console.log(JSON.stringify(protocol));
 }
} catch(error){Object.assign(protocol,{status:'failed',error:String(error)});await write();throw error;}
finally{if(browser)await browser.close();if(server)await server.close();await fs.rm(path.join(out,'module-cache'),{recursive:true,force:true});}

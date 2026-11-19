/** Fresh original-photo body inference using current source modules only.
 * No historical detections, prepared layers, edited images or UI build input.
 */
import fs from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {createServer} from '../../apps/web/node_modules/vite/dist/node/index.js';
import {chromium} from '../../apps/web/node_modules/playwright/index.mjs';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const [sourceArgument,outputArgument,part='waist']=process.argv.slice(2);
if(!sourceArgument||!outputArgument)throw new Error('source-photo new-output-directory [body-part] required');
const source=path.resolve(sourceArgument),output=path.resolve(outputArgument);
await fs.mkdir(output,{recursive:false});
const sha=async file=>{const hash=createHash('sha256');for await(const bytes of createReadStream(file))hash.update(bytes);return hash.digest('hex');};
const scopePath=path.join(output,'..','source-scope.json');
const scope=JSON.parse(await fs.readFile(scopePath,'utf8'));
if(scope.sourceSHA256!==await sha(source)||!scope.frozenBeforeDetectionAndEdits)throw new Error('a frozen scope bound to the original source is required');
const paths=['apps/web/src/core/landmarks.ts','apps/web/src/core/bodyParsing.ts','apps/web/src/core/bodyParsing.worker.ts',
  'apps/web/public/models/bodyparser.onnx','apps/web/public/models/pose_landmarker_lite.task'];
const inputs=Object.fromEntries(await Promise.all(paths.map(async file=>[file,await sha(path.join(root,file))])));
const report={source,sourceSHA256:await sha(source),part,inputs,status:'starting',
  inferenceInput:'original JPEG decoded once by browser at its original dimensions',
  algorithmOnly:true,applicationBuilt:false,editorOpened:false,oneSourceObservation:true,
  historicalPreparedInput:false,manualMeasurement:false,modelDownload:false,
  posteriorCapture:'read-only listener on the same FASHN worker message; no second inference',
  sourceScopeSHA256:await sha(scopePath),
  scriptSHA256:await sha(fileURLToPath(import.meta.url))};
const write=()=>fs.writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
await write();
let server,browser;
try {
  server=await createServer({configFile:false,root:path.join(root,'apps/web'),
    cacheDir:path.join(output,'detector-module-cache'),appType:'custom',logLevel:'error',
    server:{host:'127.0.0.1',port:0,strictPort:false,open:false,hmr:false,
      headers:{'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'}},
    plugins:[{name:'body-algorithm-only',configureServer(instance){instance.middlewares.use('/algorithm-only',(_request,response)=>{
      response.setHeader('Content-Type','text/html');response.end('<!doctype html><meta charset="utf-8"><title>Body source inference</title>');
    });}}]});
  await server.listen();const port=server.httpServer.address().port;
  browser=await chromium.launch({headless:true});const page=await browser.newPage();
  const diagnostics=[];page.on('pageerror',error=>diagnostics.push(error.message));
  await page.route('**/*',route=>{const url=new URL(route.request().url());
    return url.hostname==='127.0.0.1'||['data:','blob:'].includes(url.protocol)?route.continue():route.abort();});
  await page.goto(`http://127.0.0.1:${port}/algorithm-only`);
  const observation=await page.evaluate(async({encoded,part})=>{
    const OriginalWorker=window.Worker;const captured=[];
    window.Worker=class extends OriginalWorker {
      constructor(url,options) {
        super(url,options);
        if(String(url).includes('bodyParsing.worker'))this.addEventListener('message',event=>{
          const value=event.data;
          if(value.logits)captured.push({dims:Array.from(value.dims),logits:new Float32Array(value.logits),runtime:value.runtime});
        });
      }
    };
    const image=new Image();image.src='data:image/jpeg;base64,'+encoded;await image.decode();
    const canvas=document.createElement('canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
    canvas.getContext('2d').drawImage(image,0,0);
    const {detectPeople}=await import('/src/core/landmarks.ts');
    const {bodyParserRuntime,decodeBodyLogits,BODY_CLASSES}=await import('/src/core/bodyParsing.ts');
    const started=performance.now();const people=await detectPeople(canvas,'body');
    const milliseconds=performance.now()-started;
    if(captured.length!==1)throw new Error(`expected one FASHN result, captured ${captured.length}`);
    const capturedInference=captured[0];
    const classes=decodeBodyLogits(capturedInference.logits,capturedInference.dims,canvas.width,canvas.height);
    const classWidth=classes[BODY_CLASSES[0]].width,classHeight=classes[BODY_CLASSES[0]].height;
    const classBytes=new Uint8Array(BODY_CLASSES.length*classWidth*classHeight);
    BODY_CLASSES.forEach((name,c)=>classBytes.set(classes[name].data,c*classWidth*classHeight));
    const toBase64=bytes=>{let value='';for(let i=0;i<bytes.length;i+=8192)value+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(value);};
    if(people.length!==1)throw new Error(`one-person source requires one detection, got ${people.length}`);
    const person=people[0],selected=person.parts.find(p=>p.id===part);
    const available=person.parts.map(p=>p.id);
    const result={dimensions:{width:canvas.width,height:canvas.height},milliseconds,runtime:bodyParserRuntime(),
      available,unavailable:person.unavailableParts??[],points:person.points,selected,
      classEvidence:{classNames:BODY_CLASSES,decodedDimensions:[18,classHeight,classWidth],
        decodedEncoding:'per-class Math.round(softmax(interpolated logits)*255), same decodeBodyLogits as detectPeople',
        logitsDimensions:capturedInference.dims,logitsEncoding:'Float32 little endian NCHW',inferenceMessages:captured.length,
        rawLogitsBase64:toBase64(new Uint8Array(capturedInference.logits.buffer)),decodedClassesBase64:toBase64(classBytes)},
      sourcePNG:canvas.toDataURL('image/png').split(',')[1]};
    if(selected) {
      const key=part==='waist'?'waist':part.endsWith('thigh')?'thighs':part.endsWith('calf')?'calves':part.endsWith('upper-arm')?'arms':null;
      if(!key)throw new Error('explicit adjustment mapping required');
      result.graph={version:'1.0',sourceAssetId:'fresh-original-body',source:result.dimensions,
        operations:[{id:part,type:'body.adjust',version:'1.0',enabled:true,controlPoints:[],params:{
          opacity:100,blendMode:'normal',bodyPart:part,region:selected.region,
          mask:{shape:'polygon',points:selected.points.map(({x,y})=>({x,y})),feather:8},
          landmarks:selected.landmarks.map(({x,y})=>({x,y})),subjectMask:person.subjectMask,
          partMask:selected.partMask,protectMask:selected.protectMask,adjustments:{[key]:0},
        }}]};
    }
    return result;
  },{encoded:(await fs.readFile(source)).toString('base64'),part});
  await fs.writeFile(path.join(output,'source.png'),Buffer.from(observation.sourcePNG,'base64'));delete observation.sourcePNG;
  await fs.writeFile(path.join(output,'fashn-logits.f32'),Buffer.from(observation.classEvidence.rawLogitsBase64,'base64'));
  await fs.writeFile(path.join(output,'fashn-classes.u8'),Buffer.from(observation.classEvidence.decodedClassesBase64,'base64'));
  delete observation.classEvidence.rawLogitsBase64;delete observation.classEvidence.decodedClassesBase64;
  observation.classEvidence.logitsSHA256=await sha(path.join(output,'fashn-logits.f32'));
  observation.classEvidence.classesSHA256=await sha(path.join(output,'fashn-classes.u8'));
  await fs.writeFile(path.join(output,'class-evidence.json'),JSON.stringify(observation.classEvidence,null,2)+'\n');
  delete observation.classEvidence;
  if(observation.graph) {await fs.writeFile(path.join(output,'source-graph.json'),JSON.stringify(observation.graph));delete observation.graph;}
  await fs.writeFile(path.join(output,'observation.json'),JSON.stringify(observation));
  for(const [file,hash] of Object.entries(inputs))if(await sha(path.join(root,file))!==hash)throw new Error(`detector changed: ${file}`);
  Object.assign(report,{status:observation.selected?'completed':'unavailable-target',dimensions:observation.dimensions,
    milliseconds:observation.milliseconds,runtime:observation.runtime,available:observation.available,
    unavailable:observation.unavailable,canonicalSourceSHA256:await sha(path.join(output,'source.png')),
    classEvidenceSHA256:await sha(path.join(output,'class-evidence.json')),diagnostics});
  if(observation.selected)report.graphSHA256=await sha(path.join(output,'source-graph.json'));
  await write();console.log(JSON.stringify(report));
} catch(error) {Object.assign(report,{status:'failed',error:String(error)});await write();throw error;}
finally {
  if(browser)await browser.close();if(server)await server.close();
  // Only this invocation's disposable module transform cache; source hashes,
  // observations and original-size photo evidence remain in the output.
  await fs.rm(path.join(output,'detector-module-cache'),{recursive:true,force:true});
}

/** Replay cached Pose detections through the current native body engine.
 * This avoids repeated ONNX inference during contour development. It checks
 * support/background locality and emits full-resolution visual evidence;
 * these numerical checks alone do not rate naturalness. */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const execute=promisify(execFile);
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const manifest=JSON.parse(await fs.readFile(path.join(root,'test/images/manifest.json'),'utf8'));
const ids=['fashion-standing','fullbody-skirt','fullbody-side-fashion','tilted-shirt','action-suit'];
const results=[];
const browser=await chromium.launch({headless:true});
try {
  const page=await browser.newPage();
  await page.goto(process.env.RETOUCH_WEB_URL??'http://127.0.0.1:5173/');
  for(const id of ids) {
    const sample=manifest.find(sample=>sample.id===id);
    const cache=JSON.parse(await fs.readFile(path.join(root,'test/results/body-algorithms',id,'quality.json'),'utf8'));
    const encoded=(await fs.readFile(path.join(root,'test/images',sample.path))).toString('base64');
    const prepared=await page.evaluate(async({encoded,cache})=>{
      const {decodeImage,pixelsToCanvas}=await import('/src/core/engine.ts');
      const {measuredLimbSupport,measuredTorsoSupport}=await import('/src/core/landmarks.ts');
      const decoded=await decodeImage(new File([Uint8Array.from(atob(encoded),c=>c.charCodeAt(0))],'input.jpg',{type:'image/jpeg'}));
      const canvas=document.createElement('canvas');canvas.width=cache.source.width;canvas.height=cache.source.height;
      canvas.getContext('2d').drawImage(pixelsToCanvas(decoded),0,0,canvas.width,canvas.height);
      const checks=cache.checks.map(check=>{
        const graph=structuredClone(check.graph);const p=graph.operations[0].params;
        const part={id:p.bodyPart,label:p.bodyPart,region:p.region,points:p.mask.points,landmarks:p.landmarks};
        const paired=cache.checks.find(other=>other.part===(p.bodyPart.startsWith('left')?'right-thigh':'left-thigh'));
        const pairedPart=paired?{...part,id:paired.part,landmarks:paired.graph.operations[0].params.landmarks}:undefined;
        const limb=/(?:upper-arm|forearm|thigh|calf)$/.test(p.bodyPart);
        const support=limb?measuredLimbSupport(part,p.subjectMask,canvas.width,canvas.height,pairedPart):measuredTorsoSupport(part,p.subjectMask,canvas.width,canvas.height);
        if(!support)return {part:check.part,status:'UNAVAILABLE-hidden-contour'};
        p.mask.points=support;
        const left=Math.max(0,Math.min(...support.map(v=>v.x))-.008),top=Math.max(0,Math.min(...support.map(v=>v.y))-.008);
        const right=Math.min(1,Math.max(...support.map(v=>v.x))+.008),bottom=Math.min(1,Math.max(...support.map(v=>v.y))+.008);
        p.region={x:left,y:top,width:right-left,height:bottom-top};
        return {part:check.part,graph};
      });
      return {source:canvas.toDataURL().split(',')[1],checks};
    },{encoded,cache});
    const dir=path.join(root,'test/results/body-contour-v2',id);await fs.mkdir(dir,{recursive:true});
    const input=path.join(dir,'source.png');await fs.writeFile(input,Buffer.from(prepared.source,'base64'));
    const checks=[];
    for(const check of prepared.checks) {
      if(!check.graph){checks.push(check);continue;}
      const graphFile=path.join(dir,`${check.part}.json`),output=path.join(dir,`${check.part}-output.png`);
      await fs.writeFile(graphFile,JSON.stringify(check.graph));
      try {
        const start=performance.now();
        await execute(path.join(root,'core/image-core/target/release/image-render'),['--input',input,'--output',output,'--graph',graphFile]);
        const elapsedMs=Math.round(performance.now()-start);
        const rendered=(await fs.readFile(output)).toString('base64');
        const review=await page.evaluate(async({source,rendered,check})=>{
          const load=async encoded=>{const image=new Image();image.src='data:image/png;base64,'+encoded;await image.decode();return image;};
          const [original,edited]=await Promise.all([load(source),load(rendered)]);
          const w=original.width,h=original.height;
          const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');
          ctx.drawImage(original,0,0);const before=ctx.getImageData(0,0,w,h);ctx.clearRect(0,0,w,h);ctx.drawImage(edited,0,0);const after=ctx.getImageData(0,0,w,h),diff=ctx.createImageData(w,h);
          const p=check.graph.operations[0].params,r=p.region,m=p.subjectMask;
          const mask=(x,y)=>{if(x<0||y<0||x>1||y>1)return 0;const px=Math.max(0,Math.min(m.width-1,x*m.width-.5)),py=Math.max(0,Math.min(m.height-1,y*m.height-.5));const x0=Math.floor(px),y0=Math.floor(py),x1=Math.min(m.width-1,x0+1),y1=Math.min(m.height-1,y0+1),tx=px-x0,ty=py-y0;return (m.data[y0*m.width+x0]*(1-tx)*(1-ty)+m.data[y0*m.width+x1]*tx*(1-ty)+m.data[y1*m.width+x0]*(1-tx)*ty+m.data[y1*m.width+x1]*tx*ty)/255;};
          const metrics={changedPixels:0,changedOriginalBackgroundPixels:0,changedOutsideSupportPixels:0};
          for(let y=0;y<h;y++)for(let x=0;x<w;x++) {
            const i=(y*w+x)*4;let delta=0;for(let c=0;c<4;c++)delta+=Math.abs(before.data[i+c]-after.data[i+c]);
            const nx=(x+.5)/w,ny=(y+.5)/h;
            if(delta){metrics.changedPixels++;if(nx<r.x||nx>r.x+r.width||ny<r.y||ny>r.y+r.height)metrics.changedOutsideSupportPixels++;
              if(Math.max(...[[0,0],[-2,0],[2,0],[0,-2],[0,2]].map(([dx,dy])=>mask((x+.5+dx)/w,(y+.5+dy)/h)))<.01)metrics.changedOriginalBackgroundPixels++;}
            diff.data.set([Math.min(255,delta*8),0,0,255],i);
          }
          const difference=document.createElement('canvas');difference.width=w;difference.height=h;difference.getContext('2d').putImageData(diff,0,0);
          const review=document.createElement('canvas');review.width=w*3;review.height=h+36;const output=review.getContext('2d');output.fillStyle='#161b22';output.fillRect(0,0,review.width,review.height);output.fillStyle='#fff';output.font='16px sans-serif';
          [[original,'Original'],[edited,`${check.part}: 60`],[difference,'Difference x8']].forEach(([image,label],i)=>{output.fillText(label,i*w+10,25);output.drawImage(image,i*w,36);});
          return {metrics,png:review.toDataURL().split(',')[1]};
        },{source:prepared.source,rendered,check});
        await fs.writeFile(path.join(dir,`${check.part}.png`),Buffer.from(review.png,'base64'));
        const status=review.metrics.changedPixels===0||review.metrics.changedOriginalBackgroundPixels||review.metrics.changedOutsideSupportPixels?'FAIL-locality':'NEEDS-visual-review';
        checks.push({part:check.part,status,metrics:review.metrics,elapsedMs});
      } catch(error) {checks.push({part:check.part,status:'FAIL-render',error:String(error)});}
    }
    const record={id,source:cache.source,checks,artifactDirectory:dir};
    await fs.writeFile(path.join(dir,'quality.json'),JSON.stringify(record,null,2));results.push(record);console.log(JSON.stringify(record));
  }
  await fs.writeFile(path.join(root,'doc/qa/body-contour-v2-metrics.json'),JSON.stringify(results,null,2));
  if(results.some(sample=>sample.checks.some(check=>check.status.startsWith('FAIL'))))process.exitCode=1;
} finally {await browser.close();}

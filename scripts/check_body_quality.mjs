/** Body segmentation/contour diagnostics on real photographs. Independent
 * re-segmentation measurements are recorded, including failures and missing
 * parts, rather than treating any pixel change as successful slimming. */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const manifest=JSON.parse(await fs.readFile(path.join(root,'test/images/manifest.json'),'utf8'));
const ids=['fullbody-skirt','fullbody-side-fashion','fashion-standing','tilted-shirt','action-suit'];
const wanted=['waist','left-upper-arm','right-upper-arm','left-thigh','right-thigh','left-calf','right-calf'];
const records=[];const browser=await chromium.launch({headless:true});
try {
 const page=await browser.newPage();await page.goto(process.env.RETOUCH_WEB_URL??'http://127.0.0.1:5173/');
 for(const sample of manifest.filter(item=>ids.includes(item.id))) {
  const encoded=(await fs.readFile(path.join(root,'test/images',sample.path))).toString('base64');
  const result=await page.evaluate(async({encoded,name,wanted})=>{
   const {decodeImage,renderImage,pixelsToCanvas}=await import('/src/core/engine.ts');
   const {detectPeople}=await import('/src/core/landmarks.ts');
   let source=await decodeImage(new File([Uint8Array.from(atob(encoded),c=>c.charCodeAt(0))],name,{type:'image/jpeg'}));
   if(Math.max(source.width,source.height)>768){const canvas=document.createElement('canvas');const factor=768/Math.max(source.width,source.height);canvas.width=Math.round(source.width*factor);canvas.height=Math.round(source.height*factor);canvas.getContext('2d').drawImage(pixelsToCanvas(source),0,0,canvas.width,canvas.height);source={width:canvas.width,height:canvas.height,pixels:new Uint8Array(canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data)};}
   const original=(await detectPeople(pixelsToCanvas(source),'body'))[0];
   if(!original?.subjectMask)return {source:{width:source.width,height:source.height},error:'Body or segmentation not detected',checks:[],missing:wanted};
   const maskValue=(mask,x,y)=>{if(x<0||y<0||x>1||y>1)return 0;const px=Math.max(0,Math.min(mask.width-1,x*mask.width-.5)),py=Math.max(0,Math.min(mask.height-1,y*mask.height-.5));const x0=Math.floor(px),y0=Math.floor(py),x1=Math.min(mask.width-1,x0+1),y1=Math.min(mask.height-1,y0+1),tx=px-x0,ty=py-y0;return (mask.data[y0*mask.width+x0]*(1-tx)*(1-ty)+mask.data[y0*mask.width+x1]*tx*(1-ty)+mask.data[y1*mask.width+x0]*(1-tx)*ty+mask.data[y1*mask.width+x1]*tx*ty)/255;};
   const widths=(mask,part)=>{
    let first=part.landmarks[0],last=part.landmarks.at(-1);
    if(part.id==='waist'){first=part.landmarks[0];last=part.landmarks[2];}
    const ax=first.x*source.width,ay=first.y*source.height,bx=last.x*source.width,by=last.y*source.height,dx=bx-ax,dy=by-ay,length=Math.hypot(dx,dy);
    if(length<1)return [];
    const nx=-dy/length,ny=dx/length;
    return [.4,.5,.6].map(t=>{
     let cx=ax+dx*t,cy=ay+dy*t;if(part.id==='waist'){cx=part.landmarks[1].x*source.width;cy=part.landmarks[1].y*source.height+(t-.5)*length*.4;}
     const limit=Math.max(8,Math.min(source.width,source.height)*.25);let lo=null,hi=null;
     // Measure the connected subject run containing the part axis, not the
     // distant arm/person hit by the same infinite line.
     if(maskValue(mask,cx/source.width,cy/source.height)<.5)return {t,width:null};
     for(let d=0;d<=limit;d++){if(maskValue(mask,(cx+nx*d)/source.width,(cy+ny*d)/source.height)<.5){hi=d;break;}}
     for(let d=0;d<=limit;d++){if(maskValue(mask,(cx-nx*d)/source.width,(cy-ny*d)/source.height)<.5){lo=-d;break;}}
     return {t,width:lo===null||hi===null?null:hi-lo};
    });
   };
   const checks=[];const parts=original.parts.filter(part=>wanted.includes(part.id));
   for(const part of parts){
    const key=part.id==='waist'?'waist':part.id.endsWith('upper-arm')?'arms':part.id.endsWith('thigh')?'thighs':'calves';
    const graph={version:'1.0',sourceAssetId:'local',operations:[{id:'quality',type:'body.adjust',version:'1.0',enabled:true,params:{region:part.region,mask:{shape:'polygon',points:part.points.map(({x,y})=>({x,y})),feather:8},landmarks:part.landmarks.map(({x,y})=>({x,y})),bodyPart:part.id,subjectMask:original.subjectMask,...(part.protectMask?{protectMask:part.protectMask}:{}),adjustments:{[key]:60}},controlPoints:[]}]};
    try{
     const output=await renderImage(source,graph,{kind:'export'});const after=(await detectPeople(pixelsToCanvas(output),'body'))[0];
     const metrics={changedPixels:0,changedOriginalBackgroundPixels:0,changedOutsideSupportPixels:0,originalWidths:widths(original.subjectMask,part),outputWidths:after?.subjectMask?widths(after.subjectMask,part):[]};
     const difference={...source,pixels:new Uint8Array(source.pixels.length)};
     for(let y=0;y<source.height;y++)for(let x=0;x<source.width;x++){
      const i=(y*source.width+x)*4;let delta=0;for(let c=0;c<3;c++)delta+=Math.abs(source.pixels[i+c]-output.pixels[i+c]);
      const nx=(x+.5)/source.width,ny=(y+.5)/source.height,r=part.region;
      if(delta){metrics.changedPixels++;if(nx<r.x||nx>r.x+r.width||ny<r.y||ny>r.y+r.height)metrics.changedOutsideSupportPixels++;
       // A conservative 2px expansion avoids mislabelling antialiased source
       // boundaries as fixed background.
       let maxAlpha=0;for(const [ox,oy] of [[0,0],[-2,0],[2,0],[0,-2],[0,2]])maxAlpha=Math.max(maxAlpha,maskValue(original.subjectMask,(x+.5+ox)/source.width,(y+.5+oy)/source.height));
       if(maxAlpha<.01)metrics.changedOriginalBackgroundPixels++;
      }
      difference.pixels.set([Math.min(255,delta*8),0,0,255],i);
     }
     const scale=Math.min(1,420/source.width);const w=Math.round(source.width*scale),h=Math.round(source.height*scale);
     const canvas=document.createElement('canvas');canvas.width=w*3;canvas.height=h+36;const ctx=canvas.getContext('2d');ctx.fillStyle='#161b22';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='white';ctx.font='16px sans-serif';
     for(const [j,[image,label]]of[[source,'Original'],[output,`${part.id}: 60`],[difference,'Difference x8']].entries()){ctx.fillText(label,j*w+10,25);ctx.drawImage(pixelsToCanvas(image),j*w,36,w,h);}
     const pairs=metrics.originalWidths.map((v,i)=>v.width!==null&&metrics.outputWidths[i]?.width!==null?metrics.outputWidths[i]?.width-v.width:null).filter(v=>Number.isFinite(v));
     const median=pairs.length?[...pairs].sort((a,b)=>a-b)[Math.floor(pairs.length/2)]:null;
     const status=metrics.changedPixels===0||metrics.changedOutsideSupportPixels||metrics.changedOriginalBackgroundPixels?'FAIL-pixel-locality':median===null?'UNVERIFIED-no-contour-measurement':median<0?'contour-resegmentation-shrank':'UNVERIFIED-no-measured-shrink';
     checks.push({part:part.id,key,status,metrics,medianWidthDelta:median,graph,png:canvas.toDataURL().split(',')[1]});
    }catch(error){checks.push({part:part.id,key,status:'FAIL-render',error:String(error),graph});}
   }
   return {source:{width:source.width,height:source.height},checks,missing:wanted.filter(id=>!parts.some(part=>part.id===id))};
  },{encoded,name:sample.path,wanted});
  const dir=path.join(root,'test/results/body-algorithms',sample.id);await fs.mkdir(dir,{recursive:true});
  for(const check of result.checks){if(check.png){await fs.writeFile(path.join(dir,`${check.part}.png`),Buffer.from(check.png,'base64'));delete check.png;}}
  await fs.writeFile(path.join(dir,'quality.json'),JSON.stringify(result,null,2));
  records.push({id:sample.id,...result,checks:result.checks.map(({graph,...check})=>check),artifactDirectory:dir});
  console.log(JSON.stringify({id:sample.id,error:result.error,missing:result.missing,checks:result.checks.map(({part,status,metrics,medianWidthDelta,error})=>({part,status,metrics,medianWidthDelta,error}))}));
 }
 await fs.mkdir(path.join(root,'doc/qa'),{recursive:true});await fs.writeFile(path.join(root,'doc/qa/body-multisample-review.json'),JSON.stringify(records,null,2));
 const rows=records.flatMap(sample=>sample.checks.map(check=>`| ${sample.id} | ${check.part} | ${check.status} | ${check.medianWidthDelta??'未测得'} | ${check.metrics?.changedOriginalBackgroundPixels??'-'} |`));
 await fs.writeFile(path.join(root,'doc/qa/body-multisample-review.md'),`# 多样本身形效果记录\n\n日期：2026-10-13。最长边768px。输出重新Pose分割，测量原部位轴线的连续前景宽度，测量只作辅助判断；表中UNVERIFIED必须视觉检查，不能当成收缩验收通过。\n\n| 照片 | 部位 | 状态 | 宽度变化px（重新分割） | 原背景变化 |\n|---|---|---|---|---|\n${rows.join('\n')}\n\n缺失部位/检测失败：\n\n${records.filter(sample=>sample.error||sample.missing.length).map(sample=>`- ${sample.id}：${sample.error??sample.missing.join(', ')}`).join('\n')}\n\n原图、输出、差分图与操作图保存于test/results/body-algorithms。\n`);
 if(records.some(sample=>sample.error||sample.checks.some(check=>check.status.startsWith('FAIL'))))process.exitCode=1;
}finally{await browser.close();}

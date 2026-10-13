/** Real fabric comparisons with hand-marked clothing-only ROIs. Fine-fold
 * softening is not geometric crease removal; colour metrics and pixel
 * locality are recorded separately from the visual quality assessment. */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const samples=[
 {id:'clothing-texture',region:{x:.48,y:.56,width:.14,height:.3}},
 {id:'tilted-shirt',region:{x:.49,y:.46,width:.1,height:.16}},
 {id:'fashion-standing',region:{x:.34,y:.58,width:.25,height:.08}},
];
const browser=await chromium.launch({headless:true}),records=[];
try {
 const page=await browser.newPage();await page.goto(process.env.RETOUCH_WEB_URL??'http://127.0.0.1:5173/');
 for(const sample of samples) {
  const encoded=(await fs.readFile(path.join(root,'test/images',sample.id+'.jpg'))).toString('base64');
  const result=await page.evaluate(async({encoded,id,region})=>{
   const {decodeImage,renderImage,pixelsToCanvas}=await import('/src/core/engine.ts');
   let source=await decodeImage(new File([Uint8Array.from(atob(encoded),c=>c.charCodeAt(0))],id+'.jpg',{type:'image/jpeg'}));
   if(Math.max(source.width,source.height)>1024){const c=document.createElement('canvas'),scale=1024/Math.max(source.width,source.height);c.width=Math.round(source.width*scale);c.height=Math.round(source.height*scale);c.getContext('2d').drawImage(pixelsToCanvas(source),0,0,c.width,c.height);source={width:c.width,height:c.height,pixels:new Uint8Array(c.getContext('2d').getImageData(0,0,c.width,c.height).data)};}
   const measure=image=>{
    let count=0,luminance=0,chroma=0,gradient=0;
    const luma=i=>.2126*image.pixels[i]+.7152*image.pixels[i+1]+.0722*image.pixels[i+2];
    for(let y=Math.ceil(region.y*image.height)+6;y<(region.y+region.height)*image.height-6;y++)for(let x=Math.ceil(region.x*image.width)+6;x<(region.x+region.width)*image.width-6;x++){
     const i=(y*image.width+x)*4,rgb=Array.from(image.pixels.slice(i,i+3));luminance+=luma(i);chroma+=Math.max(...rgb)-Math.min(...rgb);gradient+=Math.abs(luma(i)-luma(i+4))+Math.abs(luma(i)-luma(i+image.width*4));count++;
    }
    return {count,meanLuminance:luminance/count,meanChroma:chroma/count,gradientEnergy:gradient/count};
   };
   const cases=[{key:'smooth',adjustments:{smooth:60,texture:70}},{key:'texture-low',adjustments:{smooth:80,texture:0}},{key:'texture-high',adjustments:{smooth:80,texture:100}},{key:'hue',adjustments:{hue:60}},{key:'saturation',adjustments:{saturation:60}},{key:'brightness',adjustments:{brightness:30}},{key:'texture-without-smooth',adjustments:{smooth:0,texture:100}}];
   const checks=[];
   for(const item of cases){
    const graph={version:'1.0',sourceAssetId:'local',operations:[{id:'cloth',type:'clothing.adjust',version:'1.0',enabled:true,params:{region,mask:{shape:'rect',region,feather:0},adjustments:item.adjustments},controlPoints:[]}]};
    const output=await renderImage(source,graph,{kind:'export'}),diff={...source,pixels:new Uint8Array(source.pixels.length)};
    let changedInside=0,changedOutside=0;
    for(let y=0;y<source.height;y++)for(let x=0;x<source.width;x++){
     const i=(y*source.width+x)*4,nx=(x+.5)/source.width,ny=(y+.5)/source.height;let delta=0;for(let c=0;c<3;c++)delta+=Math.abs(source.pixels[i+c]-output.pixels[i+c]);
     if(delta){if(nx>=region.x&&nx<=region.x+region.width&&ny>=region.y&&ny<=region.y+region.height)changedInside++;else changedOutside++;}
     diff.pixels.set([Math.min(255,delta*8),0,0,255],i);
    }
    const crop={x:Math.max(0,Math.floor((region.x-.02)*source.width)),y:Math.max(0,Math.floor((region.y-.02)*source.height)),w:Math.ceil((region.width+.04)*source.width),h:Math.ceil((region.height+.04)*source.height)};
    const canvas=document.createElement('canvas');canvas.width=crop.w*9;canvas.height=crop.h*3+36;const ctx=canvas.getContext('2d');ctx.fillStyle='#161b22';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='white';ctx.font='16px sans-serif';
    for(const [index,[image,label]]of[[source,'Original'],[output,item.key],[diff,'Difference x8']].entries()){ctx.fillText(label,index*crop.w*3+5,25);ctx.drawImage(pixelsToCanvas(image),crop.x,crop.y,crop.w,crop.h,index*crop.w*3,36,crop.w*3,crop.h*3);}
    checks.push({...item,changedInside,changedOutside,metrics:measure(output),png:canvas.toDataURL().split(',')[1]});
   }
   return {width:source.width,height:source.height,region,sourceMetrics:measure(source),checks};
  },{encoded,...sample});
  const dir=path.join(root,'test/results/clothing-algorithms',sample.id);await fs.mkdir(dir,{recursive:true});
  for(const check of result.checks){await fs.writeFile(path.join(dir,check.key+'.png'),Buffer.from(check.png,'base64'));delete check.png;}
  records.push({id:sample.id,...result,artifactDirectory:dir});console.log(JSON.stringify(records.at(-1)));
 }
 await fs.mkdir(path.join(root,'doc/qa'),{recursive:true});await fs.writeFile(path.join(root,'doc/qa/clothing-multisample-review.json'),JSON.stringify(records,null,2));
 const rows=records.flatMap(sample=>sample.checks.map(check=>`| ${sample.id} | ${check.key} | ${check.changedInside} | ${check.changedOutside} | ${check.metrics.gradientEnergy.toFixed(3)} |`));
 await fs.writeFile(path.join(root,'doc/qa/clothing-multisample-review.md'),`# 衣物局部算法真实样本记录\n\n日期：2026-10-13。三个手工衣物内部选区，最长边1024px。只检查局部调色、细褶纹理弱化与纹理保留；这组能力不能宣称几何去褶皱。纹理高/低均固定smooth=80；smooth=0时纹理保留滑杆应保持原图。高频梯度降低不能代表大褶皱消失，仍需视觉检查。\n\n| 照片 | 参数 | 选区内变化 | 选区外变化 | 梯度能量 |\n|---|---|---|---|---|\n${rows.join('\n')}\n\n原图/输出/差分图保存于test/results/clothing-algorithms。灰白衣物缺少彩色像素时色相响应可能很小，不能单靠changed断言效果好坏。\n`);
 if(records.some(r=>r.checks.some(c=>c.changedOutside!==0||c.key==='texture-without-smooth'&&c.changedInside!==0)))process.exitCode=1;
}finally{await browser.close();}

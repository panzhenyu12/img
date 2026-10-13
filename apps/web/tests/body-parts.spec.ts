import {expect,test} from '@playwright/test';

test('真部位alpha的非对称宽袖不能被旧近侧启发式剪成骨轴细条',async({page})=>{
  await page.goto('/');
  const support=await page.evaluate(async()=>{
    const {measuredLimbSupport}=await import('/src/core/landmarks.ts');
    const width=400,height=400,mask={shape:'raster' as const,width,height,data:Array.from({length:width*height},(_,i)=>i%width>=100&&i%width<207&&Math.floor(i/width)>=50&&Math.floor(i/width)<350?255:0)};
    return measuredLimbSupport({id:'right-upper-arm',label:'袖子',region:{x:.2,y:.1,width:.4,height:.8},points:[],landmarks:[{x:.5,y:.2},{x:.5,y:.8}],partMask:mask},mask,width,height);
  });
  expect(support).toHaveLength(4);
  expect(Math.min(...support!.map(p=>p.x))).toBeLessThan(.25);
  expect(Math.max(...support!.map(p=>p.x))).toBeGreaterThan(.5175);
});

test('像素分类和骨轴共同分区，手与持物不进入身形目标层',async({page})=>{
  await page.goto('/');
  const result=await page.evaluate(async()=>{
    const {deriveBodyLayers,BODY_CLASSES}=await import('/src/core/bodyParsing.ts');
    const canvas=document.createElement('canvas');canvas.width=64;canvas.height=96;const ctx=canvas.getContext('2d')!;ctx.fillStyle='#ddd';ctx.fillRect(0,0,64,96);
    const n=64*96,classes=Object.fromEntries(BODY_CLASSES.map(name=>[name,{shape:'raster' as const,width:64,height:96,data:Array(n).fill(0)}]));
    for(let y=0;y<96;y++)for(let x=0;x<64;x++){const i=y*64+x;const c=x>=10&&x<54&&y>=15&&y<85?'top':'background';classes[c].data[i]=255;}
    for(let y=48;y<68;y++)for(let x=30;x<51;x++){const i=y*64+x;classes.top.data[i]=0;classes.bag.data[i]=255;}
    for(let y=35;y<45;y++)for(let x=8;x<17;x++){const i=y*64+x;classes.top.data[i]=0;classes.hands.data[i]=255;classes.background.data[i]=0;}
    const points=Array.from({length:33},()=>({x:.5,y:.5,visibility:0}));
    for(const [i,x,y]of [[11,.3,.2],[12,.7,.2],[13,.2,.5],[14,.8,.5],[15,.18,.7],[16,.82,.7],[23,.4,.8],[24,.6,.8]])points[i]={x,y,visibility:1};
    const instance={shape:'raster' as const,width:64,height:96,data:Array.from({length:n},(_,i)=>{const x=i%64,y=Math.floor(i/64);return x>=18&&x<50&&y>=12&&y<88?255:0})};
    const layers=deriveBodyLayers(canvas,points,classes),gated=deriveBodyLayers(canvas,points,classes,instance),i=55*64+40,j=40*64+12,outside=40*64+8,inside=40*64+30,edge=40*64+17,gap=40*64+16;
    return {protection:[layers.protectMask.data[i],layers.protectMask.data[j]],targets:Object.values(layers.partMasks).map(mask=>[mask.data[i],mask.data[j]]),torsoPixels:layers.partMasks.torso.data.filter(v=>v>127).length,gated:[gated.partMasks['full-body'].data[outside],gated.partMasks['full-body'].data[inside]],edge:gated.partMasks['full-body'].data[edge],gap:gated.partMasks['full-body'].data[gap]};
  });
  expect(result.protection).toEqual([255,255]);
  for(const target of result.targets)expect(target).toEqual([0,0]);
  expect(result.torsoPixels).toBeGreaterThan(100);
  expect(result.gated).toEqual([0,255]);
  // A one-pixel coarse-instance miss may be a real photographed garment
  // edge; recover that uncertainty band while leaving a two-pixel gap out.
  expect(result.edge).toBeGreaterThan(127);
  expect(result.gap).toBe(0);
});

test('双色与宽袖分区保持真实轮廓且相等路径不会反复灌入heap',async({page})=>{
  await page.goto('/');
  const result=await page.evaluate(async()=>{
    const {deriveBodyLayers,BODY_CLASSES}=await import('/src/core/bodyParsing.ts');
    const canvas=document.createElement('canvas');canvas.width=120;canvas.height=180;
    const ctx=canvas.getContext('2d')!;ctx.fillStyle='#24343a';ctx.fillRect(0,0,120,180);ctx.fillStyle='#dddddd';ctx.fillRect(46,25,46,130);
    const n=120*180,classes=Object.fromEntries(BODY_CLASSES.map(name=>[name,{shape:'raster' as const,width:120,height:180,data:Array(n).fill(0)}]));
    // Two flat colour regions create many exactly equal geodesic paths.
    // Mixing JS double heap costs with rounded Float32 best distances made
    // an equal path repeatedly count as a shorter one (>2M heap entries).
    // The valid parsed sleeve is much wider than any canonical bone capsule.
    for(let y=25;y<155;y++)for(let x=12;x<92;x++){const i=y*120+x;classes.top.data[i]=255;classes.background.data[i]=0;}
    const points=Array.from({length:33},()=>({x:.5,y:.5,visibility:0}));
    points[12]={x:.56,y:.2,visibility:1};points[14]={x:.56,y:.8,visibility:1};
    const layers=deriveBodyLayers(canvas,points,classes);
    const mask=layers.partMasks['right-upper-arm'];
    const row=90,at=(x:number)=>mask.data[Math.floor(row*mask.height/180)*mask.width+Math.floor(x*mask.width/120)];
    // Repaint the identical parsed geometry as one continuous wide sleeve:
    // it extends 55px from a 108px bone (>0.35L) and must retain its true edge.
    ctx.fillStyle='#dddddd';ctx.fillRect(12,25,80,130);
    const wide=deriveBodyLayers(canvas,points,classes).partMasks['right-upper-arm'];
    return {outer:at(12),inner:at(46),wideEdge:wide.data[90*120+12],width:mask.width,height:mask.height};
  });
  expect(result.outer).toBeGreaterThan(127);
  expect(result.inner).toBeGreaterThan(127);
  expect(result.wideEdge).toBeGreaterThan(127);
});

import { expect, test } from '@playwright/test';

test('腰部支持区跨过真实宽衣轮廓，不能只包住骨架内部', async ({ page }) => {
  await page.goto('/');
  const support=await page.evaluate(async()=>{
    const {measuredTorsoSupport}=await import('/src/core/landmarks.ts');
    const width=400,height=600;
    const mask={shape:'raster' as const,width,height,data:Array.from({length:width*height},(_,i)=>i%width>=80&&i%width<320&&Math.floor(i/width)>=100&&Math.floor(i/width)<500?255:0)};
    const part={id:'waist',label:'腰',region:{x:.4,y:.4,width:.2,height:.2},points:[],landmarks:[{x:.5,y:.2},{x:.5,y:.5},{x:.5,y:.8}]};
    return measuredTorsoSupport(part,mask,width,height);
  });
  expect(support).toHaveLength(4);
  expect(Math.min(...support!.map(p=>p.x))).toBeLessThan(.2);
  expect(Math.max(...support!.map(p=>p.x))).toBeGreaterThan(.8);
});

test('胯部骨轴使用像素法线，在长宽照片中都垂直于两侧胯点', async ({ page }) => {
  await page.goto('/');
  const errors=await page.evaluate(async()=>{
    const {measuredTorsoSupport}=await import('/src/core/landmarks.ts');
    return [[200,1000],[1000,200],[500,500]].map(([width,height])=>{
      const radius=Math.min(width,height)*.1;
      const mask={shape:'raster' as const,width,height,data:Array.from({length:width*height},(_,i)=>Math.hypot(i%width+.5-width*.5,Math.floor(i/width)+.5-height*.5)<radius?255:0)};
      const part={id:'hips',label:'胯',region:{x:.4,y:.48,width:.2,height:.04},points:[],landmarks:[{x:.4,y:.48},{x:.6,y:.52}]};
      const support=measuredTorsoSupport(part,mask,width,height)!;
      const hipDx=(part.landmarks[1].x-part.landmarks[0].x)*width,hipDy=(part.landmarks[1].y-part.landmarks[0].y)*height;
      const axisDx=(support[1].x-support[0].x)*width,axisDy=(support[1].y-support[0].y)*height;
      return Math.abs(hipDx*axisDx+hipDy*axisDy)/(Math.hypot(hipDx,hipDy)*Math.hypot(axisDx,axisDy));
    });
  });
  for(const error of errors)expect(error).toBeLessThan(1e-8);
});

test('可见非对称肢体两侧都获得超过轮廓的独立支持宽度', async ({ page }) => {
  await page.goto('/');
  const support=await page.evaluate(async()=>{
    const {measuredLimbSupport}=await import('/src/core/landmarks.ts');
    const width=400,height=600;
    const mask={shape:'raster' as const,width,height,data:Array.from({length:width*height},(_,i)=>i%width>=80&&i%width<240&&Math.floor(i/width)>=100&&Math.floor(i/width)<500?255:0)};
    const part={id:'left-upper-arm',label:'上臂',region:{x:.4,y:.2,width:.2,height:.6},points:[],landmarks:[{x:.5,y:.2},{x:.5,y:.8}]};
    return measuredLimbSupport(part,mask,width,height);
  });
  expect(support).toHaveLength(4);
  expect(Math.min(...support!.map(p=>p.x))).toBeLessThan(.2);
  expect(Math.max(...support!.map(p=>p.x))).toBeGreaterThan(.6);
});

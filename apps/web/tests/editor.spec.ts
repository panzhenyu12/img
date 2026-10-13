import { expect, test, type Page } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';

async function fixture(page: Page): Promise<Buffer> {
  const encoded = await page.evaluate(() => {
    const canvas = document.createElement('canvas'); canvas.width = 128; canvas.height = 128;
    const ctx = canvas.getContext('2d')!; const data = ctx.createImageData(128, 128);
    for (let y = 0; y < 128; y++) for (let x = 0; x < 128; x++) {
      const i = (y * 128 + x) * 4;
      data.data.set([(x * 5 + y * 3) % 256, (x * 11 + y * 7) % 256, (x * 17 + y * 13) % 256, 255], i);
    }
    ctx.putImageData(data, 0, 0); return canvas.toDataURL().split(',')[1];
  });
  return Buffer.from(encoded, 'base64');
}
async function ready(page: Page, renderBudget = 15_000) {
  await expect(page.locator('canvas[aria-label="图片预览"]')).toBeVisible();
  await expect(page.getByText('正在渲染…', { exact: true })).toHaveCount(0, { timeout: renderBudget });
  await expect(page.locator('.status-error')).toHaveCount(0);
}
async function pixels(page: Page): Promise<string> {
  await ready(page);
  return page.locator('canvas[aria-label="图片预览"]').evaluate((node) => {
    const canvas = node as HTMLCanvasElement;
    return Array.from(canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height).data).join(',');
  });
}
async function importFixture(page: Page) {
  await page.goto('/'); const buffer = await fixture(page);
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles({ name: 'grid.png', mimeType: 'image/png', buffer });
  await ready(page); return buffer;
}
async function drag(page: Page, from: [number, number], to: [number, number]) {
  const box = await page.locator('.canvas-overlay').boundingBox(); expect(box).not.toBeNull();
  await page.mouse.move(box!.x + box!.width * from[0], box!.y + box!.height * from[1]);
  await page.mouse.down();
  await page.mouse.move(box!.x + box!.width * to[0], box!.y + box!.height * to[1], { steps: 8 });
  await page.mouse.up(); await ready(page);
}

test('无后端时实际调色、开关、撤销重做和工程/PNG导出', async ({ page }) => {
  await page.route(/https?:\/\/(localhost|127\.0\.0\.1):8080\//, route => route.abort());
  await importFixture(page); await expect(page.locator('.statusbar')).toContainText('本地修图模式');
  const original = await pixels(page);
  await page.getByRole('button', { name: '光影与调色', exact: true }).click();
  await page.getByRole('spinbutton', { name: '亮度数值', exact: true }).fill('35');
  await expect.poll(() => pixels(page)).not.toBe(original); const edited = await pixels(page);
  await page.getByRole('button', { name: '撤销', exact: true }).click(); await expect.poll(() => pixels(page)).toBe(original);
  await page.getByRole('button', { name: '重做', exact: true }).click(); await expect.poll(() => pixels(page)).toBe(edited);
  await page.getByRole('button', { name: '关闭光影与调色', exact: true }).click(); await expect.poll(() => pixels(page)).toBe(original);
  await page.getByRole('button', { name: '启用光影与调色', exact: true }).click(); await expect.poll(() => pixels(page)).toBe(edited);
  const [project] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '保存工程', exact: true }).click()]);
  const graph = JSON.parse(await fs.readFile((await project.path())!, 'utf8'));
  expect(graph.operations[0].params.brightness).toBe(35); expect(graph.operations[0].enabled).toBe(true);
  const [output] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '导出 PNG', exact: true }).click()]);
  expect(output.suggestedFilename()).toBe('grid-edited.png'); expect((await fs.stat((await output.path())!)).size).toBeGreaterThan(100);
});

test('人像参数需要真实定位，并明确隐藏尚未实现的算法', async ({ page }) => {
  await importFixture(page);
  const capabilities = await page.evaluate(async () => {
    const { portraitParametersFor } = await import('/src/core/registry.ts');
    const keys = (tool: any, context: any) => portraitParametersFor(tool, context).map((parameter: any) => parameter.key);
    const region = { x: .1, y: .1, width: .8, height: .8 };
    const polygon = (x: number,y: number,width: number,height: number) => ({shape:'polygon',points:[{x,y},{x:x+width,y},{x:x+width,y:y+height},{x,y:y+height}]});
    const semanticMasks = { face:polygon(.2,.15,.6,.7), leftEye:polygon(.24,.3,.14,.08),rightEye:polygon(.62,.36,.14,.08),leftBrow:polygon(.24,.22,.14,.05),rightBrow:polygon(.62,.29,.14,.05),lips:polygon(.41,.65,.2,.09),nose:polygon(.45,.4,.1,.2) };
    return {
      unlocated: ['face','feature','skin','body','clothing'].map(tool => keys(tool,{region:{x:0,y:0,width:1,height:1}})),
      rectangleFace: keys('face',{region,shape:'rect'}), manualBody: keys('body',{region}),
      semanticFace: keys('face',{region,semanticMasks}), semanticFeature:keys('feature',{region,semanticMasks}),
      semanticBody:keys('body',{region,bodyPart:'waist',subjectMask:{shape:'raster',width:1,height:1,data:[255]}}),
      manualSkin:keys('skin',{region}), clothing:keys('clothing',{region}),
      manualOutline:keys('face',{region,shape:'polygon',points:semanticMasks.face.points}),
    };
  });
  expect(capabilities.unlocated).toEqual([[],[],[],[],[]]);
  expect(capabilities.rectangleFace).toEqual([]); expect(capabilities.manualBody).toEqual([]);
  expect(capabilities.semanticFace).toContain('faceWidth'); expect(capabilities.manualOutline).toContain('faceWidth');
  expect(capabilities.semanticFace.filter(key=>['faceHeight','midface','doubleChin'].includes(key))).toEqual([]);
  expect(capabilities.semanticFeature.filter(key=>['eyeBag','tearTrough','noseHeight','noseLipAngle'].includes(key))).toEqual([]);
  expect(capabilities.semanticBody).toEqual(['waist','belly']);
  expect(capabilities.manualSkin).toEqual(['smooth','texture','whiten','warmth','redness']);
  expect(capabilities.clothing).toEqual(['smooth','texture','hue','saturation','brightness']);
  await page.getByRole('button', { name: '脸型', exact: true }).click();
  await expect(page.getByRole('spinbutton',{name:'脸宽 / 瘦脸数值',exact:true})).toHaveCount(0);
  await page.getByRole('button',{name:'为选区新增调整层',exact:true}).click();
  await expect(page.locator('.statusbar')).toContainText('请先检测人脸');
  expect((await savedGraph(page)).operations).toHaveLength(0);
});

test('像素轮廓保留凹边界并选择最大闭合分量', async ({ page }) => {
  await page.goto('/');
  const result=await page.evaluate(async()=>{
    const {alphaOutline}=await import('/src/core/parsing.ts');
    const mask={shape:'raster',width:40,height:40,data:Array.from({length:1600},(_,i)=>{const x=i%40,y=Math.floor(i/40);return x>=8&&x<30&&y>=8&&y<30&&!(x>=20&&y>=20)||x>=34&&x<37&&y>=34&&y<37?255:0;})};
    const points=alphaOutline(mask);
    const inside=(x:number,y:number)=>{let value=false;for(let i=0,j=points.length-1;i<points.length;j=i++){const a=points[i],b=points[j];if((a.y>y)!==(b.y>y)&&x<(b.x-a.x)*(y-a.y)/(b.y-a.y)+a.x)value=!value;}return value;};
    return {length:points.length,filled:inside(.3,.6),notch:inside(.65,.65),otherComponent:inside(.875,.875),bounds:[Math.min(...points.map(p=>p.x)),Math.max(...points.map(p=>p.x)),Math.min(...points.map(p=>p.y)),Math.max(...points.map(p=>p.y))]};
  });
  expect(result.length).toBeGreaterThanOrEqual(6); expect(result.length).toBeLessThanOrEqual(256);
  expect(result.filled).toBe(true);expect(result.notch).toBe(false);expect(result.otherComponent).toBe(false);
  expect(result.bounds).toEqual([.2,.75,.2,.75]);
});

test('身体选区法线按真实像素保持垂直，不受照片长宽比影响', async ({ page }) => {
  await page.goto('/');
  const errors=await page.evaluate(async()=>{
    const {bodyParts}=await import('/src/core/landmarks.ts');
    const points=Array.from({length:33},()=>({x:.5,y:.5,visibility:.99}));
    points[11]={x:.3,y:.3,visibility:.99};points[13]={x:.5,y:.42,visibility:.99};
    return [[200,1000],[1000,200],[500,500]].map(([width,height])=>{
      const part=bodyParts(points,width,height).find(part=>part.id==='left-upper-arm');
      const a=part.landmarks[0],b=part.landmarks[1],p=part.points[0],q=part.points[3];
      const dx=(b.x-a.x)*width,dy=(b.y-a.y)*height,nx=(p.x-q.x)*width,ny=(p.y-q.y)*height;
      return Math.abs(dx*nx+dy*ny)/(Math.hypot(dx,dy)*Math.hypot(nx,ny));
    });
  });
  for(const error of errors)expect(error).toBeLessThan(1e-10);
});

test('裙下预测大腿和交叠腿不开放独立重塑，分离可见腿可定位', async ({ page }) => {
  await page.goto('/');
  const result=await page.evaluate(async()=>{
    const {measuredLimbSupport}=await import('/src/core/landmarks.ts');
    const part=(id:string,x:number)=>({id,label:id,region:{x:x-.1,y:.2,width:.2,height:.5},points:[],landmarks:[{x,y:.2},{x,y:.7}]});
    const left=part('left-thigh',.35),right=part('right-thigh',.65);
    const raster=(separate:boolean)=>({shape:'raster',width:100,height:100,data:Array.from({length:10_000},(_,i)=>{const x=i%100,y=Math.floor(i/100);return y>=20&&y<=72&&(separate?x>=29&&x<=41||x>=59&&x<=71:x>=29&&x<=71)?255:0;})});
    const calf=part('left-calf',.5);
    const skirtCalf={shape:'raster',width:100,height:100,data:Array.from({length:10_000},(_,i)=>{const x=i%100,y=Math.floor(i/100);return y>=20&&y<=72&&(y<48?x>=29&&x<=71:x>=46&&x<=54)?255:0;})};
    return {merged:!!measuredLimbSupport(left,raster(false),100,100,right),separated:!!measuredLimbSupport(left,raster(true),100,100,right),skirtCalf:!!measuredLimbSupport(calf,skirtCalf,100,100)};
  });
  expect(result).toEqual({merged:false,separated:true,skirtCalf:false});
});

test('手工区域、液化、仿制与保护蒙版改变目标像素', async ({ page }) => {
  await importFixture(page); const original = await pixels(page);
  await page.getByRole('button', { name: '脸型', exact: true }).click(); await page.getByRole('button', { name: '套索', exact: true }).click();
  const outlineBox = (await page.locator('.canvas-overlay').boundingBox())!;
  await page.mouse.move(outlineBox.x+outlineBox.width*.25,outlineBox.y+outlineBox.height*.25); await page.mouse.down();
  for (const [x,y] of [[.75,.25],[.8,.6],[.65,.8],[.35,.8],[.2,.6],[.25,.25]]) await page.mouse.move(outlineBox.x+outlineBox.width*x,outlineBox.y+outlineBox.height*y,{steps:3});
  await page.mouse.up(); await ready(page);
  await page.getByRole('spinbutton', { name: '脸宽 / 瘦脸数值', exact: true }).fill('60');
  await expect.poll(() => pixels(page)).not.toBe(original);
  await page.getByRole('button', { name: '清空操作', exact: true }).click(); await expect.poll(() => pixels(page)).toBe(original);
  await page.getByRole('button', { name: '液化', exact: true }).click(); await drag(page, [.35, .45], [.60, .52]);
  await expect.poll(() => pixels(page)).not.toBe(original); const liquid = await pixels(page);
  await page.getByRole('button', { name: '蒙版与冻结', exact: true }).click();
  await page.getByRole('button', { name: '冻结保护', exact: true }).click();
  await page.getByRole('spinbutton', { name: '笔刷大小 %数值', exact: true }).fill('25');
  await page.getByRole('spinbutton', { name: '笔刷强度数值', exact: true }).fill('100'); await drag(page, [.35, .45], [.60, .52]);
  await expect.poll(() => pixels(page)).not.toBe(liquid);
  await page.getByRole('button', { name: '清空操作', exact: true }).click(); await ready(page);
  await page.getByRole('button', { name: '修复与仿制', exact: true }).click(); await page.getByRole('button', { name: '仿制图章', exact: true }).click();
  const box = (await page.locator('.canvas-overlay').boundingBox())!;
  await page.keyboard.down('Alt'); await page.mouse.click(box.x + box.width * .2, box.y + box.height * .2); await page.keyboard.up('Alt');
  await drag(page, [.6, .6], [.7, .7]); await expect.poll(() => pixels(page)).not.toBe(original);
});

test('同一 EditGraph 的 WASM 与 native PNG 输出逐像素一致', async ({ page, request }) => {
  await page.goto('/'); const buffer = await fixture(page);
  const created = await request.post('http://127.0.0.1:8080/api/v1/sessions'); expect(created.ok()).toBeTruthy(); const session = await created.json();
  const uploaded = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/assets`, { multipart: { file: { name: 'grid.png', mimeType: 'image/png', buffer } } });
  expect(uploaded.ok()).toBeTruthy(); const asset = await uploaded.json();
  const graph = { version: '1.0', sourceAssetId: asset.id, operations: [
    { id: 'face', type: 'face.adjust', version: '1.0', enabled: true, params: { region: { x: .15, y: .1, width: .7, height: .8 }, adjustments: { faceWidth: 30 }, mask:{shape:'polygon',points:[{x:.3,y:.2},{x:.7,y:.2},{x:.78,y:.5},{x:.6,y:.82},{x:.4,y:.82},{x:.22,y:.5}],feather:0}, landmarks: [{ x: .25, y: .5 }, { x: .5, y: .5 }, { x: .75, y: .5 }] }, controlPoints: [] },
    { id: 'skin', type: 'skin.retouch', version: '1.0', enabled: true, params: { whiten: 20, warmth: 10, semanticMasks: { skin: { shape: 'polygon', points: [{ x: .2, y: .1 }, { x: .8, y: .1 }, { x: .8, y: .8 }, { x: .2, y: .8 }], feather: 2 }, leftEye: { shape: 'ellipse', region: { x: .3, y: .3, width: .16, height: .08 } } } } },
    { id: 'body', type: 'body.adjust', version: '1.0', enabled: true, params: { bodyPart: 'waist', region: { x: .2, y: .2, width: .6, height: .6 }, adjustments: { waist: 20 }, subjectMask: { shape: 'raster', width: 8, height: 8, data: Array.from({ length: 64 }, (_,i) => i%8>=2&&i%8<=5&&Math.floor(i/8)>=1&&Math.floor(i/8)<=6 ? 255 : 0) }, landmarks: [{ x: .5, y: .2 }, { x: .5, y: .5 }, { x: .5, y: .8 }] } },
    { id: 'color', type: 'color.basic', version: '1.0', enabled: true, params: { brightness: 15, temperature: 20 } },
    { id: 'film', type: 'filter.film', version: '1.0', enabled: true, params: { preset: 'classic', intensity: 45, grain: 10, seed: 42 } },
    { id: 'crop', type: 'crop', version: '1.0', enabled: true, params: { x: 8, y: 8, width: 112, height: 112 } },
  ] };
  const rendered = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/render`, { data: { editGraph: graph, format: 'png' } });
  expect(rendered.status(), await rendered.text()).toBe(200);
  const result = await page.evaluate(async ({ src, native, graph }) => {
    const { decodeImage, renderImage } = await import('/src/core/engine.ts');
    const file = (b64: string) => new File([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], 'grid.png', { type: 'image/png' });
    const source = await decodeImage(file(src)); const expected = await decodeImage(file(native)); const actual = await renderImage(source, graph);
    return { width: actual.width, height: actual.height, expectedWidth: expected.width, expectedHeight: expected.height, mismatches: actual.pixels.reduce((sum: number, value: number, i: number) => sum + Number(value !== expected.pixels[i]), 0) };
  }, { src: buffer.toString('base64'), native: (await rendered.body()).toString('base64'), graph });
  expect(result).toEqual({ width: 112, height: 112, expectedWidth: 112, expectedHeight: 112, mismatches: 0 });
  await request.delete(`http://127.0.0.1:8080/api/v1/sessions/${session.id}`);
});

test('所有公开滑杆在有效定位输入下 WASM 与 native 输出逐像素一致', async ({ page, request }) => {
  test.setTimeout(120_000); await page.goto('/'); const buffer = await fixture(page);
  const created = await request.post('http://127.0.0.1:8080/api/v1/sessions'); expect(created.ok()).toBeTruthy(); const session = await created.json();
  const uploaded = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/assets`, { multipart: { file: { name: 'grid.png', mimeType: 'image/png', buffer } } });
  expect(uploaded.ok()).toBeTruthy(); const asset = await uploaded.json();
  const cases = await page.evaluate(async () => {
    const { TOOLS, portraitParametersFor } = await import('/src/core/registry.ts');
    const region={x:.08,y:.08,width:.84,height:.84};
    const polygon=(x:number,y:number,width:number,height:number)=>({shape:'polygon',points:[{x,y},{x:x+width,y},{x:x+width,y:y+height},{x,y:y+height}]});
    const face={shape:'polygon',points:Array.from({length:36},(_,i)=>({x:.5+Math.sin(i/36*Math.PI*2)*.3,y:.48-Math.cos(i/36*Math.PI*2)*.35}))};
    const semantics={face,skin:face,leftEye:polygon(.29,.33,.13,.07),rightEye:polygon(.58,.34,.13,.07),leftBrow:polygon(.27,.25,.16,.04),rightBrow:polygon(.57,.26,.16,.04),lips:polygon(.4,.65,.2,.08),teeth:polygon(.43,.68,.14,.025),nose:polygon(.44,.4,.12,.2),noseBridge:polygon(.46,.41,.08,.1),noseWings:polygon(.42,.53,.16,.06),noseTip:polygon(.46,.52,.08,.08),noseRoot:polygon(.47,.39,.06,.04),noseBase:polygon(.44,.57,.12,.03),leftNostril:polygon(.43,.56,.045,.025),rightNostril:polygon(.525,.56,.045,.025)};
    const pose=Array.from({length:33},()=>({x:.5,y:.5}));
    Object.entries({11:[.32,.24],12:[.68,.24],13:[.24,.39],14:[.76,.39],15:[.2,.55],16:[.8,.55],23:[.39,.55],24:[.61,.55],25:[.4,.72],26:[.6,.72],27:[.4,.88],28:[.6,.88]}).forEach(([index,[x,y]])=>pose[Number(index)]={x,y});
    const subjectMask={shape:'raster',width:16,height:16,data:Array.from({length:256},(_,i)=>i%16>=3&&i%16<=12&&Math.floor(i/16)>=3&&Math.floor(i/16)<=14?255:0)};
    return TOOLS.filter((tool:any)=>tool.type&&tool.parameters.length).flatMap((tool:any)=>portraitParametersFor(tool.id,{region,semanticMasks:semantics,bodyPart:'full-body',subjectMask}).map((parameter:any)=>{
      const value=parameter.key==='gamma'?2:parameter.default===parameter.max?parameter.min+(parameter.max-parameter.min)*.6:parameter.default===70?35:Math.min(parameter.max,60);
      const params:any={};
      if(tool.region)params.region=region;
      if(['face','feature','body','clothing'].includes(tool.id))params.adjustments={[parameter.key]:value};else params[parameter.key]=value;
      if(['face','feature','skin'].includes(tool.id)){params.semanticMasks=semantics;params.landmarks=face.points;}
      if(tool.id==='body')Object.assign(params,{bodyPart:'full-body',landmarks:pose,subjectMask,mask:polygon(.08,.08,.84,.84)});
      if(tool.id==='film')Object.assign(params,{preset:'classic',seed:42});
      return {key:tool.type+'.'+parameter.key,operation:{id:'check',type:tool.type,version:'1.0',enabled:true,params,controlPoints:[]}};
    }));
  });
  const evidence:Array<{key:string;mismatches:number}>=[];
  try {
    for (const item of cases) {
      const graph={version:'1.0',sourceAssetId:asset.id,operations:[item.operation]};
      const rendered=await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/render`,{data:{editGraph:graph,format:'png'}});
      expect(rendered.status(),`${item.key}: ${await rendered.text()}`).toBe(200);
      const mismatches=await page.evaluate(async({source,native,graph})=>{
        const {decodeImage,renderImage}=await import('/src/core/engine.ts');
        const file=(b64:string)=>new File([Uint8Array.from(atob(b64),c=>c.charCodeAt(0))],'grid.png',{type:'image/png'});
        const input=await decodeImage(file(source));const expected=await decodeImage(file(native));const actual=await renderImage(input,graph,{kind:'export'});
        return actual.pixels.reduce((sum:number,value:number,i:number)=>sum+Number(value!==expected.pixels[i]),0);
      },{source:buffer.toString('base64'),native:(await rendered.body()).toString('base64'),graph});
      evidence.push({key:item.key,mismatches}); expect(mismatches,item.key).toBe(0);
    }
    expect(evidence.length).toBeGreaterThan(60);
    await fs.writeFile(test.info().outputPath('public-parameter-parity.json'),JSON.stringify(evidence,null,2));
  } finally { await request.delete(`http://127.0.0.1:8080/api/v1/sessions/${session.id}`); }
});

test('真实人像本地 MediaPipe 检测以及参数编辑截图', async ({ page }) => {
  await page.goto('/');
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(path.join(import.meta.dirname, 'fixtures/portrait.jpg'));
  await ready(page); await page.getByRole('button', { name: '脸型', exact: true }).click();
  await page.getByRole('button', { name: '自动检测区域', exact: true }).click();
  await expect(page.locator('.statusbar')).toContainText(/检测到 \d+ 个语义区域/, { timeout: 45_000 });
  await expect(page.locator('.detected-list button')).not.toHaveCount(0);
  await page.getByRole('spinbutton', { name: '脸宽 / 瘦脸数值', exact: true }).fill('18'); await ready(page,60_000);
  const graph = await savedGraph(page);
  const face = graph.operations.at(-1) as any;
  expect(face.type).toBe('face.adjust');
  expect(face.params.mask.shape).toBe('polygon');
  expect(face.params.mask.points.length).toBeGreaterThanOrEqual(3);
  expect(face.params.mask.feather).toBe(8);
  expect(face.params.landmarks.length).toBe(36);
  expect(face.params.semanticMasks.face.shape).toBe('raster');
  expect(face.params.semanticMasks.faceOutline.points.length).toBeGreaterThanOrEqual(3);
  await fs.mkdir(path.resolve(import.meta.dirname, '../../../doc/qa'), { recursive: true });
  await page.screenshot({ path: path.resolve(import.meta.dirname, '../../../doc/qa/phase1-editor.png'), fullPage: true });
  const buffer = await fs.readFile(path.join(import.meta.dirname, 'fixtures/portrait.jpg'));
  const comparison = await page.evaluate(async ({ encoded, graph, region }) => {
    const { decodeImage, renderImage, pixelsToCanvas } = await import('/src/core/engine.ts');
    const source = await decodeImage(new File([Uint8Array.from(atob(encoded),c=>c.charCodeAt(0))],'portrait.jpg',{type:'image/jpeg'}));
    const output = await renderImage(source,graph,{kind:'export'});
    const x=Math.max(0,Math.floor((region.x-.05)*source.width)); const y=Math.max(0,Math.floor((region.y-.05)*source.height));
    const w=Math.min(source.width-x,Math.ceil((region.width+.1)*source.width)); const h=Math.min(source.height-y,Math.ceil((region.height+.1)*source.height));
    const canvas=document.createElement('canvas'); canvas.width=w*4; canvas.height=h*2+40;
    const ctx=canvas.getContext('2d')!; ctx.fillStyle='#191c22'; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#fff'; ctx.font='20px sans-serif';
    ctx.fillText('Original',16,28); ctx.fillText('Face width: 18',w*2+16,28);
    ctx.drawImage(pixelsToCanvas(source),x,y,w,h,0,40,w*2,h*2); ctx.drawImage(pixelsToCanvas(output),x,y,w,h,w*2,40,w*2,h*2);
    return canvas.toDataURL('image/png').split(',')[1];
  },{encoded:buffer.toString('base64'),graph,region:face.params.region});
  await fs.writeFile(path.resolve(import.meta.dirname,'../../../doc/qa/portrait-face-comparison.png'),Buffer.from(comparison,'base64'));
});

test('真实人像 Pose 语义部位使用独立 polygon mask', async ({ page }) => {
  await page.goto('/');
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(path.join(import.meta.dirname, 'fixtures/portrait.jpg'));
  await ready(page); await page.getByRole('button', { name: '身形', exact: true }).click();
  await page.getByRole('button', { name: '自动检测区域', exact: true }).click();
  await expect(page.locator('.statusbar')).toContainText(/检测到 \d+ 个语义区域/, { timeout: 45_000 });
  await expect(page.getByRole('button', { name: /腰部/ })).toBeVisible();
  await page.getByRole('button', { name: /腰部/ }).click();
  await page.getByRole('button', { name: '为选区新增调整层', exact: true }).click();
  await page.getByRole('spinbutton', { name: '腰围数值', exact: true }).fill('28'); await ready(page,60_000);
  const graph = await savedGraph(page);
  const body = graph.operations.at(-1) as any;
  expect(body.type).toBe('body.adjust');
  expect(body.params.bodyPart).toBe('waist');
  expect(body.params.mask.shape).toBe('polygon');
  expect(body.params.mask.points.length).toBeGreaterThanOrEqual(3);
  expect(body.params.landmarks.length).toBeGreaterThanOrEqual(2);
  expect(body.params.subjectMask.shape).toBe('raster');
  expect(body.params.subjectMask.data.length).toBe(body.params.subjectMask.width*body.params.subjectMask.height);
  expect(body.params.subjectMask.data.some((v: number) => v === 0)).toBe(true);
  expect(body.params.subjectMask.data.some((v: number) => v === 255)).toBe(true);
  const buffer = await fs.readFile(path.join(import.meta.dirname, 'fixtures/portrait.jpg'));
  const evidence = await page.evaluate(async ({ encoded, graph }) => {
    const { decodeImage, renderImage, pixelsToCanvas } = await import('/src/core/engine.ts');
    const source = await decodeImage(new File([Uint8Array.from(atob(encoded), c=>c.charCodeAt(0))], 'portrait.jpg', { type: 'image/jpeg' }));
    const output = await renderImage(source, graph, { kind: 'export' });
    const montage = document.createElement('canvas'); montage.width=source.width*2; montage.height=source.height+40;
    const ctx=montage.getContext('2d')!; ctx.fillStyle='#191c22'; ctx.fillRect(0,0,montage.width,montage.height);
    ctx.fillStyle='#fff'; ctx.font='20px sans-serif'; ctx.fillText('Original',16,28); ctx.fillText('Body: waist 28 / segmented foreground',source.width+16,28);
    ctx.drawImage(pixelsToCanvas(source),0,40); ctx.drawImage(pixelsToCanvas(output),source.width,40);
    const protectedCenters=(graph.operations.at(-1) as any).params.protectMask.strokes.every((point: any)=>{
      const x=Math.min(source.width-1,Math.floor(point.x*source.width)); const y=Math.min(source.height-1,Math.floor(point.y*source.height)); const i=(y*source.width+x)*4;
      return source.pixels.slice(i,i+4).every((v,c)=>v===output.pixels[i+c]);
    });
    return { changed: output.pixels.reduce((sum,v,i)=>sum+Number(v!==source.pixels[i]),0), protectedCenters, cornerUnchanged: source.pixels.slice(0,4).every((v,i)=>v===output.pixels[i]), png: montage.toDataURL('image/png').split(',')[1] };
  }, { encoded: buffer.toString('base64'), graph });
  expect(evidence.changed).toBeGreaterThan(100); expect(evidence.cornerUnchanged).toBe(true); expect(evidence.protectedCenters).toBe(true);
  await fs.writeFile(path.resolve(import.meta.dirname,'../../../doc/qa/portrait-body-comparison.png'),Buffer.from(evidence.png,'base64'));
});

test('真实人像肤色与妆容保存语义蒙版并保护五官', async ({ page }) => {
  await page.route(/https?:\/\/(localhost|127\.0\.0\.1):8080\//, route => route.abort());
  await page.goto('/');
  const buffer = await fs.readFile(path.join(import.meta.dirname, 'fixtures/portrait.jpg'));
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles({ name: 'portrait.jpg', mimeType: 'image/jpeg', buffer });
  await ready(page);
  await page.getByRole('button', { name: '皮肤与质感', exact: true }).click();
  await page.getByRole('button', { name: '自动检测区域', exact: true }).click();
  await expect(page.locator('.statusbar')).toContainText(/检测到 \d+ 个语义区域/, { timeout: 45_000 });
  await page.getByRole('spinbutton', { name: '美白数值', exact: true }).fill('30');
  await page.getByRole('spinbutton', { name: '肤色冷暖数值', exact: true }).fill('20');
  await ready(page,60_000);
  const skinGraph = await savedGraph(page);
  const skin: any = skinGraph.operations.at(-1);
  expect(skin.type).toBe('skin.retouch');
  expect(skin.params.semanticMasks.skin.shape).toBe('raster');
  expect(skin.params.semanticMasks.lips.points.length).toBeGreaterThan(3);
  expect(skin.params.semanticMasks.leftEye.points.length).toBeGreaterThan(3);
  const evidence = await page.evaluate(async ({ encoded, graph }) => {
    const { decodeImage, renderImage, pixelsToCanvas } = await import('/src/core/engine.ts');
    const source = await decodeImage(new File([Uint8Array.from(atob(encoded), c => c.charCodeAt(0))], 'portrait.jpg', { type: 'image/jpeg' }));
    const output = await renderImage(source, graph, { kind: 'export' });
    const masks = graph.operations.at(-1).params.semanticMasks;
    const unchanged = (points: Array<{x: number; y: number}>) => {
      const x = Math.floor(points.reduce((sum, point) => sum + point.x, 0) / points.length * source.width);
      const y = Math.floor(points.reduce((sum, point) => sum + point.y, 0) / points.length * source.height);
      const i = (y * source.width + x) * 4;
      return Array.from(source.pixels.slice(i, i + 4)).every((v, c) => v === output.pixels[i + c]);
    };
    const montage = document.createElement('canvas'); montage.width = source.width * 2; montage.height = source.height + 40;
    const ctx = montage.getContext('2d')!; ctx.fillStyle = '#191c22'; ctx.fillRect(0, 0, montage.width, montage.height);
    ctx.fillStyle = '#fff'; ctx.font = '20px sans-serif'; ctx.fillText('Original', 16, 28); ctx.fillText('Skin: whiten 30 / warmth 20', source.width + 16, 28);
    ctx.drawImage(pixelsToCanvas(source), 0, 40); ctx.drawImage(pixelsToCanvas(output), source.width, 40);
    return { changed: output.pixels.reduce((sum, v, i) => sum + Number(v !== source.pixels[i]), 0), eyesProtected: unchanged(masks.leftEye.points) && unchanged(masks.rightEye.points), cornerProtected: Array.from(source.pixels.slice(0, 4)).every((v, i) => v === output.pixels[i]), png: montage.toDataURL('image/png').split(',')[1] };
  }, { encoded: buffer.toString('base64'), graph: skinGraph });
  expect(evidence.changed).toBeGreaterThan(100); expect(evidence.eyesProtected).toBe(true); expect(evidence.cornerProtected).toBe(true);
  await fs.mkdir(path.resolve(import.meta.dirname, '../../../doc/qa'), { recursive: true });
  await fs.writeFile(path.resolve(import.meta.dirname, '../../../doc/qa/portrait-skin-comparison.png'), Buffer.from(evidence.png, 'base64'));
  // Changing tools within the face family retains feature polygons; adding a
  // manual makeup paint layer must not leak semantic params into paint.brush.
  await page.getByRole('button', { name: '五官与妆容', exact: true }).click();
  await page.getByRole('spinbutton', { name: '唇色数值', exact: true }).fill('35'); await ready(page,60_000);
  const featureGraph = await savedGraph(page);
  expect((featureGraph.operations.at(-1) as any).params.semanticMasks.lips.shape).toBe('polygon');
  await page.getByRole('button', { name: '新增绘色图层', exact: true }).click(); await ready(page);
  const paint = (await savedGraph(page)).operations.at(-1) as any;
  expect(paint.type).toBe('paint.brush'); expect(paint.params.semanticMasks).toBeUndefined();
});


test('裁剪后检测采用当前图层输入且不累积本层变形', async ({ page }) => {
  await page.goto('/');
  const buffer = await fs.readFile(path.join(import.meta.dirname,'fixtures/portrait.jpg'));
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles({ name:'portrait.jpg', mimeType:'image/jpeg',buffer });
  await ready(page); await page.getByRole('button',{name:'裁剪与几何',exact:true}).click();
  await page.getByRole('spinbutton',{name:'裁剪 X',exact:true}).fill('100');
  await page.getByRole('spinbutton',{name:'裁剪 Y',exact:true}).fill('40');
  await page.getByRole('spinbutton',{name:'裁剪宽',exact:true}).fill('600');
  await page.getByRole('spinbutton',{name:'裁剪高',exact:true}).fill('850');
  await page.getByRole('button',{name:'应用裁剪',exact:true}).click(); await ready(page);
  await page.getByRole('button',{name:'皮肤与质感',exact:true}).click();
  await page.getByRole('button',{name:'自动检测区域',exact:true}).click();
  await expect(page.locator('.statusbar')).toContainText(/检测到 \d+ 个语义区域/,{timeout:45_000});
  await page.getByRole('spinbutton',{name:'美白数值',exact:true}).fill('20'); await ready(page,60_000);
  const graph=await savedGraph(page);
  const expected = await page.evaluate(async ({ encoded,graph })=>{
    const {decodeImage,renderImage,pixelsToCanvas}=await import('/src/core/engine.ts');
    const {detectPeople}=await import('/src/core/landmarks.ts');
    const source=await decodeImage(new File([Uint8Array.from(atob(encoded),c=>c.charCodeAt(0))],'portrait.jpg',{type:'image/jpeg'}));
    const input=await renderImage(source,{...graph,operations:graph.operations.slice(0,-1)},{kind:'export'});
    return (await detectPeople(pixelsToCanvas(input),'face'))[0].semanticMasks.lips;
  },{encoded:buffer.toString('base64'),graph});
  expect((graph.operations.at(-1) as any).params.semanticMasks.lips).toEqual(expected);
  await page.getByRole('button',{name:'自动检测区域',exact:true}).click();
  await expect(page.locator('.statusbar')).toContainText(/检测到 \d+ 个语义区域/,{timeout:45_000}); await ready(page,60_000);
  expect(((await savedGraph(page)).operations.at(-1) as any).params.semanticMasks.lips).toEqual(expected);
});

test('网格、四角透视、裁剪、尺寸与JPEG质量导出', async ({ page }) => {
  await importFixture(page); const original = await pixels(page);
  await page.getByRole('button', { name: '网格与透视', exact: true }).click();
  await page.getByRole('button', { name: '建立 3×3 网格', exact: true }).click(); await ready(page);
  await drag(page, [.5, .5], [.66, .58]); await expect.poll(() => pixels(page)).not.toBe(original);
  await page.getByRole('button', { name: '清空操作', exact: true }).click(); await ready(page);
  await page.getByRole('button', { name: '四角透视', exact: true }).click(); await ready(page);
  await drag(page, [.005, .005], [.15, .12]); await expect.poll(() => pixels(page)).not.toBe(original);
  await page.getByRole('button', { name: '清空操作', exact: true }).click(); await ready(page);
  await page.getByRole('button', { name: '裁剪与几何', exact: true }).click();
  await page.getByRole('spinbutton', { name: '裁剪宽', exact: true }).fill('96');
  await page.getByRole('spinbutton', { name: '裁剪高', exact: true }).fill('80');
  await page.getByRole('button', { name: '应用裁剪', exact: true }).click(); await ready(page);
  await expect.poll(() => page.locator('canvas[aria-label="图片预览"]').evaluate((c: HTMLCanvasElement) => [c.width, c.height])).toEqual([96, 80]);
  await page.getByRole('spinbutton', { name: '输出宽', exact: true }).fill('48');
  await page.getByRole('button', { name: '应用尺寸', exact: true }).click(); await ready(page);
  await expect.poll(() => page.locator('canvas[aria-label="图片预览"]').evaluate((c: HTMLCanvasElement) => [c.width, c.height])).toEqual([48, 40]);
  await page.getByRole('combobox', { name: '导出格式', exact: true }).selectOption('jpeg');
  await page.getByRole('spinbutton', { name: 'JPEG 质量', exact: true }).fill('85');
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '导出 JPEG', exact: true }).click()]);
  expect(download.suggestedFilename()).toMatch(/\.jpg$/);
  const bytes = await fs.readFile((await download.path())!); expect([...bytes.subarray(0, 2)]).toEqual([255, 216]);
});

test('曲线、HSL、胶片缩略图、自定义预设、LUT与图层顺序', async ({ page }) => {
  await importFixture(page); const original = await pixels(page);
  await page.getByRole('button', { name: '色调曲线', exact: true }).click();
  await page.getByRole('spinbutton', { name: '输出 2', exact: true }).fill('80');
  await expect.poll(() => pixels(page)).not.toBe(original);
  await page.getByRole('button', { name: 'HSL 色彩', exact: true }).click();
  await page.getByRole('spinbutton', { name: '饱和度数值', exact: true }).fill('-80'); await ready(page);
  await page.getByRole('button', { name: '胶片模拟', exact: true }).click();
  await page.locator('.film-presets button').filter({ hasText: '暖光人像' }).click(); await ready(page);
  await expect(page.locator('.preset-thumbnail')).toHaveCount(8);
  page.once('dialog', dialog => dialog.accept('验收预设'));
  await page.getByRole('button', { name: '保存预设', exact: true }).click();
  await expect(page.getByRole('button', { name: '验收预设', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '复制预设', exact: true }).click();
  await expect(page.locator('.custom-preset')).toHaveCount(2);
  const beforeOrder = await pixels(page);
  await page.getByRole('button', { name: '下移图层', exact: true }).first().click();
  await expect.poll(() => pixels(page)).not.toBe(beforeOrder);
  const cube = 'TITLE "Invert"\nLUT_3D_SIZE 2\n1 1 1\n0 1 1\n1 0 1\n0 0 1\n1 1 0\n0 1 0\n1 0 0\n0 0 0\n';
  await page.locator('input[type=file][accept=".cube"]').setInputFiles({ name: 'invert.cube', mimeType: 'text/plain', buffer: Buffer.from(cube) });
  await ready(page); const inverted = await pixels(page);
  await page.getByRole('spinbutton', { name: 'LUT 强度数值', exact: true }).fill('0');
  await expect.poll(() => pixels(page)).not.toBe(inverted);
  await page.getByRole('button', { name: '叠加胶片模拟', exact: true }).click(); await ready(page);
});

test('绘色橡皮、修复、内容填补、渐变蒙版与工程恢复', async ({ page }) => {
  await importFixture(page); const original = await pixels(page);
  await page.getByRole('button', { name: '皮肤与质感', exact: true }).click();
  await page.getByRole('button', { name: '绘色画笔', exact: true }).click(); await drag(page, [.4, .4], [.55, .55]);
  await expect.poll(() => pixels(page)).not.toBe(original); const painted = await pixels(page);
  await page.getByRole('button', { name: '绘色橡皮', exact: true }).click(); await drag(page, [.4, .4], [.55, .55]);
  await expect.poll(() => pixels(page)).not.toBe(painted);
  await page.getByRole('button', { name: '清空操作', exact: true }).click(); await ready(page);
  await page.getByRole('button', { name: '修复与仿制', exact: true }).click();
  await page.getByRole('button', { name: '修复 / 去斑', exact: true }).click(); await drag(page, [.4, .4], [.55, .55]);
  await expect.poll(() => pixels(page)).not.toBe(original);
  await page.getByRole('button', { name: '矩形', exact: true }).first().click(); await drag(page, [.3, .3], [.5, .5]);
  await page.getByRole('button', { name: '填补选区', exact: true }).click(); await ready(page);
  await page.getByRole('button', { name: '光影与调色', exact: true }).click();
  await page.getByRole('spinbutton', { name: '曝光数值', exact: true }).fill('50'); await ready(page); const full = await pixels(page);
  await page.getByRole('button', { name: '蒙版与冻结', exact: true }).click();
  await page.getByRole('button', { name: '线性渐变', exact: true }).click(); await drag(page, [.1, .1], [.9, .9]);
  await expect.poll(() => pixels(page)).not.toBe(full); const linear = await pixels(page);
  await page.getByRole('button', { name: '径向渐变', exact: true }).click(); await drag(page, [.2, .2], [.8, .8]);
  await expect.poll(() => pixels(page)).not.toBe(linear); const expected = await pixels(page);
  const [project] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '保存工程', exact: true }).click()]);
  await page.getByRole('button', { name: '清空操作', exact: true }).click(); await ready(page);
  await page.locator('input[type=file][accept=".json"]').setInputFiles((await project.path())!);
  await expect.poll(() => pixels(page)).toBe(expected);
  await page.locator('input[type=file][accept=".json"]').setInputFiles({ name: 'bad.edit.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ operations: [{ id: 'bad', type: 'color.basic', version: '1.0', enabled: true, params: { brightnes: 80 } }] })) });
  await expect(page.locator('.statusbar')).toContainText('unknown'); expect(await pixels(page)).toBe(expected);
});

test('AI提案预览、接受、拒绝、撤销与保护新的手工编辑', async ({ page, request }) => {
  const upload = page.waitForResponse(response => response.url().includes('/assets') && response.request().method() === 'POST');
  await importFixture(page); const response = await upload;
  const sessionId = new URL(response.url()).pathname.split('/')[4];
  await expect(page.locator('.statusbar')).toContainText('图片已就绪'); const original = await pixels(page);
  const operation = { id: 'ai-color', type: 'color.basic', version: '1.0', enabled: true, origin: 'ai', label: 'AI 亮度', params: { brightness: 30 }, controlPoints: [] };
  const create = async (id: string) => {
    const session = await (await request.get(`http://127.0.0.1:8080/api/v1/sessions/${sessionId}`)).json();
    const created = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${sessionId}/proposals`, { data: { baseRevision: session.revision, operations: [{ ...operation, id }], summary: '测试调亮提案' } });
    expect(created.status(), await created.text()).toBe(201); return created.json();
  };
  await create('first'); await page.locator('.ai-panel summary').click();
  await page.getByRole('button', { name: '刷新提案', exact: true }).click();
  await page.getByRole('button', { name: '预览效果', exact: true }).click();
  await expect.poll(() => pixels(page)).not.toBe(original); await expect(page.locator('.counter')).toHaveText('0');
  await page.getByRole('button', { name: '接受', exact: true }).click();
  await expect(page.locator('.counter')).toHaveText('1'); await expect(page.locator('.proposal-card')).toHaveCount(0);
  await page.getByRole('button', { name: '撤销', exact: true }).click(); await expect.poll(() => pixels(page)).toBe(original);
  const rejected = await create('second'); await page.getByRole('button', { name: '刷新提案', exact: true }).click();
  await page.getByRole('button', { name: '拒绝', exact: true }).click(); await expect(page.locator('.proposal-card')).toHaveCount(0);
  const proposals = await (await request.get(`http://127.0.0.1:8080/api/v1/sessions/${sessionId}/proposals`)).json();
  expect(proposals.proposals.find((p: any) => p.id === rejected.id).status).toBe('rejected');
  await create('third'); await page.getByRole('button', { name: '刷新提案', exact: true }).click();
  await page.getByRole('button', { name: '光影与调色', exact: true }).click();
  await page.getByRole('spinbutton', { name: '亮度数值', exact: true }).fill('-25'); await ready(page); const manual = await pixels(page);
  await page.getByRole('button', { name: '接受', exact: true }).click();
  await expect(page.locator('.statusbar')).toContainText('手工编辑已改变'); expect(await pixels(page)).toBe(manual);
});

test('Worker取消旧预览、保留导出快照并合并连续预览', async ({ page }) => {
  await page.goto('/');
  const result = await page.evaluate(async () => {
    const { renderImage, getRenderSchedulerState } = await import('/src/core/engine.ts');
    const source = { width: 256, height: 256, pixels: new Uint8Array(256 * 256 * 4).fill(100) };
    for (let i = 3; i < source.pixels.length; i += 4) source.pixels[i] = 255;
    const graph = (brightness: number) => ({ version: '1.0', sourceAssetId: 'local', operations: [{ id: 'color', type: 'color.basic', version: '1.0', enabled: true, params: { brightness } }] });
    const controller = new AbortController();
    const first = renderImage(source, graph(10), { kind: 'preview', signal: controller.signal }).then(() => 'succeeded', (error: Error) => error.name);
    const started = getRenderSchedulerState(); controller.abort();
    const cancelled = await first;
    const exportGraph = graph(15); const exported = renderImage(source, exportGraph, { kind: 'export' });
    exportGraph.operations[0].params.brightness = 80;
    const previews = Array.from({ length: 25 }, (_, i) => renderImage(source, graph(i + 1), { kind: 'preview' }).then(image => ({ status: 'ok', pixel: image.pixels[0] }), (error: Error) => ({ status: error.name, pixel: -1 })));
    const queued = getRenderSchedulerState(); const exportImage = await exported; const outcomes = await Promise.all(previews);
    const expected = await renderImage(source, graph(15), { kind: 'background' });
    return { cancelled, started, queued, ended: getRenderSchedulerState(), exportPixel: exportImage.pixels[0], expectedPixel: expected.pixels[0], outcomes };
  });
  expect(result.cancelled).toBe('AbortError'); expect(result.started.active?.kind).toBe('preview');
  expect(result.queued.active?.kind).toBe('export'); expect(result.queued.preview).not.toBeNull();
  expect(result.exportPixel).toBe(result.expectedPixel); expect(result.outcomes.filter(o => o.status === 'ok')).toHaveLength(1);
  expect(result.outcomes.at(-1)?.status).toBe('ok'); expect(result.ended.active).toBeNull(); expect(result.ended.cancelledPreviews).toBeGreaterThan(0);
});

test('WASM模块加载失败后能够重新初始化恢复', async ({ page }) => {
  await page.route('**/wasm/image_core.js', route => route.abort()); await page.goto('/');
  const error = await page.evaluate(async () => {
    const { renderImage } = await import('/src/core/engine.ts');
    try { await renderImage({ width: 1, height: 1, pixels: new Uint8Array([10, 20, 30, 255]) }, { version: '1.0', sourceAssetId: 'local', operations: [] }); return ''; }
    catch (error) { return String(error); }
  });
  expect(error).toContain('Failed'); await page.unroute('**/wasm/image_core.js');
  const recovered = await page.evaluate(async () => {
    const { renderImage } = await import('/src/core/engine.ts');
    const result = await renderImage({ width: 1, height: 1, pixels: new Uint8Array([10, 20, 30, 255]) }, { version: '1.0', sourceAssetId: 'local', operations: [] });
    return Array.from(result.pixels);
  });
  expect(recovered).toEqual([10, 20, 30, 255]);
});

async function savedGraph(page: Page) {
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '保存工程', exact: true }).click()]);
  return JSON.parse(await fs.readFile((await download.path())!, 'utf8'));
}
async function clickImage(page: Page, x: number, y: number, alt = false) {
  const box = (await page.locator('.canvas-overlay').boundingBox())!;
  if (alt) await page.keyboard.down('Alt');
  await page.mouse.click(box.x + box.width * x, box.y + box.height * y);
  if (alt) await page.keyboard.up('Alt');
  await ready(page);
}

test('双向与对称液化、重建笔刷以及选区取样修补', async ({ page }) => {
  await importFixture(page); const original = await pixels(page);
  await page.getByRole('button', { name: '液化', exact: true }).click();
  await page.getByRole('button', { name: '旋转', exact: true }).click();
  await page.getByRole('combobox', { name: '液化旋转方向', exact: true }).selectOption('-1');
  await page.getByRole('checkbox', { name: '对称液化', exact: true }).check();
  await page.getByRole('spinbutton', { name: '笔刷大小 %数值', exact: true }).fill('20');
  await page.getByRole('spinbutton', { name: '笔刷强度数值', exact: true }).fill('100');
  await clickImage(page, .3, .5); await expect.poll(() => pixels(page)).not.toBe(original);
  const mirrored = await savedGraph(page); const strokes = mirrored.operations[0].params.strokes;
  expect(strokes).toHaveLength(2); expect(strokes.map((s: any) => s.direction)).toEqual([-1, 1]);
  expect(strokes[0].x + strokes[1].x).toBeCloseTo(1); expect(strokes[0].toX + strokes[1].toX).toBeCloseTo(1);
  const reverse = await pixels(page);
  await page.getByRole('button', { name: '清空操作', exact: true }).click(); await ready(page);
  await page.getByRole('checkbox', { name: '对称液化', exact: true }).uncheck();
  await page.getByRole('combobox', { name: '液化旋转方向', exact: true }).selectOption('1');
  await clickImage(page, .3, .5); await expect.poll(() => pixels(page)).not.toBe(reverse);
  const rotated = await pixels(page);
  await page.getByRole('button', { name: '重建笔刷', exact: true }).click();
  await clickImage(page, .3, .5); await expect.poll(() => pixels(page)).not.toBe(rotated);
  const reconstructed = await savedGraph(page);
  expect(reconstructed.operations).toHaveLength(1); expect(reconstructed.operations[0].params.strokes.at(-1).mode).toBe('reconstruct');
  const before = original.split(',').map(Number), after = (await pixels(page)).split(',').map(Number);
  // Full-strength reconstruction restores the input at the stamp center.
  const center = (64 * 128 + 38) * 4;
  expect(after.slice(center, center + 4)).toEqual(before.slice(center, center + 4));
  await page.getByRole('button', { name: '清空操作', exact: true }).click(); await ready(page);
  await page.getByRole('button', { name: '修复与仿制', exact: true }).click();
  await page.getByRole('button', { name: '矩形', exact: true }).first().click();
  await drag(page, [.6, .6], [.8, .8]); await clickImage(page, .2, .2, true);
  await expect(page.getByRole('button', { name: '从取样点修补选区', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: '从取样点修补选区', exact: true }).click();
  await expect.poll(() => pixels(page)).not.toBe(original);
  const patched = await savedGraph(page); const patch = patched.operations[0];
  expect(patch.type).toBe('repair.clone'); expect(patch.params.mask.shape).toBe('rect');
  expect(patch.params.strokes.length).toBeGreaterThan(1);
  expect(patch.params.strokes.every((s: any) => s.sourceX >= 0 && s.sourceX <= 1 && s.sourceY >= 0 && s.sourceY <= 1)).toBe(true);
  const patchPixels = (await pixels(page)).split(',').map(Number);
  expect(patchPixels.slice(0, 4)).toEqual(before.slice(0, 4));
  await page.getByRole('button', { name: '撤销', exact: true }).click(); await expect.poll(() => pixels(page)).toBe(original);
});

test('多边形去除背景杂物并用冻结笔刷保护主体', async ({ page }) => {
  await importFixture(page); const original = await pixels(page);
  await page.getByRole('button', { name: '修复与仿制', exact: true }).click();
  await page.getByRole('button', { name: '多边形', exact: true }).first().click();
  const box = (await page.locator('.canvas-overlay').boundingBox())!;
  const point = (x: number, y: number) => page.mouse.click(box.x + box.width * x, box.y + box.height * y);
  point(.60, .35); point(.82, .35); point(.82, .65); await page.mouse.dblclick(box.x + box.width * .60, box.y + box.height * .65);
  await expect(page.getByRole('button', { name: '去除多余人物 / 背景杂物', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: '去除多余人物 / 背景杂物', exact: true }).click();
  await expect.poll(() => pixels(page)).not.toBe(original);
  await page.getByRole('button', { name: '保护主体（冻结画笔）', exact: true }).click();
  await drag(page, [.35, .35], [.48, .65]);
  const graph = await savedGraph(page);
  const removal = graph.operations.at(-1) as any;
  expect(removal.type).toBe('repair.inpaint');
  expect(removal.params.mask.shape).toBe('polygon');
  expect(removal.params.mask.points.length).toBeGreaterThanOrEqual(3);
  expect(removal.params.protectMask.shape).toBe('brush');
  expect(removal.params.protectMask.strokes.length).toBeGreaterThan(0);
});

test('参数双击复位与蒙版擦除反选实际生效', async ({ page }) => {
  await importFixture(page); const original = await pixels(page);
  await page.getByRole('button', { name: '光影与调色', exact: true }).click();
  const brightness = page.getByRole('spinbutton', { name: '亮度数值', exact: true });
  await brightness.fill('40'); await expect.poll(() => pixels(page)).not.toBe(original);
  await brightness.dblclick(); await expect(brightness).toHaveValue('0'); await expect.poll(() => pixels(page)).toBe(original);
  await brightness.fill('40'); await ready(page); const full = await pixels(page);
  await page.getByRole('button', { name: '蒙版与冻结', exact: true }).click();
  await page.getByRole('button', { name: '涂抹蒙版', exact: true }).click();
  await page.getByRole('spinbutton', { name: '笔刷大小 %数值', exact: true }).fill('25');
  await page.getByRole('spinbutton', { name: '笔刷强度数值', exact: true }).fill('100');
  await clickImage(page, .5, .5); await expect.poll(() => pixels(page)).not.toBe(full); const masked = await pixels(page);
  await page.getByRole('button', { name: '擦除蒙版', exact: true }).click();
  await clickImage(page, .5, .5); await expect.poll(() => pixels(page)).not.toBe(masked);
  await page.getByRole('button', { name: '反选蒙版', exact: true }).click();
  await expect.poll(() => pixels(page)).not.toBe(original);
  const graph = await savedGraph(page); expect(graph.operations[0].params.mask.invert).toBe(true);
  expect(graph.operations[0].params.mask.strokes.at(-1).erase).toBe(true);
  await expect(page.locator('.mask-overlay')).toHaveCount(1);
});

test('EXIF方向照片在浏览器、上传元数据和native导出尺寸一致', async ({ page, request }) => {
  const file = path.resolve(import.meta.dirname, '../../../services/api/internal/api/testdata/portrait-exif6.jpg');
  const buffer = await fs.readFile(file); await page.goto('/');
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(file); await ready(page);
  await expect.poll(() => page.locator('canvas[aria-label="图片预览"]').evaluate((c: HTMLCanvasElement) => [c.width, c.height])).toEqual([1024, 820]);
  const session = await (await request.post('http://127.0.0.1:8080/api/v1/sessions')).json();
  try {
    const response = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/assets`, { multipart: { file: { name: 'portrait-exif6.jpg', mimeType: 'image/jpeg', buffer } } });
    expect(response.ok()).toBeTruthy(); const asset = await response.json(); expect([asset.width, asset.height]).toEqual([1024, 820]);
    const graph = { version: '1.0', sourceAssetId: asset.id, operations: [] };
    const rendered = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/render`, { data: { editGraph: graph, format: 'png' } });
    expect(rendered.status(), await rendered.text()).toBe(200);
    const nativeSize = await page.evaluate(async (b64) => {
      const { decodeImage } = await import('/src/core/engine.ts');
      const result = await decodeImage(new File([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], 'output.png', { type: 'image/png' }));
      return [result.width, result.height];
    }, (await rendered.body()).toString('base64'));
    expect(nativeSize).toEqual([1024, 820]);
  } finally { await request.delete(`http://127.0.0.1:8080/api/v1/sessions/${session.id}`); }
});

test('窄屏与桌面视口工具、参数和图层入口可访问', async ({ page }) => {
  await importFixture(page);
  for (const width of [320, 375, 768, 1440, 1920]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.locator('.tool-list').getByRole('button', { name: '光影与调色', exact: true }).click();
    await page.getByRole('spinbutton', { name: '亮度数值', exact: true }).fill('15'); await ready(page);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    await expect(page.getByRole('button', { name: '导出 PNG', exact: true })).toBeVisible();
    if (width <= 780) await expect(page.getByRole('combobox', { name: '选择操作图层', exact: true })).toBeVisible();
    else await expect(page.locator('.operation-row').first()).toBeVisible();
  }
});

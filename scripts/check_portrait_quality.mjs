/** Real-photo diagnostics. Pixel equality checks protect localized edits;
 * detector measurements and exported comparisons still require visual review.
 * Run: node scripts/check_portrait_quality.mjs [input.jpg]
 */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const batch = process.argv.includes('--batch');
const featureBatch = process.argv.includes('--features');
const input = path.resolve(process.argv[2] && !batch ? process.argv[2] : path.join(root, 'apps/web/tests/fixtures/portrait.jpg'));
const manifest = batch || featureBatch ? JSON.parse(await fs.readFile(path.join(root,'test/images/manifest.json'),'utf8')) : [];
const samples = batch || featureBatch ? manifest.filter(item => featureBatch ? ['portrait-makeup','portrait-dark','portrait-roll'].includes(item.id) : !['clothes-on-rail','cityscape','street-crowd'].includes(item.id)).map(item=>({...item,input:path.join(root,'test/images',item.path)})) : [{id:'fixture-portrait',input,labels:[]}];
const coreSamples = ['portrait-woman','portrait-makeup','portrait-dark','portrait-roll','profile-hoodie'];
const batchEvidence = [];
const outDir = path.join(root, 'test/results',featureBatch?'feature-skin-algorithms':'portrait-algorithms');
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  await page.goto(process.env.RETOUCH_WEB_URL ?? 'http://127.0.0.1:5173/');
  for (const sample of samples) {
  const encoded = (await fs.readFile(sample.input)).toString('base64');
  let result;
  try { result = await page.evaluate(async ({ encoded, name, allParameters, maxEdge, featureBatch }) => {
    const { decodeImage, renderImage, pixelsToCanvas } = await import('/src/core/engine.ts');
    const { detectPeople } = await import('/src/core/landmarks.ts');
    const { portraitParametersFor } = await import('/src/core/registry.ts');
    let source = await decodeImage(new File([Uint8Array.from(atob(encoded), c => c.charCodeAt(0))], name, { type: 'image/jpeg' }));
    if (Math.max(source.width,source.height)>maxEdge) { const resized=document.createElement('canvas');const scale=maxEdge/Math.max(source.width,source.height);resized.width=Math.round(source.width*scale);resized.height=Math.round(source.height*scale);resized.getContext('2d').drawImage(pixelsToCanvas(source),0,0,resized.width,resized.height);source={width:resized.width,height:resized.height,pixels:new Uint8Array(resized.getContext('2d').getImageData(0,0,resized.width,resized.height).data)}; }
    const person = (await detectPeople(pixelsToCanvas(source), 'face'))[0];
    if (!person) return {source:{name,width:source.width,height:source.height},date:new Date().toISOString(),error:'No face detected at evaluation resolution',checks:[]};
    const region = person.region;
    const contour = person.semanticMasks.faceContour.points;
    const headContour = person.semanticMasks.faceOutline.points;
    const support = { shape: 'rect', region, feather: 0 };
    const cases = (featureBatch?['feature','skin']:['face']).flatMap(tool=>portraitParametersFor(tool, { region, semanticMasks: person.semanticMasks }).filter(p => p.key !== 'symmetry' && (featureBatch || allParameters || p.key === 'faceWidth')).map(parameter=>({tool,key:parameter.key,value:parameter.key==='texture'?0:60})));
    const maskValue=(mask,x,y)=>{if(x<0||y<0||x>1||y>1)return 0;const px=Math.max(0,Math.min(mask.width-1,x*mask.width-.5)),py=Math.max(0,Math.min(mask.height-1,y*mask.height-.5)),x0=Math.floor(px),y0=Math.floor(py),x1=Math.min(mask.width-1,x0+1),y1=Math.min(mask.height-1,y0+1),tx=px-x0,ty=py-y0;return (mask.data[y0*mask.width+x0]*(1-tx)*(1-ty)+mask.data[y0*mask.width+x1]*tx*(1-ty)+mask.data[y1*mask.width+x0]*(1-tx)*ty+mask.data[y1*mask.width+x1]*tx*ty)/255;};
    // Classify source pixels once against the actual alpha, independent of
    // rendering. Do not repeatedly use a landmark/hair bounding polygon.
    const sourceClasses=new Uint8Array(source.width*source.height);
    for(let y=0;y<source.height;y++)for(let x=0;x<source.width;x++){
      const nx=(x+.5)/source.width,ny=(y+.5)/source.height;
      let value=maskValue(person.semanticMasks.faceContourMask,nx,ny)>=.5?1:0;
      if(nx<region.x||nx>region.x+region.width||ny<region.y||ny>region.y+region.height)value|=4;
      else {let maxAlpha=0;for(const [ox,oy]of[[0,0],[-2,0],[2,0],[0,-2],[0,2]])maxAlpha=Math.max(maxAlpha,maskValue(person.semanticMasks.face,(x+.5+ox)/source.width,(y+.5+oy)/source.height));if(maxAlpha<.01)value|=2;}
      sourceClasses[y*source.width+x]=value;
    }
    const crop = { x: Math.max(0, Math.floor(region.x * source.width)), y: Math.max(0, Math.floor(region.y * source.height)), w: Math.ceil(region.width * source.width), h: Math.ceil(region.height * source.height) };
    const checks = [];
    for (const item of cases) {
      const adjustment=item.tool==='skin'?{[item.key]:item.value,...(item.key==='texture'?{smooth:80}:{})}:{adjustments:{[item.key]:item.value}};
      const graph = { version: '1.0', sourceAssetId: 'local', operations: [{ id: 'quality', type: item.tool==='skin'?'skin.retouch':item.tool==='feature'?'feature.adjust':'face.adjust', version: '1.0', enabled: true, params: { region, mask: support, landmarks: person.points.map(({ x, y }) => ({ x, y })), subjectMask: person.subjectMask, semanticMasks: person.semanticMasks,...adjustment }, controlPoints: [] }] };
      try {
      const output = await renderImage(source, graph, { kind: 'export' });
      const detected = featureBatch ? undefined : (await detectPeople(pixelsToCanvas(output), 'face'))[0];
      const metrics = { changedFacePixels: 0, changedOutsideSupportPixels: 0, changedBackgroundBeyondOriginalFaceTwoPixels: 0, outsideSupportPixels: 0, backgroundBeyondOriginalFaceTwoPixels: 0, protectedFeatureCenters: {}, faceWidthAtCheek: { source: (person.points[8].x - person.points[28].x) * source.width, output: detected ? (detected.points[8].x - detected.points[28].x) * source.width : null } };
      const difference = { ...source, pixels: new Uint8Array(source.pixels.length) };
      for (let y = 0; y < source.height; y++) for (let x = 0; x < source.width; x++) {
        const index = (y * source.width + x) * 4; const nx = (x + .5) / source.width, ny = (y + .5) / source.height;
        let delta = 0; for (let c = 0; c < 3; c++) delta += Math.abs(source.pixels[index + c] - output.pixels[index + c]);
        const category=sourceClasses[y*source.width+x];
        if ((category&1) && delta) metrics.changedFacePixels++;
        if (category&4) { metrics.outsideSupportPixels++; if (delta) metrics.changedOutsideSupportPixels++; }
        else if (category&2) { metrics.backgroundBeyondOriginalFaceTwoPixels++; if (delta) metrics.changedBackgroundBeyondOriginalFaceTwoPixels++; }
        difference.pixels.set([Math.min(255, delta * 8), 0, 0, 255], index);
      }
      for (const name of ['leftEye', 'rightEye', 'nose', 'lips']) {
        const mask = person.semanticMasks[name]; if (mask.shape !== 'polygon') continue; const points = mask.points;
        const x = Math.floor(points.reduce((sum, p) => sum + p.x, 0) / points.length * source.width), y = Math.floor(points.reduce((sum, p) => sum + p.y, 0) / points.length * source.height);
        const index = (y * source.width + x) * 4;
        metrics.protectedFeatureCenters[name] = Array.from(output.pixels.slice(index, index + 4)).every((channel, i) => channel === source.pixels[index + i]);
      }
      const canvas = document.createElement('canvas'); canvas.width = crop.w * 6; canvas.height = crop.h * 2 + 36;
      const ctx = canvas.getContext('2d'); ctx.fillStyle = '#161b22'; ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.fillStyle = 'white'; ctx.font = '18px sans-serif';
      for (const [i, [image, label]] of [[source, 'Original'], [output, `${item.key}: ${item.value}`], [difference, 'Difference x8']].entries()) {
        ctx.fillText(label, crop.w * i * 2 + 10, 25); ctx.drawImage(pixelsToCanvas(image), crop.x, crop.y, crop.w, crop.h, crop.w * i * 2, 36, crop.w * 2, crop.h * 2);
      }
      checks.push({ ...item, metrics, graph, png: canvas.toDataURL().split(',')[1], sourceContour: contour, headSourceContour:headContour, anatomicalSourceContour: person.points, detectedOutputContour: detected?.semanticMasks.faceContour.points });
      } catch (error) { checks.push({...item,error:String(error),graph}); }
    }
    return { source: { name, width: source.width, height: source.height }, date: new Date().toISOString(), checks };
  }, { encoded, name: path.basename(sample.input),allParameters:!batch || coreSamples.includes(sample.id),maxEdge:768,featureBatch });
  } catch(error) { result={source:{name:path.basename(sample.input)},date:new Date().toISOString(),error:String(error),checks:[]}; }
  const sampleDir=path.join(outDir,sample.id);
  await fs.mkdir(sampleDir, { recursive: true });
  for (const item of result.checks) {
    if (!item.png) continue;
    await fs.writeFile(path.join(sampleDir, `${item.tool}-${item.key}-${item.value}.png`), Buffer.from(item.png, 'base64'));
    delete item.png;
  }
  await fs.writeFile(path.join(sampleDir, 'real-face-quality.json'), JSON.stringify(result, null, 2));
  const failures = result.checks.filter(item => item.error || item.metrics.changedOutsideSupportPixels !== 0 || (featureBatch||item.key==='faceWidth')&&item.metrics.changedBackgroundBeyondOriginalFaceTwoPixels!==0 || !featureBatch&&(item.metrics.changedFacePixels===0||Object.values(item.metrics.protectedFeatureCenters).some(equal=>!equal)));
  batchEvidence.push({id:sample.id,labels:sample.labels,source:result.source,error:result.error,checks:result.checks.map(({tool,key,value,metrics,error})=>({tool,key,value,metrics,error})),failures:failures.map(item=>item.key),artifactDirectory:sampleDir});
  console.log(JSON.stringify({id:sample.id,error:result.error,cases:result.checks.length,failures:failures.map(item=>({key:item.key,error:item.error,metrics:item.metrics}))}));
  }
  await fs.mkdir(outDir,{recursive:true});
  await fs.mkdir(path.join(root,'doc/qa'),{recursive:true});
  await fs.writeFile(path.join(root,'doc/qa',featureBatch?'feature-skin-multisample-review.json':'portrait-multisample-review.json'),JSON.stringify(batchEvidence,null,2));
  const rows=batchEvidence.flatMap(sample=>sample.checks.map(item=>`| ${sample.id} | ${item.key}=${item.value} | ${sample.failures.includes(item.key)?'FAIL':'通过像素保护检查，待视觉复核'} | ${item.metrics?.faceWidthAtCheek.source.toFixed(2)??'未测得'} → ${item.metrics?.faceWidthAtCheek.output?.toFixed(2)??'未检出'} | ${item.metrics?.changedOutsideSupportPixels??'-'} | ${item.metrics?.changedBackgroundBeyondOriginalFaceTwoPixels??'-'} |`));
  const errors=batchEvidence.filter(sample=>sample.error).map(sample=>`- ${sample.id}：${sample.error}`);
  await fs.writeFile(path.join(root,'doc/qa',featureBatch?'feature-skin-multisample-review.md':'portrait-multisample-review.md'),`# 多样本人脸算法效果记录

日期：2026-10-13。最长边缩至768px。像素脸部控制轮廓与整头前景alpha分开；有序36点只作解剖定位。输出再检测的脸宽只作辅助测量，不能代替照片边缘视觉验收。

| 照片 | 参数 | 像素检查 | 颧侧宽度px（重新检测） | 支持区外变化 | 原轮廓外2px背景变化 |
|---|---|---|---|---|---|
${rows.join('\n')}

${errors.length?'检测失败记录：\n\n'+errors.join('\n'):''}

逐参数原图、输出、差分与原/输出检测轮廓保存于 ${outDir}。像素检查通过不代表自然度已达标。五官与皮肤批测中零变化参数仅记录，需要判断照片是否存在目标（如牙齿）、当前参数是否有上下文（texture需smooth>0），不能当作自然效果通过。
`);
  if(batchEvidence.some(sample=>sample.error||sample.failures.length))process.exitCode=1;
} finally { await browser.close(); }

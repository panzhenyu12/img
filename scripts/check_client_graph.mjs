/** Real ONNX inside the editor's render Worker, with Rust graph/composite parity. */
import {chromium} from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
const source = (await fs.readFile(new URL('../doc/qa/background-lama/street-source.png',import.meta.url))).toString('base64');
const fixture = JSON.parse(await fs.readFile(new URL('../doc/qa/background-lama/street-model-graph.json',import.meta.url),'utf8'));
const browser = await chromium.launch({headless:true});
try {
  const page = await browser.newPage(), requests = [], diagnostics = [];
  await page.route(/https?:\/\/[^/]+\/api\//,route => {requests.push(route.request().url());return route.abort();});
  page.on('console',msg => {if (msg.text().includes('client-inpaint')) diagnostics.push(msg.text());});
  await page.goto(process.env.RETOUCH_WEB_URL ?? 'http://127.0.0.1:5173/');
  const result = await page.evaluate(async ({source,fixture}) => {
    const {decodeImage,renderImage} = await import('/src/core/engine.ts');
    const image = await decodeImage(new File([Uint8Array.from(atob(source),c=>c.charCodeAt(0))],'street.png',{type:'image/png'}));
    const wasm = await import('/wasm/image_core.js');await wasm.default();
    const prefix = {id:'before',type:'color.basic',enabled:true,params:{brightness:5}};
    const suffix = {id:'after',type:'color.basic',enabled:true,params:{contrast:7}};
    const model = {...fixture.operations[0],params:{...fixture.operations[0].params,execution:'client',maxSide:512,opacity:100}};
    const graph = operations => ({version:'1.0',sourceAssetId:'local',operations});
    const native = (source,operations) => wasm.render_rgba(source.width,source.height,source.pixels,JSON.stringify(graph(operations)));
    const before = native(image,[prefix]), started = performance.now();
    const first = await renderImage(image,graph([prefix,model]),{kind:'export'}), firstMs = performance.now() - started;
    const partial = {...model,params:{...model.params,opacity:40}}, cacheStart = performance.now();
    const second = await renderImage(image,graph([prefix,partial,suffix]),{kind:'export'}), cachedMs = performance.now() - cacheStart;
    const mixed = {...before,pixels:wasm.composite_inpaint(before.width,before.height,before.pixels,first.pixels,JSON.stringify(partial))};
    const expected = native(mixed,[suffix]);
    const zero = await renderImage(image,graph([prefix,{...model,params:{...model.params,opacity:0}},suffix]),{kind:'export'});
    const noModel = native(image,[prefix,suffix]);
    const masks = wasm.inpaint_masks(before.width,before.height,before.pixels,JSON.stringify(model));
    let outside = 0, alpha = 0, changed = 0, replayDifference = 0, zeroDifference = 0;
    for(let i=0;i<image.width*image.height;i++) {
      let diff=false;for(let c=0;c<4;c++) {diff ||= first.pixels[i*4+c] !== before.pixels[i*4+c]; replayDifference += second.pixels[i*4+c] !== expected.pixels[i*4+c] ? 1 : 0;zeroDifference += zero.pixels[i*4+c] !== noModel.pixels[i*4+c] ? 1 : 0;}
      if(diff) {changed++;if(!masks.target[i])outside++;}
      if(first.pixels[i*4+3] !== before.pixels[i*4+3])alpha++;
    }
    return {firstMs,cachedMs,outside,alpha,changed,replayDifference,zeroDifference,isolated:crossOriginIsolated};
  },{source,fixture});
  const report = {...result,requests,diagnostics,quality:'Execution and composition verified; perceptual quality remains below full acceptance'};
  await fs.mkdir(new URL('../test/results/client-inpaint/',import.meta.url),{recursive:true});
  await fs.writeFile(new URL('../test/results/client-inpaint/graph-report.json',import.meta.url),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report));
  if (result.outside || result.alpha || result.replayDifference || result.zeroDifference || !result.changed || requests.length) throw new Error('Client graph contract failed');
} finally {await browser.close();}

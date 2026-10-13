/** Real browser inference, disconnected from API. Retains outputs and timings;
 * success here verifies execution/masks, not an 8/10 perceptual quality score. */
import {chromium} from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outputDir = path.join(root, 'test/results/client-inpaint');
await fs.mkdir(outputDir, {recursive:true});
const browser = await chromium.launch({headless:true});
const reports = [];
try {
  const page = await browser.newPage();
  const diagnostics = [];
  page.on('console', msg => {if (msg.type() === 'error' || msg.text().includes('client-inpaint')) diagnostics.push(msg.text());});
  const uploads = [];
  await page.route(/https?:\/\/[^/]+\/api\//, route => {uploads.push(route.request().url()); return route.abort();});
  await page.goto(process.env.RETOUCH_WEB_URL ?? 'http://127.0.0.1:5173/');
  for (const name of (process.argv.slice(2).length ? process.argv.slice(2) : ['street', 'crowd', 'architecture'])) {
    const files = Object.fromEntries(await Promise.all(['source','mask','protect'].map(async kind => {
      const filename = path.join(root, 'doc/qa/background-lama', `${name}-${kind}.png`);
      try {return [kind, (await fs.readFile(filename)).toString('base64')];} catch (error) {if (kind === 'protect') return [kind, null]; throw error;}
    })));
    const result = await page.evaluate(async ({files}) => {
      const decode = async encoded => {
        const bitmap = await createImageBitmap(new Blob([Uint8Array.from(atob(encoded), c => c.charCodeAt(0))], {type:'image/png'}));
        const canvas = new OffscreenCanvas(bitmap.width,bitmap.height), ctx = canvas.getContext('2d');ctx.drawImage(bitmap,0,0);bitmap.close();
        return {width:canvas.width,height:canvas.height,pixels:new Uint8Array(ctx.getImageData(0,0,canvas.width,canvas.height).data)};
      };
      const source = await decode(files.source), mask = await decode(files.mask), protect = files.protect ? await decode(files.protect) : null;
      const n = source.width * source.height, target = new Uint8Array(n), protection = new Uint8Array(n);
      for (let i = 0; i < n; i++) {protection[i] = protect?.pixels[i * 4] ?? 0; target[i] = mask.pixels[i * 4] && !protection[i] && source.pixels[i * 4 + 3] ? 255 : 0;}
      const moduleURL = new URL('/src/core/clientInpaint.ts', location.origin).href;
      const workerURL = URL.createObjectURL(new Blob([`import {clientInpaint} from ${JSON.stringify(moduleURL)}; self.onmessage = async ({data}) => {try {const result=await clientInpaint(data.source,data.target,data.protection); self.postMessage({pixels:result},[result.buffer]);} catch (e) {self.postMessage({error:String(e)});}};`], {type:'text/javascript'}));
      const worker = new Worker(workerURL, {type:'module'});
      const started = performance.now();
      const execute = () => new Promise((resolve,reject) => {worker.onmessage = ({data}) => data.error ? reject(new Error(data.error)) : resolve(data.pixels);worker.onerror = e => reject(new Error(e.message));worker.postMessage({source,target,protection});});
      try {
        const out = await execute(), coldMs = performance.now() - started;
        const warmStart = performance.now(), cached = await execute(), cacheMs = performance.now() - warmStart;
        let outside = 0, frozen = 0, alpha = 0, changed = 0, cacheMismatch = 0;
        for (let i = 0; i < n; i++) {
          let differs = false;
          for (let c = 0; c < 4; c++) {if (out[i * 4 + c] !== source.pixels[i * 4 + c]) differs = true; if (out[i * 4 + c] !== cached[i * 4 + c]) cacheMismatch++;}
          if (differs) {changed++;if (!target[i]) outside++;if (protection[i]) frozen++;}
          if (out[i * 4 + 3] !== source.pixels[i * 4 + 3]) alpha++;
        }
        const canvas = new OffscreenCanvas(source.width,source.height);canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(out),source.width,source.height),0,0);
        const png = await canvas.convertToBlob({type:'image/png'}), bytes = new Uint8Array(await png.arrayBuffer());
        let encoded = '';for(let i=0;i<bytes.length;i+=32768) encoded += String.fromCharCode(...bytes.subarray(i,i+32768));
        return {width:source.width,height:source.height,coldMs,cacheMs,outside,frozen,alpha,changed,cacheMismatch,png:btoa(encoded)};
      } finally {worker.terminate();URL.revokeObjectURL(workerURL);}
    }, {files});
    const {png,...metrics} = result;
    await fs.writeFile(path.join(outputDir,`${name}-client.png`),Buffer.from(png,'base64'));
    const report = {name,...metrics,uploads:[...uploads],diagnostics:[...diagnostics],quality:'UNVERIFIED: inspect result and source; execution is not a perceptual pass'};
    reports.push(report);console.log(JSON.stringify(report));
    await fs.writeFile(path.join(outputDir,'report.json'),JSON.stringify(reports,null,2));
    if (metrics.outside || metrics.frozen || metrics.alpha || metrics.cacheMismatch || uploads.length) throw new Error('Client inpainting contract failed');
  }
} finally {await browser.close();}

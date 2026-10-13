import {expect, test} from '@playwright/test';

test('broken model disk cache is removed and retry fetches the file again', async ({page}) => {
  let fetches = 0;
  await page.route('**/models/lama-fp32-512.onnx?*', route => {fetches++; return route.fulfill({body:Buffer.alloc(321),contentType:'application/octet-stream'});});
  await page.goto('/');
  const result = await page.evaluate(async () => {
    const url = new URL('/models/lama-fp32-512.onnx?sha=1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6',location.origin).href;
    const cache = await caches.open('retouch-inpaint-model-v1');await cache.put(url,new Response(new Uint8Array(123)));
    const {loadClientInpaintModel} = await import('/src/core/clientInpaint.ts');
    const errors: string[] = [];
    for (let i = 0; i < 2; i++) try {await loadClientInpaintModel();} catch(e) {errors.push(String(e));}
    return {errors,entries:(await cache.keys()).length};
  });
  expect(result.errors).toHaveLength(2);expect(result.errors.every(e => e.includes('不完整'))).toBeTruthy();expect(result.entries).toBe(0);expect(fetches).toBe(1);
});

test('candidate RGB packing supports a 12MP photo and survives caller buffer transfer', async ({page}) => {
  await page.goto('/');
  const result = await page.evaluate(async () => {
    const {packInpaintRgb,restoreInpaintRgb} = await import('/src/core/inpaintPreprocess.ts');
    const width = 4000, height = 3000, pixels = new Uint8Array(width * height * 4).fill(255);
    const target = new Uint8Array(width * height), protection = new Uint8Array(target.length);
    for (let y = 1000; y < 1100; y++) for (let x = 1500; x < 1600; x++) target[y * width + x] = 255;
    protection[1000 * width + 1500] = 1;
    const source = {width,height,pixels}, candidate = new Uint8Array(pixels);
    for (let i = 0; i < target.length; i++) if (target[i] && !protection[i]) candidate.set([80,120,160],i * 4);
    const packed = packInpaintRgb(source,target,protection,candidate,32 * 1024 * 1024);
    const first = restoreInpaintRgb(source,target,protection,packed);
    structuredClone(first,{transfer:[first.buffer]});
    const second = restoreInpaintRgb(source,target,protection,packed);
    let difference = 0;for (let i = 0; i < candidate.length; i++) difference += second[i] === candidate[i] ? 0 : 1;
    return {packedBytes:packed.byteLength,detached:first.byteLength,difference};
  });
  expect(result.packedBytes).toBe(9999 * 3);expect(result.detached).toBe(0);expect(result.difference).toBe(0);
});

test('browser inpaint preserves aspect, removes unknown RGB and retains known context without halos', async ({page}) => {
  await page.goto('/');
  const result = await page.evaluate(async () => {
    const {prepareClientInpaint} = await import('/src/core/inpaintPreprocess.ts');
    const width = 1024, height = 256;
    const pixels = new Uint8Array(width * height * 4), target = new Uint8Array(width * height), protection = new Uint8Array(target.length);
    for (let i = 0; i < target.length; i++) pixels.set([180,120,90,255], i * 4);
    for (let y = 60; y < 180; y++) for (let x = 400; x < 500; x++) target[y * width + x] = 255;
    protection[30 * width + 20] = 1; pixels[20 * 4 + 3] = 0;
    const a = prepareClientInpaint({width,height,pixels}, target, protection, {maxSide:512});
    const changed = new Uint8Array(pixels);
    for (let i = 0; i < target.length; i++) if (target[i] || protection[i] || !pixels[i * 4 + 3]) changed.set([0,255,0], i * 4);
    const b = prepareClientInpaint({width,height,pixels:changed}, target, protection, {maxSide:512});
    let difference = 0, knownError = 0, unknownRgb = 0, padError = 0;
    const n = 512 * 512;
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
      difference = Math.max(difference, Math.abs(a.image[c * n + i] - b.image[c * n + i]));
      if (a.mask[i]) unknownRgb = Math.max(unknownRgb, Math.abs(a.image[c * n + i]));
      else knownError = Math.max(knownError, Math.abs(a.image[c * n + i] - [180,120,90][c] / 255));
    }
    for (let x = 0; x < 512; x++) padError = Math.max(padError, Math.abs(a.image[128 * 512 + x] - a.image[127 * 512 + x]));
    return {width:a.scaledWidth,height:a.scaledHeight, difference, knownError, unknownRgb, padError};
  });
  expect(result.width).toBe(512); expect(result.height).toBe(128);
  expect(result.difference).toBe(0); expect(result.unknownRgb).toBe(0); expect(result.knownError).toBeLessThan(1e-6); expect(result.padError).toBe(0);
});

test('client restoration respects NCHW byte units, target, soft protection, alpha and crop coordinates', async ({page}) => {
  await page.goto('/');
  const result = await page.evaluate(async () => {
    const {prepareClientInpaint,restoreClientInpaint} = await import('/src/core/inpaintPreprocess.ts');
    const width = 200, height = 180, pixels = new Uint8Array(width * height * 4);
    for (let i = 0; i < width * height; i++) pixels.set([20,40,60,i % 251],i * 4);
    const target = new Uint8Array(width * height), protection = new Uint8Array(target.length);
    target[100 * width + 150] = 255; target[100 * width + 151] = 255; target[100 * width + 152] = 255;
    protection[100 * width + 151] = 1; pixels[(100 * width + 152) * 4 + 3] = 0;
    const source = {width,height,pixels};
    const prepared = prepareClientInpaint(source,target,protection,{contextMode:'crop',contextPadding:64,maskDilation:0});
    const prediction = new Float32Array(512 * 512 * 3);
    for (let c = 0; c < 3; c++) prediction.fill([135.4,210.7,35.2][c],c * 512 * 512,(c + 1) * 512 * 512);
    const out = restoreClientInpaint(source,target,protection,prepared,prediction);
    let forbidden = 0, alpha = 0;
    for (let i = 0; i < target.length; i++) {
      alpha += out[i * 4 + 3] !== pixels[i * 4 + 3] ? 1 : 0;
      if (!target[i] || protection[i] || !pixels[i * 4 + 3]) for (let c = 0; c < 4; c++) forbidden += out[i * 4 + c] !== pixels[i * 4 + c] ? 1 : 0;
    }
    return {forbidden,alpha,pixel:Array.from(out.slice((100 * width + 150) * 4,(100 * width + 150) * 4 + 3)),box:prepared.box};
  });
  expect(result.forbidden).toBe(0); expect(result.alpha).toBe(0); expect(result.pixel).toEqual([135,211,35]); expect(result.box).toEqual([86,36,200,165]);
});

test('import and ordinary manual editing make no image upload', async ({page}) => {
  const uploads: string[] = [];
  page.on('request', request => {if (request.method() === 'POST' && /\/api\//.test(request.url())) uploads.push(request.url());});
  await page.goto('/');
  await page.locator('input[type="file"]').first().setInputFiles('../../test/images/portrait-woman.jpg');
  await expect(page.getByText('本地修图模式', {exact:false})).toBeVisible();
  expect(uploads).toEqual([]);
});

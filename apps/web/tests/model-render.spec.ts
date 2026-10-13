import {expect, test, type Page} from '@playwright/test';

async function candidatePng(page: Page): Promise<Buffer> {
  const encoded = await page.evaluate(async () => {
    const canvas = new OffscreenCanvas(8, 8), context = canvas.getContext('2d')!;
    context.fillStyle = 'rgb(200,200,200)'; context.fillRect(0, 0, 8, 8);
    const bytes = new Uint8Array(await (await canvas.convertToBlob({type: 'image/png'})).arrayBuffer());
    return btoa(String.fromCharCode(...bytes));
  });
  return Buffer.from(encoded, 'base64');
}

test('slow model previews finish while opacity and suffix changes replace callers', async ({page}) => {
  await page.goto('/');
  const candidate = await candidatePng(page);
  let requests = 0, completed = 0;
  let release!: () => void;
  const pendingResponse = new Promise<void>(resolve => {release = resolve;});
  await page.route('**/api/inpaint/model', async route => {
    requests++;
    if (requests === 1) await pendingResponse;
    await route.fulfill({status: 200, contentType: 'image/png', headers: {'Access-Control-Allow-Origin': '*'}, body: candidate});
    completed++;
  });
  await page.evaluate(async () => {
    const {renderImage, getRenderSchedulerState} = await import('/src/core/engine.ts');
    const source = {width: 8, height: 8, pixels: new Uint8Array(8 * 8 * 4)};
    for (let i = 0; i < 64; i++) source.pixels.set([100,100,100,255], i * 4);
    const graph = (opacity: number, brightness = 0) => ({version: '1.0', sourceAssetId: 'local', operations: [
      {id: 'model', type: 'repair.inpaint', enabled: true, params: {method: 'lama', execution: 'server', opacity, region: {x: .25, y: .25, width: .5, height: .5}}},
      {id: 'suffix', type: 'color.basic', enabled: true, params: {brightness}},
    ]});
    const controller = new AbortController();
    const first = renderImage(source, graph(100), {signal: controller.signal}).then(() => 'ok', (error: Error) => error.name);
    (window as any).modelTest = {source, graph, renderImage, getRenderSchedulerState, controller, first};
  });
  await expect.poll(() => requests).toBe(1);
  const queued = await page.evaluate(() => {
    const state = (window as any).modelTest;
    state.controller.abort();
    state.previews = Array.from({length: 25}, (_, i) => state.renderImage(state.source, state.graph(i === 24 ? 50 : i + 1, i === 24 ? 25 : i)).then(
      (image: any) => ({status: 'ok', pixels: Array.from(image.pixels)}), (error: Error) => ({status: error.name}),
    ));
    return state.getRenderSchedulerState();
  });
  expect(queued.workerGenerations).toBe(1);
  expect(queued.active?.kind).toBe('preview'); expect(queued.preview).not.toBeNull();
  expect(requests).toBe(1); expect(completed).toBe(0);
  release();
  const result = await page.evaluate(async () => {
    const state = (window as any).modelTest;
    const first = await state.first, outcomes = await Promise.all(state.previews);
    // Independently construct the 50% target blend, then ask ordinary Rust
    // to apply only the color suffix. This also checks stale results never win.
    const expectedInput = {...state.source, pixels: new Uint8Array(state.source.pixels)};
    for (let y = 2; y < 6; y++) for (let x = 2; x < 6; x++) expectedInput.pixels.set([150,150,150,255], (y * 8 + x) * 4);
    const expected = await state.renderImage(expectedInput, {version: '1.0', sourceAssetId: 'local', operations: [state.graph(50, 25).operations[1]]}, {kind: 'background'});
    return {first, outcomes, expected: Array.from(expected.pixels), ended: state.getRenderSchedulerState()};
  });
  expect(result.first).toBe('AbortError');
  expect(result.outcomes.slice(0, -1).every((outcome: any) => outcome.status === 'AbortError')).toBe(true);
  expect(result.outcomes.at(-1)?.status).toBe('ok'); expect(result.outcomes.at(-1)?.pixels).toEqual(result.expected);
  expect(requests).toBe(2); expect(completed).toBe(2);
  expect(result.ended.workerGenerations).toBe(1); expect(result.ended.active).toBeNull();
});

test('a new photograph interrupts an old model preview instead of waiting', async ({page}) => {
  await page.goto('/');
  let modelStarted = false;
  let release!: () => void;
  const pendingResponse = new Promise<void>(resolve => {release = resolve;});
  const candidate = await candidatePng(page);
  await page.route('**/api/inpaint/model', async route => {
    modelStarted = true; await pendingResponse;
    await route.fulfill({status: 200, contentType: 'image/png', headers: {'Access-Control-Allow-Origin': '*'}, body: candidate}).catch(() => {});
  });
  try {
    await page.evaluate(async () => {
      const {renderImage} = await import('/src/core/engine.ts');
      const source = {width: 8, height: 8, pixels: new Uint8Array(8 * 8 * 4).fill(100)};
      for (let i = 3; i < source.pixels.length; i += 4) source.pixels[i] = 255;
      (window as any).oldModel = renderImage(source, {version: '1.0', sourceAssetId: 'old', operations: [{id: 'model', type: 'repair.inpaint', enabled: true, params: {method: 'lama', execution: 'server', region: {x: .25, y: .25, width: .5, height: .5}}}]}).then(() => 'ok', (error: Error) => error.name);
    });
    await expect.poll(() => modelStarted).toBe(true);
    const result = await page.evaluate(async () => {
      const {renderImage, getRenderSchedulerState} = await import('/src/core/engine.ts');
      const source = {width: 1, height: 1, pixels: new Uint8Array([35,70,105,255])};
      const image = await renderImage(source, {version: '1.0', sourceAssetId: 'new', operations: []});
      return {old: await (window as any).oldModel, pixels: Array.from(image.pixels), state: getRenderSchedulerState()};
    });
    expect(result.old).toBe('AbortError'); expect(result.pixels).toEqual([35,70,105,255]);
    expect(result.state.workerGenerations).toBe(2);
  } finally {release();}
});

test('a failed client dependency import resets its Worker and recovers on retry', async ({page}) => {
  let dependencyRequests = 0;
  await page.route('**/src/core/clientInpaint.ts*', route => route.fulfill({contentType: 'application/javascript', body: `
    import {inpaint} from '/node_modules/.vite/deps/onnxruntime-web_webgpu.js?import-recovery-test';
    export async function clientInpaint(source, target) { return inpaint(source, target); }
  `}));
  await page.route('**/onnxruntime-web_webgpu.js?import-recovery-test', route => {
    dependencyRequests++;
    if (dependencyRequests === 1) return route.abort();
    return route.fulfill({contentType: 'application/javascript', body: `
      export function inpaint(source, target) {
        const pixels = new Uint8Array(source.pixels);
        for (let i = 0; i < target.length; i++) if (target[i]) pixels.set([200,200,200], i * 4);
        return pixels;
      }
    `});
  });
  await page.goto('/');
  const result = await page.evaluate(async () => {
    const {renderImage, getRenderSchedulerState} = await import('/src/core/engine.ts');
    const source = {width: 8, height: 8, pixels: new Uint8Array(8 * 8 * 4)};
    for (let i = 0; i < 64; i++) source.pixels.set([100,100,100,255], i * 4);
    const graph = {version: '1.0', sourceAssetId: 'local', operations: [{id: 'model', type: 'repair.inpaint', enabled: true, params: {method: 'lama', execution: 'client', region: {x: .25, y: .25, width: .5, height: .5}}}]};
    let error = '';
    try {await renderImage(source, graph);} catch (caught) {error = String(caught);}
    const image = await renderImage(source, graph);
    return {error, pixels: Array.from(image.pixels), state: getRenderSchedulerState()};
  });
  expect(result.error).toContain('客户端去人模块加载失败'); expect(dependencyRequests).toBe(2);
  expect(result.state.workerGenerations).toBe(2);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
    const value = x >= 2 && x < 6 && y >= 2 && y < 6 ? 200 : 100;
    expect(result.pixels.slice((y * 8 + x) * 4, (y * 8 + x + 1) * 4)).toEqual([value,value,value,255]);
  }
});

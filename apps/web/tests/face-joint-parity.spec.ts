import { expect, test } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';

// Real detector polygons are tracked in the Rust fixture. Raster mattes and
// texture are generated here, so this regression needs no cached photo/model.
// Byte parity is the contract; pixel colors are not aesthetic quality goldens.
test('脸宽默认联合脸颈域：真实检测坐标、正负30/100、UI导出与native逐字节一致', async ({ page, request }) => {
  test.setTimeout(180_000);
  const fixture = JSON.parse(await fs.readFile(path.resolve(import.meta.dirname,
    '../../../core/image-core/tests/fixtures/user-brow-geometry-20261014.json'), 'utf8'));
  await page.goto('/');
  const scene = await page.evaluate(params => {
    const size = 256;
    const inside = (x: number, y: number, points: Array<{ x: number; y: number }>) => {
      let found = false;
      for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
        const a = points[i], b = points[j];
        if ((a.y > y) !== (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x) found = !found;
      }
      return found;
    };
    const head: number[] = [], face: number[] = [], neck: number[] = [], skin: number[] = [], subject: number[] = [];
    const canvas = document.createElement('canvas'); canvas.width = size; canvas.height = size;
    const context = canvas.getContext('2d')!, pixels = context.createImageData(size, size);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const nx = (x + .5) / size, ny = (y + .5) / size;
      const inFace = inside(nx, ny, params.semanticMasks.faceContour.points);
      const inHead = inside(nx, ny, params.semanticMasks.faceOutline.points);
      // The observed neck begins within the lower face and remains connected
      // to it. faceContourMask alpha .5 matches its detector polygon boundary.
      const inNeck = ((nx - .495) / .12) ** 2 + ((ny - .69) / .18) ** 2 < 1;
      head.push(inHead ? 255 : 0); face.push(inFace ? 255 : 0); neck.push(inNeck ? 255 : 0);
      skin.push(inFace || inNeck ? 255 : 0); subject.push(inHead || inNeck ? 255 : 0);
      pixels.data.set([(x * 5 + y * 3) % 256, (x * 11 + y * 7) % 256,
        (x * 17 + y * 13) % 256, 255], (y * size + x) * 4);
    }
    context.putImageData(pixels, 0, 0);
    const raster = (data: number[]) => ({ shape: 'raster', width: size, height: size, data });
    params.semanticMasks.face = raster(head);
    params.semanticMasks.faceContourMask = raster(face);
    params.semanticMasks.skin = raster(skin);
    params.semanticMasks.neck = raster(neck);
    params.subjectMask = raster(subject);
    params.protectMask = { shape: 'rect', region: { x: .475, y: .635, width: .04, height: .025 } };
    params.adjustments = {};
    return { source: canvas.toDataURL('image/png').split(',')[1], params };
  }, fixture.params);
  const source = Buffer.from(scene.source, 'base64');
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles({
    name: 'detected-face-neck.png', mimeType: 'image/png', buffer: source,
  });
  const imported = { version: '1.0', sourceAssetId: '', operations: [{
    id: 'joint-face', type: 'face.adjust', enabled: true, version: '1.0',
    params: { opacity: 100, blendMode: 'normal', ...scene.params },
  }] };
  await page.locator('input[type=file][accept=".json"]').setInputFiles({
    name: 'joint-face.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(imported)),
  });
  await expect(page.locator('.statusbar')).toContainText('操作工程已载入，可撤销');
  await page.getByRole('button', { name: '脸型', exact: true }).click();
  const control = page.getByRole('spinbutton', { name: '脸宽 / 瘦脸数值', exact: true });
  await expect(control).toBeVisible();
  const created = await request.post('http://127.0.0.1:8080/api/v1/sessions');
  expect(created.ok()).toBeTruthy(); const session = await created.json();
  try {
    const upload = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/assets`, {
      multipart: { file: { name: 'detected-face-neck.png', mimeType: 'image/png', buffer: source } },
    });
    expect(upload.ok()).toBeTruthy(); const asset = await upload.json();
    for (const amount of [-100, -30, 30, 100]) {
      await control.fill(String(amount));
      const [saved] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '保存工程', exact: true }).click()]);
      const graph = JSON.parse(await fs.readFile((await saved.path())!, 'utf8'));
      expect(graph.operations).toHaveLength(1);
      expect(graph.operations[0].params.adjustments.faceWidth).toBe(amount);
      expect(graph.operations[0].params.semanticMasks).toEqual(scene.params.semanticMasks);
      expect(graph.operations[0].params.protectMask).toEqual(scene.params.protectMask);
      expect(Object.keys(graph.operations[0].params).some(key => /experiment|joint/i.test(key))).toBe(false);
      const [exported] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '导出 PNG', exact: true }).click()]);
      const actual = await fs.readFile((await exported.path())!);
      const native = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/render`, {
        data: { editGraph: { ...graph, sourceAssetId: asset.id }, format: 'png' },
      });
      expect(native.status(), `faceWidth=${amount}: native API render failed`).toBe(200);
      const check = await page.evaluate(async ({ source, actual, native, params }) => {
        const { decodeImage } = await import('/src/core/engine.ts');
        const file = (base64: string) => new File([Uint8Array.from(atob(base64), c => c.charCodeAt(0))], 'image.png', { type: 'image/png' });
        const original = await decodeImage(file(source)), browser = await decodeImage(file(actual)), expected = await decodeImage(file(native));
        let differingBytes = 0, alphaChanges = 0, changedPixels = 0, neckChanges = 0, protectedChanges = 0;
        for (let i = 0; i < original.pixels.length; i += 4) {
          const x = ((i / 4) % original.width + .5) / original.width;
          const y = (Math.floor(i / 4 / original.width) + .5) / original.height;
          const changed = [0, 1, 2].some(c => original.pixels[i + c] !== browser.pixels[i + c]);
          changedPixels += Number(changed);
          neckChanges += Number(changed && params.semanticMasks.neck.data[i / 4] === 255 && y > .59 && y < .69);
          protectedChanges += Number(changed && x >= .475 && x < .515 && y >= .635 && y < .66);
          for (let c = 0; c < 4; c++) differingBytes += Number(browser.pixels[i + c] !== expected.pixels[i + c]);
          alphaChanges += Number(browser.pixels[i + 3] !== original.pixels[i + 3]);
        }
        return { sameSize: browser.width === expected.width && browser.height === expected.height,
          differingBytes, alphaChanges, changedPixels, neckChanges, protectedChanges };
      }, { source: scene.source, actual: actual.toString('base64'), native: (await native.body()).toString('base64'), params: scene.params });
      expect(check.sameSize).toBe(true); expect(check.differingBytes, `faceWidth=${amount}`).toBe(0);
      expect(check.alphaChanges).toBe(0); expect(check.protectedChanges).toBe(0);
      expect(check.changedPixels).toBeGreaterThan(100); expect(check.neckChanges, `faceWidth=${amount}: joint domain must move observed neck`).toBeGreaterThan(0);
      await expect(page.locator('.status-error')).toHaveCount(0);
    }
  } finally { await request.delete(`http://127.0.0.1:8080/api/v1/sessions/${session.id}`); }
});

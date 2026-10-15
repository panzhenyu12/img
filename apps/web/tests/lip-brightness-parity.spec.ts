import { expect, test } from '@playwright/test';

// The original additive lightening painted fully black lip texture gray.
// Verify the material-lighting behavior through Go/native and browser/WASM,
// including the same teeth, selection and protection masks used by the editor.
test('唇部明亮保留暗纹，Go/native 与浏览器 WASM 输出一致', async ({ page, request }) => {
  await page.goto('/');
  const source = await page.evaluate(() => {
    const canvas = document.createElement('canvas'); canvas.width = 99; canvas.height = 71;
    const context = canvas.getContext('2d')!;
    const data = context.createImageData(canvas.width, canvas.height);
    const colors = [[0, 0, 0], [30, 8, 12], [140, 40, 55], [235, 180, 192]];
    for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
      const rgb = colors[Math.floor(y / 18) % colors.length];
      data.data.set([...rgb, x > 90 ? 0 : 255], (y * canvas.width + x) * 4);
    }
    context.putImageData(data, 0, 0);
    return canvas.toDataURL('image/png').split(',')[1];
  });
  const creation = await request.post('http://127.0.0.1:8080/api/v1/sessions');
  expect(creation.ok()).toBeTruthy();
  const session = await creation.json();
  try {
    const upload = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/assets`, {
      multipart: { file: { name: 'lip-material.png', mimeType: 'image/png', buffer: Buffer.from(source, 'base64') } },
    });
    expect(upload.ok()).toBeTruthy();
    const asset = await upload.json();
    for (const amount of [0, 30, -30, 100, -100]) {
      const graph = { version: '1.0', sourceAssetId: asset.id, operations: [{
        id: 'lip-lighting', type: 'feature.adjust', version: '1.0', enabled: true,
        params: {
          adjustments: { lipBrightness: amount },
          region: { x: .1, y: .1, width: .8, height: .8 },
          protectMask: { shape: 'rect', region: { x: .7, y: 0, width: .1, height: 1 } },
          semanticMasks: {
            lips: { shape: 'rect', region: { x: .1, y: .1, width: .8, height: .8 } },
            teeth: { shape: 'rect', region: { x: .4, y: 0, width: .1, height: 1 } },
          },
        },
      }] };
      const exported = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/render`, {
        data: { editGraph: graph, format: 'png' },
      });
      expect(exported.status(), await exported.text()).toBe(200);
      const result = await page.evaluate(async ({ source, native, graph }) => {
        const { decodeImage, renderImage } = await import('/src/core/engine.ts');
        const file = (b64: string) => new File([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], 'lip.png', { type: 'image/png' });
        const input = await decodeImage(file(source));
        const expected = await decodeImage(file(native));
        const actual = await renderImage(input, graph, { kind: 'export' });
        let differences = 0, alphaChanges = 0, outsideChanges = 0, blackLifted = 0, changedPixels = 0;
        for (let i = 0; i < input.pixels.length; i += 4) {
          const x = ((i / 4) % input.width + .5) / input.width;
          const y = (Math.floor(i / 4 / input.width) + .5) / input.height;
          let changed = false;
          for (let c = 0; c < 4; c++) {
            if (expected.pixels[i + c] !== actual.pixels[i + c]) differences++;
            if (c < 3 && actual.pixels[i + c] !== input.pixels[i + c]) changed = true;
          }
          if (changed) changedPixels++;
          if (actual.pixels[i + 3] !== input.pixels[i + 3]) alphaChanges++;
          if (changed && (x < .1 || x > .9 || y < .1 || y > .9 || (x >= .4 && x < .5) || (x >= .7 && x < .8))) outsideChanges++;
          if (input.pixels[i] === 0 && input.pixels[i + 1] === 0 && input.pixels[i + 2] === 0 && changed) blackLifted++;
        }
        return { differences, alphaChanges, outsideChanges, blackLifted, changedPixels };
      }, { source, native: (await exported.body()).toString('base64'), graph });
      expect(result.differences, `amount=${amount}`).toBe(0);
      expect(result.alphaChanges).toBe(0);
      expect(result.outsideChanges).toBe(0);
      expect(result.blackLifted, `amount=${amount}`).toBe(0);
      if (amount === 0) expect(result.changedPixels).toBe(0);
      else expect(result.changedPixels).toBeGreaterThan(100);
    }
  } finally {
    await request.delete(`http://127.0.0.1:8080/api/v1/sessions/${session.id}`);
  }
});

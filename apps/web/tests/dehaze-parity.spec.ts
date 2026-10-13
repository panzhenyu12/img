import { expect, test } from '@playwright/test';

// Regression for a one-byte native/WASM difference at (96,121), red channel,
// dehaze=60. Compare decoded RGBA bytes strictly; a one-byte tolerance would
// hide the original defect. Signed strength and downsampled analysis are covered.
test('去雾数值回归：正负强度、遮罩和分析缩放的 native/WASM 严格一致', async ({ page, request }) => {
  await page.goto('/');
  for (const item of [
    { width: 128, height: 128, params: { dehaze: 60 } },
    { width: 128, height: 128, params: { dehaze: -60 } },
    { width: 271, height: 139, params: { dehaze: 100, protectMask: { shape: 'rect', region: { x: 0, y: 0, width: .2, height: 1 } } } },
  ]) {
    const encoded = await page.evaluate(({ width, height }) => {
      const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
      const context = canvas.getContext('2d')!; const data = context.createImageData(width, height);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        data.data.set([(x * 5 + y * 3) % 256, (x * 11 + y * 7) % 256, (x * 17 + y * 13) % 256, 255], (y * width + x) * 4);
      }
      context.putImageData(data, 0, 0); return canvas.toDataURL('image/png').split(',')[1];
    }, item);
    const creation = await request.post('http://127.0.0.1:8080/api/v1/sessions');
    expect(creation.ok()).toBeTruthy(); const session = await creation.json();
    try {
      const upload = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/assets`, {
        multipart: { file: { name: 'dehaze-regression.png', mimeType: 'image/png', buffer: Buffer.from(encoded, 'base64') } },
      });
      expect(upload.ok()).toBeTruthy(); const asset = await upload.json();
      const graph = { version: '1.0', sourceAssetId: asset.id, operations: [{ id: 'dehaze', type: 'color.basic', version: '1.0', enabled: true, params: item.params }] };
      const exported = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/render`, { data: { editGraph: graph, format: 'png' } });
      expect(exported.status(), await exported.text()).toBe(200);
      const differences = await page.evaluate(async ({ source, native, graph }) => {
        const { decodeImage, renderImage } = await import('/src/core/engine.ts');
        const file = (b64: string) => new File([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], 'dehaze.png', { type: 'image/png' });
        const input = await decodeImage(file(source)); const expected = await decodeImage(file(native));
        const actual = await renderImage(input, graph, { kind: 'export' });
        const differences: Array<{ x: number; y: number; channel: number; native: number; wasm: number }> = [];
        for (let i = 0; i < expected.pixels.length; i++) if (actual.pixels[i] !== expected.pixels[i]) {
          differences.push({ x: Math.floor(i / 4) % actual.width, y: Math.floor(i / 4 / actual.width), channel: i % 4, native: expected.pixels[i], wasm: actual.pixels[i] });
        }
        return differences;
      }, { source: encoded, native: (await exported.body()).toString('base64'), graph });
      expect(differences, `${item.width}x${item.height}, dehaze=${item.params.dehaze}`).toEqual([]);
    } finally {
      await request.delete(`http://127.0.0.1:8080/api/v1/sessions/${session.id}`);
    }
  }
});

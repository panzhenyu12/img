import { expect, test } from '@playwright/test';

// Low-yellow pale/deep skin exposed the original cold slider's purple cast.
// Check its corrected effect and full-byte native/WASM parity with geometric
// selection, protected makeup, frozen skin, shading and transparent pixels.
test('肤色冷暖：低黄色肤色、保护区域和 native/WASM 逐字节一致', async ({ page, request }) => {
  await page.goto('/');
  const source = await page.evaluate(() => {
    const canvas = document.createElement('canvas'); canvas.width = 99; canvas.height = 71;
    const context = canvas.getContext('2d')!; const data = context.createImageData(canvas.width, canvas.height);
    const colors = [[118, 105, 99], [100, 75, 68], [184, 160, 155], [175, 135, 100], [80, 60, 50], [245, 239, 230]];
    for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
      const color = colors[Math.floor(y / 12) % colors.length];
      const detail = x % 2 === 0 ? 3 : -3;
      data.data.set([...color.map(c => c + detail), x > 90 ? 0 : 255], (y * canvas.width + x) * 4);
    }
    context.putImageData(data, 0, 0); return canvas.toDataURL('image/png').split(',')[1];
  });
  const creation = await request.post('http://127.0.0.1:8080/api/v1/sessions');
  expect(creation.ok()).toBeTruthy(); const session = await creation.json();
  try {
    const upload = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/assets`, {
      multipart: { file: { name: 'skin-temperature.png', mimeType: 'image/png', buffer: Buffer.from(source, 'base64') } },
    });
    expect(upload.ok()).toBeTruthy(); const asset = await upload.json();
    for (const warmth of [-100, -40, 0, 40, 100]) {
      const graph = { version: '1.0', sourceAssetId: asset.id, operations: [{ id: 'skin', type: 'skin.retouch', version: '1.0', enabled: true, params: {
        warmth, smooth: 0,
        region: { x: .1, y: 0, width: .8, height: 1 },
        protectMask: { shape: 'rect', region: { x: .7, y: 0, width: .1, height: 1 } },
        semanticMasks: {
          skin: { shape: 'rect', region: { x: 0, y: 0, width: 1, height: 1 } },
          lips: { shape: 'rect', region: { x: .4, y: 0, width: .1, height: 1 } },
          skinProtection: { shape: 'rect', region: { x: .5, y: 0, width: .1, height: 1 } },
        },
      } }] };
      const exported = await request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/render`, { data: { editGraph: graph, format: 'png' } });
      expect(exported.status(), await exported.text()).toBe(200);
      const result = await page.evaluate(async ({ source, native, graph, warmth }) => {
        const { decodeImage, renderImage } = await import('/src/core/engine.ts');
        const file = (b64: string) => new File([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], 'skin.png', { type: 'image/png' });
        const input = await decodeImage(file(source)); const expected = await decodeImage(file(native));
        const actual = await renderImage(input, graph, { kind: 'export' });
        const differences: Array<{ x: number; y: number; channel: number; native: number; wasm: number }> = [];
        let coldBlueCrossings = 0, alphaChanges = 0, outsideChanges = 0, maxLumaShift = 0, changedPixels = 0;
        for (let i = 0; i < expected.pixels.length; i++) if (actual.pixels[i] !== expected.pixels[i]) {
          differences.push({ x: Math.floor(i / 4) % actual.width, y: Math.floor(i / 4 / actual.width), channel: i % 4, native: expected.pixels[i], wasm: actual.pixels[i] });
        }
        for (let i = 0; i < input.pixels.length; i += 4) {
          const x = ((i / 4) % input.width + .5) / input.width;
          const changed = actual.pixels[i] !== input.pixels[i] || actual.pixels[i + 1] !== input.pixels[i + 1] || actual.pixels[i + 2] !== input.pixels[i + 2];
          if (changed) changedPixels++;
          if (changed && (x < .1 || x > .9 || (x >= .4 && x < .6) || (x >= .7 && x < .8))) outsideChanges++;
          if (actual.pixels[i + 3] !== input.pixels[i + 3]) alphaChanges++;
          if (warmth < 0 && input.pixels[i + 1] - input.pixels[i + 2] >= 2 && actual.pixels[i + 1] < actual.pixels[i + 2]) coldBlueCrossings++;
          const delta = [.2126, .7152, .0722].reduce((sum, coefficient, c) => sum + coefficient * (actual.pixels[i + c] - input.pixels[i + c]), 0);
          maxLumaShift = Math.max(maxLumaShift, Math.abs(delta));
        }
        return { differences, coldBlueCrossings, alphaChanges, outsideChanges, maxLumaShift, changedPixels };
      }, { source, native: (await exported.body()).toString('base64'), graph, warmth });
      expect(result.differences, `warmth=${warmth}`).toEqual([]);
      expect(result.alphaChanges).toBe(0);
      expect(result.outsideChanges).toBe(0);
      expect(result.coldBlueCrossings).toBe(0);
      expect(result.maxLumaShift).toBeLessThan(.8);
      if (warmth === 0) expect(result.changedPixels).toBe(0);
      else expect(result.changedPixels).toBeGreaterThan(100);
    }
  } finally {
    await request.delete(`http://127.0.0.1:8080/api/v1/sessions/${session.id}`);
  }
});

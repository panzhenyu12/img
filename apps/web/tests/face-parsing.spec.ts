import { expect, test } from '@playwright/test';

test('face contour control classes exclude ears, glasses, earrings, hair and neck', async ({ page }) => {
  await page.goto('/');
  const layers = await page.evaluate(async () => {
    const { decodeFaceClasses } = await import('/src/core/parsing.ts');
    // One certain pixel per CelebAMask-HQ class, including accessories that
    // previously expanded the jaw control contour on actual street photos.
    const count = 19, logits = new Float32Array(count * count).fill(-20);
    for (let c = 0; c < count; c++) logits[c * count + c] = 20;
    return Object.fromEntries(Object.entries(decodeFaceClasses(logits, count)).map(([name, values]) => [name, Array.from(values)]));
  });
  for (const c of [6, 7, 8, 9, 14, 17, 18]) {
    expect(layers.faceControl[c]).toBeLessThan(0.001);
    expect(layers.personAlpha[c]).toBeGreaterThan(0.999);
  }
  for (const c of [6, 7, 8, 9, 17, 18]) expect(layers.head[c]).toBeGreaterThan(0.999);
  for (const c of [1, 2, 3, 4, 5, 10, 11, 12, 13]) expect(layers.faceControl[c]).toBeGreaterThan(0.999);
  for (const c of [6, 9]) expect(layers.accessories[c]).toBeGreaterThan(0.999);
  for (const c of [7, 8]) expect(layers.ears[c]).toBeGreaterThan(0.999);
  for (const c of [6, 7, 8, 9, 17]) expect(layers.skin[c]).toBeLessThan(0.001);
  expect(layers.neck[14]).toBeGreaterThan(0.999);
  expect(layers.personAlpha[0]).toBeLessThan(0.001);
});

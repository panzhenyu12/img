import { expect, test, type Page } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';

const photo = path.resolve('tests/fixtures/portrait.jpg');
const tool = (page: Page, label: string) => page.getByLabel(label, { exact: true }).click();
const eyes = (page: Page) => page.getByRole('spinbutton', { name: '眼睛大小数值', exact: true });
async function graph(page: Page) {
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '保存工程', exact: true }).click()]);
  return JSON.parse(await fs.readFile((await download.path())!, 'utf8'));
}
async function detectedFace(page: Page) {
  await page.goto('/');
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(photo);
  await tool(page, '脸型');
  await page.getByRole('button', { name: '自动检测区域', exact: true }).click();
  await expect(page.locator('.statusbar')).toContainText(/检测到 \d+ 个语义区域/, { timeout: 180_000 });
  await page.getByRole('button', { name: '为选区新增调整层', exact: true }).click();
  const saved = await graph(page);
  expect(saved.operations[0].params.semanticMasks.leftEye.shape).toBe('polygon');
  return saved;
}

test('真实检测后跨工具返回保留五官上下文和实际滑杆图层', async ({ page }) => {
  test.setTimeout(240_000);
  const initial = await detectedFace(page);
  await tool(page, '五官与妆容');
  await expect(eyes(page)).toBeVisible();
  await tool(page, '光影与调色'); await tool(page, '脸型'); await tool(page, '五官与妆容');
  await expect(eyes(page)).toBeVisible();
  await eyes(page).fill('30');
  const saved = await graph(page);
  expect(saved.operations.at(-1).type).toBe('feature.adjust');
  expect(saved.operations.at(-1).params.adjustments.eyeSize).toBe(30);
  expect(saved.operations.at(-1).params.semanticMasks).toEqual(initial.operations[0].params.semanticMasks);
  await expect(page.locator('.status-error')).toHaveCount(0);
});

test('载入已有工程后工具栏切换与历史图层点击恢复相同上下文', async ({ page }) => {
  test.setTimeout(240_000);
  const initial = await detectedFace(page);
  // Reload a fresh editor so no detector state can mask importGraph failures.
  await page.reload();
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(photo);
  await page.locator('input[type=file][accept=".json"]').setInputFiles({ name: 'face.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(initial)) });
  await expect(page.locator('.statusbar')).toContainText('操作工程已载入，可撤销');
  await tool(page, '光影与调色'); await tool(page, '脸型'); await tool(page, '五官与妆容');
  await expect(eyes(page)).toBeVisible();
  await page.getByRole('button', { name: '为选区新增调整层', exact: true }).click();
  const saved = await graph(page);
  expect(saved.operations).toHaveLength(2);
  expect(saved.operations.at(-1).params.semanticMasks).toEqual(initial.operations[0].params.semanticMasks);
  await tool(page, '光影与调色'); await tool(page, '五官与妆容');
  await expect(eyes(page)).toBeVisible();
  await eyes(page).fill('40');
  expect((await graph(page)).operations.at(-1).params.adjustments.eyeSize).toBe(40);
});

test('上游几何改变后既有五官蒙版失效，重新检测才能恢复参数', async ({ page }) => {
  test.setTimeout(240_000);
  await detectedFace(page);
  await tool(page, '五官与妆容'); await eyes(page).fill('20');
  await tool(page, '脸型');
  await page.getByRole('spinbutton', { name: '脸宽 / 瘦脸数值', exact: true }).fill('30');
  await tool(page, '五官与妆容');
  await expect(eyes(page)).toHaveCount(0);
  await expect(page.locator('.statusbar')).toContainText('重新检测');
  await page.getByRole('button', { name: '自动检测区域', exact: true }).click();
  await expect(page.locator('.statusbar')).toContainText(/检测到 \d+ 个语义区域/, { timeout: 180_000 });
  await expect(eyes(page)).toBeVisible();
  await expect(page.locator('.status-error')).toHaveCount(0);
});

test('现有形变后新增同类图层使用新的输入检测，不复制陈旧五官蒙版', async ({ page }) => {
  test.setTimeout(240_000);
  await detectedFace(page);
  await tool(page, '五官与妆容'); await eyes(page).fill('20');
  await page.getByRole('button', { name: '为选区新增调整层', exact: true }).click();
  await expect(page.locator('.statusbar')).toContainText('待定位图层');
  await expect(eyes(page)).toHaveCount(0);
  const pending = await graph(page);
  expect(pending.operations).toHaveLength(3);
  expect(pending.operations.at(-1).params.adjustments).toEqual({});
  expect(pending.operations.at(-1).params.semanticMasks).toBeUndefined();
  expect(pending.operations.at(-1).params.landmarks).toBeUndefined();
  await page.getByRole('button', { name: '自动检测区域', exact: true }).click();
  await expect(page.locator('.statusbar')).toContainText(/检测到 \d+ 个语义区域/, { timeout: 180_000 });
  await expect(eyes(page)).toBeVisible(); await eyes(page).fill('25');
  const saved = await graph(page);
  expect(saved.operations.at(-1).params.adjustments.eyeSize).toBe(25);
  expect(saved.operations.at(-1).params.semanticMasks.leftEye.shape).toBe('polygon');
  await expect(page.locator('.status-error')).toHaveCount(0);
});

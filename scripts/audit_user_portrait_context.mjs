/** Reproduce tool switching and new portrait-layer context using real UI. */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runLabel = process.argv[2] ?? 'user-feature-context-20261014';
const output = path.join(root, 'test/results', runLabel);
const doc = path.join(root, 'doc/qa/user-feature-audit-20261014');
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const rows = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.setDefaultTimeout(30000);
  await page.goto('http://127.0.0.1:5173/');
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(path.join(root, 'test/results/user-face-contour-20261014/1024/canonical-source.png'));
  const tool = label => page.getByLabel(label, { exact: true }).click();
  const snapshot = async name => {
    const noticeBeforeSave = await page.locator('.statusbar > span').textContent();
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '保存工程', exact: true }).click()]);
    const graph = JSON.parse(await fs.readFile(await download.path(), 'utf8'));
    const file = path.join(output, name + '.json'); await fs.writeFile(file, JSON.stringify(graph));
    const row = { name, tool: await page.locator('.tool-inspector').getAttribute('data-tool'),
      featureSliders: await page.getByRole('spinbutton', { name: '眼睛大小数值', exact: true }).count(),
      notice: await page.locator('.statusbar > span').textContent(), noticeBeforeSave,
      operations: graph.operations.map(op => ({ id: op.id, type: op.type, adjustments: op.params.adjustments,
        semantics: Object.keys(op.params.semanticMasks ?? {}), landmarks: op.params.landmarks?.length ?? 0, region: op.params.region })), graph: file };
    rows.push(row); console.log(JSON.stringify(row)); return graph;
  };
  // Imported persisted face context should remain usable by another portrait tool.
  await page.locator('input[type=file][accept=".json"]').setInputFiles(path.join(root, 'test/results/user-face-contour-20261014/1024/face-graph.json'));
  await page.getByText('操作工程已载入，可撤销', { exact: true }).waitFor();
  await tool('脸型'); await tool('五官与妆容');
  await snapshot('import-face-switch-feature');
  // Repeat from a real detector session, preserving its output separately.
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(path.join(root, 'test/results/user-face-contour-20261014/1024/canonical-source.png'));
  await tool('脸型');
  await page.getByRole('button', { name: '自动检测区域', exact: true }).click();
  await page.getByText(/检测到 \d+ 个语义区域/).waitFor({ timeout: 240000 });
  await page.getByRole('button', { name: '为选区新增调整层', exact: true }).click();
  await snapshot('fresh-face-detect');
  await tool('五官与妆容');
  await snapshot('fresh-face-direct-feature');
  await tool('光影与调色'); await tool('脸型'); await tool('五官与妆容');
  await snapshot('fresh-face-color-face-feature');
  // Clicking the stored face layer invokes a different restoration path.
  const faceLayer = page.locator('.operation-select').filter({ hasText: '脸型选区图层' });
  if (await faceLayer.count()) await faceLayer.first().click();
  else await page.getByRole('button', { name: '脸型选区图层', exact: true }).click();
  await tool('五官与妆容');
  await snapshot('stored-face-layer-click-feature');
  await page.getByRole('spinbutton', { name: '眼睛大小数值', exact: true }).fill('50');
  await snapshot('fresh-feature-eye50');
  await tool('光影与调色'); await tool('五官与妆容');
  await snapshot('existing-feature-return');
  await page.getByRole('button', { name: '为选区新增调整层', exact: true }).click();
  await snapshot('existing-feature-new-layer-after-return');
  if (await page.getByRole('spinbutton', { name: '眼睛大小数值', exact: true }).count() === 0) {
    await page.getByRole('button', { name: '自动检测区域', exact: true }).click();
    await page.getByText(/检测到 \d+ 个语义区域/).waitFor({ timeout: 240000 });
    await page.getByRole('spinbutton', { name: '眼睛大小数值', exact: true }).fill('25');
    await snapshot('new-feature-layer-redetected-eye25');
  }
  await page.screenshot({ path: path.join(doc, runLabel + '-panel.png'), fullPage: true });
} catch (error) {
  rows.push({ error: String(error) }); console.error(error); process.exitCode = 1;
} finally {
  await fs.writeFile(path.join(doc, runLabel === 'user-feature-context-20261014' ? 'context-report.json' : runLabel + '-report.json'), JSON.stringify({ rows, scope: 'Real detector and UI tool-switch context; independent of fixed single-operation algorithm audit.' }, null, 2));
  await browser.close();
}

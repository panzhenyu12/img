/** Verify real editor slider -> saved feature.adjust graph -> PNG export.
 * Uses one fixed fresh source graph; does not rerun detection or overwrite QA.
 */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const baseDir = path.join(root, 'test/results/user-face-contour-20261014/1024');
const output = path.join(root, 'test/results/user-feature-ui-audit-20261014');
const doc = path.join(root, 'doc/qa/user-feature-audit-20261014');
await fs.mkdir(output, { recursive: true }); await fs.mkdir(doc, { recursive: true });
const graph = JSON.parse(await fs.readFile(path.join(baseDir, 'face-graph.json'), 'utf8'));
const params = Object.fromEntries(Object.entries(graph.operations[0].params).filter(([key]) => ['region', 'mask', 'protectMask', 'semanticMasks', 'landmarks', 'opacity', 'blendMode'].includes(key)));
const base = { ...graph, operations: [{ id: 'ui-feature-audit', type: 'feature.adjust', enabled: true, params: { ...params, adjustments: {} } }] };
const graphFile = path.join(output, 'base-feature.json'); await fs.writeFile(graphFile, JSON.stringify(base));
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = []; page.on('pageerror', error => errors.push(String(error)));
  await page.goto('http://127.0.0.1:5173/');
  await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(path.join(baseDir, 'canonical-source.png'));
  await page.getByRole('button', { name: '五官与妆容', exact: true }).click();
  await page.locator('input[type=file][accept=".json"]').setInputFiles(graphFile);
  await page.getByText('操作工程已载入，可撤销', { exact: true }).waitFor();
  const controls = await page.evaluate(async params => {
    const { TOOL_BY_ID, portraitParametersFor, DEFERRED_PORTRAIT_PARAMETERS } = await import('/src/core/registry.ts');
    const available = portraitParametersFor('feature', { region: params.region, semanticMasks: params.semanticMasks });
    return { available, deferred: DEFERRED_PORTRAIT_PARAMETERS.feature, listed: TOOL_BY_ID.feature.parameters.map(p => p.key) };
  }, params);
  const rows = []; let previous = null;
  const representative = new Set(['eyeSize', 'eyeWidth', 'eyeHeight', 'noseWings', 'mouthWidth', 'browColor', 'lipColor', 'lipBrightness', 'lipSaturation', 'teethWhiten', 'darkCircle']);
  for (const parameter of controls.available) {
    if (previous) await page.getByRole('spinbutton', { name: previous.label + '数值', exact: true }).fill('0');
    const input = page.getByRole('spinbutton', { name: parameter.label + '数值', exact: true });
    const visible = await input.isVisible(); await input.fill('100');
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '保存工程', exact: true }).click()]);
    const saved = JSON.parse(await fs.readFile(await download.path(), 'utf8'));
    const operation = saved.operations.at(-1);
    const expected = Object.fromEntries(Object.entries(operation.params.adjustments).filter(([, value]) => value !== 0));
    const row = { key: parameter.key, label: parameter.label, visible, inputValue: await input.inputValue(), operationType: operation.type,
      writtenValue: operation.params.adjustments[parameter.key], enabled: operation.enabled, opacity: operation.params.opacity, blendMode: operation.params.blendMode,
      nonzeroAdjustments: expected, semanticNames: Object.keys(operation.params.semanticMasks ?? {}), region: operation.params.region,
      semanticsEqualToFixedGraph: JSON.stringify(operation.params.semanticMasks) === JSON.stringify(params.semanticMasks),
      compareButtonActive: await page.getByRole('button', { name: '原图对比', exact: true }).evaluate(node => node.classList.contains('active')),
      status: await page.locator('.statusbar > span').textContent() };
    if (representative.has(parameter.key)) {
      const graphPath = path.join(output, `${parameter.key}-100-ui.json`); await fs.writeFile(graphPath, JSON.stringify(saved));
      const [png] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '导出 PNG', exact: true }).click()]);
      const pngPath = path.join(output, `${parameter.key}-100-ui.png`); await png.saveAs(pngPath);
      row.uiGraph = graphPath; row.uiExport = pngPath;
      row.exportSha256 = crypto.createHash('sha256').update(await fs.readFile(pngPath)).digest('hex');
      row.statusAfterExport = await page.locator('.statusbar > span').textContent();
    }
    rows.push(row); previous = parameter;
    if (operation.type !== 'feature.adjust' || row.writtenValue !== 100 || Object.keys(expected).length !== 1 || !row.semanticsEqualToFixedGraph) throw new Error(`UI parameter binding failed: ${parameter.key}`);
  }
  await page.screenshot({ path: path.join(doc, 'feature-panel.png'), fullPage: true });
  // A selected original comparison intentionally presents the source even
  // while graph exports remain edited; record this separately from algorithms.
  await page.getByRole('button', { name: '原图对比', exact: true }).click();
  const originalComparison = await page.getByRole('button', { name: '原图对比', exact: true }).evaluate(node => node.classList.contains('active'));
  const report = { independentOriginals: 1, fixedCanonical: path.join(baseDir, 'canonical-source.png'),
    availableKeys: controls.available.map(p => p.key), deferredKeys: controls.deferred, rows, errors,
    originalComparisonCanHidePreviewEffect: originalComparison,
    scope: 'UI binding and actual exported PNG, fixed original detection; this is not an aesthetic acceptance score.' };
  await fs.writeFile(path.join(doc, 'ui-report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ sliders: rows.length, nonzeroBindings: rows.filter(r => r.writtenValue === 100).length, errors, deferred: controls.deferred }));
} finally { await browser.close(); }

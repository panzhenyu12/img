/** Original-photo editor/API/native parity for explicitly selected curve candidates.
 * Usage: node scripts/audit_anatomical_curve_ui.mjs manifest.json
 * Manifest: { binary, output, cases: [{ name, input, graph, expectedPNG, control? }] }
 * Each graph has one face.adjust, feature.adjust or body.adjust operation.
 * Beauty is assessed from the exported photographs separately from this check.
 */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify, isDeepStrictEqual } from 'node:util';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
if (!process.argv[2]) throw new Error('Pass an explicit candidate manifest');
const manifestPath = path.resolve(root, process.argv[2]);
const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
const binary = path.resolve(root, manifest.binary);
const output = path.resolve(root, manifest.output);
await fs.mkdir(output, { recursive: true });
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const signature = async file => sha(await fs.readFile(file));
const tracked = new Map();
const track = async file => {
  file = path.resolve(root, file);
  if (!tracked.has(file)) tracked.set(file, await signature(file));
  return file;
};
await track(manifestPath); await track(binary);
await track('apps/web/public/wasm/image_core_bg.wasm');
for (const file of await fs.readdir(path.join(root, 'core/image-core/src'))) {
  if (file.endsWith('.rs')) await track(path.join('core/image-core/src', file));
}
const cases = [];
for (const item of manifest.cases) {
  if (!/^[a-zA-Z0-9._-]+$/.test(item.name)) throw new Error('Invalid case name');
  cases.push({ ...item, input: await track(item.input), graph: await track(item.graph),
    expectedPNG: await track(item.expectedPNG) });
}
if (!cases.length) throw new Error('No candidate cases selected');
const browser = await chromium.launch({ headless: true });
const report = { scope: 'Actual original-resolution editor upload, graph import, control, save and PNG export; decoded RGBA compared with immutable native, Go API and the reviewed experimental output. This does not establish aesthetic quality or real-time performance.',
  manifest: path.relative(root, manifestPath), rows: [], errors: [], success: false };
let page;
const sessions = [];
try {
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on('pageerror', error => report.errors.push(String(error)));
  await page.goto('http://127.0.0.1:5173/');
  for (const item of cases) {
    const graph = JSON.parse(await fs.readFile(item.graph, 'utf8'));
    const operation = graph.operations?.[0];
    const tool = { 'face.adjust': 'face', 'feature.adjust': 'feature', 'body.adjust': 'body' }[operation?.type];
    if (graph.operations.length !== 1 || !tool) throw new Error(`${item.name}: unsupported graph`);
    const key = item.control ?? Object.keys(operation.params.adjustments).find(k => operation.params.adjustments[k] !== 0);
    if (!key) throw new Error(`${item.name}: specify a control for an identity case`);
    await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(item.input);
    await page.locator('input[type=file][accept=".json"]').setInputFiles(item.graph);
    await page.getByText('操作工程已载入，可撤销', { exact: true }).waitFor();
    const labels = await page.evaluate(async ({ tool, key }) => {
      const { TOOL_BY_ID } = await import('/src/core/registry.ts');
      const definition = TOOL_BY_ID[tool];
      return { tool: definition.label, parameter: definition.parameters.find(p => p.key === key)?.label };
    }, { tool, key });
    if (!labels.parameter) throw new Error(`${item.name}: missing registered parameter`);
    await page.getByRole('button', { name: labels.tool, exact: true }).click();
    await page.getByRole('spinbutton', { name: labels.parameter + '数值', exact: true }).fill(String(operation.params.adjustments[key]));
    const [saved] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '保存工程', exact: true }).click()]);
    const actualGraphPath = path.join(output, `${item.name}-ui.json`);
    await saved.saveAs(actualGraphPath);
    const actualGraph = JSON.parse(await fs.readFile(actualGraphPath, 'utf8'));
    if (actualGraph.operations.length !== 1 || actualGraph.operations[0].type !== operation.type) throw new Error(`${item.name}: operation changed`);
    for (const [key, value] of Object.entries(operation.params)) {
      if (!isDeepStrictEqual(actualGraph.operations[0].params[key], value)) throw new Error(`${item.name}: imported ${key} changed`);
    }
    const ui = path.join(output, `${item.name}-ui.png`);
    const started = performance.now();
    const [download] = await Promise.all([page.waitForEvent('download', { timeout: 180000 }), page.getByRole('button', { name: '导出 PNG', exact: true }).click()]);
    await download.saveAs(ui);
    const uiExportMs = Math.round(performance.now() - started);
    const native = path.join(output, `${item.name}-native.png`);
    await promisify(execFile)(binary, ['--input', item.input, '--graph', actualGraphPath, '--output', native]);
    const created = await page.request.post('http://127.0.0.1:8080/api/v1/sessions');
    if (!created.ok()) throw new Error('API session creation failed');
    const session = await created.json(); sessions.push(session.id);
    const uploaded = await page.request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/assets`, {
      multipart: { file: { name: 'source.png', mimeType: 'image/png', buffer: await fs.readFile(item.input) } },
    });
    if (!uploaded.ok()) throw new Error('API upload failed');
    const asset = await uploaded.json();
    const response = await page.request.post(`http://127.0.0.1:8080/api/v1/sessions/${session.id}/render`, {
      data: { editGraph: { ...actualGraph, sourceAssetId: asset.id }, format: 'png' }, timeout: 180000,
    });
    if (!response.ok()) throw new Error(`${item.name}: API render ${response.status()}`);
    const api = path.join(output, `${item.name}-api.png`); await fs.writeFile(api, await response.body());
    const encoded = await Promise.all([ui, native, api, item.expectedPNG].map(async file => (await fs.readFile(file)).toString('base64')));
    const differences = await page.evaluate(async encoded => {
      const { decodeImage } = await import('/src/core/engine.ts');
      const images = await Promise.all(encoded.map(b64 => decodeImage(new File([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], 'image.png', { type: 'image/png' }))));
      const reference = images[1];
      return images.map((im, i) => {
        let bytes = 0, maximum = 0;
        for (let j = 0; j < im.pixels.length; j++) {
          const diff = Math.abs(im.pixels[j] - reference.pixels[j]);
          if (diff) { bytes++; maximum = Math.max(maximum, diff); }
        }
        return { runtime: ['UI', 'native', 'API', 'reviewed candidate'][i], width: im.width, height: im.height,
          sameSize: im.width === reference.width && im.height === reference.height, differingBytes: bytes, maximumDifference: maximum };
      });
    }, encoded);
    report.rows.push({ name: item.name, uiExportMs, differences, source: path.relative(root, item.input), graph: path.relative(root, actualGraphPath), graphSha256: await signature(actualGraphPath),
      outputs: await Promise.all([ui, native, api, item.expectedPNG].map(async file => ({ file: path.relative(root, file), sha256: await signature(file) }))) });
    console.log(JSON.stringify(report.rows.at(-1)));
    if (differences.some(d => !d.sameSize || d.differingBytes)) throw new Error(`${item.name}: pixel mismatch`);
  }
  if (report.errors.length) throw new Error('Browser page errors');
  for (const [file, expected] of tracked) if (await signature(file) !== expected) throw new Error(`Evidence changed during audit: ${file}`);
  report.success = true;
} catch (error) {
  report.failure = String(error);
  throw error;
} finally {
  report.signatures = Object.fromEntries([...tracked].map(([file, hash]) => [path.relative(root, file), hash]));
  await fs.writeFile(path.join(output, 'ui-parity.json'), JSON.stringify(report, null, 2));
  for (const id of sessions) await page?.request.delete(`http://127.0.0.1:8080/api/v1/sessions/${id}`).catch(() => {});
  await browser.close();
}

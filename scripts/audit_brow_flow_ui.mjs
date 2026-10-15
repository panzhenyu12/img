/** Actual editor import -> parameter control -> PNG export vs immutable Rust. */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.resolve(root, process.env.FEATURE_QA_OUTPUT ?? 'test/results/brow-flow-v4-20261014');
const doc = path.resolve(root, process.env.FEATURE_QA_DOC ?? 'doc/qa/brow-flow-v4-20261014');
const binary = path.resolve(root, process.env.FEATURE_QA_BINARY ?? path.join(output, 'image-render-final'));
await fs.mkdir(output, { recursive: true });
await fs.mkdir(doc, { recursive: true });
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const nativeHash = hash(await fs.readFile(binary));
const wasmPath = path.join(root, 'apps/web/public/wasm/image_core_bg.wasm');
const wasmHash = hash(await fs.readFile(wasmPath));
const selected = new Set(['user-browPeak--100', 'user-browPeak-100', 'user-browTail--100', 'user-browTail-100', 'user-brow-plus-eyeSize-100', 'user-brow-plus-noseStraight-100', 'portrait-roll-browPeak-100', 'portrait-dark-browTail--100', 'user-eyeSize--100', 'user-eyeSize-100', 'user-eyeTilt-100', 'user-noseWings-100', 'user-mouthWidth-100', 'user-continuity-eyeSize-browPeak-0', 'user-continuity-eyeSize-browPeak-0.001', 'user-continuity-eyeSize-noseBridge-0', 'user-continuity-eyeSize-noseBridge-0.001', 'user-continuity-browPeak-eyeSize-0', 'user-continuity-browPeak-eyeSize-0.001']);
const casePath = path.resolve(root, process.env.FEATURE_QA_CASES ?? path.join(output, 'cases-ui-final.json'));
const labels = process.env.FEATURE_QA_LABELS ? new Set(process.env.FEATURE_QA_LABELS.split(',')) : selected;
const cases = JSON.parse(await fs.readFile(casePath, 'utf8')).filter(row => labels.has(row.label));
if (cases.length !== labels.size) throw new Error('Missing requested parity cases');
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = []; page.on('pageerror', error => errors.push(String(error)));
  await page.goto('http://127.0.0.1:5173/');
  const rows = [];
  for (const item of cases) {
    const graph = JSON.parse(await fs.readFile(item.graph, 'utf8'));
    await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(item.input);
    await page.getByRole('button', { name: '五官与妆容', exact: true }).click();
    await page.locator('input[type=file][accept=".json"]').setInputFiles(item.graph);
    await page.getByText('操作工程已载入，可撤销', { exact: true }).waitFor();
    const key = Object.keys(graph.operations[0].params.adjustments).find(key => graph.operations[0].params.adjustments[key] !== 0);
    const label = await page.evaluate(async key => {
      const { TOOL_BY_ID } = await import('/src/core/registry.ts');
      return TOOL_BY_ID.feature.parameters.find(item => item.key === key).label;
    }, key);
    const control = page.getByRole('spinbutton', { name: label + '数值', exact: true });
    if (!(await control.isVisible())) throw new Error(`${item.label}: missing control`);
    await control.fill(String(graph.operations[0].params.adjustments[key]));
    const [saved] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '保存工程', exact: true }).click()]);
    const uiGraph = JSON.parse(await fs.readFile(await saved.path(), 'utf8'));
    if (JSON.stringify(uiGraph.operations.at(-1).params.adjustments) !== JSON.stringify(graph.operations[0].params.adjustments)) throw new Error(`${item.label}: graph binding mismatch`);
    const started = performance.now();
    const [download] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), page.getByRole('button', { name: '导出 PNG', exact: true }).click()]);
    const uiPath = path.join(output, item.label + '-ui-final-export.png'); await download.saveAs(uiPath);
    const uiExportMs = performance.now() - started;
    const nativePath = path.join(output, item.label + '-ui-final-native.png');
    if (hash(await fs.readFile(binary)) !== nativeHash) throw new Error('native worker changed');
    await promisify(execFile)(binary, ['--input', item.input, '--graph', item.graph, '--output', nativePath]);
    const result = await page.evaluate(async ({ actual, expected }) => {
      const { decodeImage } = await import('/src/core/engine.ts');
      const file = b64 => new File([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], 'image.png', { type: 'image/png' });
      const a = await decodeImage(file(actual)); const b = await decodeImage(file(expected));
      let changedBytes = 0, maximumDifference = 0;
      for (let i = 0; i < a.pixels.length; i++) if (a.pixels[i] !== b.pixels[i]) { changedBytes++; maximumDifference = Math.max(maximumDifference, Math.abs(a.pixels[i] - b.pixels[i])); }
      return { width: a.width, height: a.height, sameSize: a.width === b.width && a.height === b.height, changedBytes, maximumDifference };
    }, { actual: (await fs.readFile(uiPath)).toString('base64'), expected: (await fs.readFile(nativePath)).toString('base64') });
    rows.push({ label: item.label, ...result, uiExportMs: Math.round(uiExportMs), uiPNG: path.relative(root, uiPath), nativePNG: path.relative(root, nativePath), graphSha256: hash(await fs.readFile(item.graph)), sourceSha256: hash(await fs.readFile(item.input)) });
    console.log(JSON.stringify(rows.at(-1)));
    if (!result.sameSize || result.changedBytes) throw new Error(`${item.label}: WASM/native byte mismatch`);
  }
  if (hash(await fs.readFile(binary)) !== nativeHash || hash(await fs.readFile(wasmPath)) !== wasmHash) throw new Error('worker or WASM mutated during parity audit');
  await page.screenshot({ path: path.join(doc, 'feature-panel-v4.png'), fullPage: true });
  await fs.writeFile(path.join(doc, 'ui-parity.json'), JSON.stringify({ scope: 'Real editor parameter/import/export flow; decoded full RGBA compared with immutable native worker. Export wall times include UI/encoding and concurrency; not a realtime benchmark.', nativeSha256: nativeHash, wasmSha256: wasmHash, rows, errors }, null, 2));
  if (errors.length) throw new Error('page errors: ' + errors.join('; '));
} finally { await browser.close(); }

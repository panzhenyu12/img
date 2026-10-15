/** Raw RGBA WASM/native hotfix parity; canvas decoding is intentionally bypassed. */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'test/results/feature-svf-v4b-20261014');
const doc = path.join(root, 'doc/qa/brow-flow-v4-20261014/hotfix-v4b');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const binary = path.join(output, 'image-render-final');
const wasm = path.join(root, 'apps/web/public/wasm/image_core_bg.wasm');
const nativeHash = hash(await fs.readFile(binary));
const wasmHash = hash(await fs.readFile(wasm));
const cases = JSON.parse(await fs.readFile(path.join(output, 'alpha-cases.json'), 'utf8'));
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  await page.goto('http://127.0.0.1:5173/');
  const rows = [];
  for (const item of cases) {
    const source = await fs.readFile(item.sourceRaw);
    const expected = await fs.readFile(item.nativeRaw);
    const graph = await fs.readFile(item.graph, 'utf8');
    const result = await page.evaluate(async ({ width, height, source, expected, graph }) => {
      const wasm = await import('/wasm/image_core.js');
      await wasm.default();
      const decode = b64 => Uint8Array.from(atob(b64), c => c.charCodeAt(0));
      const original = decode(source);
      const native = decode(expected);
      const output = wasm.render_rgba(width, height, original, graph);
      let changedBytes = 0, hiddenRGBChangesInFrozenRect = 0;
      for (let i = 0; i < native.length; i++) changedBytes += Number(native[i] !== output.pixels[i]);
      for (let y = 76; y < 85; y++) for (let x = 76; x < 85; x++) {
        const i = (y * width + x) * 4;
        if (original[i + 3] === 0) for (let c = 0; c < 3; c++) {
          hiddenRGBChangesInFrozenRect += Number(original[i + c] !== output.pixels[i + c]);
        }
      }
      return { sameSize: width === output.width && height === output.height, changedBytes, hiddenRGBChangesInFrozenRect };
    }, { width: item.width, height: item.height, source: source.toString('base64'), expected: expected.toString('base64'), graph });
    if (!result.sameSize || result.changedBytes || result.hiddenRGBChangesInFrozenRect) throw new Error(JSON.stringify(result));
    rows.push({ label: item.label, ...result, frozenPixelsChecked: item.frozenPixelsChecked,
      sourceRawSha256: hash(source), nativeRawSha256: hash(expected), graphSha256: hash(Buffer.from(graph)) });
  }
  if (hash(await fs.readFile(binary)) !== nativeHash || hash(await fs.readFile(wasm)) !== wasmHash) throw new Error('Worker changed');
  const report = { scope: 'Raw varying-alpha RGBA passed directly to browser WASM and native PNG worker, compared over all output bytes. Bypasses Canvas decoding, which can discard hidden RGB on imported PNGs; this verifies the core boundary, not a lossless editor import contract.', nativeSha256: nativeHash, wasmSha256: wasmHash, rows };
  await fs.writeFile(path.join(doc, 'alpha-parity.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await browser.close(); }

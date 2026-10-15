/** Optional original-photo UI/native/WASM V19 integration QA, with immutable workers. */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sourceDirectory = path.resolve(root, process.env.FACE_JOINT_QA_SOURCE ?? 'test/results/user-face-contour-20261014/1024');
const output = path.resolve(root, process.env.FACE_JOINT_QA_OUTPUT ?? 'test/results/face-joint-v19-production-20261014');
const doc = path.resolve(root, process.env.FACE_JOINT_QA_DOC ?? 'doc/qa/face-joint-v19-integration');
const binary = path.resolve(root, process.env.FACE_JOINT_QA_BINARY ?? path.join(output, 'image-render'));
const source = path.join(sourceDirectory, 'canonical-source.png');
const originalGraphPath = path.join(sourceDirectory, 'face-graph.json');
const candidateDirectory = path.join(sourceDirectory, 'face-joint-v19-candidate');
const wasmPath = path.join(root, 'apps/web/public/wasm/image_core_bg.wasm');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const hashFile = async filename => hash(await fs.readFile(filename));
const relative = filename => path.relative(root, filename);
await fs.mkdir(output, { recursive: true }); await fs.mkdir(doc, { recursive: true });
const nativeHash = await hashFile(binary), wasmHash = await hashFile(wasmPath);
const sourceHash = await hashFile(source), originalGraphHash = await hashFile(originalGraphPath);
const candidateHashes = Object.fromEntries(await Promise.all([-100, -30, 30, 100].map(async amount => {
  const filename = path.join(candidateDirectory, `faceWidth${amount}.png`);
  return [filename, await hashFile(filename)];
})));
const sourceFiles = ['src/face.rs', 'src/face_flow.rs', 'src/geometry.rs', 'src/color.rs', 'src/mask.rs', 'Cargo.toml', 'Cargo.lock'];
const coreHashes = Object.fromEntries(await Promise.all(sourceFiles.map(async file => [file, await hashFile(path.join(root, 'core/image-core', file))])));
const originalGraph = JSON.parse(await fs.readFile(originalGraphPath, 'utf8'));
const params = originalGraph.operations[0].params;
for (const key of ['face', 'faceOutline', 'faceContour', 'faceContourMask', 'skin', 'neck']) {
  if (!params.semanticMasks?.[key]) throw new Error(`Missing source semantic ${key}`);
}
if (Object.keys(params).some(key => /experiment|joint/i.test(key))) throw new Error('Unexpected experimental public parameter');
const browser = await chromium.launch({ headless: true });
let apiSession;
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = []; page.on('pageerror', error => errors.push(String(error)));
  await page.goto('http://127.0.0.1:5173/');
  const creation = await page.request.post('http://127.0.0.1:8080/api/v1/sessions');
  if (!creation.ok()) throw new Error('API session creation failed');
  apiSession = await creation.json();
  const upload = await page.request.post(`http://127.0.0.1:8080/api/v1/sessions/${apiSession.id}/assets`, {
    multipart: { file: { name: 'canonical-source.png', mimeType: 'image/png', buffer: await fs.readFile(source) } },
  });
  if (!upload.ok()) throw new Error('Canonical API upload failed');
  const asset = await upload.json();
  const rows = [];
  for (const amount of [-100, -30, 30, 100]) {
    const label = `faceWidth${amount}`;
    const graph = structuredClone(originalGraph);
    graph.operations[0].params.adjustments = { faceWidth: amount };
    const requestedGraphPath = path.join(output, `${label}-request.json`);
    await fs.writeFile(requestedGraphPath, JSON.stringify(graph));
    await page.locator('input[type=file][accept="image/jpeg,image/png,image/webp"]').setInputFiles(source);
    await page.locator('input[type=file][accept=".json"]').setInputFiles(requestedGraphPath);
    await page.getByText('操作工程已载入，可撤销', { exact: true }).waitFor();
    await page.getByRole('button', { name: '脸型', exact: true }).click();
    const control = page.getByRole('spinbutton', { name: '脸宽 / 瘦脸数值', exact: true });
    await control.fill(String(amount));
    const [saved] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '保存工程', exact: true }).click()]);
    const actualGraphPath = path.join(output, `${label}-saved-ui.json`);
    await saved.saveAs(actualGraphPath);
    const actualGraph = JSON.parse(await fs.readFile(actualGraphPath, 'utf8'));
    if (actualGraph.operations.length !== 1 || actualGraph.operations[0].params.adjustments.faceWidth !== amount) throw new Error(`${label}: UI parameter binding mismatch`);
    if (JSON.stringify(actualGraph.operations[0].params.semanticMasks) !== JSON.stringify(params.semanticMasks)) throw new Error(`${label}: detector mattes changed on import/control`);
    if (Object.keys(actualGraph.operations[0].params).some(key => /experiment|joint/i.test(key))) throw new Error('Requires experimental parameter');
    const start = performance.now();
    const [download] = await Promise.all([page.waitForEvent('download', { timeout: 120_000 }), page.getByRole('button', { name: '导出 PNG', exact: true }).click()]);
    const uiPath = path.join(output, `${label}-ui.png`); await download.saveAs(uiPath);
    const uiExportMs = Math.round(performance.now() - start);
    const nativePath = path.join(output, `${label}-native.png`);
    await promisify(execFile)(binary, ['--input', source, '--graph', actualGraphPath, '--output', nativePath]);
    const apiResponse = await page.request.post(`http://127.0.0.1:8080/api/v1/sessions/${apiSession.id}/render`, {
      data: { editGraph: { ...actualGraph, sourceAssetId: asset.id }, format: 'png' },
      timeout: 120_000,
    });
    if (!apiResponse.ok()) throw new Error(`${label}: API render ${apiResponse.status()} ${await apiResponse.text()}`);
    const apiPath = path.join(output, `${label}-api.png`);
    await fs.writeFile(apiPath, await apiResponse.body());
    const candidatePath = path.join(candidateDirectory, `${label}.png`);
    const compared = await page.evaluate(async ({ source, actual, native, candidate, api }) => {
      const { decodeImage } = await import('/src/core/engine.ts');
      const file = base64 => new File([Uint8Array.from(atob(base64), c => c.charCodeAt(0))], 'image.png', { type: 'image/png' });
      const [input, ui, expected, prior, server] = await Promise.all([source, actual, native, candidate, api].map(base64 => decodeImage(file(base64))));
      const difference = (a, b) => {
        let differingBytes = 0, maximumDifference = 0;
        for (let i = 0; i < a.pixels.length; i++) if (a.pixels[i] !== b.pixels[i]) {
          differingBytes++; maximumDifference = Math.max(maximumDifference, Math.abs(a.pixels[i] - b.pixels[i]));
        }
        return { sameSize: a.width === b.width && a.height === b.height, differingBytes, maximumDifference };
      };
      let changedPixels = 0, alphaChanges = 0;
      for (let i = 0; i < input.pixels.length; i += 4) {
        changedPixels += Number([0, 1, 2].some(c => input.pixels[i + c] !== expected.pixels[i + c]));
        alphaChanges += Number(input.pixels[i + 3] !== expected.pixels[i + 3]);
      }
      return { width: ui.width, height: ui.height, uiNative: difference(ui, expected), candidateNative: difference(prior, expected), apiNative: difference(server, expected), changedPixels, alphaChanges };
    }, { source: (await fs.readFile(source)).toString('base64'), actual: (await fs.readFile(uiPath)).toString('base64'), native: (await fs.readFile(nativePath)).toString('base64'), candidate: (await fs.readFile(candidatePath)).toString('base64'), api: (await fs.readFile(apiPath)).toString('base64') });
    if ([compared.uiNative, compared.candidateNative, compared.apiNative].some(row => !row.sameSize || row.differingBytes)) throw new Error(`${label}: ${JSON.stringify(compared)}`);
    if (!compared.changedPixels || compared.alphaChanges) throw new Error(`${label}: invalid edit/alpha`);
    rows.push({ label, amount, ...compared, uiExportMs,
      source: relative(source), sourceSha256: sourceHash, requestedGraph: relative(requestedGraphPath), requestedGraphSha256: await hashFile(requestedGraphPath),
      savedUIGraph: relative(actualGraphPath), savedUIGraphSha256: await hashFile(actualGraphPath),
      uiPNG: relative(uiPath), uiPNGsha256: await hashFile(uiPath), nativePNG: relative(nativePath), nativePNGsha256: await hashFile(nativePath),
      apiPNG: relative(apiPath), apiPNGsha256: await hashFile(apiPath),
      candidatePNG: relative(candidatePath), candidatePNGsha256: await hashFile(candidatePath) });
    console.log(JSON.stringify(rows.at(-1)));
    await page.screenshot({ path: path.join(doc, `${label}-editor.png`), fullPage: true });
  }
  if (errors.length) throw new Error(errors.join('; '));
  if (await hashFile(binary) !== nativeHash || await hashFile(wasmPath) !== wasmHash || await hashFile(source) !== sourceHash || await hashFile(originalGraphPath) !== originalGraphHash) throw new Error('Frozen evidence changed during parity');
  for (const [file, digest] of Object.entries(candidateHashes)) if (await hashFile(file) !== digest) throw new Error(`Candidate output changed: ${file}`);
  for (const [file, digest] of Object.entries(coreHashes)) if (await hashFile(path.join(root, 'core/image-core', file)) !== digest) throw new Error(`Core source changed: ${file}`);
  await fs.writeFile(path.join(doc, 'ui-parity.json'), JSON.stringify({
    scope: 'Optional original 1024px canonical photo; real editor upload/import/control/save/PNG export, Go native API, current immutable production native and browser WASM full decoded RGBA parity, plus prior immutable V19 candidate output parity. No experimental public parameter or environment flag. Four strengths on one photograph are not four independent samples; byte parity does not establish aesthetic quality. Wall times include UI/encoding and concurrent load, not realtime benchmarks.',
    nativeBinary: relative(binary), nativeSha256: nativeHash, wasmSha256: wasmHash, coreSourceHashes: coreHashes,
    originalGraph: relative(originalGraphPath), originalGraphSha256: originalGraphHash, rows, errors,
  }, null, 2));
} finally {
  if (apiSession) await browser.contexts()[0]?.request.delete(`http://127.0.0.1:8080/api/v1/sessions/${apiSession.id}`);
  await browser.close();
}

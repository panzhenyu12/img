/** Fresh source-only parsing for the user contour report. Native replay uses
 * an immutable binary copy; no output image is redetected or fed back. */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.resolve(process.argv[2] ?? path.join(root, 'test/images/user-face-contour-20261014.jpg'));
const output = path.resolve(process.argv[3] ?? path.join(root, 'test/results/user-face-contour-20261014'));
const digest = data => createHash('sha256').update(data).digest('hex');
await fs.mkdir(output, { recursive: true });
try {
  await fs.access(path.join(output, 'provenance.json'));
  throw new Error(`Frozen diagnosis already exists at ${output}; supply a new output directory to preserve its inputs and baseline.`);
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}
const input = await fs.readFile(source);
const binary = path.join(root, 'core/image-core/target/release/image-render');
const binaryData = await fs.readFile(binary);
const baseline = path.join(output, 'baseline-image-render');
await fs.writeFile(baseline, binaryData, { mode: 0o755 });
await fs.chmod(baseline, 0o755);
if (digest(await fs.readFile(baseline)) !== digest(binaryData)) throw new Error('Baseline copy SHA mismatch');
const provenance = {
  createdUtc: new Date().toISOString(), source, sourceSha256: digest(input),
  baseline, baselineSha256: digest(binaryData),
  scope: 'Fresh browser source detection, one call per resolution. The 768 image is a resized derivative of the independent original; output images are never detected.',
  sourceHashes: {}, resolutions: [],
};
for (const relative of ['core/image-core/src/face.rs', 'core/image-core/src/face_mls.rs', 'core/image-core/src/face_matting.rs', 'apps/web/src/core/parsing.ts', 'apps/web/src/core/landmarks.ts', 'apps/web/public/models/faceparser.onnx', 'apps/web/public/models/face_landmarker.task']) {
  provenance.sourceHashes[relative] = digest(await fs.readFile(path.join(root, relative)));
}
await fs.writeFile(path.join(output, 'provenance.json'), JSON.stringify(provenance, null, 2));
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  await page.goto(process.env.RETOUCH_WEB_URL ?? 'http://127.0.0.1:5173/');
  for (const longestSide of [1024, 768]) {
    const prepared = await page.evaluate(async ({ encoded, longestSide }) => {
      const { decodeImage, pixelsToCanvas } = await import('/src/core/engine.ts');
      const { detectPeople } = await import('/src/core/landmarks.ts');
      const decoded = await decodeImage(new File([Uint8Array.from(atob(encoded), c => c.charCodeAt(0))], 'original.jpg', { type: 'image/jpeg' }));
      const original = pixelsToCanvas(decoded);
      const canvas = document.createElement('canvas');
      const scale = Math.min(1, longestSide / Math.max(decoded.width, decoded.height));
      canvas.width = Math.round(decoded.width * scale); canvas.height = Math.round(decoded.height * scale);
      canvas.getContext('2d').drawImage(original, 0, 0, canvas.width, canvas.height);
      const canonical = canvas.toDataURL('image/png').split(',')[1];
      const started = performance.now();
      const people = await detectPeople(canvas, 'face');
      const detectionElapsedMs = performance.now() - started;
      if (people.length !== 1) throw new Error(`Expected one face, detected ${people.length}`);
      const person = people[0], r = person.region;
      const points = [{ x: r.x, y: r.y }, { x: r.x + r.width, y: r.y }, { x: r.x + r.width, y: r.y + r.height }, { x: r.x, y: r.y + r.height }];
      const graph = { version: '1.0', sourceAssetId: 'user-face-contour-20261014', source: { width: canvas.width, height: canvas.height }, operations: [{
        id: 'face', type: 'face.adjust', enabled: true, version: '1.0', params: {
          opacity: 100, blendMode: 'normal', region: r,
          mask: { shape: 'polygon', points, feather: 8, invert: false },
          adjustments: { faceWidth: 0 }, landmarks: person.points.map(({ x, y }) => ({ x, y })),
          semanticMasks: person.semanticMasks, subjectMask: person.subjectMask,
        },
      }] };
      return { width: canvas.width, height: canvas.height, originalWidth: decoded.width, originalHeight: decoded.height, detectionElapsedMs, canonical, graph, people };
    }, { encoded: input.toString('base64'), longestSide });
    const dir = path.join(output, String(longestSide)); await fs.mkdir(dir, { recursive: true });
    const canonical = Buffer.from(prepared.canonical, 'base64'); delete prepared.canonical;
    const graphData = Buffer.from(JSON.stringify(prepared.graph));
    await fs.writeFile(path.join(dir, 'canonical-source.png'), canonical);
    await fs.writeFile(path.join(dir, 'face-graph.json'), graphData);
    await fs.writeFile(path.join(dir, 'detection.json'), JSON.stringify(prepared));
    const record = { longestSide, width: prepared.width, height: prepared.height, originalWidth: prepared.originalWidth, originalHeight: prepared.originalHeight, detectionElapsedMs: prepared.detectionElapsedMs, canonicalSha256: digest(canonical), graphSha256: digest(graphData), directory: dir, graph: path.join(dir, 'face-graph.json'), canonical: path.join(dir, 'canonical-source.png') };
    provenance.resolutions.push(record);
    await fs.writeFile(path.join(output, 'provenance.json'), JSON.stringify(provenance, null, 2));
    console.log(JSON.stringify(record));
  }
} finally { await browser.close(); }

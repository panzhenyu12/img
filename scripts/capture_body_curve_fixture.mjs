/** Capture source-only production localization for an independent body fixture. */
import { chromium } from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const input = path.resolve(root, process.argv[2] ?? 'test/images/body-yoga-mediapipe.jpg');
const output = path.resolve(root, process.argv[3] ?? 'test/results/body-anatomical-inputs/body-yoga-mediapipe');
await fs.mkdir(output, { recursive: true });
const bytes = await fs.readFile(input);
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  const errors = []; page.on('pageerror', error => errors.push(String(error)));
  await page.goto('http://127.0.0.1:5173/');
  const captured = await page.evaluate(async encoded => {
    const { decodeImage, pixelsToCanvas } = await import('/src/core/engine.ts');
    const { detectPeople } = await import('/src/core/landmarks.ts');
    const source = await decodeImage(new File([Uint8Array.from(atob(encoded), c => c.charCodeAt(0))], 'source.jpg', { type: 'image/jpeg' }));
    const canvas = pixelsToCanvas(source);
    const started = performance.now();
    const people = await detectPeople(canvas, 'body');
    return { width: source.width, height: source.height, people,
      detectionMs: performance.now() - started, canonical: canvas.toDataURL('image/png').split(',')[1] };
  }, bytes.toString('base64'));
  await fs.writeFile(path.join(output, 'source.png'), Buffer.from(captured.canonical, 'base64'));
  delete captured.canonical;
  const graphs = [];
  for (const [personIndex, person] of captured.people.entries()) {
    for (const part of person.parts ?? []) {
      const key = part.id === 'waist' ? 'waist' : part.id === 'hips' ? 'hips'
        : part.id?.endsWith('thigh') ? 'thighs' : part.id?.endsWith('calf') ? 'calves'
        : part.id?.endsWith('upper-arm') ? 'arms' : undefined;
      if (!key || !person.subjectMask) continue;
      const params = { region: part.region,
        mask: { shape: 'polygon', points: part.points.map(({ x, y }) => ({ x, y })), feather: 8 },
        landmarks: part.landmarks.map(({ x, y }) => ({ x, y })), bodyPart: part.id,
        subjectMask: person.subjectMask, ...(part.partMask ? { partMask: part.partMask } : {}),
        ...(part.protectMask ? { protectMask: part.protectMask } : {}), adjustments: { [key]: 30 } };
      const graph = { version: '1.0', operations: [{ id: `person-${personIndex}-${part.id}`, type: 'body.adjust', enabled: true, params }] };
      const filename = `person-${personIndex}-${part.id}.json`;
      await fs.writeFile(path.join(output, filename), JSON.stringify(graph));
      graphs.push({ part: part.id, key, filename, hasPartMask: Boolean(part.partMask) });
    }
  }
  const sha = value => crypto.createHash('sha256').update(value).digest('hex');
  const report = { input: path.relative(root, input), sourceSha256: sha(bytes),
    canonicalSha256: sha(await fs.readFile(path.join(output, 'source.png'))), ...captured, graphs, errors,
    scope: 'Fresh production browser source-only detector at original resolution. No edited-output detection, handcrafted body parts or forced opening of unavailable parts. Detection time includes initialization and concurrent development load.' };
  await fs.writeFile(path.join(output, 'localization.json'), JSON.stringify(report));
  console.log(JSON.stringify({ width: captured.width, height: captured.height, people: captured.people.length, graphs, errors, detectionMs: captured.detectionMs }));
} finally { await browser.close(); }

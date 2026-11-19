"""Ordinary native replay of one freshly observed original body photograph.

The input directory must be produced by detect_source.mjs. No old source
arrays, edited-output detections, target masks, or model calls are used here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sample_mask(mask, width, height):
    a = np.asarray(mask['data'], float).reshape(mask['height'], mask['width']) / 255
    yy, xx = np.mgrid[:height, :width]
    x = np.clip((xx + .5) * mask['width'] / width - .5, 0, mask['width'] - 1)
    y = np.clip((yy + .5) * mask['height'] / height - .5, 0, mask['height'] - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, mask['width'] - 1), np.minimum(y0 + 1, mask['height'] - 1)
    dx, dy = x - x0, y - y0
    return a[y0, x0] * (1 - dx) * (1 - dy) + a[y0, x1] * dx * (1 - dy) + a[y1, x0] * (1 - dx) * dy + a[y1, x1] * dx * dy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output-name', default='ordinary')
    parser.add_argument('--graph', type=Path, help='Source-only ownership/support candidate from rebuild_graph.mjs')
    args = parser.parse_args()
    directory = args.directory.resolve()
    report = json.loads((directory / 'report.json').read_text())
    scope_path=directory.parent/'source-scope.json'
    scope=json.loads(scope_path.read_text())
    if sha(scope_path)!=report['sourceScopeSHA256'] or scope['sourceSHA256']!=report['sourceSHA256']:
        raise ValueError('original-image review scope changed')
    if report['status'] != 'completed' or report['historicalPreparedInput']:
        raise ValueError('fresh completed source observation required')
    if sha(directory / 'source.png') != report['canonicalSourceSHA256'] or sha(directory / 'source-graph.json') != report['graphSHA256']:
        raise ValueError('fresh source or graph changed')
    graph = json.loads((directory / 'source-graph.json').read_text())
    candidate_evidence=None
    if args.graph:
        candidate_path=args.graph.resolve()
        candidate=json.loads(candidate_path.read_text())
        evidence_path=candidate_path.parent/'report.json'
        candidate_evidence=json.loads(evidence_path.read_text())
        if candidate_evidence.get('status')!='completed' or candidate_evidence.get('candidateSourceBound') is not True or candidate_evidence.get('sourceSHA256')!=report['canonicalSourceSHA256'] or candidate_evidence.get('originalGraphSHA256')!=report['graphSHA256'] or candidate_evidence.get('graphSHA256')!=sha(candidate_path):
            raise ValueError('candidate geometry must be bound to the original source/graph')
        # Nothing except source material ownership and its remeasured support
        # may change. Pose, original protection, source and amount stay fixed.
        restored=json.loads(json.dumps(candidate))
        for key in ['partMask','region']:
            restored['operations'][0]['params'][key]=graph['operations'][0]['params'][key]
        restored['operations'][0]['params']['mask']['points']=graph['operations'][0]['params']['mask']['points']
        if restored!=graph:raise ValueError('candidate changed fields beyond part ownership/support')
        graph=candidate
    p = graph['operations'][0]['params']
    source = np.asarray(Image.open(directory / 'source.png').convert('RGBA'))
    h, w = source.shape[:2]
    renderer = ROOT / 'core/image-core/target/release/image-render'
    renderer_sha = sha(renderer)
    if not args.output_name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.output_name):
        raise ValueError('output-name must contain lowercase letters, digits, or hyphens')
    output = directory / args.output_name
    output.mkdir(exist_ok=False)
    # Freeze the source-only visual audit scope before rendering any edit.
    protocol = dict(source=report['source'], originalSourceSHA256=report['sourceSHA256'],
        canonicalSourceSHA256=report['canonicalSourceSHA256'], graphSHA256=report['graphSHA256'],
        dimensions=[w, h], target=scope['target'],
        amounts=[0, -30, 30], key='waist', reviewCrop=scope['reviewCrop'],
        sourceOcclusions=scope['sourceOcclusions'],sourceScopeSHA256=sha(scope_path),
        partMaskMeaning='automatic garment/body ownership, not optical alpha truth',
        successRequires=scope['successRequires'],
        noHistoricalPreparedInputs=True, detectorRuns=1, outputDetectionRuns=0,
        resized=False, renderer=str(renderer), rendererSHA256=renderer_sha,
        bodyImplementationSHA256=sha(ROOT / 'core/image-core/src/body.rs'),
        scriptSHA256=sha(__file__),candidateGraphSHA256=sha(args.graph) if args.graph else None,
        candidateGeometryEvidence=candidate_evidence)
    write(output / 'protocol.json', protocol)
    part = sample_mask(p['partMask'], w, h)
    protected = sample_mask(p['protectMask'], w, h)
    overlays = []
    for mask, color, label in [(part, [30, 230, 80], 'automatic target ownership'), (protected, [245, 30, 220], 'automatic protected material')]:
        rgb = source[..., :3].astype(float)
        blend = .55 * mask[..., None]
        rgb = np.rint(rgb * (1 - blend) + np.asarray(color) * blend).astype(np.uint8)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        points = [(pt['x'] * w, pt['y'] * h) for pt in p['mask']['points']]
        draw.line(points + [points[0]], fill=(20, 130, 255), width=2)
        draw.line([(pt['x'] * w, pt['y'] * h) for pt in p['landmarks']], fill=(255, 220, 20), width=2)
        draw.text((12, 12), label, fill='white', stroke_width=2, stroke_fill='black')
        image.save(output / ('source-target.png' if len(overlays) == 0 else 'source-protection.png'))
        overlays.append(image)
    results = []
    images = {}
    for amount in protocol['amounts']:
        current = json.loads(json.dumps(graph))
        current['operations'][0]['params']['adjustments'] = {'waist': amount}
        stem = f'{amount:+d}'
        graph_file, image_file = output / f'{stem}.json', output / f'{stem}.png'
        write(graph_file, current)
        if sha(renderer) != renderer_sha:
            raise ValueError('native renderer changed during source replay')
        command = [str(renderer), '--input', str(directory / 'source.png'), '--graph', str(graph_file), '--output', str(image_file)]
        started = time.monotonic()
        run = subprocess.run(command, text=True, capture_output=True)
        row = dict(amount=amount, graphSHA256=sha(graph_file), milliseconds=(time.monotonic() - started) * 1000,
            command=command, exitCode=run.returncode, stdout=run.stdout, stderr=run.stderr)
        if run.returncode == 0:
            actual = np.asarray(Image.open(image_file).convert('RGBA'))
            changed = np.any(actual != source, axis=-1)
            row.update(output=str(image_file), outputSHA256=sha(image_file),
                rgbaSHA256=hashlib.sha256(actual.tobytes()).hexdigest(),
                identityExact=bool(np.array_equal(actual, source)), changedPixels=int(changed.sum()),
                protectedPixels=int((protected > 0).sum()), changedProtectedPixels=int((changed & (protected > 0)).sum()),
                alphaUnchanged=bool(np.array_equal(actual[..., 3], source[..., 3])))
            row['sourceFrozenProbes']={}
            for name,box in {**scope['fixedMaterialProbeBoxes'],**scope['sceneProbeBoxes']}.items():
                x0,y0,x1,y1=box;diff=np.abs(actual[y0:y1,x0:x1,:3].astype(int)-source[y0:y1,x0:x1,:3].astype(int))
                row['sourceFrozenProbes'][name]={'box':box,'changedPixels':int(np.any(diff>0,axis=-1).sum()),'maxByte':int(diff.max())}
            images[amount] = Image.fromarray(actual[..., :3])
        results.append(row)
    write(output / 'report.json', dict(protocol=protocol, results=results,
        timingScope='one native process including decode and encode; not an algorithm-only benchmark',
        naturalnessAccepted=False))
    if any(row['exitCode'] != 0 for row in results):
        raise RuntimeError('native replay rejected input: '+next(row['stderr'].strip() for row in results if row['exitCode'] != 0))
    if len(images) == 3:
        x0, y0, x1, y1 = protocol['reviewCrop']
        cw, ch = x1 - x0, y1 - y0
        sheet = Image.new('RGB', (cw * 3, ch + 30), '#222')
        for index, amount in enumerate([0, -30, 30]):
            sheet.paste(images[amount].crop((x0, y0, x1, y1)), (index * cw, 30))
            ImageDraw.Draw(sheet).text((index * cw + 8, 8), f'waist {amount:+d}', fill='white')
        sheet.save(output / 'waist-native-triptych.png')
        for name,box in scope['crops'].items():
            cw,ch=box[2]-box[0],box[3]-box[1]
            sheet=Image.new('RGB',(cw*3,ch+25),'#222');draw=ImageDraw.Draw(sheet)
            for index,amount in enumerate([0,-30,30]):
                sheet.paste(images[amount].crop(box),(index*cw,25));draw.text((index*cw+5,5),f'{amount:+d}',fill='white')
            sheet.save(output/(name+'-triptych.png'))
    print(json.dumps(dict(directory=str(output), results=[{k:v for k,v in r.items() if k not in ['command','stdout','stderr']} for r in results])))


if __name__ == '__main__':
    main()

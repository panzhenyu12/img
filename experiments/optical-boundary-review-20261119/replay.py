"""Compare optical-boundary hypotheses through the ordinary Rust face API.

Old/new official model estimates are both unquantized and use the same source,
geometry and motion evidence. The prior production replay is a fourth display
reference, not falsely labelled a pure-alpha ablation.
"""
from pathlib import Path
import argparse
import copy
import json
import subprocess
import hashlib

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return np.asarray(Image.open(path).convert('RGBA'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sheet(path, panels, box):
    x, y, r, b = box
    w, h = r-x, b-y
    canvas = Image.new('RGB', (w*len(panels), h+24), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (name, pixels) in enumerate(panels):
        canvas.paste(Image.fromarray(pixels).crop(box).convert('RGB'), (i*w, 24))
        draw.text((i*w+3, 5), name, fill='black')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-case', required=True,
                        choices=['portrait-614810','portrait-762020'])
    parser.add_argument('--extended-scene', type=Path, required=True,
                        help='Full-canvas, real guard-band neural prediction')
    args = parser.parse_args()
    base = ROOT/'test/results/face-generalization-20261118'/args.source_case
    out = base/'optical-boundary-review-20261119'
    spec = json.loads((base/'ordinary-rlib-spec.json').read_text())
    parent = spec['cases'][0]
    guard = base/'context-guard-valid-20261119'
    source = read(parent['source'])
    report = json.loads((out/'report.json').read_text())
    x, y, r, b = report['sourceCropBox']
    old = base/'source-preparation/head-proposal/component-1/inference/alpha-official.npy'
    alpha = json.loads(Path(parent['physicalAlpha']).read_text())
    for name, path in [('old', old), ('new', out/'alpha-official.npy')]:
        a = np.zeros(source.shape[:2], np.float32)
        a[y:b, x:r] = np.load(path)
        value = copy.deepcopy(alpha)
        value['data'] = (a*255.0).ravel().tolist()
        dump(out/f'{name}-physical-alpha.json', value)
    descriptor = json.loads((guard/'guard4-descriptor.json').read_text())
    descriptor['image'] = str(args.extended_scene.resolve())
    dump(out/'extended-descriptor.json', descriptor)
    arms = [('old-matte', 'old', guard/'guard4-descriptor.json'),
            ('new-matte-only', 'new', guard/'guard4-descriptor.json'),
            ('new-pair', 'new', out/'extended-descriptor.json')]
    cases = []
    for name, estimate, desc in arms:
        c = copy.deepcopy(parent)
        c.update(id=name, amounts=[0,30,60,-30,-60],
                 physicalAlpha=str(out/f'{estimate}-physical-alpha.json'),
                 sourcePlatePrior=str(desc))
        cases.append(c)
    spec.update(cases=cases, outputDir=str(out/'replay'))
    dump(out/'spec.json', spec)
    binary = ROOT/'core/image-core/target/release/examples/face-source-replay'
    command = [str(binary), str(out/'spec.json')]
    execution = {'command': command, 'binarySHA256': sha(binary),
                 'unchangedGeometryAndMotion': {name: sha(parent[name]) for name in
                  ['source','graph','sourceMotionInterface','sourceMovingAttachmentSupport']},
                 'alphaVariants': {name: sha(path) for name, path in [('old',old),('new',out/'alpha-official.npy')]},
                 'extendedSceneSHA256': sha(args.extended_scene),
                 'status': 'running', 'cases': []}
    dump(out/'replay-report.json', execution)
    with (out/'replay.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('public replay failed; see replay.log')
    boxes = {'whole-source':(0,0,source.shape[1],source.shape[0])}
    if args.source_case == 'portrait-614810':
        boxes.update({'left-edge':(265,280,390,695),'right-edge':(650,280,790,695)})
    else:
        boxes.update({'left-edge':(340,180,460,515),'right-edge':(640,180,770,515)})
    for i, amount in enumerate([0,30,60,-30,-60]):
        token = str(amount) if amount>=0 else 'minus'+str(-amount)
        previous = read(guard/f'replay/guard4-{i}-{token}-candidate.png')
        images = {name: read(out/f'replay/{name}-{i}-{token}-candidate.png') for name,_,_ in arms}
        row = {'amount': amount, 'zeroExact': all(np.array_equal(a,source) for a in images.values()) if amount==0 else None,
               'points': [], 'oldOfficialVsPriorPairedChangedPixels': int(np.any(images['old-matte']!=previous,2).sum())}
        if args.source_case == 'portrait-614810':
            for xx,yy in [(300,571),(302,571),(304,571),(302,580),(318,580),(688,580)]:
                row['points'].append({'xy':[xx,yy], 'source':source[yy,xx].tolist(),
                    'previousGuard':previous[yy,xx].tolist(), **{n:a[yy,xx].tolist() for n,a in images.items()}})
        execution['cases'].append(row)
        if amount:
            for name,box in boxes.items():
                sheet(out/f'{name}-{amount:+}.png', [('source',source),('previous paired',previous)]+list(images.items()),box)
    execution['status']='complete; independent visual review required'
    dump(out/'replay-report.json', execution)
    print(json.dumps(execution,indent=2))


if __name__ == '__main__':
    main()

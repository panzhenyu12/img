"""Source-bound manual positive/negative SAM2 proposals, with a frozen protocol.

The model invocation is extracted from the existing frozen probe.py. That
historical script stays byte-identical; this entry has no fixture globals.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'experiments/sam2matting-tiny-probe-20261015'
WEIGHTS = ROOT / '.models/sam2matting-20261015/SAM2Matting-SAM2.1Tiny.pt'
CONFIG = 'configs/sam2matting-sam2.1tiny.yaml'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_queries(request, size, excluded_boxes):
    if request.get('manualCorrection') is not True:
        raise ValueError('manualCorrection must be explicit')
    width, height = size
    queries = request.get('queries', [])
    if not queries or len(queries) > 3:
        raise ValueError('one to three frozen role queries required')
    ids = set()
    for query in queries:
        name = query['id']
        if not re.fullmatch(r'[a-z][a-z0-9-]*', name) or name in ids:
            raise ValueError('unique safe role id required')
        ids.add(name)
        seen = set()
        for label in ('positive', 'negative'):
            points = query[label]
            if not 1 <= len(points) <= 2:
                raise ValueError('each role has at most two positive and two negative points')
            for point in points:
                if len(point) != 2 or not all(isinstance(value, int) for value in point):
                    raise ValueError('integer source pixel coordinates required')
                x, y = point
                if not 0 <= x < width or not 0 <= y < height:
                    raise ValueError('prompt outside source')
                if tuple(point) in seen:
                    raise ValueError('duplicate or conflicting point labels')
                seen.add(tuple(point))
                if any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in excluded_boxes):
                    raise ValueError('prompt may not use frozen evaluation core')
    return queries


def prepare(source, request_path, scope_path, output):
    source, request_path, scope_path = [path.resolve() for path in (source, request_path, scope_path)]
    request = json.loads(request_path.read_text())
    scope = json.loads(scope_path.read_text())
    image = Image.open(source).convert('RGB')
    if request['sourceSHA256'] != sha(source) or list(image.size) != scope['dimensions']:
        raise ValueError('source binding differs')
    original = Path(scope['source'])
    if sha(original) != scope['sourceSHA256']:
        raise ValueError('evaluation scope original source changed')
    observation_report = source.parent / 'report.json'
    observation = json.loads(observation_report.read_text())
    if observation['sourceSHA256'] != scope['sourceSHA256'] or observation['canonicalSourceSHA256'] != sha(source):
        raise ValueError('canonical image does not match the frozen original observation')
    boxes = list(scope['fixedMaterialProbeBoxes'].values()) + list(scope['sceneProbeBoxes'].values())
    queries = validate_queries(request, image.size, boxes)
    files = {'source': source, 'originalSource': original, 'observationReport': observation_report, 'manualRequest': request_path,
             'evaluationExclusion': scope_path, 'checkpoint': WEIGHTS, 'script': Path(__file__),
             'predictor': RUNTIME / 'vendor/sam2/sam2_image_predictor.py',
             'config': RUNTIME / 'vendor/sam2' / CONFIG}
    protocol = {'status': 'frozen_before_inference', 'manualCorrection': True,
                'automaticGeneralizationClaim': False, 'sourceObservation': request['sourceObservation'],
                'pointSelection': 'visible source RGB interiors; no prior masks or edited images used as point inputs',
                'priorExposure': 'Researcher has seen earlier failures; this is explicit manual correction, not a blind test.',
                'files': {key: {'path': str(path), 'sha256': sha(path)} for key, path in files.items()},
                'dimensions': list(image.size), 'queries': queries,
                'reviewCrop': request['reviewCrop'],
                'rules': {'imageEncodings': 1, 'allMultimasks': True, 'alphaHeadCalls': 0,
                          'maskThreshold': 0, 'postfilter': None, 'additionalQueries': 0,
                          'candidateScoreSelection': False},
                'evaluationUse': 'Only exclude prompt overlap; no evaluation pixel or mask enters model.',
                'acceptance': 'Independent full visible sleeve/cuff and torso review; correct point labels alone do not establish complete material identity.'}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    # Plot the first reciprocal query once to avoid duplicate labels.
    for label, color in [('positive', 'cyan'), ('negative', 'yellow')]:
        for index, point in enumerate(queries[0][label]):
            x, y = point
            draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline='black')
            draw.text((x+8, y-8), f'{label}{index+1}', fill=color, stroke_width=1, stroke_fill='black')
    preview.save(output / 'source-points.png')
    preview.crop(tuple(request['reviewCrop'])).save(output / 'source-points-detail.png')
    print(json.dumps({'protocol': str(output / 'protocol.json'), 'sha256': sha(output / 'protocol.json'), 'manualCorrection': True}), flush=True)


def run(output):
    if (output / 'run.json').exists():
        raise ValueError('refusing a second run')
    protocol = json.loads((output / 'protocol.json').read_text())
    for entry in protocol['files'].values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('frozen input changed: ' + entry['path'])
    sys.path[:0] = [str(RUNTIME / 'deps'), str(RUNTIME / 'vendor')]
    import torch
    from sam2.build_sam import build_sam2matting
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    report = {'status': 'running', 'manualCorrection': True, 'automaticGeneralizationClaim': False,
              'protocolSHA256': sha(output / 'protocol.json'), 'torch': torch.__version__,
              'device': 'cpu', 'imageEncodings': 0, 'alphaHeadCalls': 0, 'candidates': []}
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    model = build_sam2matting(CONFIG, str(WEIGHTS), device='cpu')
    original_forward = model.forward_image
    def encode(*args, **kwargs):
        report['imageEncodings'] += 1
        assert report['imageEncodings'] == 1
        return original_forward(*args, **kwargs)
    def forbidden(*args, **kwargs):
        report['alphaHeadCalls'] += 1
        raise AssertionError('No alpha head in this diagnostic')
    model.forward_image = encode
    model._forward_alpha_heads = forbidden
    predictor = SAM2ImagePredictor(model, mask_threshold=0, max_hole_area=0, max_sprinkle_area=0)
    source = Image.open(protocol['files']['source']['path']).convert('RGB')
    rgb = np.array(source)
    all_masks, all_logits, all_scores = [], [], []
    with torch.inference_mode():
        start = time.perf_counter()
        predictor.set_image(source)
        report['encodeSeconds'] = time.perf_counter() - start
        for query in protocol['queries']:
            start = time.perf_counter()
            points = query['positive'] + query['negative']
            labels = [1] * len(query['positive']) + [0] * len(query['negative'])
            masks, scores, logits = predictor.predict(point_coords=np.asarray(points, np.float32),
                point_labels=np.asarray(labels, np.int32), multimask_output=True, return_logits=False)
            seconds = time.perf_counter() - start
            assert masks.shape == (3, source.height, source.width)
            assert np.isfinite(logits).all() and np.isfinite(scores).all()
            for index in range(3):
                name = f'{query["id"]}-m{index}'
                mask = masks[index].astype(bool)
                Image.fromarray(mask.astype(np.uint8)*255).save(output / f'{name}-mask.png')
                preview = np.where(mask[..., None], rgb, np.rint(rgb*.2)).astype(np.uint8)
                Image.fromarray(preview).save(output / f'{name}-preview.png')
                Image.fromarray(preview).crop(tuple(protocol['reviewCrop'])).save(output / f'{name}-detail.png')
                all_masks.append(np.packbits(mask.reshape(-1)))
                all_logits.append(logits[index])
                all_scores.append(scores[index])
                report['candidates'].append({'id': name, 'query': query['id'], 'maskIndex': index,
                    'predictedIOUNotTruth': float(scores[index]), 'pixels': int(mask.sum()),
                    'queryDecodeSeconds': seconds, 'maskSHA256': sha(output / f'{name}-mask.png')})
            print(query['id'], seconds, flush=True)
    np.savez_compressed(output / 'candidates.npz', masksPacked=np.stack(all_masks),
        lowResolutionLogits=np.stack(all_logits), predictedIOU=np.asarray(all_scores),
        imageShape=[source.height, source.width])
    assert report['imageEncodings'] == 1 and report['alphaHeadCalls'] == 0
    report.update(status='completed_independent_review_pending', candidateBankSHA256=sha(output / 'candidates.npz'))
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'candidates'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    create = sub.add_parser('prepare')
    for name in ('source', 'request', 'scope', 'output'):
        create.add_argument(name, type=Path)
    execute = sub.add_parser('run')
    execute.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.source, args.request, args.scope, args.output)
    else:
        run(args.output)

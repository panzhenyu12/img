#!/usr/bin/env python3
"""Automatic original-image ZIM hair/accessory edge audit; no graph replacement.

Both prompt protocols are fixed before inference. Every official candidate is kept;
each protocol is reviewed using the official score argmax, never a visual choice.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cached_zim_probe', ROOT / 'experiments/zim-20261014/run.py')
zim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zim)
np, cv2, Image, ImageDraw = zim.np, zim.cv2, zim.Image, zim.ImageDraw

SOURCES = {
    'portrait-woman': (
        'experiments/face-anatomical-v22-20261015/inputs/portrait-woman-source.png',
        'test/results/face-core-fresh/portrait-woman-faceWidth-60.json'),
    'user-face-contour': (
        'test/results/user-face-contour-20261014/1024/canonical-source.png',
        'test/results/user-face-contour-20261014/1024/face-graph.json'),
    'portrait-hair': (
        'test/results/face-core-review/portrait-hair-source.png',
        'test/results/face-core-review/portrait-hair-faceWidth-60.json'),
}


def png(path, alpha):
    Image.fromarray(np.rint(np.clip(alpha, 0, 1) * 255).astype('uint8')).save(path)


def face_crop(params, width, height):
    pts = np.array([[p['x'] * width, p['y'] * height] for p in params['landmarks']])
    low, high = pts.min(axis=0), pts.max(axis=0)
    delta = high - low
    return (max(0, int(low[0] - .22 * delta[0])), max(0, int(low[1] - .15 * delta[1])),
            min(width, int(high[0] + .22 * delta[0])), min(height, int(high[1] + .30 * delta[1])))


def deepest(binary):
    d = cv2.distanceTransform(binary.astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return [int(x), int(y)], float(d[y, x])


def prompts_from_components(confidence, skin, kind, chin):
    """Same component box + interior distance-maximum rule as accessory-v1.

    Hair uses the largest head-overlapping components; its centroid can lie below
    the chin for long hair. This is a semantic rule, not a per-photo crop/window.
    """
    h, w = confidence.shape
    count, labels, stats, centers = cv2.connectedComponentsWithStats((confidence >= .5).astype('uint8'))
    components = [i for i in range(1, count) if stats[i, 4] > 30
                  and (centers[i][1] < chin if kind == 'accessories' else stats[i, 1] < chin)]
    components.sort(key=lambda i: int(stats[i, 4]), reverse=True)
    result = []
    for i in components[:2]:
        x, y, cw, ch, area = map(int, stats[i])
        box = [max(0, x - 4), max(0, y - 4), min(w - 1, x + cw + 3), min(h - 1, y + ch + 3)]
        positive, depth = deepest(labels == i)
        result.append(dict(kind=kind, component=int(i), box=box, positive=positive,
                           positiveInteriorDistancePx=depth, sourceReferenceArea=area,
                           protocol='component-box-positive', negative=[]))
        if kind == 'hair':
            valid = (skin >= .85) & (confidence < .05)
            inside = np.zeros_like(valid)
            inside[box[1]:box[3] + 1, box[0]:box[2] + 1] = True
            negative, negative_depth = deepest(valid & inside)
            if negative_depth > 1:
                result.append({**result[-1], 'protocol': 'component-box-positive-skin-negative',
                               'negative': [negative], 'negativeInteriorDistancePx': negative_depth})
    return result, labels


def cutout(rgb, alpha):
    bg = np.array([45, 80, 60], dtype=np.float32)
    return Image.fromarray(np.rint(rgb * alpha[..., None] + bg * (1 - alpha[..., None])).clip(0, 255).astype('uint8'))


def make_panel(path, rgb, masks, alphas, scores, prompt, crop):
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    sheet = Image.new('RGB', (cw * 6, ch + 28), '#20242a')
    draw = ImageDraw.Draw(sheet)
    panels = [('Original', Image.fromarray(rgb)), ('Parser ' + prompt['kind'], cutout(rgb, masks[prompt['kind']]))]
    panels += [(f'{i}: {scores[i]:.3f}' + (' OFFICIAL' if i == int(np.argmax(scores)) else ''), cutout(rgb, a))
               for i, a in enumerate(alphas)]
    for i, (label, im) in enumerate(panels):
        sheet.paste(im.crop(crop), (cw * i, 28))
        draw.text((cw * i + 4, 8), label, fill='white')
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', nargs='+', default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument('--out', type=Path, default=HERE / 'outputs')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    Encoder, Decoder, Zim, Predictor, vendor = zim.load_vendor()
    weights = ROOT / '.models/zim-20261014'
    digests = {'encoder.onnx': '6123ea10722dd43a0f8bbb96cd6a0532d8a65e2b1e8339f0cd6593d494807809',
               'decoder.onnx': 'a99d14bac3dc28b17809020b1711b1991bd5b7eb18f678c3624ce508748cfa11'}
    for file, digest in digests.items():
        assert zim.sha(weights / file) == digest, file
    zim.torch.set_num_threads(3)
    start = time.perf_counter()
    predictor = Predictor(Zim(Encoder(str(weights / 'encoder.onnx'), num_threads=3),
                              Decoder(str(weights / 'decoder.onnx'), num_threads=3)))
    report = dict(scope='Original RGB only; no output redetection, graph changes, warp, or production integration. '
                        'Raw ZIM alpha is a candidate, not semantic/physical ground truth.',
                  runnerSha256=zim.sha(Path(__file__)), vendorCommit=vendor['commit'],
                  weights=digests, loadMs=(time.perf_counter() - start) * 1000, images=[])
    for name in args.names:
        source, graph_path = [ROOT / p for p in SOURCES[name]]
        rgb = np.asarray(Image.open(source).convert('RGB'))
        h, w = rgb.shape[:2]
        params = json.loads(graph_path.read_text())['operations'][0]['params']
        masks = {k: zim.coverage(v, w, h) for k, v in params['semanticMasks'].items()}
        folder = args.out / name
        folder.mkdir()
        crop = face_crop(params, w, h)
        chin = max(p['y'] * h for p in params['landmarks'])
        prompts = []
        for kind in ['hair', 'accessories']:
            png(folder / f'parser-{kind}.png', masks[kind])
            selected, _ = prompts_from_components(masks[kind], masks['skin'], kind, chin)
            prompts += selected
        (folder / 'prompts.json').write_text(json.dumps(prompts, indent=2) + '\n')
        start = time.perf_counter()
        predictor.set_image(rgb)
        row = dict(name=name, source=str(source.relative_to(ROOT)), sourceSha256=zim.sha(source),
                   graph=str(graph_path.relative_to(ROOT)), graphSha256=zim.sha(graph_path), size=[w, h],
                   faceCrop=crop, encodeMs=(time.perf_counter() - start) * 1000, rows=[])
        for prompt in prompts:
            case = folder / f"{prompt['kind']}-{prompt['component']}-{prompt['protocol']}"
            case.mkdir()
            coords = [prompt['positive']] + prompt['negative']
            labels = [1] + [0] * len(prompt['negative'])
            start = time.perf_counter()
            alphas, scores, _ = predictor.predict(box=np.array(prompt['box'], dtype=np.float32),
                point_coords=np.array(coords, dtype=np.float32), point_labels=np.array(labels, dtype=np.float32),
                multimask_output=True)
            elapsed = (time.perf_counter() - start) * 1000
            assert alphas.shape == (4, h, w) and np.isfinite(alphas).all()
            np.save(case / 'alphas.npy', alphas, allow_pickle=False)
            best = int(np.argmax(scores))
            summaries = []
            skin_core = (masks['skin'] >= .9) & (masks['hair'] < .05) & (masks['accessories'] < .05)
            parser_core = masks[prompt['kind']] >= .9
            for i, alpha in enumerate(alphas):
                png(case / f'alpha-{i}.png', alpha)
                cutout(rgb, alpha).crop(crop).save(case / f'face-cutout-{i}-1x.png')
                summaries.append(dict(index=i, score=float(scores[i]), foregroundPixels=int((alpha >= .5).sum()),
                    positiveAlpha=float(alpha[prompt['positive'][1], prompt['positive'][0]]),
                    negativeAlpha=[float(alpha[y, x]) for x, y in prompt['negative']],
                    parserCoreRetainedFraction=float((alpha[parser_core] >= .5).mean()) if parser_core.any() else None,
                    skinCoreRetainedFraction=float((alpha[skin_core] >= .5).mean()) if skin_core.any() else None))
            make_panel(case / 'face-comparison-1x.png', rgb, masks, alphas, scores, prompt, crop)
            box = prompt['box']
            if prompt['kind'] == 'accessories':
                accessory_crop = (max(0, box[0] - 12), max(0, box[1] - 12), min(w, box[2] + 13), min(h, box[3] + 13))
                make_panel(case / 'component-comparison-1x.png', rgb, masks, alphas, scores, prompt, accessory_crop)
            summary = dict(prompt=prompt, directory=str(case.relative_to(args.out)), decodeMs=elapsed,
                           officialScoreArgmax=best, alphas=summaries, alphaSha256=zim.sha(case / 'alphas.npy'))
            row['rows'].append(summary)
            print(json.dumps(dict(name=name, protocol=prompt['protocol'], kind=prompt['kind'],
                                 component=prompt['component'], decodeMs=elapsed, official=best,
                                 selected=summaries[best])), flush=True)
        report['images'].append(row)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('OUTPUT', args.out, flush=True)


if __name__ == '__main__':
    main()

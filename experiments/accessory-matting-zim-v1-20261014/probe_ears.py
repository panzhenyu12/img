"""Source-only physical ear alpha probe; never infer from a warped image."""
from pathlib import Path
import importlib.util
import json
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('zim_probe', ROOT / 'experiments/zim-20261014/run.py')
zim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zim)
np, cv2, Image, ImageDraw = zim.np, zim.cv2, zim.Image, zim.ImageDraw


def main():
    out = ROOT / 'test/results/accessory-matting-zim-v1-20261014/ears-probe'
    out.mkdir(exist_ok=False)
    source = ROOT / 'test/results/user-face-contour-20261014/1024/canonical-source.png'
    graph_path = source.with_name('face-graph.json')
    rgb = np.asarray(Image.open(source).convert('RGB'))
    h, w = rgb.shape[:2]
    params = json.loads(graph_path.read_text())['operations'][0]['params']
    mask = zim.coverage(params['semanticMasks']['ears'], w, h)
    face = zim.coverage(params['semanticMasks']['faceContourMask'], w, h)
    count, labels, stats, centers = cv2.connectedComponentsWithStats((mask >= .5).astype('uint8'))
    components = sorted(range(1, count), key=lambda i: int(stats[i, 4]), reverse=True)[:2]
    Encoder, Decoder, Zim, Predictor, vendor = zim.load_vendor()
    weights = ROOT / '.models/zim-20261014'
    digests = {'encoder.onnx': '6123ea10722dd43a0f8bbb96cd6a0532d8a65e2b1e8339f0cd6593d494807809',
               'decoder.onnx': 'a99d14bac3dc28b17809020b1711b1991bd5b7eb18f678c3624ce508748cfa11'}
    for name, digest in digests.items():
        assert zim.sha(weights / name) == digest
    zim.torch.set_num_threads(3)
    start = time.perf_counter()
    predictor = Predictor(Zim(Encoder(str(weights / 'encoder.onnx'), num_threads=3), Decoder(str(weights / 'decoder.onnx'), num_threads=3)))
    load_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    predictor.set_image(rgb)
    encode_ms = (time.perf_counter() - start) * 1000
    report = dict(scope='Automatic original parser ear boxes/positive interior and nearest confident face negative. All four raw alphas retained; no clipping or graph mutation.',
                  sourceSha256=zim.sha(source), graphSha256=zim.sha(graph_path), runnerSha256=zim.sha(Path(__file__)),
                  vendorCommit=vendor['commit'], weights=digests, loadMs=load_ms, encodeMs=encode_ms, rows=[])
    yy, xx = np.indices((h, w))
    for index in components:
        x, y, cw, ch, area = map(int, stats[index])
        # Padding is relative to parser pitch: 4 px at the 1024 source.
        box = [max(0, x - 4), max(0, y - 4), min(w - 1, x + cw + 3), min(h - 1, y + ch + 3)]
        distance = cv2.distanceTransform((labels == index).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        py, px = np.unravel_index(np.argmax(distance), distance.shape)
        face_inside = cv2.distanceTransform((face >= .85).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) > 8
        candidates = ((xx - px) ** 2 + (yy - py) ** 2).astype(float)
        candidates[~face_inside] = np.inf
        ny, nx = np.unravel_index(np.argmin(candidates), candidates.shape)
        points = [[int(px), int(py)], [int(nx), int(ny)]]
        start = time.perf_counter()
        alphas, scores, _ = predictor.predict(box=np.array(box, dtype=np.float32), point_coords=np.array(points, dtype=np.float32), point_labels=np.array([1, 0], dtype=np.float32), multimask_output=True)
        elapsed = (time.perf_counter() - start) * 1000
        folder = out / f'component-{index}'
        folder.mkdir()
        np.save(folder / 'alphas.npy', alphas, allow_pickle=False)
        crop = (max(0, x - 15), max(0, y - 15), min(w, x + cw + 15), min(h, y + ch + 15))
        pw, ph = crop[2] - crop[0], crop[3] - crop[1]
        sheet = Image.new('RGB', (pw * 6, ph + 24), '#222')
        draw = ImageDraw.Draw(sheet)
        for k, (label, alpha) in enumerate([('Original', np.ones_like(mask)), ('Parser', mask)] + [(f'{i}: {scores[i]:.3f}', a) for i, a in enumerate(alphas)]):
            pixels = np.rint(rgb * alpha[..., None] + np.array([45, 80, 60]) * (1 - alpha[..., None])).clip(0, 255).astype('uint8')
            sheet.paste(Image.fromarray(pixels).crop(crop), (k * pw, 24))
            draw.text((k * pw + 3, 5), label, fill='white')
        sheet.save(folder / 'comparison-1x.png')
        sheet.resize((sheet.width * 2, sheet.height * 2), Image.Resampling.NEAREST).save(folder / 'comparison-2x.png')
        row = dict(component=index, box=box, points=points, pointLabels=[1, 0], sourceArea=area, scores=scores.tolist(), officialScoreArgmax=int(np.argmax(scores)), decodeMs=elapsed)
        report['rows'].append(row)
        print(json.dumps(row), flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

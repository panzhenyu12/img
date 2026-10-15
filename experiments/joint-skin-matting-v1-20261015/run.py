"""Direct connected face/neck matte, independently prompted on source RGB.

Unlike accessory subtraction, a low accessory score is never labelled skin.
Prompts come from source semantic interiors. Raw candidates are all retained;
the official score argmax is reported rather than hand choosing an output.
This is an isolated hypothesis, not a production mask replacement.
"""
from pathlib import Path
import importlib.util
import json
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load('skin_probe', ROOT / 'experiments/hair-accessory-matting-zim-v2-20261015/run.py')
adapter = load('skin_adapter', ROOT / 'experiments/zim-combined-prompts-20261015/adapter.py')
zim, np, cv2, Image, ImageDraw = probe.zim, probe.np, probe.cv2, probe.Image, probe.ImageDraw


def component_points(mask, max_components=4):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    ids = sorted(range(1, count), key=lambda i: int(stats[i, 4]), reverse=True)
    result = []
    for i in ids[:max_components]:
        if stats[i, 4] < 12:
            continue
        point, distance = probe.deepest(labels == i)
        if distance >= 2:
            result.append(point)
    return result


def prompts(params, masks, w, h):
    pts = np.array([[p['x']*w, p['y']*h] for p in params['landmarks']])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = hi-lo
    box = [max(0, int(lo[0]-.22*size[0])), max(0, int(lo[1]-.12*size[1])),
           min(w, int(hi[0]+.22*size[0])), min(h, int(hi[1]+.34*size[1]))]
    roi = np.zeros((h, w), bool)
    roi[box[1]:box[3], box[0]:box[2]] = True
    fixed = np.maximum.reduce([masks[k] for k in ['hair', 'ears', 'accessories']])
    core = (masks['skin'] >= .9) & (fixed < .01) & roi
    # Spatially distributed interior samples within the source face and neck.
    positive = []
    for ya, yb in zip(np.linspace(box[1], box[3], 5, dtype=int)[:-1], np.linspace(box[1], box[3], 5, dtype=int)[1:]):
        for xa, xb in zip(np.linspace(box[0], box[2], 4, dtype=int)[:-1], np.linspace(box[0], box[2], 4, dtype=int)[1:]):
            tile = np.zeros_like(core)
            tile[ya:yb, xa:xb] = True
            positive += component_points(core & tile, 1)
    negative = []
    for key in ['hair', 'ears', 'accessories']:
        negative += component_points((masks[key] >= .9) & roi, 2)
    # Skin excludes eyes, lips and teeth by design. Its complement cannot
    # establish scene ownership. A conservative head/person veto is suitable
    # for negative prompts even though it is not physical matting truth.
    subject = zim.coverage(params['subjectMask'],w,h) if 'subjectMask' in params else np.zeros((h,w))
    occupied = [subject]+[masks[k] for k in ['skin','hair','ears','accessories','neck','face','faceOutline','lips','teeth','nose','leftEye','rightEye','leftBrow','rightBrow'] if k in masks]
    background = roi & (np.maximum.reduce(occupied) < .01)
    negative += component_points(background, 2)
    return box, positive, negative, core


def main(name='user-face-contour', out=None):
    out = out or HERE / 'outputs'
    out.mkdir(exist_ok=False)
    source, graph = [ROOT / p for p in probe.SOURCES[name]]
    rgb = np.asarray(Image.open(source).convert('RGB'))
    h, w = rgb.shape[:2]
    params = json.loads(graph.read_text())['operations'][0]['params']
    masks = {k: zim.coverage(v, w, h) for k, v in params['semanticMasks'].items()}
    box, positive, negative, core = prompts(params, masks, w, h)
    Encoder, Decoder, Zim, Predictor, vendor = zim.load_vendor()
    weights = ROOT / '.models/zim-20261014'
    zim.torch.set_num_threads(3)
    decoder = Decoder(str(weights/'decoder.onnx'), num_threads=3)
    model = Zim(Encoder(str(weights/'encoder.onnx'), num_threads=3), decoder)
    predictor = Predictor(model)
    report = dict(scope='Direct source face+neck alpha; automatic source-only semantic prompts; no accessory-complement-as-skin.',
                  source=str(source.relative_to(ROOT)), sourceSha256=zim.sha(source), graphSha256=zim.sha(graph),
                  scriptSha256=zim.sha(Path(__file__)), vendorCommit=vendor['commit'],
                  box=box, positive=positive, negative=negative, rows=[], accepted=False)
    for mode in ['full', 'crop']:
        crop = [0, 0, w, h] if mode == 'full' else box
        x0, y0, x1, y1 = crop
        start = time.perf_counter()
        predictor.set_image(rgb[y0:y1, x0:x1])
        encode_ms = (time.perf_counter()-start)*1000
        coords = np.asarray(positive+negative, dtype=np.float32)-np.array([x0, y0], dtype=np.float32)
        local_box = np.asarray(box, dtype=np.float32)-np.array([x0, y0, x0, y0], dtype=np.float32)
        for attention in ['point', 'box']:
            model.decoder = adapter.CombinedPromptDecoder(decoder, attention=attention, bbox_attention=model.bbox_attn_mask)
            start = time.perf_counter()
            alphas, scores, _ = predictor.predict(point_coords=coords, point_labels=np.asarray([1]*len(positive)+[0]*len(negative), dtype=np.float32), box=local_box, multimask_output=True)
            decode_ms = (time.perf_counter()-start)*1000
            mapped = np.zeros((4, h, w), dtype=np.float32)
            mapped[:, y0:y1, x0:x1] = alphas
            dest = out / f'{mode}-{attention}'
            dest.mkdir()
            np.save(dest/'alphas-full.npy', mapped, allow_pickle=False)
            official = int(np.argmax(scores))
            items = [('Source', Image.fromarray(rgb)), ('Parser skin', probe.cutout(rgb, masks['skin']))]
            rows = []
            for i, alpha in enumerate(mapped):
                probe.png(dest/f'alpha-{i}.png', alpha)
                items.append((f'{i}: {scores[i]:.3f}'+(' official' if i == official else ''), probe.cutout(rgb, alpha)))
                rows.append(dict(index=i, score=float(scores[i]), positive=[float(alpha[y,x]) for x,y in positive],
                                 negative=[float(alpha[y,x]) for x,y in negative],
                                 coreRetained=float((alpha[core]>.5).mean()),
                                 accessoriesMean=float(alpha[masks['accessories']>.9].mean()) if (masks['accessories']>.9).any() else None))
            panels=[('face',box)]
            if name=='user-face-contour':panels += [('left',[320,400,410,540]),('right',[600,400,700,540])]
            for label, panel_box in panels:
                px0, py0, px1, py1 = panel_box
                pw, ph = px1-px0, py1-py0
                panel = Image.new('RGB', (pw*len(items), ph+24), '#20242a')
                draw = ImageDraw.Draw(panel)
                for i, (title, im) in enumerate(items):
                    panel.paste(im.crop(panel_box), (i*pw, 24))
                    draw.text((i*pw+3, 5), title, fill='white')
                panel.save(dest/f'{label}-1x.png')
            entry = dict(mode=mode, attention=attention, crop=crop, officialScoreArgmax=official, candidates=rows,
                         encodeMs=encode_ms, decodeMs=decode_ms, feed=model.decoder.last_prompt,
                         alphaSha256=zim.sha(dest/'alphas-full.npy'))
            report['rows'].append(entry)
            (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(dict(mode=mode, attention=attention, official=official, scores=scores.tolist())), flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--names',nargs='+',default=['user-face-contour'])
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    for name in args.names:
        main(name,(args.out/name) if args.out else None)

#!/usr/bin/env python3
"""Official ZIM CPU alpha probe on the original user photo, isolated from production.

No edited image is detected. Detection masks are references, not alpha ground truth.
The vendor code is unmodified; only namespace bootstrapping avoids unrelated demo/SAM
dependencies. Encoder/decoder thread counts are set to 3 instead of upstream's 16.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import resource
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.run/zim-deps'))
sys.path.insert(0, str(ROOT / '.run/inpaint4drag-deps'))
sys.path.insert(0, str(ROOT / 'scripts'))
import cv2
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort
import torch
from replay_user_face_contour import raster_coverage


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_vendor():
    vendor = HERE / 'vendor'
    manifest = json.loads((vendor / 'manifest.json').read_text())
    for rel, record in manifest['files'].items():
        assert sha(vendor / rel) == record['sha256'], rel
    # Do not import upstream package __init__, which pulls the unrelated
    # automatic_mask_generator, SAM and matplotlib demo dependencies.
    for name in ['zim_anything', 'zim_anything.modeling', 'zim_anything.utils']:
        module = types.ModuleType(name)
        module.__path__ = [str(vendor / name.replace('.', '/'))]
        sys.modules[name] = module
    def load(name, rel):
        spec = importlib.util.spec_from_file_location(name, vendor / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    utils = load('zim_anything.utils.utils', 'zim_anything/utils/utils.py')
    sys.modules['zim_anything.utils'].ResizeLongestSide = utils.ResizeLongestSide
    encoder = load('zim_anything.modeling.encoder', 'zim_anything/modeling/encoder.py')
    decoder = load('zim_anything.modeling.decoder', 'zim_anything/modeling/decoder.py')
    zim = load('zim_anything.modeling.zim', 'zim_anything/modeling/zim.py')
    predictor = load('zim_anything.predictor', 'zim_anything/predictor.py')
    return encoder.ZIM_Encoder, decoder.ZIM_Decoder, zim.Zim, predictor.ZimPredictor, manifest


def coverage(mask, width, height):
    if mask['shape'] == 'raster':
        return raster_coverage(mask, width, height)
    # Polygon reference is deliberately labelled as a rounded binary interior.
    result = np.zeros((height, width), dtype=np.uint8)
    points = np.rint([[p['x'] * width, p['y'] * height] for p in mask['points']]).astype(np.int32)
    cv2.fillPoly(result, [points], 1)
    return result.astype(np.float32)


def pick(mask, region=None):
    binary = (mask >= .85).astype(np.uint8)
    if region is not None:
        x0, y0, x1, y1 = region
        window = np.zeros_like(binary)
        window[y0:y1, x0:x1] = 1
        binary *= window
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    if distance[y, x] <= 1:
        raise RuntimeError(f'No robust original detection point in region {region}')
    return [int(x), int(y)]


def prompt_sets(masks):
    positives = [('nose', pick(masks['nose']))]
    positives += [(name, pick(masks[name])) for name in ['leftBrow', 'rightBrow', 'lips', 'leftEye', 'rightEye']]
    # These windows are original-image face subregions, never edited-output coordinates.
    positives += [('skin-'+name, pick(masks['skin'], window)) for name, window in [
        ('forehead', (380, 200, 635, 315)),
        ('left-cheek', (360, 390, 460, 505)),
        ('right-cheek', (560, 390, 650, 505)),
        ('chin', (445, 550, 565, 602)),
    ]]
    negatives = [(name+'-'+side, pick(masks[name], window)) for name, side, window in [
        ('hair', 'top', (300, 0, 700, 195)),
        ('hair', 'left', (320, 190, 400, 305)),
        ('hair', 'right', (600, 180, 695, 310)),
        ('ears', 'left', (310, 295, 385, 437)),
        ('ears', 'right', (640, 295, 711, 437)),
        ('neck', 'left', (355, 575, 510, 718)),
        ('neck', 'right', (510, 575, 667, 718)),
    ]]
    y, x = np.nonzero(masks['faceContourMask'] >= .5)
    box = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
    sets = [
        {'name': 'single-face-positive', 'positive': positives[:1], 'negative': []},
        {'name': 'face-anatomy-positives', 'positive': positives, 'negative': []},
        {'name': 'face-signed-prompts', 'positive': positives, 'negative': negatives},
        {'name': 'face-box', 'box': box, 'positive': [], 'negative': []},
    ]
    # Add forbidden-region points near the face sides, retaining original parser provenance.
    dense = negatives + [(name+'-edge-'+str(i), pick(masks[name], window))
        for i, (name, window) in enumerate([
            ('hair', (340, 205, 390, 270)), ('hair', (635, 215, 688, 285)),
            ('neck', (420, 602, 485, 650)), ('neck', (535, 602, 602, 650)),
            ('accessories', (330, 430, 384, 510)), ('accessories', (625, 430, 690, 510)),
        ])]
    sets.append({'name': 'face-dense-signed-prompts', 'positive': positives, 'negative': dense})
    return sets


def metrics(alpha, masks, references, prompts):
    foreground = alpha >= .5
    values = {'finite': bool(np.isfinite(alpha).all()), 'min': float(alpha.min()), 'max': float(alpha.max()),
              'foreground_pixels': int(foreground.sum()), 'fractional_alpha_pixels_01_99': int(((alpha > .01) & (alpha < .99)).sum()),
              'regions': {}, 'reference_disagreement': {}, 'prompt_alpha': []}
    for name, coverage_ in masks.items():
        region = coverage_ >= .85
        if region.any():
            values['regions'][name] = {'reference_pixels': int(region.sum()), 'mean_alpha': float(alpha[region].mean()),
                'retained_at_05_fraction': float(foreground[region].mean()), 'retained_at_099_fraction': float((alpha[region] >= .99).mean())}
    for name, ref in references.items():
        binary = ref >= .5
        union = (foreground | binary).sum()
        values['reference_disagreement'][name] = {'threshold_iou': float((foreground & binary).sum() / max(1, union)),
              'alpha_mean_absolute_difference': float(np.mean(np.abs(alpha - ref))),
              'zim_only_foreground_pixels': int((foreground & ~binary).sum()), 'reference_only_foreground_pixels': int((~foreground & binary).sum())}
    for label, points in [(1, prompts.get('positive', [])), (0, prompts.get('negative', []))]:
        for region, (x, y) in points:
            values['prompt_alpha'].append({'label': label, 'region': region, 'xy': [x, y], 'alpha': float(alpha[y, x])})
    return values


def visualize(rgb, alpha, prompts):
    composite = np.rint(rgb.astype(np.float32) * alpha[..., None] + np.array([36, 45, 56], dtype=np.float32) * (1-alpha[..., None])).astype(np.uint8)
    image = Image.fromarray(composite)
    draw = ImageDraw.Draw(image)
    for label, points in [(1, prompts.get('positive', [])), (0, prompts.get('negative', []))]:
        for _, (x, y) in points:
            draw.ellipse((x-5,y-5,x+5,y+5),fill=(40,255,80) if label else (255,40,40),outline='white',width=1)
    if 'box' in prompts:
        draw.rectangle(prompts['box'],outline=(40,255,80),width=2)
    return image


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--out', type=Path, default=ROOT / 'test/results/zim-20261014')
    args = args.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    torch.manual_seed(0)
    Encoder, Decoder, Zim, Predictor, manifest = load_vendor()
    model_dir = ROOT / '.models/zim-20261014'
    weights = {'encoder.onnx': (360680416, '6123ea10722dd43a0f8bbb96cd6a0532d8a65e2b1e8339f0cd6593d494807809'),
               'decoder.onnx': (19330176, 'a99d14bac3dc28b17809020b1711b1991bd5b7eb18f678c3624ce508748cfa11')}
    for name, (size, digest) in weights.items():
        assert (model_dir/name).stat().st_size == size and sha(model_dir/name) == digest, name
    source_path = ROOT / 'test/results/user-face-contour-20261014/1024/canonical-source.png'
    assert sha(source_path) == '5f15e92b14c44a468d924af5ac985a7cc7dc1c87d8ab235f2c94b2e88b173c73'
    rgb = np.array(Image.open(source_path).convert('RGB'))
    height, width = rgb.shape[:2]
    graph_path = ROOT / 'test/results/user-face-contour-20261014/1024/face-graph.json'
    graph = json.loads(graph_path.read_text())
    masks = {name: coverage(mask, width, height) for name, mask in graph['operations'][0]['params']['semanticMasks'].items()
             if name in ['face', 'faceContourMask', 'skin', 'lips', 'leftBrow', 'rightBrow', 'leftEye', 'rightEye', 'nose', 'hair', 'ears', 'neck', 'accessories']}
    debug = ROOT / 'test/results/user-face-contour-20261014/1024/trimap-candidate/debug100'
    geometry = json.loads((debug/'geometry.json').read_text())
    local = np.array(Image.open(debug/'old-alpha.png').convert('L'),dtype=np.float32)/255
    physical = np.zeros((height,width),dtype=np.float32)
    x,y = geometry['origin']
    physical[y:y+local.shape[0],x:x+local.shape[1]] = local
    references = {'bisenet-face': masks['face'], 'bisenet-movable-face': masks['faceContourMask'], 'recorded-physical-alpha': physical}
    for name, ref in references.items():
        Image.fromarray(np.rint(ref*255).astype(np.uint8)).save(args.out/(name+'.png'))
    sets = prompt_sets(masks)
    (args.out/'prompts.json').write_text(json.dumps(sets,indent=2)+'\n')
    start = time.perf_counter()
    encoder = Encoder(str(model_dir/'encoder.onnx'),num_threads=3)
    decoder = Decoder(str(model_dir/'decoder.onnx'),num_threads=3)
    model = Zim(encoder, decoder)
    predictor = Predictor(model)
    load_ms = (time.perf_counter()-start)*1000
    encoder_timings = []
    for _ in range(2):
        start=time.perf_counter()
        predictor.set_image(rgb)
        encoder_timings.append((time.perf_counter()-start)*1000)
    print('model-load-ms',round(load_ms,2),'encode-ms',encoder_timings,flush=True)
    report = {'status':'EXPERIMENT ONLY; no production integration; matting reference disagreement is not accuracy',
              'source_sha256':sha(source_path),'graph_sha256':sha(graph_path),'runner_sha256':sha(Path(__file__)),
              'official_commit':manifest['commit'],'vendor_manifest':manifest,
              'hf_model_revision':'883e9bbb15ca35a6afe16b72ae7e55cc13efbc78',
              'weights':{name:{'bytes':size,'sha256':digest} for name,(size,digest) in weights.items()},
              'environment':{'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),
                            'torch':torch.__version__,'numpy':np.__version__,'onnxruntime':ort.__version__,
                            'providers':encoder.ort_session.get_providers(),'threads':3},
              'timings':{'session_and_model_load_ms':load_ms,'encoder_set_image_first_ms':encoder_timings[0],
                         'encoder_set_image_repeat_ms':encoder_timings[1]},
              'reference_hashes':{'physical-alpha-png':sha(debug/'old-alpha.png'),'geometry':sha(debug/'geometry.json')},
              'notes':['Official preprocessing, padded point semantics, attention mask, postprocess and sigmoid are executed unchanged.',
                       'Point sets derive from original parser coverage and named original-image face windows; no output redetection.',
                       'All four official mask outputs are retained. Official score argmax and first output are both named; neither is ground-truth selection.',
                       'No trimap/semantic clipping was applied to raw ZIM alpha. Freezing hair/ears/neck afterwards is separate adaptation.',
                       'Matting cannot repair the old deformation field alpha=1 to 0 discontinuity; no warp was executed here.',
                       'One independent source image; no alpha ground truth, no accuracy claim, no browser or GPU timing.'], 'rows':[]}
    for case in sets:
        case_dir=args.out/case['name']
        case_dir.mkdir()
        kwargs={'multimask_output':True}
        if 'box' in case:
            kwargs['box']=np.array(case['box'],dtype=np.float32)
        else:
            coords=[xy for _,xy in case['positive']]+[xy for _,xy in case['negative']]
            kwargs['point_coords']=np.array(coords,dtype=np.float32)
            kwargs['point_labels']=np.array([1]*len(case['positive'])+[0]*len(case['negative']),dtype=np.float32)
        times=[]
        for _ in range(2):
            start=time.perf_counter()
            alphas,scores,_=predictor.predict(**kwargs)
            times.append((time.perf_counter()-start)*1000)
        np.save(case_dir/'alpha-all-f32.npy',alphas,allow_pickle=False)
        row={'name':case['name'],'prompt':case,'decode_postprocess_first_ms':times[0],
             'decode_postprocess_repeat_ms':times[1],'official_iou_scores':scores.tolist(),
             'official_score_argmax':int(np.argmax(scores)),'alphas':[]}
        for i,alpha in enumerate(alphas):
            path=case_dir/f'alpha-{i}.png'
            Image.fromarray(np.rint(alpha*255).astype(np.uint8)).save(path)
            visualize(rgb,alpha,case).save(case_dir/f'cutout-{i}.png')
            row['alphas'].append({'index':i,'png_sha256':sha(path), **metrics(alpha,masks,references,case)})
        panel=Image.new('RGB',(width*4,height+28),'white')
        draw=ImageDraw.Draw(panel)
        for i,alpha in enumerate(alphas):
            panel.paste(visualize(rgb,alpha,case),(i*width,28))
            draw.text((i*width+5,8),f'ZIM {i} / score {scores[i]:.4f}',fill='black')
        panel.resize((width*2,height//2+14),Image.Resampling.LANCZOS).save(case_dir/'four-mask-review.png')
        report['rows'].append(row)
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(case['name'], 'decode-ms', [round(t,2) for t in times], 'scores',scores.tolist(),flush=True)
    report['peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    report['files_sha256']={str(p.relative_to(args.out)):sha(p) for p in sorted(args.out.rglob('*')) if p.is_file() and p.name!='report.json'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()

"""Source-only frozen-photo decomposition; no new inference or production edits."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time
import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from decompose import decompose, encode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('zim_probe', ROOT / 'experiments/hair-accessory-matting-zim-v2-20261015/run.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rgb_save(path, rgb):
    Image.fromarray(np.floor(np.clip(rgb, 0, 1) * 255 + .5).astype('uint8')).save(path)


def mask_save(path, mask):
    Image.fromarray(np.floor(np.clip(mask, 0, 1) * 255 + .5).astype('uint8')).save(path)


def panel(path, rgb, result):
    hair = encode(result['hair']) if result['space'] == 'linear' else result['hair']
    skin = encode(result['skin']) if result['space'] == 'linear' else result['skin']
    marked = rgb.copy()
    marked[result['trusted']] = .35 * marked[result['trusted']] + .65 * np.array([0, 1, .2])
    im = Image.new('RGB', (rgb.shape[1] * 4, rgb.shape[0] + 28), '#20242a')
    draw = ImageDraw.Draw(im)
    # Show inferred colors only where an actual source mixed-edge solve ran.
    for i, (label, value) in enumerate([('Source', rgb), ('Mixed edge green', marked), ('Hair RGB estimate', hair), ('Skin RGB estimate', skin)]):
        if i >= 2:
            value = np.where(result['domain'][..., None], value, rgb)
        image = Image.fromarray(np.floor(np.clip(value, 0, 1) * 255 + .5).astype('uint8'))
        im.paste(image, (i * rgb.shape[1], 28)); draw.text((i * rgb.shape[1] + 4, 8), label, fill='white')
    im.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE / 'source-decomposition')
    parser.add_argument('--corrected-prompts', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    raw_root = ROOT / 'experiments/hair-accessory-matting-zim-v2-20261015/outputs'
    selection_root = ROOT / 'experiments/zim-semantic-selection-20261015/outputs'
    raw_report = json.loads((raw_root / 'report.json').read_text())
    selection_report = json.loads((selection_root / 'report.json').read_text())
    corrected_root = ROOT / 'experiments/zim-combined-prompts-20261015/outputs'
    corrected_report = json.loads((corrected_root / 'report.json').read_text()) if args.corrected_prompts else None
    report = dict(scope='Original source-only local soft alpha/color solve; no graph replacement or broad skin union.',
        sourceScriptSha256=sha(Path(__file__)), solverSha256=sha(HERE / 'decompose.py'),
        frozenZimReportSha256=sha(raw_root / 'report.json'), selectorReportSha256=sha(selection_root / 'report.json'),
        correctedPromptReportSha256=sha(corrected_root / 'report.json') if args.corrected_prompts else None, rows=[])
    for image in raw_report['images']:
        name = image['name']
        selection = next(v for v in selection_report['rows'] if v['photo'] == name and v['directory'].endswith('hair-1-component-box-positive'))
        assert selection['selectedIndex'] is not None
        alpha_path = selection_root / selection['directory'] / 'selected-alpha.npy'
        if args.corrected_prompts:
            fixed_image = next(v for v in corrected_report['images'] if v['name'] == name)
            fixed_case = next(v for v in fixed_image['rows'] if v['directory'].endswith('component-box-positive-skin-negative-1-combined-box'))
            alpha_path = corrected_root / fixed_case['directory'] / 'alphas.npy'
            chosen_index = fixed_case['officialScoreArgmax']
            alpha_full = np.load(alpha_path)[chosen_index]
        else:
            chosen_index = selection['selectedIndex']
            alpha_full = np.load(alpha_path)
        source = ROOT / image['source']; graph_path = ROOT / image['graph']
        assert sha(source) == image['sourceSha256'] and sha(graph_path) == image['graphSha256']
        rgb_full = np.asarray(Image.open(source).convert('RGB'), dtype='float64') / 255
        h, w = rgb_full.shape[:2]
        params = json.loads(graph_path.read_text())['operations'][0]['params']
        masks = {k: probe.zim.coverage(v, w, h) for k, v in params['semanticMasks'].items()}
        x0, y0, x1, y1 = image['faceCrop']; sl = np.s_[y0:y1, x0:x1]
        rgb = rgb_full[sl]; alpha = alpha_full[sl].astype('float64')
        forbidden = np.maximum.reduce([masks[k] for k in ['ears', 'accessories', 'leftEye', 'rightEye', 'leftBrow', 'rightBrow', 'lips', 'teeth']])[sl]
        raster = params['semanticMasks']['hair']; cell = max(w / raster['width'], h / raster['height'])
        # ZIM is not calibrated to produce alpha==1 (woman max=.99409). Use
        # corroborated deep original semantic interior, not an impossible .995
        # cutoff or a new broad pure-skin classification.
        hair_inside = distance_transform_edt(masks['hair'][sl] >= .95)
        hair_core = (masks['hair'][sl] >= .95) & (alpha >= .95) & (hair_inside >= 2 * cell)
        skin_core = (masks['skin'][sl] >= .9) & (alpha <= .005) & (forbidden < .05)
        candidate = (forbidden < .05) & ~hair_core & ~skin_core
        face_width = (max(p['x'] for p in params['landmarks']) - min(p['x'] for p in params['landmarks'])) * w
        radius = max(3 * cell, .035 * face_width)
        pure_path = raw_root / name / 'ownership-component-box-positive/source-evidence-reclaim-candidate.png'
        # This is only the already source-supported hair subset, never the
        # accessory subset and never a new dilation or inferred face polygon.
        pure = (np.asarray(Image.open(pure_path))[sl] > 0) & (masks['hair'][sl] >= .5) & (alpha <= .05)
        for variant, space, refine in [('fixed-zim-linear', 'linear', False), ('source-fit-linear', 'linear', True), ('source-fit-srgb', 'srgb', True)]:
            dest = out / name / variant; dest.mkdir(parents=True)
            start = time.perf_counter()
            result = decompose(rgb, alpha, hair_core, skin_core, candidate, radius,
                               space=space, refine_alpha=refine, prior_radius=max(1, int(np.ceil(cell))))
            elapsed = (time.perf_counter() - start) * 1000
            np.savez_compressed(dest / 'layers.npz', hair=result['hair'].astype('float32'), skin=result['skin'].astype('float32'),
                alpha=result['alpha'].astype('float32'), domain=result['domain'], trusted=result['trusted'], pureReclaim=pure,
                hairCore=hair_core, skinCore=skin_core)
            rgb_save(dest / 'source.png', rgb)
            rgb_save(dest / 'reconstructed-source.png', np.where(result['domain'][..., None], result['sourceReconstructed'], rgb))
            mask_save(dest / 'mixed-source-band.png', result['domain']); mask_save(dest / 'trusted-mixed-band.png', result['trusted'])
            mask_save(dest / 'original-zim-alpha.png', alpha); mask_save(dest / 'refined-hair-alpha.png', result['alpha'])
            panel(dest / 'source-layer-comparison-1x.png', rgb, result)
            stats = dict(name=name, variant=variant, source=str(source.relative_to(ROOT)), sourceSha256=sha(source),
                graph=str(graph_path.relative_to(ROOT)), graphSha256=sha(graph_path), crop=[x0, y0, x1, y1],
                selectedAlpha=str(alpha_path.relative_to(ROOT)), alphaSha256=sha(alpha_path),
                officialIndex=chosen_index if args.corrected_prompts else selection['officialIndex'], automaticSemanticIndex=chosen_index,
                originalOfficialWasRejected=False if args.corrected_prompts else selection['officialIndex'] != selection['selectedIndex'],
                promptIntegration='explicit combined box+signed points' if args.corrected_prompts else 'upstream box overrides point labels; historical baseline',
                parserCellPx=cell, donorRadiusPx=radius, hairCorePixels=int(hair_core.sum()), skinCorePixels=int(skin_core.sum()),
                pureObservedHairReclaimPixels=int(pure.sum()), elapsedMs=elapsed, **result['stats'], directory=str(dest.relative_to(HERE)))
            report['rows'].append(stats)
            print(json.dumps(stats), flush=True)
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

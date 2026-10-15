"""Diagnostic source-matte/color correction at a frozen V36 inverse map.

Skin-over-backing hypothesis at observed cheek edges; hair RGB stays in source
coordinates. This does not infer hair/skin occlusion order universally. The
opaque observed hair core, ears and accessories remain fixed. Mixed ownership
is changed only by the source RGB/alpha solve, never by broad face-mask growth.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np
from PIL import Image, ImageDraw
from decompose import linearize, encode
from run_real import HERE, ROOT, probe, sha, rgb_save, mask_save


def sample(values, xy):
    """Bilinear zero-padded sampling of premultiplied layers."""
    h, w = values.shape[:2]
    xx, yy = xy[..., 0], xy[..., 1]
    x0, y0 = np.floor(xx).astype('int32'), np.floor(yy).astype('int32')
    tx, ty = xx - x0, yy - y0
    if values.ndim == 2:
        values = values[..., None]
    output = np.zeros((*xy.shape[:2], values.shape[2]), dtype='float64')
    for dx, dy, weight in [(0, 0, (1-tx)*(1-ty)), (1, 0, tx*(1-ty)), (0, 1, (1-tx)*ty), (1, 1, tx*ty)]:
        x, y = x0 + dx, y0 + dy
        valid = (x >= 0) & (y >= 0) & (x < w) & (y < h)
        output[valid] += values[y[valid], x[valid]] * weight[valid, None]
    return output


def montage(path, images, labels, crop):
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    canvas = Image.new('RGB', (cw * len(images), ch + 28), '#20242a'); draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(images, labels)):
        canvas.paste(im.convert('RGB').crop(crop), (cw * i, 28)); draw.text((cw * i + 3, 8), label, fill='white')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE / 'renders')
    parser.add_argument('--estimates', type=Path, default=HERE / 'source-decomposition/report.json')
    parser.add_argument('--full-backing-combination', action='store_true')
    args = parser.parse_args(); args.out = args.out.resolve(); args.out.mkdir(exist_ok=False)
    estimates = json.loads(args.estimates.read_text())
    material_root = ROOT / 'experiments/face-material-fill-v1-20261015'
    cases = [('portrait-woman', 100, 'woman-100-full'), ('portrait-woman', -100, 'woman--100-full'),
             ('user-face-contour', 100, 'user-face-contour-100-full'), ('portrait-hair', 100, 'portrait-hair-100-full')]
    report = dict(scope='Isolated source-edge layer experiment; exact frozen V36 geometry. No production changes. '
                        'Skin-over-hair/backing is a boundary-order hypothesis; negative/occluding cases require visual rejection.',
                  scriptSha256=sha(Path(__file__)), solverSha256=sha(HERE / 'decompose.py'),
                  sourceEstimateReportSha256=sha(args.estimates), fullBackingCombination=args.full_backing_combination, rows=[])
    for name, amount, directory in cases:
        material = material_root / directory
        layers = material / 'layers'
        state = json.loads((layers / 'state.json').read_text())
        cw, ch = state['width'], state['height']; ox, oy = state['origin']
        estimate_rows = [row for row in estimates['rows'] if row['name'] == name]
        source_path = ROOT / estimate_rows[0]['source']; source = np.array(Image.open(source_path).convert('RGBA'))
        base_path = material / 'candidate.png'
        if not base_path.exists():
            base_path = material / 'output.png'
        base = np.array(Image.open(base_path).convert('RGBA'))
        assert base.shape == source.shape
        source_alpha = np.fromfile(layers / 'source-alpha-f32le.bin', dtype='<f4').reshape(ch, cw).astype('float64')
        xy = np.fromfile(layers / 'inverse-source-xy-f32le.bin', dtype='<f4').reshape(ch, cw, 2)
        foreground = np.asarray(Image.open(layers / 'foreground.png').convert('RGB'), dtype='float64') / 255
        bg = np.asarray(Image.open(layers / 'background.png').convert('RGB'), dtype='float64') / 255
        exact = np.fromfile(layers / 'full-samples-f32le.bin', dtype='<f4').reshape(ch, cw, 5)
        replay_alpha = sample(source_alpha, xy)[..., 0]
        replay_rgb = sample(foreground * 255 * source_alpha[..., None], xy)
        sample_error = float(np.max(np.abs(replay_rgb - exact[..., :3])))
        alpha_error = float(np.max(np.abs(replay_alpha - exact[..., 4])))
        assert sample_error < .001 and alpha_error < 1e-5, (sample_error, alpha_error)
        params = json.loads((ROOT / estimate_rows[0]['graph']).read_text())['operations'][0]['params']
        h, w = source.shape[:2]
        masks = {k: probe.zim.coverage(v, w, h)[oy:oy+ch, ox:ox+cw] for k, v in params['semanticMasks'].items()}
        protection = np.maximum(masks['ears'], masks['accessories']) >= .05
        yy, xx = np.indices((ch, cw)); movement = np.hypot(xy[..., 0] - xx, xy[..., 1] - yy)
        for estimate in estimate_rows:
            data = np.load(HERE / estimate['directory'] / 'layers.npz')
            x0, y0, x1, y1 = estimate['crop']; local = np.s_[y0-oy:y1-oy, x0-ox:x1-ox]
            trusted = np.zeros((ch, cw), bool); trusted[local] = data['trusted']
            pure = np.zeros((ch, cw), bool); pure[local] = data['pureReclaim']
            hair_core = np.zeros((ch, cw), bool); hair_core[local] = data['hairCore']
            physical = source_alpha.copy()
            patch_alpha = physical[local].copy(); patch_alpha[data['trusted']] = 1 - data['alpha'][data['trusted']]
            patch_alpha[data['pureReclaim']] = 1; physical[local] = patch_alpha
            linear = estimate['space'] == 'linear'
            colors = linearize(foreground) if linear else foreground.copy()
            local_colors = colors[local].copy(); local_colors[data['trusted']] = data['skin'][data['trusted']]
            source_rgb_patch = source[y0:y1, x0:x1, :3].astype('float64') / 255
            pure_colors = linearize(source_rgb_patch) if linear else source_rgb_patch
            local_colors[data['pureReclaim']] = pure_colors[data['pureReclaim']]; colors[local] = local_colors
            transported_alpha = sample(physical, xy)[..., 0]
            transported_rgb = sample(colors * physical[..., None], xy)
            warped_support = sample((trusted | pure).astype('float64'), xy)[..., 0] > 1e-6
            writes = (trusted | pure | warped_support) & (movement > 1e-4) & ~protection & ~hair_core
            backing_variants = [('v36-backing', bg, base)]
            if name == 'portrait-woman' and amount == 100:
                for size in [128, 256, 512]:
                    path = ROOT / f'experiments/face-background-context-20261015/woman-100-local/{size}/background-layer.png'
                    if path.exists():
                        backing_base = np.asarray(Image.open(path.with_name('candidate.png')).convert('RGBA')) if args.full_backing_combination else base
                        backing_variants.append((f'lama-context-{size}', np.asarray(Image.open(path).convert('RGB'), dtype='float64') / 255, backing_base))
            for backing_name, backing, variant_base in backing_variants:
                dest = args.out / f'{name}-{amount}' / estimate['variant'] / backing_name
                dest.mkdir(parents=True)
                background = linearize(backing) if linear else backing.copy()
                # Original mixed RGB cannot serve as pure backing after skin
                # leaves. Use the stationary source-estimated hair RGB there.
                back_patch = background[local].copy(); back_patch[data['trusted']] = data['hair'][data['trusted']]
                # The prior is evaluated only at already source-supported
                # pure reclaimed skin texels, not throughout the face interior.
                back_patch[data['pureReclaim']] = data['hair'][data['pureReclaim']]
                background[local] = back_patch
                mixed = transported_rgb + (1 - transported_alpha[..., None]) * background
                display = encode(mixed) if linear else np.clip(mixed, 0, 1)
                candidate = variant_base.copy(); patch = candidate[oy:oy+ch, ox:ox+cw, :3].copy()
                rendered = np.floor(display * 255 + .5).clip(0, 255).astype('uint8'); patch[writes] = rendered[writes]
                candidate[oy:oy+ch, ox:ox+cw, :3] = patch
                assert np.array_equal(candidate[..., 3], source[..., 3])
                full_writes = np.zeros((h, w), bool); full_writes[oy:oy+ch, ox:ox+cw] = writes
                assert np.array_equal(candidate[~full_writes], variant_base[~full_writes])
                assert np.array_equal(candidate[oy:oy+ch, ox:ox+cw][hair_core], source[oy:oy+ch, ox:ox+cw][hair_core])
                Image.fromarray(candidate).save(dest / 'candidate.png')
                mask_save(dest / 'write-mask.png', writes); mask_save(dest / 'source-skin-alpha.png', physical)
                mask_save(dest / 'transported-skin-alpha.png', transported_alpha)
                face_crop = estimate['crop']
                montage(dest / 'face-comparison-1x.png', [Image.fromarray(source), Image.fromarray(base), Image.fromarray(candidate)],
                        ['Source', 'V36', estimate['variant'] + ' / ' + backing_name], face_crop)
                mid = (x0+x1)//2
                for side, crop in [('left', [x0,y0,mid,y1]), ('right', [mid,y0,x1,y1])]:
                    montage(dest / f'{side}-edge-1x.png', [Image.fromarray(source), Image.fromarray(base), Image.fromarray(candidate)],
                            ['Source', 'V36', 'Decomposed edge'], crop)
                delta = np.abs(candidate.astype('int16') - base.astype('int16'))
                stats = dict(name=name, amount=amount, variant=estimate['variant'], backing=backing_name,
                    sourceSha256=sha(source_path), v36OutputSha256=sha(base_path),
                    inverseMapSha256=sha(layers / 'inverse-source-xy-f32le.bin'), sourceAlphaSha256=sha(layers / 'source-alpha-f32le.bin'),
                    sourceLayersSha256=sha(HERE / estimate['directory'] / 'layers.npz'),
                    baselineSampleMaxRgbError=sample_error, baselineSampleMaxAlphaError=alpha_error,
                    writtenPixels=int(writes.sum()), changedPixels=int((delta.max(axis=-1)>0).sum()),
                    maxChannelChange=int(delta.max()), opaqueHairPixelsChanged=0, untouchedOutsideWrite=True,
                    sourceMixedPixels=int(trusted.sum()), sourcePureReclaimPixels=int(pure.sum()),
                    fullBackingCombination=args.full_backing_combination,
                    output=str((dest / 'candidate.png').relative_to(HERE)))
                report['rows'].append(stats); print(json.dumps(stats), flush=True)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Conservative source-observation ownership diagnostic, never a graph mutation.

The distance bound is only a veto on independently excluded old protected pixels:
no dilation/union grows a face or skin mask. A failed raw prompt is not replaced by
a lower-scoring candidate. Scores are not used as semantic correctness evidence.
"""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('edge_probe', HERE / 'run.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
np, cv2, Image, ImageDraw, zim = probe.np, probe.cv2, probe.Image, probe.ImageDraw, probe.zim
from scipy.ndimage import distance_transform_edt


def panel(path, items, crop):
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    sheet = Image.new('RGB', (cw * len(items), ch + 28), '#20242a')
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(items):
        sheet.paste(img.convert('RGB').crop(crop), (cw * i, 28))
        draw.text((cw * i + 4, 8), label, fill='white')
    sheet.save(path)


def tint(rgb, layers):
    result = rgb.astype(np.float32)
    for region, color in layers:
        result[region] = .35 * result[region] + .65 * np.array(color)
    return Image.fromarray(np.rint(result).clip(0, 255).astype('uint8'))


def components(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    rows = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        rows.append(dict(component=i, bbox=[x, y, x + w, y + h], pixels=area))
    return sorted(rows, key=lambda p: p['pixels'], reverse=True)


def main():
    out = HERE / 'outputs'
    report = json.loads((out / 'report.json').read_text())
    analysis = dict(scriptSha256=zim.sha(Path(__file__)), inputReportSha256=zim.sha(out / 'report.json'),
        scope='Candidate ownership only, without warp or graph replacement. Original model argmax is preserved. '
              'No alpha ground truth; semantic compatibility is a rejection gate, not proof of correct matting.',
        rule={
            'skinAnchor': 'skin>=.90; hair/accessories/ears and protected facial features<.05',
            'modelExclusion': 'old hair/accessories>=.50 AND corresponding selected ZIM alpha<=.05; both new hair and accessory alpha<=.05',
            'sourceEvidence': 'skin>=.50 OR Lab DeltaEab<=8 and abs(DeltaL)<=6 and chroma distance<=6 to nearest local confident-skin mean',
            'distanceVeto': 'nearest skin anchor <= min(3 original parser cells, .025 landmark face width)',
            'additionalExclusion': 'original ears and protected facial features<.05; remain within automatic landmark face crop',
            'failedPrompt': 'positive alpha<.5 OR negative alpha>.1 OR skin-core retained>.05 OR hair-core retained<.65 => no ownership changes for that protocol',
            'noMorphologyGrowth': 'distance is only a veto; no dilated face/skin mask or raw face-alpha union is constructed',
        }, images=[])
    for row in report['images']:
        name = row['name']
        folder = out / name
        source = probe.ROOT / row['source']
        graph_path = probe.ROOT / row['graph']
        assert zim.sha(source) == row['sourceSha256'] and zim.sha(graph_path) == row['graphSha256']
        rgb = np.asarray(Image.open(source).convert('RGB'))
        h, w = rgb.shape[:2]
        params = json.loads(graph_path.read_text())['operations'][0]['params']
        masks = {k: zim.coverage(v, w, h) for k, v in params['semanticMasks'].items()}
        skin, old_hair, old_accessory, ears = (masks[k] for k in ['skin', 'hair', 'accessories', 'ears'])
        feature_names = ['lips', 'teeth', 'leftBrow', 'rightBrow', 'leftEye', 'rightEye']
        features = np.maximum.reduce([masks[k] for k in feature_names])
        anchors = (skin >= .90) & (np.maximum.reduce([old_hair, old_accessory, ears, features]) < .05)
        distance, nearest = distance_transform_edt(~anchors, return_indices=True)
        raster = params['semanticMasks']['hair']
        cell = max(w / raster['width'], h / raster['height'])
        face_width = (max(p['x'] for p in params['landmarks']) - min(p['x'] for p in params['landmarks'])) * w
        max_distance = min(3 * cell, .025 * face_width)
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        kernel = 2 * int(np.ceil(cell)) + 1
        count = cv2.boxFilter(anchors.astype('float32'), -1, (kernel, kernel), normalize=False)
        means = np.stack([cv2.boxFilter(lab[..., i] * anchors, -1, (kernel, kernel), normalize=False)
                          / np.maximum(count, 1) for i in range(3)], axis=-1)
        nearest_color = means[nearest[0], nearest[1]]
        delta = lab - nearest_color
        color_error = np.linalg.norm(delta, axis=-1)
        color_support = (color_error <= 8) & (np.abs(delta[..., 0]) <= 6) & (np.linalg.norm(delta[..., 1:], axis=-1) <= 6)
        source_support = (skin >= .5) | color_support
        within_crop = np.zeros((h, w), dtype=bool)
        x0, y0, x1, y1 = row['faceCrop']
        within_crop[y0:y1, x0:x1] = True
        # This explicitly includes model-excluded skin-coloured pixels, not every
        # model-excluded hair pixel (background, holes and dark strands stay unknown).
        eligible = within_crop & (distance <= max_distance) & source_support & (ears < .05) & (features < .05)
        probe.png(folder / 'skin-anchor.png', anchors)
        image_summary = dict(name=name, maxSourceEvidenceDistancePx=max_distance, parserCellPx=cell,
                             skinAnchorPixels=int(anchors.sum()), protocols=[])
        for protocol in ['component-box-positive', 'component-box-positive-skin-negative']:
            selected = {}
            validity = {}
            selected_cases = []
            for kind in ['hair', 'accessories']:
                cases = [r for r in row['rows'] if r['prompt']['kind'] == kind
                         and r['prompt']['protocol'] == (protocol if kind == 'hair' else 'component-box-positive')]
                if not cases:
                    selected[kind] = masks[kind].copy()
                    validity[kind] = dict(status='no-component; original reference unchanged', passed=True)
                    continue
                alphas = [np.load(out / r['directory'] / 'alphas.npy', mmap_mode='r')[r['officialScoreArgmax']] for r in cases]
                merged = np.maximum.reduce(alphas)
                selected[kind] = merged
                skin_leak = float((merged[anchors] >= .5).mean())
                core = masks[kind] >= .9
                core_retention = float((merged[core] >= .5).mean()) if core.any() else None
                reasons = []
                for case in cases:
                    a = case['alphas'][case['officialScoreArgmax']]
                    if a['positiveAlpha'] < .5:
                        reasons.append('positive point not retained')
                    if any(v > .1 for v in a['negativeAlpha']):
                        reasons.append('negative skin point retained')
                    selected_cases.append(dict(directory=case['directory'], officialScoreArgmax=case['officialScoreArgmax']))
                if skin_leak > .05:
                    reasons.append('more than 5% confident skin retained as protected object')
                if kind == 'hair' and core_retention is not None and core_retention < .65:
                    reasons.append('less than 65% original hair core retained')
                validity[kind] = dict(passed=not reasons, reasons=reasons, skinCoreRetainedFraction=skin_leak,
                                      originalCoreRetainedFraction=core_retention)
            new_hair, new_accessory = selected['hair'], selected['accessories']
            excluded_by_kind = {'hair': (old_hair >= .5) & (new_hair <= .05),
                                'accessories': (old_accessory >= .5) & (new_accessory <= .05)}
            excluded = excluded_by_kind['hair'] | excluded_by_kind['accessories']
            passed = all(v['passed'] for v in validity.values())
            reclaim = excluded & (np.maximum(new_hair, new_accessory) <= .05) & eligible if passed else np.zeros((h, w), dtype=bool)
            unresolved = excluded & ~reclaim
            assert not (reclaim & ~excluded).any()
            assert not (reclaim & ~eligible).any()
            assert not (reclaim & (np.maximum(new_hair, new_accessory) > .05)).any()
            dest = folder / ('ownership-' + protocol)
            dest.mkdir(exist_ok=False)
            probe.png(dest / 'selected-hair-alpha.png', new_hair)
            probe.png(dest / 'selected-accessories-alpha.png', new_accessory)
            probe.png(dest / 'model-excluded-old-protection.png', excluded)
            probe.png(dest / 'source-evidence-reclaim-candidate.png', reclaim)
            probe.png(dest / 'unresolved-excluded.png', unresolved)
            difference = tint(rgb, [(unresolved, (255, 50, 190)), (reclaim, (40, 255, 80))])
            difference.save(dest / 'ownership-difference.png')
            panel(dest / 'face-hair-alpha-1x.png', [('Original', Image.fromarray(rgb)),
                ('Parser hair', probe.cutout(rgb, old_hair)), ('Official hair argmax', probe.cutout(rgb, new_hair))], row['faceCrop'])
            panel(dest / 'face-ownership-1x.png', [('Original', Image.fromarray(rgb)),
                ('Green: source-supported / pink: unknown', difference)], row['faceCrop'])
            # Per-side panels remain native size, independently of image viewer scaling.
            midpoint = (x0 + x1) // 2
            for side, crop in [('left', [x0, y0, midpoint, y1]), ('right', [midpoint, y0, x1, y1])]:
                panel(dest / f'{side}-edge-1x.png', [('Original', Image.fromarray(rgb)),
                    ('Parser hair', probe.cutout(rgb, old_hair)), ('ZIM hair', probe.cutout(rgb, new_hair)),
                    ('Reclaim / unknown', difference)], crop)
            removed_values = color_error[reclaim]
            result = dict(protocol=protocol, passedSemanticRejectionGates=passed, validity=validity,
                selectedCases=selected_cases, modelExcludedOldProtectionPixels=int(excluded.sum()),
                sourceSupportedReclaimPixels=int(reclaim.sum()),
                reclaimByOldKind={k: int((reclaim & v).sum()) for k, v in excluded_by_kind.items()},
                unresolvedExcludedPixels=int(unresolved.sum()),
                reclaimedMaxDistancePx=float(distance[reclaim].max()) if reclaim.any() else None,
                reclaimedMaxDeltaEab=float(removed_values.max()) if len(removed_values) else None,
                components=components(reclaim), directory=str(dest.relative_to(out)))
            image_summary['protocols'].append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ['components', 'selectedCases']})[:3000], flush=True)
        analysis['images'].append(image_summary)
    (out / 'ownership-report.json').write_text(json.dumps(analysis, indent=2) + '\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Evaluate independent V36 probes; coordinates never feed prompts/ownership.

These are diagnostic coordinates recorded by another audit before seeing ZIM.
No alpha selection, thresholds, source windows or graph is changed here.
"""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('probe', HERE / 'run.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
np, Image, cv2 = probe.np, probe.Image, probe.cv2
from scipy.ndimage import distance_transform_edt


def main():
    result = HERE / 'outputs'
    report = json.loads((result / 'report.json').read_text())
    # Derivative gray-alpha panels, using original 1:1 image samples.
    for item in report['images']:
        directory = result / item['name']
        source = Image.open(probe.ROOT / item['source']).convert('RGB')
        crop = item['faceCrop']
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        for protocol in ['component-box-positive', 'component-box-positive-skin-negative']:
            ownership = directory / ('ownership-' + protocol)
            sheet = Image.new('RGB', (cw * 3, ch + 28), '#20242a')
            draw = probe.ImageDraw.Draw(sheet)
            panels = [('Original', source), ('Parser hair alpha', Image.open(directory / 'parser-hair.png')),
                      ('Official hair alpha', Image.open(ownership / 'selected-hair-alpha.png'))]
            for i, (label, image) in enumerate(panels):
                sheet.paste(image.convert('RGB').crop(crop), (cw * i, 28))
                draw.text((cw * i + 4, 8), label, fill='white')
            sheet.save(ownership / 'face-raw-alpha-1x.png')
        # Tiny accessory crops need short labels to preserve readable 1x panels.
        for case in item['rows']:
            if case['prompt']['kind'] != 'accessories':
                continue
            directory_case = result / case['directory']
            box = case['prompt']['box']
            component_crop = (max(0, box[0] - 12), max(0, box[1] - 12),
                              min(source.width, box[2] + 13), min(source.height, box[3] + 13))
            pw, ph = component_crop[2] - component_crop[0], component_crop[3] - component_crop[1]
            raw = np.load(directory_case / 'alphas.npy', mmap_mode='r')
            rgb_source = np.asarray(source)
            old = np.asarray(Image.open(directory / 'parser-accessories.png'), dtype=np.float32) / 255
            panels = [('Src', source), ('Parser', probe.cutout(rgb_source, old))]
            panels += [(('*' if i == case['officialScoreArgmax'] else '') + f"{i} {case['alphas'][i]['score']:.3f}",
                        probe.cutout(rgb_source, a)) for i, a in enumerate(raw)]
            sheet = Image.new('RGB', (pw * len(panels), ph + 28), '#20242a')
            draw = probe.ImageDraw.Draw(sheet)
            for i, (label, image) in enumerate(panels):
                sheet.paste(image.convert('RGB').crop(component_crop), (pw * i, 28))
                draw.text((pw * i + 3, 8), label, fill='white')
            sheet.save(directory_case / 'component-short-label-comparison-1x.png')
    row = next(r for r in report['images'] if r['name'] == 'portrait-woman')
    out = result / row['name']
    rgb = np.asarray(Image.open(probe.ROOT / row['source']).convert('RGB'))
    h, w = rgb.shape[:2]
    params = json.loads((probe.ROOT / row['graph']).read_text())['operations'][0]['params']
    anchors = np.asarray(Image.open(out / 'skin-anchor.png')) > 0
    distance, nearest = distance_transform_edt(~anchors, return_indices=True)
    raster = params['semanticMasks']['hair']
    cell = max(w / raster['width'], h / raster['height'])
    k = 2 * int(np.ceil(cell)) + 1
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    count = cv2.boxFilter(anchors.astype('float32'), -1, (k, k), normalize=False)
    means = np.stack([cv2.boxFilter(lab[..., i] * anchors, -1, (k, k), normalize=False)
                      / np.maximum(count, 1) for i in range(3)], axis=-1)
    delta = lab - means[nearest[0], nearest[1]]
    raw = next(r for r in row['rows'] if r['prompt']['kind'] == 'hair'
               and r['prompt']['protocol'] == 'component-box-positive')
    alpha = np.load(result / raw['directory'] / 'alphas.npy')
    reclaim = np.asarray(Image.open(out / 'ownership-component-box-positive/source-evidence-reclaim-candidate.png')) > 0
    probe_source = probe.ROOT / 'experiments/face-source-curve-v36-20261015/qa/woman-left-edge-probes.json'
    observations = json.loads(probe_source.read_text())
    for value in observations:
        x, y = value['x'], value['y']
        value.update(zimOfficialAlpha=float(alpha[raw['officialScoreArgmax'], y, x]),
                     zimFourAlpha=[float(a[y, x]) for a in alpha], reclaimed=bool(reclaim[y, x]),
                     skinAnchorDistancePx=float(distance[y, x]), nearestSkin=[int(nearest[1, y, x]), int(nearest[0, y, x])],
                     labDifference=delta[y, x].tolist(), deltaEab=float(np.linalg.norm(delta[y, x])))
    output = dict(note='Independent V36 probes are evaluation-only; never used in prompts, masks, threshold choice or ownership.',
                  scriptSha256=probe.zim.sha(Path(__file__)), probeSource=str(probe_source.relative_to(probe.ROOT)),
                  probeSourceSha256=probe.zim.sha(probe_source), probes=observations)
    (out / 'independent-v36-probe-observations.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()

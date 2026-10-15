"""Fixed-V36 auxiliary-background context ablation; never edits production."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from context import ROOT, sha, compose, context_masks, montage, raster_coverage
sys.path.insert(0, str(ROOT / 'services/background-worker'))
from worker import LamaWorker


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diagnostic', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-side', type=int, default=512)
    args = ap.parse_args()
    base, out = args.diagnostic.resolve(), args.output.resolve()
    if out.exists():
        raise SystemExit('Refusing to overwrite frozen experiment output')
    out.mkdir(parents=True)
    debug = base / 'layers'
    state = json.loads((debug/'state.json').read_text())
    provenance = json.loads((base/'provenance.json').read_text())
    graph = json.loads((base/'graph.json').read_text())
    params = graph['operations'][0]['params']
    origin = tuple(state['origin']); ox, oy = origin
    h, w = state['height'], state['width']
    source_path = Path(provenance['source'])
    source = np.asarray(Image.open(source_path).convert('RGBA'))
    candidate = np.asarray(Image.open(base/'candidate.png').convert('RGBA'))
    source_crop = np.asarray(Image.open(debug/'source-crop.png').convert('RGBA'))
    background = np.asarray(Image.open(debug/'background.png').convert('RGBA'))
    sample = np.fromfile(debug/'samples-f32le.bin', '<f4').reshape(h,w,5)
    source_alpha = np.fromfile(debug/'source-alpha-f32le.bin', '<f4').reshape(h,w)
    flags = {name: np.fromfile(debug/f'{name}-u8.bin', 'uint8').reshape(h,w)>0
             for name in ['changed','holes','frozen']}
    recomposed = source_crop.copy()
    calc = compose(background, sample[:,:,:4], sample[:,:,4])
    recomposed[flags['changed']] = calc[flags['changed']]
    candidate_crop = candidate[oy:oy+h, ox:ox+w]
    assert np.array_equal(recomposed, candidate_crop), 'Exact original compositor parity is required'
    known_full, excluded_full = context_masks(params, (source.shape[1],source.shape[0]), source_alpha, origin)
    # Full-person parsing is cropped at y=1698 on this source: never condition
    # on full-image regions outside the original renderer's known crop.
    known = {name: mask[oy:oy+h,ox:ox+w] & (source_crop[:,:,3]>0)
             for name,mask in known_full.items()}
    neck = raster_coverage(params['semanticMasks']['neck'], source.shape[1], source.shape[0])[oy:oy+h,ox:ox+w] > .8
    # Conservative nearest visible material hypothesis, NOT occluded truth.
    # Leave neck-adjacent disocclusion unchanged. Hair-vs-scene model context
    # remains the only difference between the two completion candidates.
    hair_distance = distance_transform_edt(~known['hair-only'])
    neck_distance = distance_transform_edt(~neck)
    target = (flags['holes'] & flags['changed'] & ~flags['frozen'] &
              (source_alpha > sample[:,:,4] + np.float32(1e-6)) &
              (sample[:,:,4] < 1-np.float32(1e-6)) &
              (hair_distance + 8 < neck_distance))
    if not target.any():
        raise RuntimeError('No eligible material disocclusion')
    target_full = np.zeros(source.shape[:2], bool); target_full[oy:oy+h,ox:ox+w]=target
    Image.fromarray((target*255).astype('uint8')).save(out/'write-target.png')
    Image.fromarray((excluded_full*255).astype('uint8')).save(out/'foreground-exclusion.png')
    target_overlay=candidate_crop.copy(); target_overlay[target,:3]=[255,0,230]
    Image.fromarray(target_overlay).save(out/'write-target-overlay.png')
    inputs=[source_path,base/'candidate.png',base/'graph.json',Path(provenance['binary']),*sorted(debug.iterdir()),Path(__file__),Path(__file__).with_name('context.py'),ROOT/'services/background-worker/worker.py']
    report={'scope':'One source, fixed V36 source/geometry/foreground. Cached LaMa input context ablation; not a new model or accepted retouch.',
            'inputs':{str(p.relative_to(ROOT)):sha(p) for p in inputs if p.is_file()},
            'crop':state,'baselineRecompositionPixelDifferences':0,
            'targetRule':'holes & changed & !frozen & sourceAlpha>membership+1e-6 & membership<1-1e-6 & distance(known hair)+8<distance(source neck>.8)',
            'targetPixels':int(target.sum()),'rejectedNeckOrNotNewlyExposedPixels':int((flags['holes'] & ~target).sum()),'rows':[]}
    worker = LamaWorker(threads=3)
    for name, visible in known.items():
        dest=out/name;dest.mkdir()
        unknown=~visible
        preview=source_crop.copy();preview[unknown,:3]=0
        Image.fromarray(preview).save(dest/'conditioning.png')
        Image.fromarray((unknown*255).astype('uint8')).save(dest/'inference-mask.png')
        filled,meta=worker.inpaint(Image.fromarray(source_crop),unknown.astype('uint8')*255,
                                max_side=args.max_side,context_mode='full',mask_dilation=0,seam_blend='none')
        filled.save(dest/'background-layer.png')
        filled_array=np.asarray(filled)
        full_background=source.copy();full_background[oy:oy+h,ox:ox+w]=filled_array
        Image.fromarray(full_background).save(dest/'full-background-layer.png')
        new_layer=background.copy();new_layer[target,:3]=filled_array[target,:3]
        transformed=compose(new_layer,sample[:,:,:4],sample[:,:,4])
        result=candidate.copy();local=result[oy:oy+h,ox:ox+w];local[target]=transformed[target]
        # The final target never confers permission to rewrite original alpha.
        assert np.array_equal(result[:,:,3],candidate[:,:,3])
        changed=np.any(result!=candidate,axis=2)
        assert not (changed & ~target_full).any()
        assert np.array_equal(local[flags['frozen']],candidate_crop[flags['frozen']])
        Image.fromarray(result).save(dest/'candidate.png')
        row={'name':name,'knownContextPixels':int(visible.sum()),'unknownContextPixels':int(unknown.sum()),
             'changedPixels':int(changed.sum()),'outsideTargetChanges':int((changed&~target_full).sum()),
             'frozenChanges':int(np.any(local[flags['frozen']]!=candidate_crop[flags['frozen']],axis=1).sum()),
             'alphaChanges':0,'foregroundAndGeometryBuffersUnchanged':True,'meta':meta,'outputSha256':sha(dest/'candidate.png')}
        report['rows'].append(row)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    # The worker must ignore all unknown source RGB, including Fourier context
    # and downsampling. Corrupt those pixels and compare the actual model result.
    visible=known['hair-scene'];corrupted=source_crop.copy();unknown=~visible
    rng=np.random.default_rng(1729);corrupted[unknown,:3]=rng.integers(0,256,(int(unknown.sum()),3),dtype='uint8')
    invariant,_=worker.inpaint(Image.fromarray(corrupted),unknown.astype('uint8')*255,
                              max_side=args.max_side,context_mode='full',mask_dilation=0,seam_blend='none')
    original_fill=np.asarray(Image.open(out/'hair-scene/background-layer.png').convert('RGBA'))
    ignored_rgb_differences=int(np.any(np.asarray(invariant)[unknown,:3]!=original_fill[unknown,:3],axis=1).sum())
    assert ignored_rgb_differences==0, 'Excluded foreground leaked into inference'
    report['excludedRGBPerturbationOutputDifferences']=ignored_rgb_differences
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    images=[source,candidate]+[np.asarray(Image.open(out/name/'candidate.png').convert('RGBA')) for name in known]
    labels=['Original','V36 baseline','Hair context','Hair + scene context']
    montage(images,labels,out/'face-1x.png',box=(300,480,900,1170))
    montage(images,labels,out/'left-edge-2x.png',box=(340,820,485,1110),zoom=2)
    montage(images,labels,out/'right-edge-2x.png',box=(745,800,840,1080),zoom=2)


if __name__=='__main__':main()

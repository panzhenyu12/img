"""Explicit A/B layer-order hypotheses; neither is automatically selected.

A: original hair is in front of the moving skin.
B: moving skin is in front of source-stationary hair, but only in real newly
covered hair components touching the observed original mixed boundary.
Unknown order is retained as a label. Geometry, source RGB solve, original
detector and corrected ZIM candidates are frozen for the comparison.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_dilation
from compose_real import sample, montage
from run_real import HERE, ROOT, probe, sha, mask_save


def connected_coverage(new_alpha, source_alpha, hair, border, allowed):
    added = (new_alpha > source_alpha + .001) & (hair >= .05) & allowed
    labels, count = label(added, structure=np.ones((3, 3)))
    touches = binary_dilation(border, structure=np.ones((3, 3)))
    ids = np.unique(labels[touches & added])
    ids = ids[ids > 0]
    connected = np.isin(labels, ids) if len(ids) else np.zeros_like(added)
    return connected, added & ~connected, labels, ids


def write_domains(edge, hair_core, cover_a, cover_b, unknown_a, unknown_b):
    # A transported edge can be spatially disconnected from the observed
    # source edge. Explicitly reject unknown coverage from *both* write routes.
    unknown = unknown_a | unknown_b
    return ((edge | cover_a) & ~hair_core & ~unknown,
            ((edge & ~hair_core) | cover_b) & ~unknown)


def main():
    out = HERE / 'occlusion-corrected-prompts'
    out.mkdir(exist_ok=False)
    estimate_path = HERE / 'source-decomposition-corrected-prompts/report.json'
    estimates = json.loads(estimate_path.read_text())
    material_root = ROOT / 'experiments/face-material-fill-v1-20261015'
    cases = [(name, value, ('woman' if name == 'portrait-woman' else name) + f'-{value}-full')
             for name in ['portrait-woman', 'user-face-contour', 'portrait-hair'] for value in [100, -100]]
    report = dict(scope='Two explicit unknown-order hypotheses, no automatic order choice, no production changes.',
                  scriptSha256=sha(Path(__file__)), sourceEstimatesSha256=sha(estimate_path),
                  colorVariant='source-fit-srgb', sourceColorSolverFrozenSha256=sha(HERE / 'decompose.py'), rows=[])
    for name, amount, folder in cases:
        estimate = next(v for v in estimates['rows'] if v['name'] == name and v['variant'] == 'source-fit-srgb')
        material = material_root / folder; layers = material / 'layers'
        state = json.loads((layers / 'state.json').read_text()); cw, ch = state['width'], state['height']; ox, oy = state['origin']
        source_path = ROOT / estimate['source']; source = np.asarray(Image.open(source_path).convert('RGBA'))
        base = np.asarray(Image.open(material / 'candidate.png').convert('RGBA')); h, w = source.shape[:2]
        source_rgb = source[oy:oy+ch, ox:ox+cw, :3].astype('float64') / 255
        source_alpha = np.fromfile(layers / 'source-alpha-f32le.bin', dtype='<f4').reshape(ch, cw).astype('float64')
        xy = np.fromfile(layers / 'inverse-source-xy-f32le.bin', dtype='<f4').reshape(ch, cw, 2)
        person = np.fromfile(layers / 'person-coverage-f32le.bin', dtype='<f4').reshape(ch, cw)
        foreground = np.asarray(Image.open(layers / 'foreground.png').convert('RGB'), dtype='float64') / 255
        params = json.loads((ROOT / estimate['graph']).read_text())['operations'][0]['params']
        masks = {k: probe.zim.coverage(v, w, h)[oy:oy+ch, ox:ox+cw] for k, v in params['semanticMasks'].items()}
        ears_accessories = np.maximum(masks['ears'], masks['accessories']) >= .05
        yy, xx = np.indices((ch, cw)); r = params['region']
        roi = ((xx+ox+.5)/w >= r['x']) & ((xx+ox+.5)/w <= r['x']+r['width']) & ((yy+oy+.5)/h >= r['y']) & ((yy+oy+.5)/h <= r['y']+r['height'])
        allowed = roi & ~ears_accessories & (person >= .5)
        moved = np.hypot(xy[..., 0]-xx, xy[..., 1]-yy) > 1e-4
        data = np.load(HERE / estimate['directory'] / 'layers.npz')
        x0,y0,x1,y1 = estimate['crop']; sl = np.s_[y0-oy:y1-oy,x0-ox:x1-ox]
        trusted = np.zeros((ch,cw),bool); trusted[sl] = data['trusted']
        pure = np.zeros((ch,cw),bool); pure[sl] = data['pureReclaim']
        hair_core = np.zeros((ch,cw),bool); hair_core[sl] = data['hairCore']
        hair_alpha = np.load(ROOT / estimate['selectedAlpha'])[estimate['officialIndex']][oy:oy+ch, ox:ox+cw].astype('float64')
        hair_alpha[sl] = np.where(data['trusted'], data['alpha'], hair_alpha[sl])
        hair_color = source_rgb.copy(); p = hair_color[sl].copy(); p[data['trusted'] | data['pureReclaim']] = data['hair'][data['trusted'] | data['pureReclaim']]; hair_color[sl] = p
        skin_color = foreground.copy(); p = skin_color[sl].copy(); p[data['trusted']] = data['skin'][data['trusted']]
        p[data['pureReclaim']] = source[y0:y1,x0:x1,:3][data['pureReclaim']] / 255; skin_color[sl] = p
        physical = source_alpha.copy(); p = physical[sl].copy(); p[data['trusted']] = 1-data['alpha'][data['trusted']]; p[data['pureReclaim']] = 1; physical[sl] = p
        # Both hypotheses reconstruct the same source mix locally. Under the
        # hair-front hypothesis, visible skin=(1-hairAlpha)*underlying skin.
        under_skin = physical.copy(); under_skin[trusted] = np.clip(physical[trusted] / np.maximum(1-hair_alpha[trusted], .001), 0, 1)
        alpha_b = sample(physical,xy)[...,0]; rgb_b = sample(skin_color*physical[...,None],xy)
        alpha_a = sample(under_skin,xy)[...,0]; rgb_a = sample(skin_color*under_skin[...,None],xy)
        transported_border = sample((trusted|pure).astype('float64'),xy)[...,0] > 1e-6
        cover_b, unknown_b, cc, ids = connected_coverage(alpha_b,physical,masks['hair'],trusted|pure,allowed&moved)
        cover_a, unknown_a, _, _ = connected_coverage(alpha_a,under_skin,masks['hair'],trusted|pure,allowed&moved)
        edge = (trusted|pure|transported_border) & allowed & moved
        # A retains source opaque hair exactly. B may occlude it only where
        # the transported physical skin newly covers a connected boundary.
        writes_a, writes_b = write_domains(edge,hair_core,cover_a,cover_b,unknown_a,unknown_b)
        fixed_hair_alpha = np.where(masks['hair'] >= .05, hair_alpha, 0)
        fixed_hair_alpha[hair_core] = 1
        bg = np.asarray(Image.open(layers/'background.png').convert('RGB'),dtype='float64')/255
        backing_variants = [('v36-backing',bg,base)]
        if name == 'portrait-woman' and amount == 100:
            for size in [128,256,512]:
                folder_back = ROOT/f'experiments/face-background-context-20261015/woman-100-local/{size}'
                backing_variants.append((f'lama-{size}',np.asarray(Image.open(folder_back/'background-layer.png').convert('RGB'),dtype='float64')/255,
                                         np.asarray(Image.open(folder_back/'candidate.png').convert('RGBA'))))
        for backing_name, backing, initial in backing_variants:
            dest=out/f'{name}-{amount}'/backing_name;dest.mkdir(parents=True)
            backing=backing.copy();backing[trusted|pure]=hair_color[trusted|pure]
            beneath=rgb_a+(1-alpha_a[...,None])*backing
            render_a=fixed_hair_alpha[...,None]*hair_color+(1-fixed_hair_alpha[...,None])*beneath
            render_b=rgb_b+(1-alpha_b[...,None])*backing
            output_a=initial.copy();output_b=initial.copy()
            for output,render,writes in [(output_a,render_a,writes_a),(output_b,render_b,writes_b)]:
                patch=output[oy:oy+ch,ox:ox+cw,:3].copy();patch[writes]=np.floor(np.clip(render[writes],0,1)*255+.5).astype('uint8');output[oy:oy+ch,ox:ox+cw,:3]=patch
                assert np.array_equal(output[oy:oy+ch,ox:ox+cw][ears_accessories],initial[oy:oy+ch,ox:ox+cw][ears_accessories])
                full_write=np.zeros((h,w),bool);full_write[oy:oy+ch,ox:ox+cw]=writes
                assert np.array_equal(output[~full_write],initial[~full_write])
                assert np.array_equal(output[...,3],source[...,3])
            assert np.array_equal(output_a[oy:oy+ch,ox:ox+cw][hair_core],source[oy:oy+ch,ox:ox+cw][hair_core])
            uncovered_core=hair_core&~cover_b
            assert np.array_equal(output_b[oy:oy+ch,ox:ox+cw][uncovered_core],source[oy:oy+ch,ox:ox+cw][uncovered_core])
            Image.fromarray(output_a).save(dest/'A-hair-front.png');Image.fromarray(output_b).save(dest/'B-skin-front.png')
            for label_,mask in [('A-write',writes_a),('B-write',writes_b),('newly-covered-connected-hair',cover_b),('unknown-new-coverage',unknown_b|unknown_a),('source-edge',trusted|pure)]:mask_save(dest/(label_+'.png'),mask)
            montage(dest/'face-A-B-1x.png',[Image.fromarray(source),Image.fromarray(base),Image.fromarray(output_a),Image.fromarray(output_b)],['Source','V36','A hair in front','B skin in front'],estimate['crop'])
            mid=(x0+x1)//2
            for side,crop in [('left',[x0,y0,mid,y1]),('right',[mid,y0,x1,y1])]:
                montage(dest/f'{side}-A-B-1x.png',[Image.fromarray(source),Image.fromarray(base),Image.fromarray(output_a),Image.fromarray(output_b)],['Source','V36','A hair in front','B skin in front'],crop)
            result=dict(name=name,amount=amount,backing=backing_name,sourceSha256=sha(source_path),sourceLayerSha256=sha(HERE/estimate['directory']/'layers.npz'),
                inverseSha256=sha(layers/'inverse-source-xy-f32le.bin'),AWritePixels=int(writes_a.sum()),BWritePixels=int(writes_b.sum()),
                newlyCoveredConnectedHairPixels=int(cover_b.sum()),unknownCoveragePixels=int((unknown_a|unknown_b).sum()),
                AVisibleOpaqueHairChanged=0,BUncoveredOpaqueHairChanged=0,
                BSourceHairCorePixelsOccluded=int((cover_b&hair_core).sum()),earsAccessoriesNewlyChanged=0,
                inheritedEarsAccessoriesPixelsDifferentFromSource=int((ears_accessories & np.any(initial[oy:oy+ch,ox:ox+cw]!=source[oy:oy+ch,ox:ox+cw],axis=2)).sum()),
                suppressedSkinVisiblePixelsA=int(((alpha_b>.95)&(alpha_a*(1-fixed_hair_alpha)<.5)&cover_b).sum()),
                layerOrder='UNKNOWN: both counterfactual hypotheses exported; no automatic choice',
                directory=str(dest.relative_to(HERE)))
            report['rows'].append(result);print(json.dumps(result),flush=True)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

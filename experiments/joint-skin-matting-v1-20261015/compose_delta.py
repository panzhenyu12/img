"""Test a complete local source/destination domain with direct skin alpha.

The experiment deliberately changes source ownership and compositing together,
holding the full V36 inverse field fixed. It does not claim to recover hidden
texture truth. Source source-reconstruction residuals and all changed domains
are saved separately from the final photographic outputs.
"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from compose_real import sample, montage
from run_real import probe, sha, mask_save


def nearest(values, valid):
    distance, indices = distance_transform_edt(~valid, return_indices=True)
    return values[tuple(indices)], distance


def main():
    out = HERE/'composite-delta-v2'
    out.mkdir(exist_ok=False)
    source_path, graph_path = [ROOT/p for p in probe.SOURCES['user-face-contour']]
    source = np.asarray(Image.open(source_path).convert('RGBA'))
    rgb = source[...,:3].astype(float)/255
    h, w = rgb.shape[:2]
    params = json.loads(graph_path.read_text())['operations'][0]['params']
    masks = {k: probe.zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
    full = np.load(HERE/'outputs/full-point/alphas-full.npy')[2].astype(float)
    direct = np.load(HERE/'outputs/crop-point/alphas-full.npy')[2].astype(float)
    accessory = np.load(ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz')
    jewelry = accessory['alpha']
    # The component ROI has source-derived semantics; it is not a hand-drawn
    # target outline. Both moving source and destination enter the same domain.
    local = binary_dilation(masks['accessories']>.01, iterations=8)
    fixed = (masks['ears']>=.5)|(masks['hair']>=.5)
    local &= ~fixed
    bg_known = (full<.005)&(direct<.005)&(jewelry<.005)&(masks['hair']<.05)&(masks['ears']<.05)
    # The source pixels are observed. Nearest-background extension underneath
    # skin is a hypothesis, saved and reviewed rather than called ground truth.
    bg_estimate, bg_distance = nearest(rgb, bg_known)
    skin_known = (direct>.995)&(jewelry<.001)&(masks['skin']>.8)
    skin_estimate, skin_distance = nearest(rgb, skin_known)
    metal_known = (jewelry>.98)&(masks['accessories']>.05)
    metal_estimate, _ = nearest(rgb, metal_known)
    # Reconstruct anti-aliased metal from nearby observed backing before it
    # becomes a source-stationary layer. At opaque texels keep original RGB.
    beneath = direct[...,None]*skin_estimate+(1-direct[...,None])*bg_estimate
    metal = np.clip((rgb-(1-jewelry[...,None])*beneath)/np.maximum(jewelry[...,None],1e-6),0,1)
    metal[jewelry<.01]=metal_estimate[jewelry<.01]
    metal[jewelry>.98]=rgb[jewelry>.98]
    clean = np.clip((rgb-jewelry[...,None]*metal)/np.maximum(1-jewelry[...,None],1e-6),0,1)
    clean[jewelry>.98]=beneath[jewelry>.98]
    face_color=np.clip((clean-(1-direct[...,None])*bg_estimate)/np.maximum(direct[...,None],1e-6),0,1)
    face_color[direct<.01]=skin_estimate[direct<.01]
    source_reconstruction=jewelry[...,None]*metal+(1-jewelry[...,None])*(direct[...,None]*face_color+(1-direct[...,None])*bg_estimate)
    Image.fromarray(np.rint(np.clip(source_reconstruction,0,1)*255).astype('uint8')).save(out/'source-reconstruction.png')
    Image.fromarray(np.rint(bg_estimate*255).astype('uint8')).save(out/'background-estimate.png')
    for name,m in [('local-domain',local),('scene-observed',bg_known),('skin-alpha',direct),('fixed',fixed)]:mask_save(out/f'{name}.png',m)
    np.savez_compressed(out/'source-layers.npz',skinAlpha=direct,skin=face_color,background=bg_estimate,jewelry=jewelry,metal=metal,local=local,sceneDistance=bg_distance)
    report=dict(sourceSha256=sha(source_path), graphSha256=sha(graph_path), sourceReconstructionMaxByte=float(np.max(np.abs(source_reconstruction-rgb)[local])*255),
                sourceReconstructionMeanByte=float(np.mean(np.abs(source_reconstruction-rgb)[local])*255),rows=[],accepted=False)
    for amount in [100,-100]:
        material=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full'
        layers=material/'layers'
        state=json.loads((layers/'state.json').read_text())
        cw,ch=state['width'],state['height'];ox,oy=state['origin'];cs=np.s_[oy:oy+ch,ox:ox+cw]
        xy=np.fromfile(layers/'inverse-source-xy-f32le.bin',dtype='<f4').reshape(ch,cw,2)
        physical=np.fromfile(layers/'source-alpha-f32le.bin',dtype='<f4').reshape(ch,cw).astype(float)
        fg=np.asarray(Image.open(layers/'foreground.png').convert('RGB')).astype(float)/255
        # Use one smooth source transition, not isolated support islands.
        weight=np.clip(distance_transform_edt(local)/4,0,1)
        weight=weight*weight*(3-2*weight)
        alpha=physical*(1-weight[cs])+direct[cs]*weight[cs]
        color=fg*(1-weight[cs][...,None])+face_color[cs]*weight[cs][...,None]
        moved_alpha=sample(alpha,xy)[...,0]
        moved_color=sample(color*alpha[...,None],xy)
        # Preserve the observed stationary residual, including real earring
        # highlights. Only the difference in moving skin premultiplication
        # and exposed background is recomposited. This guarantees identity
        # at zero motion even when hidden colors are uncertain; it does not
        # assert that an incorrect source matte can be repaired by identity.
        skin_delta=moved_color-color*alpha[...,None]
        exposure_delta=(alpha-moved_alpha)[...,None]*bg_estimate[cs]
        result=rgb[cs]+(1-jewelry[cs][...,None])*(skin_delta+exposure_delta)
        base=np.asarray(Image.open(material/'candidate.png').convert('RGBA'))
        output=base.copy()
        transported=sample(weight[cs],xy)[...,0]
        write_weight=np.maximum(weight[cs],transported)
        # Source ear/hair and opaque metal stay exact; no output redetection.
        write_weight[fixed[cs]|(jewelry[cs]>=.98)]=0
        patch=output[cs][:,:,:3].astype(float)/255
        patch=patch*(1-write_weight[...,None])+result*write_weight[...,None]
        output[cs][:,:,:3]=np.floor(np.clip(patch,0,1)*255+.5).astype('uint8')
        d=out/str(amount);d.mkdir()
        Image.fromarray(output).save(d/'candidate.png')
        mask_save(d/'write-weight.png',write_weight)
        for label,box in [('face',[287,197,730,709]),('left',[320,400,420,550]),('right',[590,400,700,550])]:
            montage(d/f'{label}-1x.png',[Image.fromarray(source),Image.fromarray(base),Image.fromarray(output)],['Source','V36','Source residual + skin delta'],box)
        report['rows'].append(dict(amount=amount,inverseSha256=sha(layers/'inverse-source-xy-f32le.bin'),changedPixels=int(np.any(output!=base,axis=2).sum()),
                                   opaqueMetalChanged=int(np.any(output[jewelry>=.98]!=base[jewelry>=.98],axis=1).sum()),outputSha256=sha(d/'candidate.png')))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""Combine the existing explicit B hypothesis with alternative full backings.

Reconstruct the frozen prior B result exactly before substituting any backing.
This keeps the unknown depth-order limitation visible; B is not auto-selected.
"""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
HAIR=ROOT/'experiments/hair-skin-decomposition-v1-20261015'
sys.path.insert(0,str(HAIR))
from compose_real import sample
sys.path.insert(0,str(ROOT/'experiments/face-background-context-20261015'))
from context import montage,sha


def main():
    out=HERE/'combined';out.mkdir(exist_ok=False)
    material=ROOT/'experiments/face-material-fill-v1-20261015/woman-100-full';layers=material/'layers'
    state=json.loads((layers/'state.json').read_text());cw,ch=state['width'],state['height'];ox,oy=state['origin']
    estimates=json.loads((HAIR/'source-decomposition-corrected-prompts/report.json').read_text())
    estimate=next(r for r in estimates['rows'] if r['name']=='portrait-woman' and r['variant']=='source-fit-srgb')
    data_path=HAIR/estimate['directory']/'layers.npz';data=np.load(data_path)
    x0,y0,x1,y1=estimate['crop'];sl=np.s_[y0-oy:y1-oy,x0-ox:x1-ox]
    source=np.asarray(Image.open(ROOT/estimate['source']).convert('RGBA'))
    trusted=np.zeros((ch,cw),bool);trusted[sl]=data['trusted']
    pure=np.zeros((ch,cw),bool);pure[sl]=data['pureReclaim']
    physical=np.fromfile(layers/'source-alpha-f32le.bin','<f4').reshape(ch,cw).astype('float64')
    p=physical[sl].copy();p[data['trusted']]=1-data['alpha'][data['trusted']];p[data['pureReclaim']]=1;physical[sl]=p
    skin_color=np.asarray(Image.open(layers/'foreground.png').convert('RGB'),dtype='float64')/255
    p=skin_color[sl].copy();p[data['trusted']]=data['skin'][data['trusted']];p[data['pureReclaim']]=source[y0:y1,x0:x1,:3][data['pureReclaim']]/255;skin_color[sl]=p
    hair_color=source[oy:oy+ch,ox:ox+cw,:3].astype('float64')/255
    p=hair_color[sl].copy();p[data['trusted']|data['pureReclaim']]=data['hair'][data['trusted']|data['pureReclaim']];hair_color[sl]=p
    xy=np.fromfile(layers/'inverse-source-xy-f32le.bin','<f4').reshape(ch,cw,2)
    alpha_b=sample(physical,xy)[...,0];rgb_b=sample(skin_color*physical[...,None],xy)
    old_dir=HAIR/'occlusion-corrected-prompts/portrait-woman-100/lama-256'
    writes=np.asarray(Image.open(old_dir/'B-write.png'))>0

    def render(folder):
        initial=np.asarray(Image.open(folder/'candidate.png').convert('RGBA'))
        background=np.asarray(Image.open(folder/'background-layer.png').convert('RGB'),dtype='float64')/255
        background[trusted|pure]=hair_color[trusted|pure]
        rgb=rgb_b+(1-alpha_b[...,None])*background
        result=initial.copy();patch=result[oy:oy+ch,ox:ox+cw,:3]
        patch[writes]=np.floor(np.clip(rgb[writes],0,1)*255+.5).astype('uint8')
        assert np.array_equal(result[...,3],initial[...,3])
        assert np.array_equal(patch[~writes],initial[oy:oy+ch,ox:ox+cw,:3][~writes])
        return result

    previous_folder=ROOT/'experiments/face-background-context-20261015/woman-100-local/256'
    previous=render(previous_folder)
    assert np.array_equal(previous,np.asarray(Image.open(old_dir/'B-skin-front.png').convert('RGBA'))),'Existing B reproduction failed'
    report=dict(scope='B (skin in front) remains explicit UNKNOWN order hypothesis. No automatic depth result.',
        previousRecompositionRGBAExact=True,scriptSha256=sha(Path(__file__)),sourceDataSha256=sha(data_path),
        inverseSha256=sha(layers/'inverse-source-xy-f32le.bin'),writeDomainSha256=sha(old_dir/'B-write.png'),rows=[])
    images=[source,previous];labels=['Source','Prior B / LaMa256']
    for name in ['256/neural','512/neural','256/prior-only']:
        folder=HERE/'outputs'/name;result=render(folder);dest=out/name;dest.mkdir(parents=True)
        Image.fromarray(result).save(dest/'candidate.png')
        montage([source,previous,result],['Source','Previous B',f'B + {name}'],dest/'left-1x.png',box=(300,480,600,1170))
        images.append(result);labels.append(name)
        report['rows'].append(dict(name=name,inputCandidateSha256=sha(folder/'candidate.png'),inputBackingSha256=sha(folder/'background-layer.png'),outputSha256=sha(dest/'candidate.png')))
    montage(images,labels,out/'left-comparison-1x.png',box=(340,820,500,1140))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

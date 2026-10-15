"""Restrict the last cached v5 result to the frozen v2 original-strip gate.

No new inference, warp, foreground update, or destination-edge write is allowed.
The old unknown and insufficient-scene refusal masks remain authoritative.
"""
from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import sha,mask_save
from compose_real import montage


def main():
    out=HERE/'final-gated-v5';out.mkdir(exist_ok=False)
    v5=HERE/'neural-backing-v5-virtual-scene';r=json.loads((v5/'report.json').read_text())
    frozen=HERE/'outputs-v2/combined-point';layers=np.load(frozen/'source-layers.npz')
    support=layers['pure']|layers['trusted']
    source=np.asarray(Image.open(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png').convert('RGBA'))
    report={'scope':'Final strict old-strip-only hypothesis. Frozen v2 is the initial image; cached v5 pixels can replace only source support inside original v2 write permission, outside all original unknown/insufficient refusals. No destination recompose or inference.',
            'accepted':False,'visualConclusion':'Still visible doubled/broken edge; stop this branch.',
            'sourceSha256':r['sourceSha256'],'graphSha256':r['graphSha256'],'sourceLayersSha256':sha(frozen/'source-layers.npz'),
            'cachedNeuralReportSha256':sha(v5/'report.json'),'scriptSha256':sha(Path(__file__)),'rows':[]}
    for row in r['rows']:
        size,amount=row['size'],row['amount'];baseline_dir=frozen/f'{amount}-source-local-scene'
        baseline=np.asarray(Image.open(baseline_dir/'candidate.png').convert('RGBA'))
        proposal=np.asarray(Image.open(HERE/row['directory']/'candidate.png').convert('RGBA'))
        mat=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full'
        state=json.loads((mat/'layers/state.json').read_text());ox,oy=state['origin'];cw,ch=state['width'],state['height'];cs=np.s_[oy:oy+ch,ox:ox+cw]
        previous_write=np.asarray(Image.open(baseline_dir/'allowed-write.png'))>0
        unknown=np.asarray(Image.open(baseline_dir/'unknown-new-coverage.png'))>0
        insufficient=np.asarray(Image.open(baseline_dir/'insufficient-scene-support.png'))>0
        target=support[cs]&previous_write&~unknown&~insufficient
        full_target=np.zeros(source.shape[:2],bool);full_target[cs]=target
        output=baseline.copy();output[full_target]=proposal[full_target]
        changed=np.any(output!=baseline,axis=2)
        assert not (changed&~full_target).any()
        assert not (target&unknown).any() and not (target&insufficient).any()
        assert np.array_equal(output[layers['opaque']],source[layers['opaque']])
        assert np.array_equal(output[...,3],source[...,3])
        dest=out/str(size)/str(amount);dest.mkdir(parents=True)
        Image.fromarray(output).save(dest/'candidate.png')
        for n,m in [('final-write',target),('original-unknown',unknown),('original-insufficient',insufficient)]:mask_save(dest/(n+'.png'),m)
        v36=np.asarray(Image.open(mat/'candidate.png').convert('RGBA'))
        montage(dest/'face-1x.png',[Image.fromarray(source),Image.fromarray(v36),Image.fromarray(baseline),Image.fromarray(output)],['Source','V36','Frozen v2','Strict old-strip v5'],[287,197,730,709])
        for name,crop in [('left',[326,409,397,548]),('right',[620,414,692,551])]:
            montage(dest/(name+'-1x.png'),[Image.fromarray(source),Image.fromarray(v36),Image.fromarray(baseline),Image.fromarray(output)],['Source','V36','v2','Strict v5'],crop)
        report['rows'].append({'size':size,'amount':amount,'writePixels':int(target.sum()),'changedPixels':int(changed.sum()),
            'outsideOriginalSupportChanges':0,'originalUnknownChanges':0,'originalInsufficientChanges':0,'opaqueMetalChanges':0,
            'initialSha256':sha(baseline_dir/'candidate.png'),'proposalSha256':sha(HERE/row['directory']/'candidate.png'),
            'inverseSha256':sha(mat/'layers/inverse-source-xy-f32le.bin'),'outputSha256':sha(dest/'candidate.png'),'directory':str(dest.relative_to(HERE))})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'rows':len(report['rows']),'allStrictInvariants':'passed','accepted':False}))


if __name__=='__main__':main()

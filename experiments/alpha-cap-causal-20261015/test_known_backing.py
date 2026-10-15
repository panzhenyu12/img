"""Known alpha/color counterexamples, evaluated after changing the backing.

No source-fit threshold is called a visual pass. Donor-mismatch case deliberately
has a correct learned alpha but mixed fine-hair colors unlike opaque donors.
"""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import distance_transform_edt
from alpha import prepare,solve,VARIANTS
from run_real import save,panel,sha
HERE=Path(__file__).resolve().parent

def fixture(kind):
    h,w=128,192;y,x=np.indices((h,w),dtype='float64');edge=72+7*np.sin(y/27)
    t=np.clip((x-edge+4)/8,0,1);truth=1-t*t*(3-2*t)
    hair=np.zeros((h,w,3))+[.14,.095,.065];skin=np.zeros((h,w,3))+[.85,.65,.50]
    if kind!='constant-exact':
        hair+=(.035*np.sin(y/7)+.012*np.sin(x/9))[...,None]
        skin+=(.035*np.sin(y/19)+.0005*(x-90))[...,None]
    fine=np.zeros((h,w),bool)
    if kind=='fine-hair-wrong-donor':
        fine=(x>edge+6)&(x<edge+11)&(np.sin(y/9)>.1)
        truth[fine]=.65
        # Known thin warm hair lies on the donor-to-skin color line. A RGB
        # compatibility gate cannot disambiguate this from lower opacity.
        hair[fine]=.55*hair[fine]+.45*skin[fine]
    if kind=='low-separation-noisy-donor':
        hair=np.zeros((h,w,3))+[.47,.39,.31]
        skin=np.zeros((h,w,3))+[.50,.42,.34]
        hair+=(.02*np.sin(y/2)+.01*np.sin(x/2))[...,None]
        skin+=(.02*np.cos(y/2)+.01*np.cos(x/2))[...,None]
    source=truth[...,None]*hair+(1-truth[...,None])*skin
    under=np.zeros((h,w,3))+[.08,.065,.05]
    under+=(.015*np.sin(y/15))[...,None]
    expected=truth[...,None]*hair+(1-truth[...,None])*under
    raw=truth.copy();mixed=(truth>.001)&(truth<.995)
    bias={'textured-bias-minus035':-.35,'textured-bias-plus035':.35}.get(kind,0)
    raw[mixed]=np.clip(raw[mixed]+bias,.002,.994)
    hair_core=truth==1;skin_core=truth==0
    dh=distance_transform_edt(~hair_core);ds=distance_transform_edt(~skin_core)
    domain=(dh<=20)&(ds<=20)&(raw>.001)&(raw<.995)&~hair_core&~skin_core
    return locals()

def main():
    out=HERE/'synthetic';out.mkdir(exist_ok=False)
    # Neither donor combination can explain the white middle row. Reject its
    # interval, rather than clipping a nonphysical solution to an endpoint.
    impossible=np.zeros((3,3,3))+.4;impossible[0]=.2;impossible[1]=1
    hc=np.zeros((3,3),bool);hc[0]=True;sc=np.zeros((3,3),bool);sc[2]=True
    test=prepare(impossible,np.full((3,3),.5),hc,sc,~(hc|sc),1)
    assert not test['intervalCompatible'][1].any()
    assert np.array_equal(test['proposals']['donor-interval'][1],np.full(3,.5))
    report=dict(scope=__doc__,scriptSha256=sha(Path(__file__)),alphaScriptSha256=sha(HERE/'alpha.py'),
        physicallyImpossibleIntervalAbstains=True,rows=[])
    kinds=['constant-exact','textured-exact','textured-bias-minus035','textured-bias-plus035','fine-hair-wrong-donor','low-separation-noisy-donor']
    for kind in kinds:
        f=fixture(kind);folder=out/kind;folder.mkdir()
        p=prepare(f['source'],f['raw'],f['hair_core'],f['skin_core'],f['domain'],2)
        np.savez_compressed(folder/'truth.npz',alpha=f['truth'],hair=f['hair'],skin=f['skin'],source=f['source'],replacement=f['under'],
            expected=f['expected'],rawAlpha=f['raw'],domain=f['domain'],fineHair=f['fine'])
        images=[f['source'],f['expected']];labels=['Source','Known new backing']
        for variant in VARIANTS:
            r=solve(f['source'],p,variant)
            render=r['alpha'][...,None]*r['hair']+(1-r['alpha'][...,None])*f['under']
            candidate=f['expected'].copy();candidate[f['domain']]=render[f['domain']]
            error=np.abs(candidate-f['expected'])*255
            region=f['domain'];fine=f['fine']&region
            stats=dict(case=kind,variant=variant,**{k:v for k,v in r['stats'].items() if k!='variant'},
                newBackingMeanAbsByte=float(error[region].mean()),newBackingMaxAbsByte=float(error[region].max()),
                alphaMeanAbsoluteError=float(np.abs(r['alpha'][region]-f['truth'][region]).mean()),
                fineHairPixels=int(fine.sum()),fineHairNewBackingMeanAbsByte=float(error[fine].mean()) if fine.any() else None,
                fineHairMeanAlpha=float(r['alpha'][fine].mean()) if fine.any() else None,
                opacityCoreChanged=int(np.count_nonzero(r['alpha'][f['hair_core']]!=1)))
            assert stats['opacityCoreChanged']==0
            images.append(candidate);labels.append(variant)
            save(folder/f'{variant}.png',candidate)
            np.savez_compressed(folder/f'{variant}.npz',alpha=r['alpha'],sourceResidual=r['residual'],candidate=candidate)
            report['rows'].append(stats);print(json.dumps(stats),flush=True)
        panel(folder/'comparison-1x.png',images,labels)
    # These are causal observations of this frozen synthetic family, not an
    # algorithm acceptance test. Preserve counterexamples to optimistic fit.
    lookup={(r['case'],r['variant']):r for r in report['rows']}
    for kind in ['textured-bias-minus035','textured-bias-plus035']:
        assert lookup[(kind,'uncapped')]['newBackingMeanAbsByte']<lookup[(kind,'bounded-015')]['newBackingMeanAbsByte']
    bad=lookup[('fine-hair-wrong-donor','uncapped')];bound=lookup[('fine-hair-wrong-donor','bounded-015')]
    report['counterexampleObserved']=dict(sourceFitImproves=bad['sourceMeanAbsByte']<bound['sourceMeanAbsByte'],
        fineHairBackingErrorWorsens=bad['fineHairNewBackingMeanAbsByte']>bound['fineHairNewBackingMeanAbsByte'],
        learnedAlphaWasExact=True)
    assert all(report['counterexampleObserved'].values()),report['counterexampleObserved']
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

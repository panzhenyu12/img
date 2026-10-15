"""Apply frozen semantic candidate selection and audit source-only anatomy."""
from pathlib import Path
import argparse,importlib.util,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from run import runner,ROOT,HERE

spec=importlib.util.spec_from_file_location('frozen_selector',HERE/'selector-snapshot.py')
selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def panel(path,rgb,masks,alphas,official,selected,prompt,box):
    images=[Image.fromarray(rgb),runner.cutout(rgb,masks[prompt['kind']]),runner.cutout(rgb,alphas[official])]
    labels=['Source','Parser reference',f'Official {official}']
    if selected is not None:images.append(runner.cutout(rgb,alphas[selected]));labels.append(f'Automatic {selected}')
    else:images.append(Image.new('RGB',(rgb.shape[1],rgb.shape[0]),'#222'));labels.append('ABSTAIN')
    w,h=box[2]-box[0],box[3]-box[1];canvas=Image.new('RGB',(w*4,h+28),'#222');draw=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(images,labels)):
        canvas.paste(im.crop(box),(i*w,28));draw.text((i*w+4,7),label,fill='white')
    canvas.save(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,default=HERE/'raw');ap.add_argument('--out',type=Path,default=HERE/'selection');args=ap.parse_args()
    raw,out=args.raw.resolve(),args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report=json.loads((raw/'report.json').read_text())
    result={'scope':'Independent source-image audit; selection uses frozen original thresholds. Extra anatomy metrics are audit-only, not hidden selection tuning. No alpha truth or warp validation.',
        'selectorSha256':sha(HERE/'selector-snapshot.py'),'rawReportSha256':sha(raw/'report.json'),'auditSha256':sha(Path(__file__)),'images':[]}
    for photo in report['images']:
        source=ROOT/photo['source'];rgb=np.asarray(Image.open(source).convert('RGB'));h,w=rgb.shape[:2]
        assert sha(source)==photo['sourceSha256']
        p=json.loads((ROOT/photo['graph']).read_text())['operations'][0]['params'];masks={k:runner.zim.coverage(v,w,h) for k,v in p['semanticMasks'].items()}
        row={'name':photo['name'],'source':photo['source'],'sourceSha256':photo['sourceSha256'],'size':[w,h],
           'corePixels':{k:int((masks[k]>=.9).sum()) for k in ['hair','skin','ears','accessories','neck'] if k in masks},'rows':[]}
        for case in photo['rows']:
            directory=out/case['directory'];directory.mkdir(parents=True)
            path=raw/case['directory']/'alphas.npy';assert sha(path)==case['alphaSha256'];alphas=np.load(path)
            selected,decisions=selector.choose(case['alphas'],case['prompt']['kind']);official=case['officialScoreArgmax']
            audit=[]
            for i,a in enumerate(alphas):
                regions={}
                for k in ['hair','skin','ears','accessories','neck','leftEye','rightEye','lips']:
                    if k not in masks:continue
                    core=masks[k]>=.9
                    if k=='skin':core&=(masks['hair']<.05)&(masks['accessories']<.05)
                    values=a[core]
                    regions[k]={'referenceCorePixels':int(core.sum()),'meanAlpha':float(values.mean()) if len(values) else None,
                       'p95Alpha':float(np.percentile(values,95)) if len(values) else None,
                       'retainedAtHalf':float((values>=.5).mean()) if len(values) else None,
                       'nontrivialAlphaOver005Fraction':float((values>.05).mean()) if len(values) else None}
                audit.append({'index':i,'regions':regions})
            if selected is not None:
                np.save(directory/'selected-alpha.npy',alphas[selected],allow_pickle=False);runner.png(directory/'selected-alpha.png',alphas[selected]);runner.cutout(rgb,alphas[selected]).save(directory/'selected-cutout-full.png')
            runner.cutout(rgb,alphas[official]).save(directory/'official-cutout-full.png')
            panel(directory/'face-comparison-1x.png',rgb,masks,alphas,official,selected,case['prompt'],photo['faceCrop'])
            for k in ['ears','accessories']:
                if k not in masks:continue
                yy,xx=np.nonzero(masks[k]>=.5)
                if len(yy):
                    box=(max(0,int(xx.min())-20),max(0,int(yy.min())-20),min(w,int(xx.max())+21),min(h,int(yy.max())+21))
                    panel(directory/f'{k}-audit-1x.png',rgb,masks,alphas,official,selected,case['prompt'],box)
            decision={'directory':case['directory'],'prompt':case['prompt'],'alphaSha256':case['alphaSha256'],
              'officialIndex':official,'selectedIndex':selected,'decisions':decisions,'officialMetrics':case['alphas'][official],
              'selectedMetrics':next((m for m in case['alphas'] if m['index']==selected),None),'allCandidateAnatomyAudit':audit}
            row['rows'].append(decision)
            print(json.dumps({k:decision[k] for k in ['directory','officialIndex','selectedIndex','decisions','selectedMetrics']}),flush=True)
        result['images'].append(row);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()

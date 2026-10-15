"""Frozen full-resolution replay of the combined physical-layer hypothesis."""
from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from review_body_refinement import raster


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    base=ROOT/'test/results/accessory-matting-zim-v1-20261014'
    seed=base/'physical-support-replay/faceWidth100-physical-support.json'
    out=base/'full-replay';out.mkdir(exist_ok=False)
    binary=Path(json.loads((base/'physical-support-replay/manifest.json').read_text())['binary'])
    baseline=ROOT/'test/results/face-joint-v19-production-20261014/image-render'
    bh,oldh=sha(binary),sha(baseline)
    source=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
    original=Image.open(source).convert('RGBA');pix=np.asarray(original);w,h=original.size
    graph=json.loads(seed.read_text());p=graph['operations'][0]['params'];sem=p['semanticMasks']
    fixed={key:raster(sem[key],w,h)>=.5 for key in ['hair','ears','accessories']}
    reg=p['region'];yy,xx=np.indices((h,w));outside=((xx+.5)/w<reg['x'])|((xx+.5)/w>reg['x']+reg['width'])|((yy+.5)/h<reg['y'])|((yy+.5)/h>reg['y']+reg['height'])
    person=raster(p['subjectMask'],w,h)
    report={'scope':'One original, frozen source-only ZIM physical support adaptation, 512px raster protocol; no redetection or production changes. Baseline is live-v19-equivalent immutable native; attachment-control uses same refined masks and guarded V21 binary, restoring old attachment semantics. Contracts do not imply aesthetic acceptance.',
            'sourceSha256':sha(source),'seedGraphSha256':sha(seed),'binary':str(binary),'binarySha256':bh,'productionBaselineSha256':oldh,'rows':[]}
    for amount in [0,-100,-30,30,100]:
        g=copy.deepcopy(graph);g['operations'][0]['params']['adjustments']={'faceWidth':amount};gp=out/f'faceWidth{amount}.json';gp.write_text(json.dumps(g,separators=(',',':')))
        panels=[('Original',original)];row={'amount':amount,'graph':str(gp),'graphSha256':sha(gp),'renders':{}}
        for label,worker in [('production',baseline),('physical',binary),('attachment-control',binary)]:
            actual=out/f'faceWidth{amount}-{label}.png';env=dict(os.environ)
            if label=='attachment-control':env['FACE_V21_USE_ALL_ATTACHMENTS']='1'
            if label=='physical' and amount:
                debug=out/f'debug{amount}';debug.mkdir();env['FACE_V21_DEBUG_DIR']=str(debug)
            # Production comparison uses the original detector graph.
            run_graph=gp
            if label=='production':
                old=json.loads((source.with_name('face-graph.json')).read_text());old['operations'][0]['params']['adjustments']={'faceWidth':amount};run_graph=out/f'faceWidth{amount}-production.json';run_graph.write_text(json.dumps(old,separators=(',',':')))
            started=time.perf_counter();result=subprocess.run([str(worker),'--input',str(source),'--graph',str(run_graph),'--output',str(actual)],env=env,capture_output=True)
            (out/f'faceWidth{amount}-{label}.log').write_bytes(result.stdout+result.stderr);result.check_returncode()
            im=Image.open(actual).convert('RGBA');a=np.asarray(im);changed=np.any(a!=pix,axis=2)
            metrics={'output':str(actual),'outputSha256':sha(actual),'processMs':round((time.perf_counter()-started)*1000,2),'changedPixels':int(changed.sum()),
                     'alphaChanges':int((a[:,:,3]!=pix[:,:,3]).sum()),'outsideRegionChanges':int(changed[outside].sum()),
                     'fixedChanges':{key:int(changed[mask].sum()) for key,mask in fixed.items()},'personUnder001ChangedPixels':int(changed[person<.01].sum())}
            if label!='production':
                assert not metrics['alphaChanges'] and not metrics['outsideRegionChanges']
                assert not any(metrics['fixedChanges'].values()),metrics
            if amount==0:assert not changed.any()
            row['renders'][label]=metrics
            panels.append((label,im))
        box=(280,160,750,700);cw,ch=box[2]-box[0],box[3]-box[1]
        sheet=Image.new('RGB',(cw*len(panels),ch+28),'#222');d=ImageDraw.Draw(sheet)
        for i,(label,im) in enumerate(panels):sheet.paste(im.convert('RGB').crop(box),(i*cw,28));d.text((i*cw+5,8),f'{label} / {amount}',fill='white')
        sheet.save(out/f'faceWidth{amount}-comparison-1x.png');report['rows'].append(row)
        assert sha(binary)==bh and sha(baseline)==oldh
        (out/'report.json').write_text(json.dumps(report,indent=2));print(amount,flush=True)


if __name__=='__main__':main()

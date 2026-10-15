"""Same-source/graph immutable-native comparison for source-curve experiment."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time
from PIL import Image, ImageDraw
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    out=HERE/'outputs';out.mkdir(exist_ok=False)
    built=ROOT/'.run/face-source-curve-v31-target/release/image-render'
    binary=out/f'image-render-{sha(built)}';shutil.copy2(built,binary)
    base=ROOT/'test/results/accessory-matting-zim-v1-20261014/full-replay'
    source=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
    original=Image.open(source).convert('RGBA');pix=np.asarray(original)
    old=ROOT/'test/results/face-joint-v19-production-20261014/image-render'
    report={'sourceSha256':sha(source),'binarySha256':sha(binary),'binary':str(binary),'scope':'One original, two frozen source supports. V32 attachment-conditioned source-curve velocity replaces width-only field; no semantic-layer clipping/neck relighting. All source masks unchanged within each ablation.','rows':[]}
    for support in ['combined','parser']:
        for amount in [0,30,100,-30,-100]:
            dest=out/f'{support}-{amount}';dest.mkdir()
            seed=base/f'faceWidth{amount}.json' if support=='combined' else source.with_name('face-graph.json')
            graph=json.loads(seed.read_text());graph['operations'][0]['params']['adjustments']={'faceWidth':amount}
            gp=dest/'graph.json';gp.write_text(json.dumps(graph,separators=(',',':')))
            t=time.perf_counter();r=subprocess.run([str(binary),'--input',str(source),'--graph',str(gp),'--output',str(dest/'candidate.png')],env={**os.environ,'FACE_V21_DEBUG_DIR':str(dest/'debug'),'FACE_V32_DEBUG_DIR':str(dest/'debug')},capture_output=True)
            (dest/'render.log').write_bytes(r.stdout+r.stderr);r.check_returncode()
            cand=Image.open(dest/'candidate.png').convert('RGBA');a=np.asarray(cand)
            row={'support':support,'amount':amount,'graphSha256':sha(gp),'outputSha256':sha(dest/'candidate.png'),'timeMs':(time.perf_counter()-t)*1000,'changedPixels':int(np.any(a!=pix,axis=2).sum()),'newCurveUsed':(dest/'debug/source-width.json').exists()}
            if amount==0: assert np.array_equal(a,pix)
            assert sha(binary)==report['binarySha256']
            comparison=Image.open(base/f'faceWidth{amount}-physical.png').convert('RGBA') if support=='combined' else Image.open(base/f'faceWidth{amount}-production.png').convert('RGBA')
            for name,box in {'face':(280,160,750,700),'neck':(330,450,690,735),'leftEar':(315,360,420,560),'rightEar':(595,360,710,560)}.items():
                w,h=box[2]-box[0],box[3]-box[1];sheet=Image.new('RGB',(w*3,h+24),'#222');d=ImageDraw.Draw(sheet)
                for i,(label,im) in enumerate([('Original',original),('V21 same support' if support=='combined' else 'Production V19',comparison),('V32',cand)]):
                    sheet.paste(im.convert('RGB').crop(box),(i*w,24));d.text((i*w+3,5),label,fill='white')
                sheet.save(dest/f'{name}-comparison-1x.png')
            report['rows'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2));print(row,flush=True)

if __name__=='__main__':main()

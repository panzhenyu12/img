"""Repeatable optional texture-refinement comparison; never assigns scores."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, default=ROOT/'experiments/inpaint/target/release/inpaint-experiment')
    args = parser.parse_args()
    data = ROOT/'doc/qa/background-lama'
    out = ROOT/'test/results/patch-refinement'
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for case, result_file in [('street', 'street-http.png'), ('crowd', 'crowd-http-fixed.png'), ('architecture', 'architecture-http-fixed.png')]:
        source = Image.open(data/f'{case}-source.png').convert('RGBA')
        original = np.asarray(source)
        candidate = Image.open(data/result_file).convert('RGBA')
        target = np.asarray(Image.open(data/f'{case}-mask.png').convert('L'))
        protect_file = data/f'{case}-protect.png'
        protect = np.asarray(Image.open(protect_file).convert('L')) if protect_file.exists() else np.zeros(target.shape, np.uint8)
        canonical=(target>0)&(protect==0)&(original[:,:,3]>0)
        canonical_protect=protect>0
        # Restore immutable pixels in both results under the exact same
        # full-resolution PNG target, without graph-mask quantization.
        baseline=np.array(candidate)
        baseline[~canonical]=original[~canonical]
        baseline[:,:,3]=original[:,:,3]
        candidate=Image.fromarray(baseline)
        output = out/f'{case}-refined.png'
        command=[str(args.binary),'--input',str(data/f'{case}-source.png'),'--candidate',str(data/result_file),'--mask',str(data/f'{case}-mask.png'),'--output',str(output)]
        if protect_file.exists(): command += ['--protection',str(protect_file)]
        started=time.monotonic()
        subprocess.run(command,check=True,capture_output=True)
        elapsed=(time.monotonic()-started)*1000
        result=np.asarray(Image.open(output).convert('RGBA'))
        change=np.any(original!=result,axis=2)
        row={'case':case,'totalProcessMs':round(elapsed,2),'outsideChangedPixels':int(change[~canonical].sum()),'protectionChangedPixels':int(change[canonical_protect].sum()),'alphaChangedPixels':int(np.count_nonzero(result[:,:,3]!=original[:,:,3])),'maskPixelsDifferentFromPreviousPillowMask':int(np.count_nonzero(canonical!=((target>0)&(protect==0)&(original[:,:,3]>0)))),'sourceSha256':hashlib.sha256((data/f'{case}-source.png').read_bytes()).hexdigest(),'candidateSha256':hashlib.sha256((data/result_file).read_bytes()).hexdigest(),'binarySha256':hashlib.sha256(args.binary.read_bytes()).hexdigest()}
        if case=='architecture':
            clean=np.asarray(Image.open(data/'architecture-clean.png').convert('RGBA'))
            row['targetMaeBefore']=float(np.abs(np.asarray(candidate)[:,:,:3].astype(float)-clean[:,:,:3])[canonical].mean())
            row['targetMaeAfter']=float(np.abs(result[:,:,:3].astype(float)-clean[:,:,:3])[canonical].mean())
        ys,xs=np.nonzero(canonical)
        box=(max(0,int(xs.min())-35),max(0,int(ys.min())-35),min(source.width,int(xs.max())+36),min(source.height,int(ys.max())+36))
        views=[source.crop(box),candidate.crop(box),Image.fromarray(result).crop(box)]
        grid=Image.new('RGB',(views[0].width*3,views[0].height+25),'white')
        draw=ImageDraw.Draw(grid)
        for i,(view,label) in enumerate(zip(views,['Source','LaMa baseline','Guided PatchMatch'])):
            grid.paste(view,(i*view.width,25)); draw.text((i*view.width+4,6),label,fill='black')
        grid.save(out/f'{case}-comparison.png')
        report.append(row)
        print(json.dumps(row),flush=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()

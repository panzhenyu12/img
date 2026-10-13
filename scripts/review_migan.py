"""Pinned MI-GAN experiment, deliberately separate from shipped operations.

The original ONNX pipeline internally resizes RGB and binary masks with
different footprints. Feed a positive-area resized, symmetric 512 square to
avoid masked RGB leaking into known context. This does not fix model quality.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

ROOT=Path(__file__).resolve().parents[1]
REVISION='406830d0fa60666da0071c342ad2fbc8f30c5c64'
SHA='593eba0b7e04730f1b61c0a3cbca68d97d8d6a7ff5c6a44a7b9d7fcd880fc5ae'
URL=f'https://huggingface.co/andraniksargsyan/migan/resolve/{REVISION}/migan.onnx'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=ROOT/'.models/background/migan.onnx')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    if not args.model.is_file() and args.download:
        args.model.parent.mkdir(parents=True,exist_ok=True)
        partial=args.model.with_suffix('.part')
        urllib.request.urlretrieve(URL,partial)
        if hashlib.sha256(partial.read_bytes()).hexdigest()!=SHA:raise RuntimeError('MI-GAN checksum mismatch')
        partial.replace(args.model)
    if hashlib.sha256(args.model.read_bytes()).hexdigest()!=SHA:raise RuntimeError('MI-GAN checksum mismatch')
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    started=time.monotonic();session=ort.InferenceSession(str(args.model),options,providers=['CPUExecutionProvider']);load_ms=(time.monotonic()-started)*1000
    data=ROOT/'doc/qa/background-lama';out=ROOT/'test/results/migan-experiment';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case,baseline in [('street','street-http.png'),('crowd','crowd-http-fixed.png'),('architecture','architecture-http-fixed.png')]:
        source=Image.open(data/f'{case}-source.png').convert('RGBA');original=np.array(source)
        target=np.array(Image.open(data/f'{case}-mask.png').convert('L'))>0
        path=data/f'{case}-protect.png';protection=np.array(Image.open(path).convert('L'))>0 if path.exists() else np.zeros(target.shape,bool)
        target&=~protection&(original[:,:,3]>0)
        unknown=binary_dilation(target,iterations=4)|protection|(original[:,:,3]==0)
        rgb=original[:,:,:3].copy();rgb[unknown]=0
        scale=min(1,512/max(source.size));size=(round(source.width*scale),round(source.height*scale))
        rgb=np.array(Image.fromarray(rgb).resize(size,Image.Resampling.BOX))
        coverage=np.asarray(Image.fromarray(unknown.astype(np.float32)).resize(size,Image.Resampling.BOX))>0;rgb[coverage]=0
        top=(512-size[1])//2;left=(512-size[0])//2;pads=((top,512-size[1]-top),(left,512-size[0]-left))
        rgb=np.pad(rgb,(*pads,(0,0)),mode='symmetric');coverage=np.pad(coverage,pads,mode='symmetric')
        started=time.monotonic()
        prediction=session.run(None,{'image':rgb.transpose(2,0,1)[None].copy(),'mask':((~coverage)*255).astype(np.uint8)[None,None]})[0][0].transpose(1,2,0)
        inference_ms=(time.monotonic()-started)*1000
        prediction=np.array(Image.fromarray(prediction[top:top+size[1],left:left+size[0]]).resize(source.size,Image.Resampling.LANCZOS))
        result=original.copy();result[target,:3]=prediction[target];Image.fromarray(result).save(out/f'{case}-area.png')
        ys,xs=np.nonzero(target);box=(max(0,int(xs.min())-35),max(0,int(ys.min())-35),min(source.width,int(xs.max())+36),min(source.height,int(ys.max())+36))
        views=[source.crop(box),Image.open(data/baseline).crop(box),Image.fromarray(result).crop(box)]
        grid=Image.new('RGB',(views[0].width*3,views[0].height+25),'white');draw=ImageDraw.Draw(grid)
        for i,(view,label) in enumerate(zip(views,['Source','LaMa','MI-GAN area pad512'])):grid.paste(view,(i*view.width,25));draw.text((i*view.width+3,5),label,fill='black')
        grid.save(out/f'{case}-area-comparison.png')
        row={'case':case,'modelBytes':args.model.stat().st_size,'modelSHA256':SHA,'revision':REVISION,'onnxruntime':ort.__version__,'device':'CPU','threads':4,'loadMs':round(load_ms,2),'inferenceMs':round(inference_ms,2),'sourceSHA256':hashlib.sha256((data/f'{case}-source.png').read_bytes()).hexdigest(),'outsideChangedPixels':int(np.any(result!=original,axis=2)[~target].sum()),'alphaChangedPixels':int(np.count_nonzero(result[:,:,3]!=original[:,:,3]))}
        if case=='architecture':row['targetMAE']=float(np.abs(result[:,:,:3].astype(float)-np.array(Image.open(data/'architecture-clean.png').convert('RGB')))[target].mean())
        rows.append(row);print(json.dumps(row),flush=True)
    (out/'area-report.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()

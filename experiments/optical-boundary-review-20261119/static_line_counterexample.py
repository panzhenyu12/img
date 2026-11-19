"""Frozen known-truth shadow/line rejection probe, not portrait visual acceptance.

Exactly one original6 and one expanded4 MEMatte evaluation. The new band is
pure static background in this fixture. No output-selected masks or parameters.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'test/results/optical-boundary-static-line-20261119'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
def image(path, a): Image.fromarray(np.uint8(np.clip(np.floor(a+.5),0,255))).save(path)

def infer(name):
    sys.path[:0] = [str(ROOT/'.run/mematte-20261117-deps'),
                   str(ROOT/'.run/mematte-20261020/detectron2-vendor'),
                   str(ROOT/'experiments/body-matting-model-review-20261020')]
    from run_mematte import official_prediction
    weights = ROOT/'.models/mematte-20261020/MEMatte-AIM500-ViTS.pth'
    expected = '538cf871933dd0663a6bc28a4bec9b870cc2de909c7431b857b72ce427e82a09'
    assert sha(weights) == expected
    source = np.asarray(Image.open(OUT/'source.png').convert('RGB'))
    tri = np.asarray(Image.open(OUT/f'trimap-{name}.png').convert('L'))
    raw, official, evidence = official_prediction(source, tri, weights)
    for kind, a in [('raw',raw),('official',official)]:
        np.save(OUT/f'{name}-{kind}.npy',a)
        image(OUT/f'{name}-{kind}.png',a*255)
    dump(OUT/f'{name}-inference.json',dict(evidence,modelSHA256=expected,
        sourceSHA256=sha(OUT/'source.png'),trimapSHA256=sha(OUT/f'trimap-{name}.png')))

def move(a):
    out = np.zeros_like(a)
    out[:, :-8] = a[:, 8:]
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infer',choices=['original6','expanded4'])
    args=parser.parse_args()
    if args.infer:
        infer(args.infer)
        return
    OUT.mkdir(exist_ok=False)
    y,x=np.mgrid[:256,:256]
    hard=((x-128)/88)**2+((y-128)/104)**2<=1
    truth=gaussian_filter(hard.astype(np.float32),sigma=1.25,truncate=3.0)
    texture=1.4*np.sin(x*.21)+.8*np.cos(y*.17)+.5*np.sin((x+y)*.11)
    bg=np.stack([151+texture+.025*y,172+texture+.012*x,181+texture+.015*y],axis=2)
    line=np.zeros((256,256),float);line[76:181,223:226]=1
    line=gaussian_filter(line,sigma=.65,truncate=3.0)
    bg-=line[:,:,None]*np.array([65,67,67])
    bg=np.floor(np.clip(bg,0,255)+.5).astype(np.uint8)
    fg=np.stack([193+texture+.03*y,139+texture+.02*x,111+texture+.02*y],axis=2)
    fg=np.floor(np.clip(fg,0,255)+.5).astype(np.uint8)
    composite=truth[:,:,None]*fg+(1-truth[:,:,None])*bg
    source=np.floor(composite+.5).astype(np.uint8)
    tri=np.zeros((256,256),np.uint8)
    tri[distance_transform_edt(~hard)<=6]=128
    tri[hard&(distance_transform_edt(hard)>6)]=255
    demoted=(tri==0)&binary_dilation(tri>0,iterations=4)
    expanded=tri.copy();expanded[demoted]=128
    assert np.count_nonzero(truth[demoted])==0
    assert np.array_equal(expanded[tri==255],tri[tri==255])
    line_region=(line>.1)&demoted
    assert line_region.any()
    for name,a in [('source',source),('background-truth',bg),('foreground-truth',fg),
                   ('alpha-truth',truth*255),('hard-support',hard*255),
                   ('trimap-original6',tri),('trimap-expanded4',expanded),
                   ('demoted-band',demoted*255),('static-line-in-new-band',line_region*255)]:
        image(OUT/f'{name}.png',a)
    np.save(OUT/'alpha-truth.npy',truth)
    np.save(OUT/'source-float-truth.npy',composite)
    protocol=dict(status='frozen before inference',width=256,height=256,
        foreground='ellipse center128,128 radii88,104 with fixed sinusoidal texture',
        blur={'sigma':1.25,'truncate':3.0},
        background='fixed textured plane with static dark vertical line x223:226 y76:181 blurred sigma.65 truncate3',
        originalTrimap='Euclidean 6px band around binary foreground; all true soft-alpha pixels contained',
        candidateTrimap='4 iterations cross dilation of original non-BG, demote BG to unknown',
        sourceOnly=True,parameterSearch=False,synthetic=True,
        purpose='Can newly unknown true background receive nonzero alpha? Rejection test, not real-photo acceptance.',
        demotedPixels=int(demoted.sum()),trueAlphaInDemotedMax=float(truth[demoted].max()),
        staticLinePixelsInDemoted=int(line_region.sum()),
        inferenceCount=2,model='same existing MEMatte ViT-S/AIM500 CPU float32',
        translationRiskProbe='Hypothetical left8 px translation of inferred premultiplied layer against exact stationary B. Does not grant core motion permission.',
        inputsSHA256={name:sha(OUT/name) for name in ['source.png','alpha-truth.npy','background-truth.png','trimap-original6.png','trimap-expanded4.png']})
    dump(OUT/'protocol.json',protocol)
    for name in ['original6','expanded4']:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--infer',name],check=True)
    rows=[];panels=[('source',source),('true alpha',np.repeat(truth[:,:,None]*255,3,axis=2))]
    B=bg.astype(float);I=source.astype(float)
    true_output=move(truth[:,:,None]*fg)+(1-move(truth)[:,:,None])*B
    image(OUT/'true-left8-output.png',true_output)
    risk_region=(truth==0)&(move(truth)==0)
    for name in ['original6','expanded4']:
        a=np.load(OUT/f'{name}-official.npy')
        error=abs(a-truth)
        P=I-(1-a[:,:,None])*B
        shifted=move(P)+(1-move(a)[:,:,None])*B
        image(OUT/f'{name}-hypothetical-left8-output.png',shifted)
        edit_error=abs(shifted-true_output)
        row=dict(name=name,alphaMAE=float(error.mean()),
            trueBGDemotedAlphaMean=float(a[demoted].mean()),trueBGDemotedAlphaMax=float(a[demoted].max()),
            trueBGDemotedAbove01=int((a[demoted]>.01).sum()),
            trueBGDemotedAbove10=int((a[demoted]>.1).sum()),
            staticLineNewBandAlphaMean=float(a[line_region].mean()),
            staticLineNewBandAlphaMax=float(a[line_region].max()),
            staticLineNewBandAbove10=int((a[line_region]>.1).sum()),
            hypotheticalTrueBGEditMaxRGBError=float(edit_error[risk_region].max()),
            hypotheticalTrueBGEditedPixelsAbove1=int((edit_error[risk_region].max(axis=1)>1).sum()),
            hypotheticalOutOfRangeChannels=int(((shifted<0)|(shifted>255)).sum()))
        rows.append(row)
        panels.append((name+' alpha',np.repeat(a[:,:,None]*255,3,axis=2)))
    sheet=Image.new('RGB',(256*len(panels),280),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,a) in enumerate(panels):
        sheet.paste(Image.fromarray(np.uint8(np.clip(np.floor(a+.5),0,255))),(i*256,24))
        draw.text((i*256+4,4),name,fill='black')
    sheet.save(OUT/'review.png')
    for name in ['source.png','alpha-truth.npy','background-truth.png','trimap-original6.png','trimap-expanded4.png']:
        assert sha(OUT/name)==protocol['inputsSHA256'][name]
    dump(OUT/'report.json',dict(protocol,status='completed, visual review required',results=rows,
        limitations=['Synthetic distribution; one pass cannot establish real-photo generalization.',
            'False alpha is not motion permission; the hypothetical translation demonstrates risk if alpha were used as permission.',
            'Existing core/replay was not changed or built.']))
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()

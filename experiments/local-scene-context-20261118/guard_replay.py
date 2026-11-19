"""Public rlib comparison for matched 0px / 4px context-guard source scenes."""
from pathlib import Path
import argparse,copy,hashlib,json,subprocess
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[2]
def read(p):return np.asarray(Image.open(p).convert('RGBA'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sheet(path,panels,box):
    x,y,r,b=box;w,h=r-x,b-y;out=Image.new('RGB',(w*len(panels),h+24),'white');draw=ImageDraw.Draw(out)
    for i,(name,a) in enumerate(panels):out.paste(Image.fromarray(a).crop(box).convert('RGB'),(i*w,24));draw.text((i*w+4,4),name,fill='black')
    out.save(path)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-case',required=True,choices=['portrait-614810','portrait-762020']);args=parser.parse_args()
    base=ROOT/'test/results/face-generalization-20261118'/args.source_case;out=base/'context-guard-valid-20261119';neural=base/'source-preparation/head-neural-scene'
    prep=json.loads((out/'report.json').read_text());assert prep['status']=='completed; visual review required'
    spec=json.loads((base/'ordinary-rlib-spec.json').read_text());case=spec['cases'][0];descriptor=json.loads(Path(case['sourcePlatePrior']).read_text());old=read(descriptor['image'])
    region=json.loads((neural/'report.json').read_text())['sourceCropBox'];x,y,r,b=region
    mask=np.asarray(Image.open(neural/'source-inpaint-mask.png').convert('L'))>0;band=mask&(distance_transform_edt(mask)<=32)
    cases=[]
    for name in ['matched0','guard4']:
        prior=old.copy();prior[y:b,x:r][band]=read(out/f'{name}-scene.png')[band]
        Image.fromarray(prior).save(out/f'{name}-full-scene.png');desc=copy.deepcopy(descriptor);desc['image']=str(out/f'{name}-full-scene.png');dump(out/f'{name}-descriptor.json',desc)
        c=copy.deepcopy(case);c.update(id=name,sourcePlatePrior=str(out/f'{name}-descriptor.json'),amounts=[0,30,60,-30,-60]);cases.append(c)
    spec.update(outputDir=str(out/'replay'),cases=cases);dump(out/'spec.json',spec)
    binary=ROOT/'core/image-core/target/release/examples/face-source-replay'
    process=subprocess.run([str(binary),str(out/'spec.json')],capture_output=True,text=True);(out/'execution.log').write_text(process.stdout+process.stderr)
    if process.returncode:raise RuntimeError(process.stderr)
    source=read(case['source']);source_changed=int(np.any(source!=read(case['source']),axis=2).sum());assert source_changed==0
    graph_input_hashes={k:sha(case[k]) for k in ['source','graph','physicalAlpha','sourceMotionInterface','sourceMovingAttachmentSupport']}
    layout=json.loads((base/'review-layout.json').read_text());boxes={'face-neck':layout['nativeFaceNeckBox']}
    if args.source_case=='portrait-614810':boxes.update({'left-edge':[265,280,390,695],'right-edge':[650,280,790,695]})
    else:boxes.update({'left-edge':[340,180,460,515],'right-edge':[640,180,770,515]})
    rows=[]
    for idx,amount in enumerate([0,30,60,-30,-60]):
        token=str(amount) if amount>=0 else 'minus'+str(-amount)
        a0=read(out/f'replay/matched0-{idx}-{token}-candidate.png');a4=read(out/f'replay/guard4-{idx}-{token}-candidate.png')
        old_local_idx={0:0,30:1,60:2,-30:3,-60:4}[amount]
        original=read(base/f'local-scene-context/replay/{base.name}-{old_local_idx}-{token}-candidate.png')
        delta=np.abs(a0.astype(int)-a4.astype(int));selection_delta=np.abs(original.astype(int)-a0.astype(int))
        row={'amount':amount,'guardVsMatched0ChangedPixels':int(np.any(delta,axis=2).sum()),'guardVsMatched0MaxByteDelta':int(delta.max()),'matched0VsAllTiles0ChangedPixels':int(np.any(selection_delta,axis=2).sum()),'matched0VsAllTiles0MaxByteDelta':int(selection_delta.max()),'zeroExact':bool(np.array_equal(a4,source)) if amount==0 else None}
        rows.append(row)
        if amount:
            panels=[('source',source),('all-tiles 0px',original),('matched 0px',a0),('matched 4px',a4)]
            for name,box in boxes.items():sheet(out/f'{name}-{amount:+}.png',panels,box)
    points=[]
    if args.source_case=='portrait-614810':
        B0=read(out/'matched0-full-scene.png');B4=read(out/'guard4-full-scene.png');a0=read(out/'replay/matched0-2-60-candidate.png');a4=read(out/'replay/guard4-2-60-candidate.png')
        for xx,yy in [(318,580),(302,520),(340,620),(688,580),(302,571)]:
            points.append({'point':[xx,yy],'source':source[yy,xx,:3].tolist(),'matched0B':B0[yy,xx,:3].tolist(),'guard4B':B4[yy,xx,:3].tolist(),'matched0Output60':a0[yy,xx,:3].tolist(),'guard4Output60':a4[yy,xx,:3].tolist()})
    report={'binarySHA256':sha(binary),'unchangedInputHashes':graph_input_hashes,'knownScenePixelsChanged':0,'cases':rows,'points':points,'scope':'ordinary rlib; same geometry/alpha inputs. Actual consumed coverage may change through existing SceneLowerBound; no claim it is pure B-only pixel transport.'}
    dump(out/'replay-review.json',report);print(json.dumps(report))
if __name__=='__main__':main()

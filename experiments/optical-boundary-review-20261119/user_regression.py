"""Adapt saved user-photo evidence and replay a fixed source-pair regression.

No detection, source replacement, output-selected prompt or parameter search.
The previous paired-alpha/scene is the unchanged regression reference, not an
official-matte ablation. Run setup, the two fixed source models, then replay.
"""
from pathlib import Path
import argparse, copy, hashlib, json, subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'test/results/face-generalization-20261118/user-original'
SAVED=ROOT/'test/results/face-usable-20261112/motion-interface-20261116'
SPEC=Path('/tmp/face-replay-20261117/ordinary-rlib-48-photo-replay-spec.json')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def read(p):return np.asarray(Image.open(p).convert('RGBA'))

def setup():
    spec=json.loads(SPEC.read_text());cases=[c for c in spec['cases'] if c['id']=='user']
    assert len(cases)==1
    case=cases[0];source=Path(case['source']);neural=SAVED/'user-head-neural-scene'
    report=json.loads((neural/'report.json').read_text());proposal=Path(report['proposal'])
    assert proposal==SAVED/'user-head-observed-scale-full-source'
    assert report['source']==str(source) and sha(source)==report['sourceSHA256']
    assert sha(proposal/'protocol.json')==report['proposalSHA256']
    src=read(source);rgba=read(neural/'source-crop.png')
    assert np.array_equal(src,rgba) and report['sourceCropBox']==[0,0,768,768]
    assert np.array_equal(src[:,:,:3],read(proposal/'component-1/source-crop.png')[:,:,:3])
    tri=np.asarray(Image.open(proposal/'component-1/trimap.png').convert('L'))
    mask=np.asarray(Image.open(neural/'source-inpaint-mask.png').convert('L'))>0
    assert np.array_equal(mask,tri>0)
    desc=json.loads(Path(case['sourcePlatePrior']).read_text())
    assert desc['source']==str(source) and desc['sourceSha256']==sha(source)
    alpha=json.loads(Path(case['physicalAlpha']).read_text())
    old_a=np.asarray(alpha['data'],np.float32).reshape(768,768)/np.float32(255)
    official=np.load(proposal/'component-1/inference/alpha-official.npy')
    old_b=read(desc['image']);model_b=read(neural/'scene-generated.png')
    BASE.mkdir(exist_ok=False);prep=BASE/'source-preparation';prep.mkdir()
    (BASE/'canonical-source.png').symlink_to(source)
    (prep/'head-proposal').symlink_to(proposal,target_is_directory=True)
    (prep/'head-neural-scene').symlink_to(neural,target_is_directory=True)
    (prep/'runtime-current').symlink_to(Path(case['graph']).parent,target_is_directory=True)
    replay=copy.deepcopy(spec);replay['cases']=[case];replay['outputDir']=str(BASE/'original-replay')
    dump(BASE/'ordinary-rlib-spec.json',replay)
    dump(BASE/'adaptation.json',dict(source=str(source),sourceSHA256=sha(source),sourceSize=[768,768],
        savedSpec=str(SPEC),savedSpecSHA256=sha(SPEC),sourceProposal=str(proposal),
        sourceProposalSHA256=sha(proposal/'protocol.json'),sourceNeuralScene=str(neural),
        sourceOnly=True,newDetectionRun=False,geometryChanged=False,motionPermissionChanged=False,
        operation='layout symlinks to saved user evidence, not fresh automatic detection',
        baselineRole='Previous paired source estimate; not a pure-alpha causal control',
        baselineVsOriginalOfficialAlphaChanged=int((old_a!=official).sum()),
        baselineVsOriginalOfficialAlphaMaxDelta=float(abs(old_a-official).max()),
        baselineSceneVsRawSceneChanged=int(np.any(old_b!=model_b,axis=2).sum()),
        baselineSceneVsRawSceneMaxByteDelta=int(abs(old_b.astype(int)-model_b.astype(int)).max()),
        frozenInputSHA256={key:sha(case[key]) for key in ['source','graph','physicalAlpha','sourcePlatePrior']}))
    print(json.dumps({'setup':str(BASE),'sourceSHA256':sha(source),'newDetection':False}))

def sheet(path,panels,box):
    x,y,r,b=box;w,h=r-x,b-y
    image=Image.new('RGB',(len(panels)*w,h+24),'white');draw=ImageDraw.Draw(image)
    for i,(name,a) in enumerate(panels):
        image.paste(Image.fromarray(a).crop(box).convert('RGB'),(i*w,24))
        draw.text((i*w+4,4),name,fill='black')
    image.save(path)

def replay():
    out=BASE/'paired-regression-20261119';out.mkdir(exist_ok=False)
    spec=json.loads((BASE/'ordinary-rlib-spec.json').read_text());original=spec['cases'][0]
    a=json.loads(Path(original['physicalAlpha']).read_text())
    estimate=np.load(BASE/'optical-boundary-review-20261119/alpha-official.npy')
    assert estimate.shape==(768,768)
    a['data']=(estimate*np.float32(255)).ravel().tolist()
    dump(out/'candidate-alpha.json',a)
    cases=[]
    for name in ['previous','new-pair']:
        case=copy.deepcopy(original);case.update(id=name,amounts=[0,30,60,-30,-60])
        if name=='new-pair':
            case['physicalAlpha']=str(out/'candidate-alpha.json')
            case['sourcePlatePrior']=str(BASE/'context-guard-direct-20261119/guard4-descriptor.json')
        cases.append(case)
    spec.update(cases=cases,outputDir=str(out/'replay'));dump(out/'spec.json',spec)
    binary=ROOT/'core/image-core/target/release/examples/face-source-replay'
    report=dict(status='running',binarySHA256=sha(binary),sourceSHA256=sha(original['source']),
        sourceOnly=True,newDetection=False,geometryChanged=False,motionPermissionChanged=False,
        comparison='new fixed optical-unknown + local guarded B versus unchanged previous paired estimate',
        sourceInputs={k:sha(original[k]) for k in ['source','graph']},results=[])
    dump(out/'report.json',report)
    with (out/'replay.log').open('w') as log:
        subprocess.run([str(binary),str(out/'spec.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
    source=read(original['source'])
    boxes={'whole-source':(0,0,768,768),'face-neck':(210,100,580,650),
        'left-edge':(210,160,330,635),'right-edge':(460,160,585,635),
        'neck':(285,430,545,675)}
    previous_dir=Path(json.loads(SPEC.read_text())['outputDir'])
    for i,amount in enumerate([0,30,60,-30,-60]):
        token=str(amount) if amount>=0 else 'minus'+str(-amount)
        old=read(out/f'replay/previous-{i}-{token}-candidate.png')
        new=read(out/f'replay/new-pair-{i}-{token}-candidate.png')
        delta=abs(old.astype(int)-new.astype(int))
        row=dict(amount=amount,changedPixels=int(np.any(delta,axis=2).sum()),maxByteDelta=int(delta.max()),
            zeroExact=bool(np.array_equal(old,source) and np.array_equal(new,source)) if not amount else None)
        if amount:
            oi=[30,-30,60,-60].index(amount)
            cached=previous_dir/f'user-{oi}-{token}-candidate.png'
            if cached.exists():row['previousVsSavedChangedPixels']=int(np.any(old!=read(cached),axis=2).sum())
            for name,box in boxes.items():sheet(out/f'{name}-{amount:+}.png', [('source',source),('previous',old),('new-pair',new)],box)
        report['results'].append(row)
    report['status']='complete; real-user-photo visual review pending';dump(out/'report.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['setup','replay']);args=p.parse_args()
    (setup if args.stage=='setup' else replay)()

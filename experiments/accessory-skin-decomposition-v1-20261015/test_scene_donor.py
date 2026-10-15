"""Independent output/ownership regression checks for rejected scene donors."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def image(p):return np.asarray(Image.open(p).convert('RGBA'))
def mask(p):return np.asarray(Image.open(p))>0

def check(candidate,base,target,protected,safe,donor):
 changed=np.any(candidate!=base,axis=-1)
 assert not (changed&~target).any(),'outside target changed'
 assert not (changed&protected).any(),'protected foreground changed'
 assert not (donor&~safe).any(),'foreground donor used'
 assert np.array_equal(candidate[...,3],base[...,3]),'alpha changed'
 return changed

def must_reject(fn):
 try:fn()
 except AssertionError:return
 raise AssertionError('Protection gate did not reject corrupted result')

def main():
 safe=mask(HERE/'scene-donor-v1/safe-scene-mask.png');protected=mask(HERE/'scene-donor-v1/protected-mask.png');data=np.load(HERE/'outputs-v2/combined-point/source-layers.npz');source=image(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png');rows=0
 for directory in ['scene-donor-v1','scene-donor-v2-local']:
  root=HERE/directory;report=json.loads((root/'report.json').read_text());assert report['accepted'] is False;assert len(report['rows'])==2;assert report['sourceSha256']==sha(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png');assert report['graphSha256']==sha(ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json')
  for row in report['rows']:
   amount=row['amount'];d=HERE/row['directory'];target=mask(d/'target-mask.png');donor=mask(d/'donor-mask.png');candidate=image(d/'candidate.png');base=image(HERE/f'outputs-v2/combined-point/{amount}-original-backing/candidate.png');changed=check(candidate,base,target,protected,safe,donor)
   assert np.array_equal(candidate[data['opaque']],source[data['opaque']])
   L=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full/layers';st=json.loads((L/'state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height'];cs=np.s_[oy:oy+ch,ox:ox+cw]
   previous=HERE/f'outputs-v2/combined-point/{amount}-original-backing';write=mask(previous/'allowed-write.png');unknown=mask(previous/'unknown-new-coverage.png');insufficient=mask(previous/'insufficient-scene-support.png')
   assert not (target[cs]&~write).any() and not (target[cs]&unknown).any() and not (target[cs]&insufficient).any()
   assert not (target[cs]&~(data['pure']|data['trusted'])[cs]).any()
   assert row['inverseSha256']==sha(L/'inverse-source-xy-f32le.bin');assert row['outputSha256']==sha(d/'candidate.png');assert row['changedPixels']==int(changed.sum());assert row['targetPixels']==int(target.sum())
   # Measured seam regression must prevent promotion even when bytes are safe.
   assert row['boundaryAfter']['mean']>row['boundaryBefore']['mean'];assert not report['accepted']
   if directory=='scene-donor-v1':assert np.array_equal(image(d/'donor-gated-candidate.png'),base)
   broken=candidate.copy();iy,ix=np.argwhere(protected)[0];broken[iy,ix,0]^=1;must_reject(lambda:check(broken,base,target,protected,safe,donor))
   bad_donor=donor.copy();iy,ix=np.argwhere(~safe)[0];bad_donor[iy,ix]=True;must_reject(lambda:check(candidate,base,target,protected,safe,bad_donor))
   rows+=1
 print(json.dumps({'rows':rows,'status':'passed','accepted':False,'meaning':'Ownership and refusal checks pass; visual/seam failure remains deliberately rejected.'}))
if __name__=='__main__':main()

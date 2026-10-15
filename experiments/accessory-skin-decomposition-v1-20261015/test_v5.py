"""Regression checks for the stopped scene-only ear-ring experiment."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=HERE/'neural-backing-v5-virtual-scene';r=json.loads((p/'report.json').read_text())
 assert r['sourceSha256']==sha(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png')
 assert r['graphSha256']==sha(ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json')
 assert r['sourceLayersSha256']==sha(HERE/'outputs-v2/combined-point/source-layers.npz')
 source=np.asarray(Image.open(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png').convert('RGBA'))
 ok=[]
 for row in r['rows']:
  dest=p/row['directory'].split('/',1)[1]; output=np.asarray(Image.open(dest/'candidate.png').convert('RGBA'))
  amount=row['amount']; mat=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full'; base=np.asarray(Image.open(mat/'candidate.png').convert('RGBA'))
  st=json.loads((mat/'layers/state.json').read_text());ox,oy=st['origin'];cw,ch=st['width'],st['height']
  writes=np.asarray(Image.open(dest/'allowed-write.png'))>0
  full=np.zeros(source.shape[:2],bool);full[oy:oy+ch,ox:ox+cw]=writes
  changed=np.any(output!=base,axis=2)
  assert not (changed&~full).any()
  opaque=np.asarray(np.load(HERE/'outputs-v2/combined-point/source-layers.npz')['opaque'])
  assert np.array_equal(output[opaque],source[opaque])
  assert np.array_equal(output[...,3],source[...,3])
  assert int(changed.sum())==row['changedPixels']
  assert row['unknownNewCoveragePixels']==0 and row['opaqueMetalChanged']==0 and row['outsideWritesChanged']==0
  assert sha(mat/'layers/inverse-source-xy-f32le.bin')==row['inverseSha256']
  ok.append(row['directory'])
 print(json.dumps({'rows':len(ok),'status':'passed','scope':'invariants only; visual quality rejected'}))
if __name__=='__main__':main()

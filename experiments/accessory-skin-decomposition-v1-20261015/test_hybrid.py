"""Byte and ownership invariants for the rejected hybrid crop candidate."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 r=json.load(open(HERE/'hybrid-crop2/report.json'));assert not r['accepted'];assert len(r['rows'])==2
 src=np.asarray(Image.open(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png').convert('RGBA'));d=np.load(HERE/'outputs-v2/combined-point/source-layers.npz');opaque=d['opaque'];assert r['sourceSha256']==sha(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png');assert r['cropAuditReportSha256']==sha(ROOT/'experiments/zim-accessory-crop-resolution-20261015/outputs/audit-report.json')
 for row in r['rows']:
  amount=row['amount'];out=HERE/row['directory'];img=np.asarray(Image.open(out/'candidate.png'));mat=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full';st=json.load(open(mat/'layers/state.json'));ox,oy=st['origin'];cw,ch=st['width'],st['height'];w=np.asarray(Image.open(out/'allowed-write.png'))>0;full=np.zeros(src.shape[:2],bool);full[oy:oy+ch,ox:ox+cw]=w;base=np.asarray(Image.open(HERE/f'outputs-v2/combined-point/{amount}-original-backing/candidate.png'))
  changed=np.any(img!=base,axis=2);assert not (changed&~full).any();assert np.array_equal(img[opaque],src[opaque]);assert np.array_equal(img[...,3],src[...,3]);assert row['opaqueMetalChanged']==0 and row['outsideWritesChanged']==0;assert row['sourceSupportPixels']==81
  assert not (w & (np.asarray(Image.open(out/'old-unknown.png'))>0)).any();assert not (w & (np.asarray(Image.open(out/'old-insufficient.png'))>0)).any()
 print(json.dumps({'rows':2,'status':'passed','accepted':False}))
if __name__=='__main__':main()

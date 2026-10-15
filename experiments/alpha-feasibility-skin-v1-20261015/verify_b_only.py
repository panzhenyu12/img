"""Independent saved-artifact invariants for the two B-only experiments."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
src=np.asarray(Image.open(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png').convert('RGBA'));scene_path=ROOT/'experiments/joint-skin-scene-support-v1-20261015/outputs-v2-consistent/scene-support.npz';scene=np.load(scene_path);finite=np.isfinite(scene['sceneDistance']);acc=np.load(ROOT/'experiments/accessory-skin-decomposition-v1-20261015/outputs-v2/combined-point/source-layers.npz');opaque=acc['opaque'];rows=[]
for variant in ['v37-scene-b-only-v1','v37-scene-b-only-v2-feather']:
 root=HERE/'qa'/variant;report=json.loads((root/'report.json').read_text());assert report['sceneSupportSha256']==sha(scene_path)
 for rr in report['rows']:
  amount=rr['amount'];tr=HERE/f'qa/v37-trace/{amount if amount else 100}';l=tr/'layers';im=np.asarray(Image.open(root/str(amount)/'candidate.png').convert('RGBA'));baseline=np.asarray(Image.open(tr/'candidate.png').convert('RGBA')) if amount else src;allowed=np.asarray(Image.open(root/str(amount)/'allowed.png'))>0;diff=np.any(im!=baseline,axis=-1);assert not (diff&~allowed).any();assert not (diff&~finite).any();assert not (diff&opaque).any();assert np.array_equal(im[...,3],baseline[...,3]);assert sha(l/'source-alpha-f32le.bin')==rr['sourceAlphaSha256'];assert sha(l/'samples-f32le.bin')==rr['samplesSha256'];assert sha(l/'foreground.png')==rr['sourceForegroundSha256'];assert sha(tr/'debug/flow-geometry.json')==rr['geometrySha256']
  st=json.loads((l/'state.json').read_text());ox,oy=st['origin'];w,h=st['width'],st['height'];frozen=np.fromfile(l/'frozen.bin',dtype='uint8').reshape(h,w)>0;assert not diff[oy:oy+h,ox:ox+w][frozen].any()
  if not amount:assert np.array_equal(im,src)
  rows.append({'variant':variant,'amount':amount,'checksPassed':True,'changedPixels':int(diff.sum()),'sourceAlphaUnchanged':True,'foregroundUnchanged':True,'sampledPremultipliedColorUnchanged':True,'geometryUnchanged':True,'outsideFiniteUnchanged':True,'opaqueMetalUnchangedFromV37':True,'frozenUnchangedFromV37':True,'outputAlphaUnchanged':True,'zeroExactSource':bool(np.array_equal(im,src)) if not amount else None})
out={'passed':True,'acceptance':'invariants only, not visual acceptance or hidden-background truth','rows':rows};(HERE/'qa/v37-b-only-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

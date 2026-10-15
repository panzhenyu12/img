"""Independent scene purity, coordinate and reachability audit."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from run_real import probe
cv2=probe.cv2
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=HERE/'outputs-v2-consistent';report=json.loads((out/'report.json').read_text());data=np.load(out/'inputs.npz');joined=np.load(out/'scene-support.npz');source=np.asarray(Image.open(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png').convert('RGB'),dtype='float32')/255;lab=cv2.cvtColor(source,cv2.COLOR_RGB2LAB)
 assert report['sourceSha256']==sha(ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png');assert report['graphSha256']==sha(ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json');assert report['outputSha256']==sha(out/'scene-support.npz');assert report['fullSkinAlphaSha256']==sha(ROOT/'experiments/joint-skin-matting-v1-20261015/outputs/full-point/alphas-full.npy');assert report['cropSkinAlphaSha256']==sha(ROOT/'experiments/joint-skin-matting-v1-20261015/outputs/crop-point/alphas-full.npy')
 assert not report['acceptedForProduction'];eligible=(data['fullSkin']<.01)&(data['cropSkin']<.01)&~data['protected'];assert not (joined['newScene']&~eligible).any();combined=np.zeros(source.shape[:2],bool)
 for row in report['rows']:
  d=HERE/row['directory'];s=np.load(d/'scene.npz');scene=s['scene'];region=s['region'];combined|=region;added=s['added'];assert not (scene&~eligible).any();assert not (scene&~region).any();assert int(added.sum())==row['addedScenePixels'];assert np.max(np.abs(s['prior'][scene]-source[scene]))<1e-6
  iy,ix=s['nearest'];assert scene[iy[region],ix[region]].all();assert region[iy[region],ix[region]].all();yy,xx=np.indices(region.shape);dist=np.hypot(ix[region]-xx[region],iy[region]-yy[region]);assert np.max(np.abs(s['distance'][region]-dist))<1e-4
  # Independent sparse graph checks that every added pixel reaches an
  # observed seed through four-neighbor, color-consistent scene edges.
  indices=np.full(region.shape,-1,dtype=int);indices[scene]=np.arange(int(scene.sum()));edges_a=[];edges_b=[]
  for a,b in [(np.s_[:-1,:],np.s_[1:,:]),(np.s_[:,:-1],np.s_[:,1:])]:
   delta=lab[a]-lab[b];valid=scene[a]&scene[b]&(np.linalg.norm(delta,axis=-1)<=8)&(np.abs(delta[...,0])<=6)&(np.linalg.norm(delta[...,1:],axis=-1)<=6);edges_a.extend(indices[a][valid]);edges_b.extend(indices[b][valid])
  graph=coo_matrix((np.ones(len(edges_a)),(edges_a,edges_b)),shape=(int(scene.sum()),int(scene.sum())));_,labels=connected_components(graph,directed=False);seed=scene&data['strictScene'];seed_labels=np.unique(labels[indices[seed]]);assert np.isin(labels[indices[added]],seed_labels).all()
  assert not (added&data['legacyStrictScene']).any();assert not (added&(data['accessory']>=.01)).any()
 assert np.isfinite(joined['sceneDistance'][combined]).all();assert np.isinf(joined['sceneDistance'][~combined]).all()
 # The rejected first run really contained contradictory donors; regression
 # requires the consistent version to eliminate them while preserving proof.
 first=np.load(HERE/'outputs/component-1/scene.npz');assert (first['scene']&~eligible).any()
 print(json.dumps({'status':'passed','components':2,'newObservedScenePixels':int(joined['addedScene'].sum()),'newSceneForegroundConflict':0,'nearestCrossSideOrInvalid':0,'productionAccepted':False}))
if __name__=='__main__':main()

"""Fixed five-mode combined-prompt holdout with complete actual-token capture.

Single-image prompt batch only. Records tiny prompt/attention metadata through an
ORT-session proxy, without changing any tensor sent to the model or vendor code.
"""
from pathlib import Path
import hashlib,importlib.util,json,shutil,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
CODE=ROOT/'experiments/zim-combined-prompts-20261015'
sys.path.insert(0,str(CODE))
spec=importlib.util.spec_from_file_location('combined_runner',CODE/'run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={'encoder.onnx':'6123ea10722dd43a0f8bbb96cd6a0532d8a65e2b1e8339f0cd6593d494807809','decoder.onnx':'a99d14bac3dc28b17809020b1711b1991bd5b7eb18f678c3624ce508748cfa11'}
for name,digest in expected.items():assert sha(ROOT/'.models/zim-20261014'/name)==digest,name
snap=HERE/'combined-code-snapshot';snap.mkdir(exist_ok=False)
for name in ['run.py','adapter.py']:shutil.copy2(CODE/name,snap/name)
script_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),CODE/'run.py',CODE/'adapter.py',HERE/'selector-snapshot.py']}
feeds=[]
class RecordingSession:
 def __init__(self,session):self.session=session
 def __getattr__(self,name):return getattr(self.session,name)
 def run(self,output_names,inputs,*args,**kwargs):
  c,l,a=inputs['point_coords'],inputs['point_labels'],inputs['attn_mask']
  assert c.shape[0]==l.shape[0]==a.shape[0]==1,'Only B=1 is validated'
  assert c.shape[:2]==l.shape and c.shape[2]==2
  feeds.append({'pointCoords':c.tolist(),'pointLabels':l.tolist(),'pointCoordsDtype':str(c.dtype),'pointLabelsDtype':str(l.dtype),
   'attentionShape':list(a.shape),'attentionDtype':str(a.dtype),'attentionSum':float(a.sum()),'attentionSha256':hashlib.sha256(a.tobytes()).hexdigest(),
   'promptSha256':hashlib.sha256(c.tobytes()+l.tobytes()).hexdigest()})
  return self.session.run(output_names,inputs,*args,**kwargs)
original_load=runner.zim.load_vendor

def recording_load():
 Encoder,Decoder,Zim,Predictor,manifest=original_load()
 class RecordedDecoder(Decoder):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);self.ort_session=RecordingSession(self.ort_session)
 return Encoder,RecordedDecoder,Zim,Predictor,manifest
runner.zim.load_vendor=recording_load
runner.probe.SOURCES={
 'portrait-makeup':('experiments/face-anatomical-v22-20261015/inputs/portrait-makeup-source.png','experiments/face-anatomical-v22-20261015/inputs/portrait-makeup-graph.json'),
 'portrait-glasses-street':('test/images/portrait-glasses-street.jpg','test/results/face-core-fresh/portrait-glasses-street-faceWidth-60.json'),
 'portrait-dark':('experiments/zim-semantic-holdout-20261015/dark-detection/1920/canonical-source.png','experiments/zim-semantic-holdout-20261015/dark-detection/1920/face-graph.json'),
}
if __name__=='__main__':
 out=HERE/'combined-raw';sys.argv=[str(CODE/'run.py'),'--names',*runner.probe.SOURCES,'--out',str(out)];runner.main()
 report=json.loads((out/'report.json').read_text());items=[row for image in report['images'] for row in image['rows']];assert len(items)==len(feeds)
 for item,feed in zip(items,feeds):
  item['recordedActualDecoderFeed']=feed
  labels=feed['pointLabels'][0];requested=[1]+[0]*len(item['prompt']['negative'])
  if item['mode'] in ['upstream-mixed','box-only']:assert labels==[2.,3.]
  elif item['mode']=='points-only':assert labels==requested+[-1.]
  else:assert labels==requested+[2.,3.]
 report['holdoutProvenance']={'scope':'Three independent full-size original images, unused by the three-image combined-prompt development set. Original graph reused without output redetection; dark fresh source-only graph isolated.',
  'scriptHashes':script_hashes,'weightsVerifiedBeforeInference':expected,'batchLimitation':'B=1 only; no general multi-image batching claim','actualORTFeedsCaptured':len(feeds),'featureReuse':'Runner calls predictor.set_image once per photograph; all protocols and modes use those features.'}
 for name,digest in script_hashes.items():assert sha(ROOT/name)==digest,'Source changed during frozen run'
 (out/'report-with-feeds.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'cases':len(items),'allActualFeedsVerified':True}),flush=True)

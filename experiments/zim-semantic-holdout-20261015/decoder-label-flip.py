"""A real ONNX signed-label counterfactual with fixed synthetic features."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.run/zim-deps'));import onnxruntime as ort
model=ROOT/'.models/zim-20261014/decoder.onnx';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(model)=='a99d14bac3dc28b17809020b1711b1991bd5b7eb18f678c3624ce508748cfa11'
s=ort.SessionOptions();s.intra_op_num_threads=3;s.inter_op_num_threads=3
session=ort.InferenceSession(str(model),sess_options=s,providers=['CPUExecutionProvider'])
base={v.name:np.zeros(v.shape,np.float32) for v in session.get_inputs() if v.name not in ['point_coords','point_labels','attn_mask']}
coords=np.asarray([[[240.,200.],[480.,420.],[120.,100.],[700.,800.]]],np.float32)
attention=np.zeros((1,64,64),np.float32);attention[:,6:51,7:45]=1
results=[];times=[]
for labels in [[[1,0,2,3]],[[1,1,2,3]]]:
 inputs={**base,'point_coords':coords,'point_labels':np.asarray(labels,np.float32),'attn_mask':attention}
 start=time.perf_counter();results.append(session.run(None,inputs));times.append((time.perf_counter()-start)*1000)
mask_delta=np.abs(results[0][0]-results[1][0]);score_delta=np.abs(results[0][1]-results[1][1])
report={'scope':'Synthetic zero feature counterfactual verifies the real ONNX uses signed labels when sent; not image quality.',
 'modelSha256':sha(model),'scriptSha256':sha(Path(__file__)),'pointCoords':coords.tolist(),'labelsA':[[1,0,2,3]],'labelsB':[[1,1,2,3]],
 'otherInputsIdentical':True,'maxMaskLogitDelta':float(mask_delta.max()),'meanMaskLogitDelta':float(mask_delta.mean()),'changedMaskLogits':int(np.count_nonzero(mask_delta)),
 'maxScoreDelta':float(score_delta.max()),'scoresA':results[0][1].tolist(),'scoresB':results[1][1].tolist(),'ms':times}
assert report['maxMaskLogitDelta']>1e-5 and all(np.isfinite(r).all() for result in results for r in result)
(HERE/'decoder-label-flip.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

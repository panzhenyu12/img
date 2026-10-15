"""Independent wrapper capture and real ONNX dynamic-prompt contract probe.

Synthetic zero features verify tensor compatibility only; no image quality or
semantic correctness claim. Vendor code and model are unmodified.
"""
from pathlib import Path
import importlib.util,json,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('zim_probe',ROOT/'experiments/zim-20261014/run.py');zim=importlib.util.module_from_spec(spec);spec.loader.exec_module(zim)
_,Decoder,_,_,manifest=zim.load_vendor();torch=zim.torch;torch.set_num_threads(3)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class Capture:
 def __init__(self):self.inputs=[]
 def run(self,names,inputs):
  self.inputs.append({k:v.copy() for k,v in inputs.items()});return np.zeros((1,4,2,2),np.float32),np.zeros((1,4),np.float32)
cap=Capture();d=Decoder.__new__(Decoder);d.ort_session=cap
features=torch.zeros((1,1,1,1));points=torch.tensor([[[240.,200.],[480.,420.]]]);box=torch.tensor([[120.,100.,700.,800.]]);attention=torch.zeros((1,64,64))
for label in [[1.,0.],[1.,1.]]:
 d.forward([features]*3,features,(points,torch.tensor([label])),box,attention)
a,b=cap.inputs
capture={'requestedCoords':points.tolist(),'requestedLabelsA':[[1,0]],'requestedLabelsB':[[1,1]],
 'actualCoords':a['point_coords'].tolist(),'actualLabels':a['point_labels'].tolist(),
 'allActualInputsIdenticalAfterLabelFlip':all(np.array_equal(a[k],b[k]) for k in a)}
assert capture['allActualInputsIdenticalAfterLabelFlip'] and a['point_labels'].tolist()==[[2.,3.]]
model=ROOT/'.models/zim-20261014/decoder.onnx';assert sha(model)=='a99d14bac3dc28b17809020b1711b1991bd5b7eb18f678c3624ce508748cfa11'
decoder=Decoder(str(model),num_threads=3);sess=decoder.ort_session
metadata=[{'name':v.name,'shape':v.shape,'type':v.type} for v in sess.get_inputs()]
base={v.name:np.zeros(v.shape,np.float32) for v in sess.get_inputs() if v.name not in ['point_coords','point_labels','attn_mask']}
attn=np.zeros((1,64,64),np.float32);attn[:,6:51,7:45]=1
cases=[('mixed-four',[ [240.,200.],[480.,420.],[120.,100.],[700.,800.] ],[1,0,2,3]),
       ('mixed-seven',[[240.,200.],[480.,420.],[360.,220.],[500.,600.],[200.,500.],[120.,100.],[700.,800.]],[1,0,1,0,1,2,3])]
rows=[]
for name,coords,labels in cases:
 inputs={**base,'point_coords':np.asarray([coords],np.float32),'point_labels':np.asarray([labels],np.float32),'attn_mask':attn}
 t=time.perf_counter();outputs=sess.run(None,inputs);elapsed=(time.perf_counter()-t)*1000
 assert all(np.isfinite(a).all() for a in outputs)
 row={'name':name,'pointCount':len(labels),'labels':labels,'outputShapes':[list(a.shape) for a in outputs],'allFinite':True,'ms':elapsed};rows.append(row);print(json.dumps(row),flush=True)
report={'scope':'No source photograph inference. Synthetic zero features confirm dynamic token compatibility only; not image-quality evidence.',
 'vendorCommit':manifest['commit'],'wrapperSha256':sha(ROOT/'experiments/zim-20261014/vendor/zim_anything/modeling/decoder.py'),'modelSha256':sha(model),'scriptSha256':sha(Path(__file__)),
 'capture':capture,'onnxInputs':metadata,'actualRuntimeTests':rows,
 'finding':'The old mixed box+point wrapper overwrites point tokens and labels with the box corners. Dynamic ONNX accepts concatenated actual signed points plus corners without a padding point.'}
(HERE/'decoder-contract.json').write_text(json.dumps(report,indent=2)+'\n')

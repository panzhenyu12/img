"""Resolve photograph/classification evidence through recorded paths and hashes."""
import hashlib,json
from pathlib import Path
from PIL import Image,ImageOps
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def absolute(p):
 p=Path(p);return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
def resolve(base,provenance,source=None,hard=None):
 ft=provenance.get('feature_transport',{});expected=provenance['control_trace']['source_evidence']['source_sha256'];candidates=[]
 if source:candidates=[absolute(source)]
 else:
  for value in [ft.get('source'),provenance.get('ear_material_strain',{}).get('source')]:
   if value:candidates.append(absolute(value))
  for folder in [base,ft.get('base'),ft.get('comparison'),provenance.get('ear_material_strain',{}).get('baseline')]:
   if not folder:continue
   folder=absolute(folder);report=folder/'photo.jacobian.json'
   if report.exists() and read(report).get('input'):candidates.append(absolute(read(report)['input']))
   run=folder/'run.json'
   if run.exists():
    records=read(run)
    if isinstance(records,dict):records=[records] if 'command' in records or 'cmd' in records else records.get('steps',[])
    for row in records:
     if not isinstance(row,dict):continue
     cmd=row.get('cmd',row.get('command',[]))
     if cmd and 'face-photo-transport' in str(cmd[0]) and '--input' in cmd:candidates.append(absolute(cmd[cmd.index('--input')+1]))
 sources={p for p in candidates if p.exists() and sha(p)==expected}
 if len(sources)!=1:raise ValueError(f'provide one source path matching recorded hash; found {len(sources)}')
 source=sources.pop();image=ImageOps.exif_transpose(Image.open(source)).convert('RGB');expected_size=provenance['control_trace']['source_evidence']['reference']['source_dimensions']
 if list(image.size)!=expected_size:raise ValueError('source dimensions disagree with native trace')
 trace=absolute(provenance['source_trace']);graph=trace.parent/'graph.json'
 if sha(graph)!=provenance['feature_graph_sha256']:raise ValueError('feature graph hash disagreement')
 expected_hard=provenance['unknown_motion_prior']['hard_sha256']
 candidates=[absolute(hard)] if hard else ([absolute(ft['classifier_hard_path'])] if ft.get('classifier_hard_path') else list(trace.parents[1].glob('frozen-classification-*/*/candidate-hard.png')))
 matches={p.resolve() for p in candidates if p.exists() and sha(p)==expected_hard}
 if len(matches)!=1:raise ValueError(f'provide classified hard path matching recorded hash; found {len(matches)}')
 hard=matches.pop();report=hard.parent/'report.json'
 if sha(report)!=provenance['unknown_motion_prior']['report_sha256']:raise ValueError('classified hard report hash disagreement')
 classified=read(report)
 if classified['crop']!=provenance['crop'] or sha(trace/'frozen.png') not in classified['input_sha256'].values():raise ValueError('classification does not belong to recorded source trace')
 # Decode native source evidence, rather than trusting only file names/dimensions.
 from prepare_bounded_face import validate_trace_source
 validate_trace_source(trace,np.asarray(ImageOps.exif_transpose(Image.open(source)).convert('RGBA')))
 return source,image,graph,hard

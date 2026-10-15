from pathlib import Path
import subprocess,json,hashlib
from PIL import Image
import numpy as np
E=Path(__file__).resolve().parent;R=E.parents[1];binary=E/'bin/candidate-image-render';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();binary_sha=sha(binary)
fixtures=[('user-face-contour',R/'test/results/user-face-contour-20261014/1024/canonical-source.png',E/'replay')]
for name in ['portrait-woman','portrait-makeup']:fixtures.append((name,R/f'experiments/face-anatomical-v20-20261014/inputs/{name}-source.png',E/'additional'/name))
report={'binary_sha256':binary_sha,'cases':[]}
for name,source,parent in fixtures:
 for amount in [0,30,-30,100,-100]:
  case=parent/str(amount);D=E/'clean-verification'/name/str(amount);D.mkdir(parents=True,exist_ok=True);output=D/'candidate.png';p=subprocess.run([str(binary),'--input',str(source),'--graph',str(case/'graph.json'),'--output',str(output)],capture_output=True,text=True);(D/'render.log').write_text(p.stdout+p.stderr);p.check_returncode();expected=case/'candidate.png';a=np.array(Image.open(expected).convert('RGBA'));b=np.array(Image.open(output).convert('RGBA'));difference=np.abs(a.astype(int)-b.astype(int));record={'name':name,'amount':amount,'graph_sha256':sha(case/'graph.json'),'reference_sha256':sha(expected),'clean_sha256':sha(output),'changed_pixels':int(np.any(difference,axis=2).sum()),'max_rgba_error':int(difference.max())};report['cases'].append(record);assert record['changed_pixels']==0,record;assert sha(binary)==binary_sha;print(name,amount,'exact',flush=True)
(E/'clean-verification/report.json').write_text(json.dumps(report,indent=2))

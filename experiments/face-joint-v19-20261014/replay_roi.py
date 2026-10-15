from pathlib import Path
import json,os,subprocess,hashlib
root=Path(__file__).resolve().parents[2];base=root/'test/results/user-face-contour-20261014/1024';frozen=base/'face-joint-v19-candidate';out=base/'face-joint-v19-roi-probes';out.mkdir(exist_ok=True);binary=root/'.run/face-joint-v19-roi-debug-image-render';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();report={'binarySha256':sha(binary),'scope':'debug-only ROI mapper instrumentation must not alter frozen outputs','rows':[]}
for amount in [100,-100,30,-30]:
 env=os.environ.copy();env['FACE_JOINT_NECK_EXPERIMENT']='1';env['FACE_FLOW_DEBUG_DIR']=str(out/f'debug{amount}');png=out/f'faceWidth{amount}.png'
 p=subprocess.run([str(binary),'--input',str(base/'canonical-source.png'),'--graph',str(frozen/f'faceWidth{amount}.json'),'--output',str(png)],env=env,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr)
 row={'amount':amount,'frozenSha256':sha(frozen/f'faceWidth{amount}.png'),'instrumentedSha256':sha(png)};row['byteIdentical']=row['frozenSha256']==row['instrumentedSha256'];report['rows'].append(row)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 if not row['byteIdentical']:raise RuntimeError('Instrumentation changed output')
print(json.dumps(report))

from pathlib import Path
import json,hashlib,subprocess,sys
root=Path.cwd();stage=int(sys.argv[1]);base=root/'test/results/user-face-contour-20261014/1024';binary=root/f'.run/face-rbf-v{stage}-independent-image-render';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out=base/f'rbf-v{stage}-independent-rebuild';out.mkdir(exist_ok=True);frozen=base/f'rbf-v{stage}-candidate';report={'stage':stage,'binarySha256':sha(binary),'rows':[]}
for amount in [100,-100,60,-60,30,-30]:
 png=out/f'faceWidth{amount}.png';graph=frozen/f'faceWidth{amount}.json'
 p=subprocess.run([str(binary),'--input',str(base/'canonical-source.png'),'--graph',str(graph),'--output',str(png)],capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 old=sha(frozen/f'faceWidth{amount}.png');new=sha(png);report['rows'].append({'amount':amount,'sourceGraphSha256':sha(graph),'frozenOutputSha256':old,'rebuildOutputSha256':new,'byteIdentical':old==new})
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 if old!=new:raise RuntimeError(f'rebuild image differs at {amount}')
print(json.dumps(report))

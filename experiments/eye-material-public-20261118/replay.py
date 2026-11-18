"""Photographic public-graph replay of source-measured eye width.

Uses frozen source detector observations and hashes. No render test hooks,
new detections, alpha estimates, or model-generated targets are used.
"""
from pathlib import Path
import hashlib, json, subprocess, time
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'test/results/eye-material-public-20261118'
BIN=ROOT/'core/image-core/target/release/image-render'
BASE=ROOT/'test/results/eye-height-layers-20261117'
EYES={
 'leftEye':([33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7],468,[469,470,471,472]),
 'rightEye':([362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382],473,[474,475,476,477])}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 rows=[]
 for name in ['user','dark']:
  folder=OUT/name;folder.mkdir()
  observation=BASE/'raw-source-observations'/name/'observations.json'
  report=json.loads((observation.parent/'report.json').read_text())
  source=Path(report['source'])
  assert sha(source)==report['sourceSHA256'] and sha(observation)==report['observationsSHA256']
  image=Image.open(source).convert('RGBA');w,h=image.size
  mesh=json.loads(observation.read_text())['rows'][0]['sourceLandmarks'][0]
  graph=json.loads((BASE/'baseline'/name/'graph+0.json').read_text())
  params=graph['operations'][0]['params'];params['eyeGeometry']={}
  pt=lambda i:{k:mesh[i][k] for k in ['x','y']}
  for key,(polygon,center,ring) in EYES.items():
   params['semanticMasks'][key]={'shape':'polygon','points':[pt(i) for i in polygon],'feather':0}
   params['eyeGeometry'][key]={'iris':{'center':pt(center),'boundary':[pt(i) for i in ring]}}
  points=[pt(i) for polygon,_,_ in EYES.values() for i in polygon]
  x=[p['x']*w for p in points];y=[p['y']*h for p in points]
  box=(max(0,int(min(x))-24),max(0,int(min(y))-24),min(w,int(max(x))+25),min(h,int(max(y))+25))
  panels=[('source',image)]
  for amount in [0,-30,30,-60,60]:
   params['adjustments']={'eyeWidth':amount}
   input_graph=folder/f'{amount:+}.json';input_graph.write_text(json.dumps(graph))
   output=folder/f'{amount:+}.png';start=time.monotonic()
   process=subprocess.run([str(BIN),'--input',str(source),'--graph',str(input_graph),'--output',str(output)],capture_output=True,text=True)
   row=dict(name=name,amount=amount,elapsedSeconds=time.monotonic()-start,returncode=process.returncode,stderr=process.stderr,sourceSHA256=sha(source),observationsSHA256=sha(observation),graphSHA256=sha(input_graph))
   if process.returncode==0:
    result=Image.open(output).convert('RGBA');a=np.asarray(image);b=np.asarray(result)
    row.update(outputSHA256=sha(output),changedPixels=int(np.any(a!=b,axis=2).sum()),zeroExact=bool(np.array_equal(a,b)))
    panels.append((str(amount),result))
   rows.append(row);print(json.dumps(row),flush=True)
  pw,ph=box[2]-box[0],box[3]-box[1]
  review=Image.new('RGB',(pw*len(panels),ph+24),'white');draw=ImageDraw.Draw(review)
  for i,(label,im) in enumerate(panels):
   review.paste(im.crop(box).convert('RGB'),(i*pw,24));draw.text((i*pw+4,4),label,fill='black')
  review.save(folder/'eye-native.png');review.resize((review.width*2,review.height*2)).save(folder/'eye-2x.png')
 (OUT/'report.json').write_text(json.dumps({'binarySHA256':sha(BIN),'scope':'two existing photos, same-source MediaPipe measurement; no claim of automatic occluder solution','rows':rows},indent=2))
if __name__=='__main__':main()

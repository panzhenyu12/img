"""Replay the fifteen real body-part graphs at symmetric medium strengths.

The masks and source photographs come from the frozen v14 instance-edge review;
only the adjustment amount changes. This is a visual regression sweep, not an
image-quality score derived from changed-pixel counts.
"""
from copy import deepcopy
from pathlib import Path
from PIL import Image, ImageDraw
import hashlib, json, os, subprocess, time

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'test/results/body-parts-v14'
OUT=ROOT/'test/results/body-strength-v15'
QA=ROOT/'doc/qa/body-strength-v15'
BINARY=Path(os.environ.get('BODY_RENDER_BINARY',ROOT/'.run/body-candidate-v4-immutable/image-render'))
AMOUNTS=(-60,-30,30,60)
IDS=os.environ.get('BODY_IDS','fashion-standing,fullbody-skirt,fullbody-side-fashion,tilted-shirt,action-suit').split(',')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for ident in IDS:
  directory=BASE/ident
  source=directory/'source.png'
  original=Image.open(source).convert('RGB');w,h=original.size
  destination=OUT/ident;destination.mkdir(parents=True,exist_ok=True)
  review=QA/ident;review.mkdir(parents=True,exist_ok=True)
  for graph_path in sorted(directory.glob('*.json')):
    if graph_path.stem in ('layers','quality'):continue
    graph=json.loads(graph_path.read_text())
    params=graph['operations'][0]['params'];r=params['region']
    box=(max(0,int(r['x']*w)-16),max(0,int(r['y']*h)-16),min(w,int((r['x']+r['width'])*w)+16),min(h,int((r['y']+r['height'])*h)+16))
    cw,ch=box[2]-box[0],box[3]-box[1]
    sheet=Image.new('RGB',(cw*(len(AMOUNTS)+1),ch+28),'#161b22');draw=ImageDraw.Draw(sheet)
    sheet.paste(original.crop(box),(0,28));draw.text((8,8),'Original',fill='white')
    for col,amount in enumerate(AMOUNTS,1):
      candidate=deepcopy(graph);candidate['operations'][0]['params']['adjustments']={key:amount for key in params['adjustments']}
      out_graph=destination/f'{graph_path.stem}-{amount}.json';out_image=destination/f'{graph_path.stem}-{amount}.png'
      out_graph.write_text(json.dumps(candidate,separators=(',',':')))
      start=time.monotonic();subprocess.run([str(BINARY),'--input',str(source),'--output',str(out_image),'--graph',str(out_graph)],check=True,capture_output=True);elapsed=round((time.monotonic()-start)*1000)
      rendered=Image.open(out_image).convert('RGB');sheet.paste(rendered.crop(box),(col*cw,28));draw.text((col*cw+8,8),str(amount),fill='white')
      records.append({'id':ident,'part':graph_path.stem,'amount':amount,'elapsedMs':elapsed,'outputSha256':sha(out_image),'graphSha256':sha(out_graph)})
    sheet.save(review/f'{graph_path.stem}.png')
    print(ident,graph_path.stem,flush=True)
(QA/'metrics.json').write_text(json.dumps({'base':'body-parts-v14','binary':str(BINARY),'binarySha256':sha(BINARY),'amounts':AMOUNTS,'records':records},indent=2))

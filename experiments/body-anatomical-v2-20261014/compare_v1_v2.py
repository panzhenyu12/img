import json
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
E=Path(__file__).resolve().parent;V1=E.parent/'body-anatomical-v1-20261014'
rows=[]
for rp in (E/'renders').glob('*/report.json'):
 label=rp.parent.name;a=json.loads((V1/'renders'/label/'report.json').read_text());b=json.loads(rp.read_text())
 before={(r['sample'],r['part'],r['amount']):r for r in a['rows']}
 for r in b['rows']:
  old=before[(r['sample'],r['part'],r['amount'])]
  assert r['source_sha256']==old['source_sha256'] and r['graph_sha256']==old['graph_sha256']
  assert r['baseline']['output_sha256']==old['baseline']['output_sha256']
  stem=f"{r['sample']}-{r['part']}-{r['amount']}"
  p1=V1/'renders'/label/f'{stem}-candidate.png';p2=E/'renders'/label/f'{stem}-candidate.png'
  changed=int(np.any(np.asarray(Image.open(p1))!=np.asarray(Image.open(p2)),axis=2).sum())
  if not r['part'].endswith('thigh'):assert changed==0;continue
  box=tuple(r['crop']);cw,ch=box[2]-box[0],box[3]-box[1]
  paths=[(Path(r['source']),'Original'),(p1,'V1 positive difference'),(p2,'V2 absolute difference'),(E/'renders'/label/f'{stem}-matched.png',f"Old, matched {r['matched_legacy']['amount']:.2f}")]
  sheet=Image.new('RGB',(cw*4,ch+28),'#111824');draw=ImageDraw.Draw(sheet)
  for i,(p,title) in enumerate(paths):sheet.paste(Image.open(p).convert('RGB').crop(box),(i*cw,28));draw.text((i*cw+5,7),title,fill='white')
  target=E/'evidence'/f'{stem}-v1-v2.png';sheet.save(target)
  rec={'sample':r['sample'],'part':r['part'],'amount':r['amount'],'changed_v1_to_v2':changed,'v1_side_max':[p['max_abs_motion'] for p in old['candidate']['curve']],'v2_side_max':[p['max_abs_motion'] for p in r['candidate']['curve']],'v1_mean':old['matched_legacy']['candidate_mean_abs_shaft_motion'],'v2_mean':r['matched_legacy']['candidate_mean_abs_shaft_motion'],'v2_matched_old':r['matched_legacy']['amount'],'v2_matched_mean_error':r['matched_legacy']['absolute_mismatch'],'comparison':str(target)}
  rows.append(rec)
(E/'evidence/v1-v2-summary.json').write_text(json.dumps(rows,indent=2))
for r in rows:
 if r['amount']==100:print(r['sample'],r['part'],'v1',r['v1_side_max'],'v2',r['v2_side_max'],'mean',round(r['v1_mean'],3),round(r['v2_mean'],3),'old-match',round(r['v2_matched_old'],3))

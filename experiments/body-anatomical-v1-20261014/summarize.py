"""Auditable source/target curves; descriptive metrics, never quality scores."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np
E=Path(__file__).resolve().parent
out=E/'evidence';out.mkdir(exist_ok=True)
reports=[json.loads(p.read_text()) for p in (E/'renders').glob('*/report.json')]
rows=[r for d in reports for r in d['rows']]
summary={'cases':len(rows),'thigh_cases':sum(r['part'].endswith('thigh') for r in rows),'unchanged_other_parts':sum(not r['part'].endswith('thigh') and r['changed_vs_baseline']==0 for r in rows),'contracts':{},'thigh_summary':[]}
for key in ['protected_changes','outside_region_changes','known_background_changes','outside_part_changes','alpha_changes']:
 summary['contracts'][key]={b:sum(r[b][key] or 0 for r in rows) for b in ['baseline','candidate']}
colors={'source':'#b7c3d7','baseline':'#f08080','candidate':'#58d6a9','matched':'#e6b85c'}
for row in rows:
 if not row['part'].endswith('thigh'):continue
 parent=Path(row['comparison']).parent;stem=f"{row['sample']}-{row['part']}-{row['amount']}"
 ts={k:json.loads((parent/f'{stem}-{k}-curve.json').read_text())[0] for k in ['baseline','candidate','matched']}
 rec={'sample':row['sample'],'part':row['part'],'amount':row['amount'],'joint_radii':ts['candidate'].get('thigh_joint_radii'),'confidence':ts['candidate'].get('thigh_confidence'),'candidate_curve':row['candidate']['curve'],'matched_legacy':row['matched_legacy'],'source_order_reversal_counts':{}}
 # Count 2-section changes where source falls >.10px but output rises >.10px,
 # or source rises >.10px but output falls >.10px. Includes protection/anchors.
 for key,d in ts.items():
  cnt=[]
  for side in [0,1]:
   seg=[s for s in d['sections'] if .22<=s['t']<=.875 and s['sides'][side]['source_radius']>1]
   rev=0
   for a,b in zip(seg,seg[2:]):
    sr=b['sides'][side]['source_radius']-a['sides'][side]['source_radius'];tr=b['sides'][side]['effective_target_radius']-a['sides'][side]['effective_target_radius']
    if abs(sr)>.1 and abs(tr)>.1 and sr*tr<0:rev+=1
   cnt.append(rev)
  rec['source_order_reversal_counts'][key]=cnt
 summary['thigh_summary'].append(rec)
 if abs(row['amount'])!=100:continue
 sheet=Image.new('RGB',(1120,360),'#101722');draw=ImageDraw.Draw(sheet)
 draw.text((18,10),f"{row['sample']} / {row['part']} / requested {row['amount']} / matched old {row['matched_legacy']['amount']:.2f}",fill='white')
 for side in [0,1]:
  left=55+side*550;top=65;pw=475;ph=240
  series={key:[(s['t'],s['sides'][side]['effective_target_radius']) for s in d['sections'] if s['sides'][side]['source_radius']>1] for key,d in ts.items()}
  series['source']=[(s['t'],s['sides'][side]['source_radius']) for s in ts['baseline']['sections'] if s['sides'][side]['source_radius']>1]
  low=max(0,min(y for arr in series.values() for x,y in arr)-3);high=max(y for arr in series.values() for x,y in arr)+3
  X=lambda x:left+x*pw;Y=lambda y:top+ph-(y-low)/(high-low)*ph
  draw.text((left,top-20),f'Side {side} | source radius and actual target (px)',fill='white')
  for i in range(6):
   x=i/5;draw.line((X(x),top,X(x),top+ph),fill='#293246');draw.text((X(x)-8,top+ph+8),f'{x:.1f}',fill='#a6b0c1')
   y=low+(high-low)*i/5;draw.line((left,Y(y),left+pw,Y(y)),fill='#293246');draw.text((left-36,Y(y)-4),f'{y:.1f}',fill='#a6b0c1')
  for j,key in enumerate(['source','baseline','candidate','matched']):
   draw.line([(X(x),Y(y)) for x,y in series[key]],fill=colors[key],width=2)
   draw.text((left+j*118,top+ph+29),key,fill=colors[key])
 sheet.save(out/f'{stem}-curves.png')
summary['matched_max_mean_error']=max(r['matched_legacy']['absolute_mismatch'] for r in rows if r['part'].endswith('thigh'))
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='thigh_summary'},indent=2))

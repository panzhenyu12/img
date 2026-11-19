"""Source-only axis visibility diagnosis, without native target/alpha claims."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
from replay import sample_mask
from inspect_field import at

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path);ap.add_argument('--replay-name',required=True);ap.add_argument('--diagnosis-name',required=True);a=ap.parse_args();base=a.directory.resolve();out=base/a.diagnosis_name/'axis-source'
 out.mkdir(exist_ok=False);source=base/'source.png';im=Image.open(source).convert('RGB');w,h=im.size;rgb=np.array(im)
 gp=base/a.replay_name/'+0.json';g=json.loads(gp.read_text());p=g['operations'][0]['params']
 pts=np.array([[np.float32(q['x'])*np.float32(w),np.float32(q['y'])*np.float32(h)] for q in p['landmarks']],np.float32)
 delta=pts[2]-pts[0];bone_len=np.hypot(delta[0],delta[1]);direction=delta/bone_len;length=np.float32(bone_len*np.float32(.65));origin=pts[1]-direction*length*np.float32(.5);end=pts[1]+direction*length*np.float32(.5)
 delta=end-origin;length=np.hypot(delta[0],delta[1]);direction=delta/length
 arrays={k:sample_mask(p[k],w,h) for k in ['partMask','subjectMask','protectMask']}
 cm=json.loads((base/'class-evidence.json').read_text());classes=np.fromfile(base/'fashn-classes.u8',np.uint8).reshape(cm['decodedDimensions'])
 rows=[];overlay=im.copy();draw=ImageDraw.Draw(overlay)
 for i in range(65):
  t=np.float32(i)/np.float32(64);center=origin+direction*length*t;x,y=map(float,center)
  vals={k:float(at(v,x-.5,y-.5)) for k,v in arrays.items()};target=min(vals['partMask'],vals['subjectMask'])
  cx=min(classes.shape[2]-1,int(x/w*classes.shape[2]));cy=min(classes.shape[1]-1,int(y/h*classes.shape[1]));scores=classes[:,cy,cx]/255;top=sorted(zip(cm['classNames'],scores.tolist()),key=lambda x:x[1],reverse=True)[:3]
  row={'index':i,'t':float(t),'center':[x,y],**vals,'rawGeometricIntersection':target,'rawCenterAtLeastHalf':target>=.5,'sourceTopClasses':top,'sourceRGB':rgb[int(y),int(x)].tolist()};rows.append(row)
  color=(0,255,80) if target>=.5 else (255,0,20);draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
  if i%4==0:draw.text((x+5,y-5),str(i),fill=color)
 overlay.crop((360,550,485,855)).resize((375,915),Image.Resampling.NEAREST).save(out/'axis-source-crop.png')
 report={'sourceSHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'graphSHA256':hashlib.sha256(gp.read_bytes()).hexdigest(),
  'axis':{'origin':origin.tolist(),'direction':direction.tolist(),'length':float(length)},'rows':rows,
  'semantics':'raw graph masks sampled at the native read position (center-.5) after ordinary bilinear rasterization; not the refined native target',
  'limitations':['Axis construction follows current waist f32 equations; minor platform rounding may differ.','The native geometry_target refines masks; this table does not pretend to be that private refined alpha or its target displacement.','Low raw center coverage is source evidence to compare with the old native row-gate, not a new source edit.']}
 edge_path=base/a.diagnosis_name/'source-edge-field/source-edge.json'
 if edge_path.exists():
  matches=[]
  for e in json.loads(edge_path.read_text()):
   if e['sourceRGBEdgeX'] is None:continue
   x,y=e['sourceRGBEdgeX'],e['y'];t=float((np.array([x,y])-origin)@direction/length);slot=t*64;first=int(np.floor(slot));last=first+1
   if 0<=first<64:
    matches.append({'edgeY':y,'edgeX':x,'t':t,'slot':slot,'first':first,'last':last,'firstCenter':rows[first]['center'],'firstRawTarget':rows[first]['rawGeometricIntersection'],'lastRawTarget':rows[last]['rawGeometricIntersection'],'firstProtect':rows[first]['protectMask'],'lastProtect':rows[last]['protectMask']})
  report['originalEdgeMappedToSectionIndices']=matches
 (out/'report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'axis':report['axis'],'lowRawCenterRows':[{k:r[k] for k in ['index','center','rawGeometricIntersection','protectMask','sourceTopClasses']} for r in rows if not r['rawCenterAtLeastHalf']]},indent=2))
if __name__=='__main__':main()

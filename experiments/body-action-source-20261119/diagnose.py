"""Source-bound diagnosis of visible garment ownership and actual response.

Uses the frozen original source scope and ordinary graph. No source correction,
new inference, fitted target or post-output measurement-window change is made.
"""
from pathlib import Path
import hashlib,json,argparse,subprocess
import numpy as np
from PIL import Image,ImageDraw
from replay import sample_mask
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
 ap.add_argument('--replay-name',default='ordinary');ap.add_argument('--output-name',default='diagnosis')
 args=ap.parse_args();base=args.directory.resolve()
 if any(Path(name).name!=name for name in [args.replay_name,args.output_name]):raise ValueError('replay/output names must be folder names')
 scope_path=base.parent/'source-scope.json';scope=json.loads(scope_path.read_text());source=base/'source.png'
 r=json.loads((base/'report.json').read_text());cm=json.loads((base/'class-evidence.json').read_text())
 if sha(source)!=r['canonicalSourceSHA256'] or sha(scope_path)!=r['sourceScopeSHA256']:raise ValueError('source or original evaluation scope changed')
 if sha(base/'fashn-classes.u8')!=cm['classesSHA256'] or sha(base/'fashn-logits.f32')!=cm['logitsSHA256']:raise ValueError('same-inference class evidence changed')
 replay_root=base/args.replay_name;replay_report=json.loads((replay_root/'report.json').read_text())
 if any(row['exitCode']!=0 for row in replay_report['results']):raise ValueError('cannot inspect RGB from a failed native replay')
 if replay_report['protocol']['canonicalSourceSHA256']!=r['canonicalSourceSHA256'] or replay_report['protocol']['sourceScopeSHA256']!=r['sourceScopeSHA256']:
  raise ValueError('replay uses a different source or review scope')
 out=base/args.output_name;out.mkdir(exist_ok=False)
 graph=json.loads((replay_root/'+0.json').read_text());p=graph['operations'][0]['params'];src=np.array(Image.open(source).convert('RGBA'));h,w=src.shape[:2]
 classes=np.fromfile(base/'fashn-classes.u8',np.uint8).reshape(cm['decodedDimensions'])
 part=sample_mask(p['partMask'],w,h);protect=sample_mask(p['protectMask'],w,h)
 renderer=ROOT/'core/image-core/target/release/examples/body-layered-replay'
 protocol={'sourceSHA256':sha(source),'sourceScopeSHA256':sha(scope_path),'graphSHA256':sha(base/'source-graph.json'),
  'classEvidenceSHA256':sha(base/'class-evidence.json'),'rendererSHA256':sha(renderer),'sourceArcsFrozenBeforeInference':True,
  'replayDirectory':str(replay_root),'replayReportSHA256':sha(replay_root/'report.json'),
  'fieldAPI':'prepare_body_inverse_displacement','newInference':False,'sourceCorrection':False,
  'arcBandPixels':4,'bandDefinedAfterReplayForDiagnosis':True,
  'analysisMethod':'pixels within 4 pixels of frozen visible garment polylines; RGB changed count and ordinary inverse-field magnitude',
  'bandIsExactMaterialTruth':False,'scalarProbeClassEncoding':cm['decodedEncoding']}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 arc_masks={}
 for arc in scope['sourceVisibleArcs']:
  mask=Image.new('L',(w,h));draw=ImageDraw.Draw(mask);draw.line([tuple(q) for q in arc['points']],fill=255,width=9)
  arc_masks[arc['id']]=np.array(mask)>0
 rows=[]
 for amount in [-30,30]:
  label=f'{amount:+d}';gp=replay_root/f'{label}.json';fp=out/f'{label}.f32'
  row_record=next(row for row in replay_report['results'] if row['amount']==amount)
  if sha(gp)!=row_record['graphSHA256'] or sha(replay_root/f'{label}.png')!=row_record['outputSHA256']:raise ValueError('frozen replay graph or image changed')
  run=subprocess.run([str(renderer),'--source',str(source),'--graph',str(gp),'--field-only','--save-field',str(fp)],capture_output=True,text=True)
  if run.returncode:raise RuntimeError(run.stderr)
  field=np.fromfile(fp,'<f4').reshape(h,w,2);motion=np.linalg.norm(field,axis=-1);actual=np.array(Image.open(replay_root/f'{label}.png').convert('RGBA'))
  diff=np.abs(actual[...,:3].astype(int)-src[...,:3].astype(int));changed=np.any(diff>0,axis=-1)
  row={'amount':amount,'fieldSHA256':sha(fp),'fieldOnlyStdout':run.stdout,'maximumDisplacement':float(motion.max()),'arcs':{}}
  for name,mask in arc_masks.items():
   row['arcs'][name]={'pixels':int(mask.sum()),'partMaskAboveHalf':int((part[mask]>.5).sum()),'protectedPixels':int((protect[mask]>0).sum()),
     'changedPixels':int(changed[mask].sum()),'maxRGB':int(diff[mask].max()),'maxDisplacement':float(motion[mask].max()),
     'displacedAbovePoint01':int((motion[mask]>.01).sum()),'meanDisplacement':float(motion[mask].mean())}
  rows.append(row)
 # Source-only posterior samples. Fixed arcs predate all edits; offsets are
 # displayed only as classification diagnostics, never used as edit controls.
 source_probes=[];subject=sample_mask(p['subjectMask'],w,h)
 for arc in scope['sourceVisibleArcs']:
  for x,y in arc['points']:
   for offset in [0,-8,8]:
    qx=int(np.clip(x+offset,0,w-1));qy=int(y);cx=min(classes.shape[2]-1,int((qx+.5)/w*classes.shape[2]));cy=min(classes.shape[1]-1,int((qy+.5)/h*classes.shape[1]));probs=classes[:,cy,cx]/255
    top=sorted(zip(cm['classNames'],probs.tolist()),key=lambda x:x[1],reverse=True)[:4]
    source_probes.append({'arc':arc['id'],'xy':[qx,qy],'xOffset':offset,'partMask':float(part[qy,qx]),'subjectMask':float(subject[qy,qx]),'protectMask':float(protect[qy,qx]),'topClasses':top})
 summary={'protocol':protocol,'region':p['region'],'landmarks':p['landmarks'],'rows':rows,'sourcePosteriorProbes':source_probes,
   'limitations':['Polyline bands are source inspection scopes, not physical alpha truth.','Raw logits are retained for exact posterior reconstruction; class diagnostics use the actual quantized masks used by current derivation.','No hidden anatomical waist claim.']}
 (out/'report.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'rows':rows,'region':p['region']},indent=2))
if __name__=='__main__':main()

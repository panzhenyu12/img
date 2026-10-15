"""Explicit component-aware denominator correction; no new model inference."""
from pathlib import Path
import collections,hashlib,importlib.util,json,sys
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.run/inpaint4drag-deps'));import cv2
sys.path.insert(0,str(ROOT/'scripts'));from replay_user_face_contour import raster_coverage
spec=importlib.util.spec_from_file_location('thresholds',HERE/'thresholds-snapshot.py');thresholds=importlib.util.module_from_spec(spec);spec.loader.exec_module(thresholds)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES={'holdout':(ROOT/'experiments/zim-semantic-holdout-20261015/combined-raw', 'report-with-feeds.json'),
         'development':(ROOT/'experiments/zim-combined-prompts-20261015/outputs','report.json')}


def stats(alpha,reference):
 values=alpha[reference]
 return {'referencePixels':int(len(values)), 'retainedAtHalf':float((values>=.5).mean()) if len(values) else None,
         'meanAlpha':float(values.mean()) if len(values) else None,'p95Alpha':float(np.percentile(values,95)) if len(values) else None,
         'alphaOver005Fraction':float((values>.05).mean()) if len(values) else None}


def cutout(rgb,a):
 return Image.fromarray(np.rint(rgb*a[:,:,None]+np.array([45,80,60])*(1-a[:,:,None])).clip(0,255).astype('uint8'))


def panel(path,images,labels,box):
 w,h=box[2]-box[0],box[3]-box[1];out=Image.new('RGB',(len(images)*w,h+28),'#222');draw=ImageDraw.Draw(out)
 for i,(im,title) in enumerate(zip(images,labels)):
  out.paste(im.crop(box),(i*w,28));draw.text((i*w+4,7),title,fill='white')
 out.save(path)


def reference_components(mask):
 n,labels,component_stats,_=cv2.connectedComponentsWithStats((mask>=.5).astype('uint8'))
 _,near=distance_transform_edt(labels==0,return_indices=True)
 nearest=labels[near[0],near[1]]
 return n,labels,component_stats,nearest


def main():
 out=HERE/'outputs';out.mkdir(exist_ok=False)
 report={'scope':'Explicit component denominator correction, same numerical gates; no inference or production mutation. Raw alpha remains unchanged; parser agreement is not physical alpha accuracy.',
  'scripts':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'thresholds-snapshot.py']},'datasets':[]}
 for dataset,(raw,report_name) in SOURCES.items():
  original=json.loads((raw/report_name).read_text());data={'name':dataset,'rawReportSha256':sha(raw/report_name),'photos':[]}
  for photo in original['images']:
   source=ROOT/photo['source'];graph=ROOT/photo['graph'];assert sha(source)==photo['sourceSha256'] and sha(graph)==photo['graphSha256']
   image=Image.open(source).convert('RGB');rgb=np.asarray(image);w,h=image.size
   p=json.loads(graph.read_text())['operations'][0]['params'];masks={k:raster_coverage(v,w,h) for k,v in p['semanticMasks'].items() if v['shape']=='raster'}
   n,labels,cs,nearest=reference_components(masks['hair']);hair=masks['hair']>=.9;skin=(masks['skin']>=.9)&(masks['hair']<.05)&(masks['accessories']<.05)
   components={i:(labels==i)&hair for i in range(1,n) if ((labels==i)&hair).any()}
   prompted=sorted(set(c['prompt']['component'] for c in photo['rows']));dest=out/dataset/photo['name'];dest.mkdir(parents=True)
   ref={'source':photo['source'],'sourceSha256':photo['sourceSha256'],'graph':photo['graph'],'graphSha256':photo['graphSha256'],
    'name':photo['name'],'size':[w,h],'promptedComponents':prompted,'referenceComponents':[{'component':int(i),'binaryArea':int(cs[i,4]),'corePixels':int(core.sum())} for i,core in components.items()],
    'globalHairCorePixels':int(hair.sum()),'unpromptedHairCorePixels':int(sum(core.sum() for i,core in components.items() if i not in prompted)), 'cases':[],'unions':[]}
   bygroup=collections.defaultdict(list)
   for case in photo['rows']:
    cid=case['prompt']['component'];assert int(cs[cid,4])==case['prompt']['sourceReferenceArea'];x,y=case['prompt']['positive'];assert labels[y,x]==cid
    core=components.get(cid,np.zeros((h,w),bool));fringe=(masks['hair']>=.05)&(masks['hair']<.9)&(nearest==cid)
    apath=raw/case['directory']/'alphas.npy';assert sha(apath)==case['alphaSha256'];alphas=np.load(apath,mmap_mode='r')
    adapted=[];audit=[]
    for c in case['alphas']:
     a=alphas[c['index']];global_metric=float((a[hair]>=.5).mean());assert abs(global_metric-c['parserCoreRetainedFraction'])<1e-7
     component_metric=float((a[core]>=.5).mean()) if core.any() else None
     adapted.append({**c,'parserCoreRetainedFraction':component_metric if component_metric is not None else 0.})
     audit.append({'index':c['index'],'score':c['score'],'globalHair':stats(a,hair),'componentHair':stats(a,core),'componentParserFringe':stats(a,fringe),
                   'skin':stats(a,skin),'ears':stats(a,masks['ears']>=.9),'accessories':stats(a,masks['accessories']>=.9)})
    old,old_dec=thresholds.choose(case['alphas'],'hair');new,new_dec=thresholds.choose(adapted,'hair')
    if not core.any():new=None;new_dec=[{'index':c['index'],'eligible':False,'reasons':['no confident source component core']} for c in adapted]
    record={'directory':case['directory'],'mode':case['mode'],'prompt':case['prompt'],'alphaSha256':case['alphaSha256'],
      'officialIndex':case['officialScoreArgmax'],'oldSelected':old,'componentSelected':new,'selectionChanged':old!=new,
      'componentCorePixels':int(core.sum()),'oldDecisions':old_dec,'componentDecisions':new_dec,'allCandidateAudits':audit,
      'selectedAudit':audit[new] if new is not None else None}
    ref['cases'].append(record);bygroup[(case['prompt']['protocol'],case['mode'])].append((case,record))
    if old!=new:
     folder=dest/(case['prompt']['protocol']+f'-{cid}-'+case['mode']);folder.mkdir()
     box=case['prompt']['box'];box=(max(0,box[0]-12),max(0,box[1]-12),min(w,box[2]+13),min(h,box[3]+13))
     empty=Image.new('RGB',(w,h),'#222');panel(folder/'component-change-1x.png',[image,cutout(rgb,masks['hair']),cutout(rgb,alphas[old]) if old is not None else empty,cutout(rgb,alphas[new]) if new is not None else empty],
       ['Original','Parser reference',f'Old {old}' if old is not None else 'Old ABSTAIN',f'Component {new}' if new is not None else 'New ABSTAIN'],box)
     if new is not None:Image.fromarray(np.rint(alphas[new]*255).astype('uint8')).save(folder/'selected-alpha.png')
   for (protocol,mode),cases in sorted(bygroup.items()):
    folder=dest/(protocol+'-'+mode+'-union');folder.mkdir()
    unions={kind:np.zeros((h,w),np.float32) for kind in ['old','component']}
    available={kind:[] for kind in unions};negative=[];positive=[]
    for case,rec in cases:
     alphas=np.load(raw/case['directory']/'alphas.npy',mmap_mode='r')
     positive.append(case['prompt']['positive']);negative+=case['prompt']['negative']
     for kind,key in [('old','oldSelected'),('component','componentSelected')]:
      if rec[key] is not None:np.maximum(unions[kind],alphas[rec[key]],out=unions[kind]);available[kind].append(rec['prompt']['component'])
    union_report={'protocol':protocol,'mode':mode,'expectedPromptedComponents':prompted,'componentsWithCases':sorted(c['prompt']['component'] for c,_ in cases),'variants':{}}
    for kind,a in unions.items():
     selected_ids=sorted(available[kind]);complete=selected_ids==prompted
     numeric={'index':0,'score':0.,'positiveAlpha':min(float(a[y,x]) for x,y in positive),'negativeAlpha':[float(a[y,x]) for x,y in negative],
              'parserCoreRetainedFraction':float((a[hair]>=.5).mean()),'skinCoreRetainedFraction':float((a[skin]>=.5).mean())}
     passes,reasons=thresholds.choose([numeric],'hair')
     union_report['variants'][kind]={'selectedComponents':selected_ids,'missingComponents':sorted(set(prompted)-set(selected_ids)),
      'componentStatus':'all prompted components selected' if complete else 'partial' if selected_ids else 'abstain',
      'physicalMatteAccepted':False,'globalHair':stats(a,hair),'perComponentHair':{str(i):stats(a,c) for i,c in components.items()},
      'skin':stats(a,skin),'ears':stats(a,masks['ears']>=.9),'accessories':stats(a,masks['accessories']>=.9),
      'unionPassesOriginalNumericGate':passes is not None,'unionGateDecisions':reasons,'unionPromptMetrics':numeric}
    a=unions['component'];np.save(folder/'union-alpha.npy',a,allow_pickle=False);Image.fromarray(np.rint(a*255).astype('uint8')).save(folder/'union-alpha.png')
    panel(folder/'face-union-1x.png',[image,cutout(rgb,masks['hair']),cutout(rgb,unions['old']),cutout(rgb,a)],
      ['Original','Parser reference','Old union '+union_report['variants']['old']['componentStatus'],'Component union '+union_report['variants']['component']['componentStatus']],photo['faceCrop'])
    union_report['directory']=str(folder.relative_to(out));ref['unions'].append(union_report)
   data['photos'].append(ref);report['datasets']=[*report['datasets'],data] if data not in report['datasets'] else report['datasets']
   (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
   print(json.dumps({'dataset':dataset,'photo':photo['name'],'cases':len(ref['cases']),'changes':sum(c['selectionChanged'] for c in ref['cases']),'oldSelected':sum(c['oldSelected'] is not None for c in ref['cases']),'newSelected':sum(c['componentSelected'] is not None for c in ref['cases']),'unpromptedCorePixels':ref['unpromptedHairCorePixels']}),flush=True)

if __name__=='__main__':main()

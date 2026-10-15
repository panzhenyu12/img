"""Frozen public-render replay for lip brightness. Scores are deliberately absent."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from review_face_refinement import polygon_interior
from check_skin_detail_quality import comparison, sha, run

QA = ROOT / 'doc/qa/lip-lighting-v2-20261015'
OUT = HERE / 'evidence/replay'
QA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
label = sys.argv[1] if len(sys.argv) > 1 else 'candidate-a'
binaries = {'baseline': HERE / 'baseline-isolated-render', 'candidate': HERE / (label + '-render')}
hashes = {k: sha(v) for k, v in binaries.items()}
assert hashes['baseline'] != hashes['candidate'], 'A/B artifact identity error: both binaries have identical SHA'
samples = [('user', ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png', ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json')]
for name in ['portrait-makeup', 'portrait-woman', 'portrait-hair', 'portrait-dark']:
    samples.append((name, ROOT/f'test/results/native-face-contour/{name}-source.png', ROOT/f'test/results/native-face-contour/{name}-graph.json'))
cases = [(f'brightness-{a}', {'lipBrightness':a}) for a in [-100,-30,0,30,100]]
cases += [('color-only', {'lipColor':60,'lipBrightness':0}), ('saturation-only', {'lipSaturation':60,'lipBrightness':0})]
cases += [('color-bright', {'lipColor':60,'lipBrightness':60}), ('color-dark', {'lipColor':60,'lipBrightness':-60}), ('saturation-bright', {'lipSaturation':60,'lipBrightness':60}), ('desaturation-dark', {'lipSaturation':-60,'lipBrightness':-60})]
rows=[]
for name, source, parsing in samples:
    all_params=json.loads(parsing.read_text())['operations'][0]['params']
    params={k:v for k,v in all_params.items() if k in {'region','mask','protectMask','landmarks','semanticMasks'}}
    original_image=Image.open(source).convert('RGBA'); original=np.asarray(original_image)
    width,height=original_image.size
    teeth=polygon_interior(params['semanticMasks']['teeth'], original_image.size)
    points=params['semanticMasks']['lips']['points']
    feather=params['semanticMasks']['lips'].get('feather',0)/100*.2
    lip_box=(max(0,int(min(p['x'] for p in points)*width)-14),max(0,int(min(p['y'] for p in points)*height)-14),min(width,int(max(p['x'] for p in points)*width)+15),min(height,int(max(p['y'] for p in points)*height)+15))
    yy,xx=np.indices(original.shape[:2]); nx=(xx+.5)/width; ny=(yy+.5)/height
    roi=params['region']; inside_roi=(nx>=roi['x']) & (nx<=roi['x']+roi['width']) & (ny>=roi['y']) & (ny<=roi['y']+roi['height'])
    # A conservative bounding support, expanded by the full feather range,
    # verifies all distant skin/background without falsely excluding a real soft lip edge.
    support=(nx>=min(p['x'] for p in points)-feather) & (nx<=max(p['x'] for p in points)+feather) & (ny>=min(p['y'] for p in points)-feather) & (ny<=max(p['y'] for p in points)+feather)
    skin=params['semanticMasks'].get('skin')
    skin_interior=polygon_interior(skin,original_image.size) if skin and skin['shape']=='polygon' else None
    candidate_sequence=[]
    panels_for_contact=[]
    for case,adjustments in cases:
        graph={'version':'1.0','sourceAssetId':name,'operations':[{'id':'lip-lighting','type':'feature.adjust','enabled':True,'params':{**params,'adjustments':adjustments}}]}
        graph_path=OUT/f'{name}-{case}.json'; graph_path.write_text(json.dumps(graph))
        row={'sample':name,'case':case,'adjustments':adjustments,'source':str(source.relative_to(ROOT)),'source_sha256':sha(source),'parsing':str(parsing.relative_to(ROOT)),'parsing_sha256':sha(parsing),'graph':str(graph_path.relative_to(ROOT)),'graph_sha256':sha(graph_path),'semantic_masks':sorted(params['semanticMasks']), 'width':width,'height':height}
        panels=[(original_image,'Original')]
        results={}
        for variant,binary in binaries.items():
            output=OUT/f'{name}-{case}-{variant}.png'
            elapsed,result=run(binary,hashes[variant],source,graph_path,output)
            assert result.returncode==0,result.stderr
            edited_image=Image.open(output).convert('RGBA'); edited=np.asarray(edited_image)
            changed=np.any(edited!=original,axis=2); results[variant]=edited
            contract={'outside_roi':int(changed[~inside_roi].sum()),'distant_skin_background':int(changed[~support].sum()),'inner_mouth_teeth':int(changed[teeth].sum()),'alpha':int((edited[:,:,3]!=original[:,:,3]).sum())}
            if skin_interior is not None: contract['distant_skin']=int(changed[skin_interior & ~support].sum())
            assert not any(contract.values()),(name,case,variant,contract)
            if case=='brightness-0': assert not changed.any()
            row[variant]={'output':str(output.relative_to(ROOT)),'output_sha256':sha(output),'native_total_ms':round(elapsed,2),'changed_pixels':int(changed.sum()),'contracts':contract}
            panels.append((edited_image,variant+' '+case))
        row['candidate_vs_baseline_pixels']=int(np.any(results['candidate']!=results['baseline'],axis=2).sum())
        if case in {'color-only','saturation-only','brightness-0'}: assert row['candidate_vs_baseline_pixels']==0
        if case.startswith('brightness-'):
            candidate_sequence.append(results['candidate'][:,:,:3].astype(np.int16))
            panels_for_contact.append((case,[im.crop(lip_box) for im,_ in panels]))
        comparison(panels,lip_box,QA/f'{name}-{case}.png')
        rows.append(row)
    monotonic_violations=sum(int((b<a).any(axis=2).sum()) for a,b in zip(candidate_sequence,candidate_sequence[1:]))
    assert monotonic_violations==0,(name,monotonic_violations)
    w,h=panels_for_contact[0][1][0].size
    sheet=Image.new('RGB',(3*w,5*(h+28)),'#171c22'); draw=ImageDraw.Draw(sheet)
    for y,(case,images) in enumerate(panels_for_contact):
        for x,(variant,img) in enumerate(zip(['original','baseline','candidate'],images)):
            draw.text((x*w+4,y*(h+28)+5),f'{variant} {case}',fill='white');sheet.paste(img,(x*w,y*(h+28)+28))
    sheet.save(QA/f'{name}-contact-100pct.png')
    print(json.dumps({'sample':name,'cases':len(cases),'monotonic_violations':monotonic_violations}),flush=True)
report={'date':'2026-10-15','candidate':label,'independent_originals':len(samples),'binary_sha256':hashes,'source_sha256':{k:sha(HERE/(('baseline' if k=='baseline' else 'candidate')+'/src/color.rs')) for k in binaries},'rows':rows,'timing_scope':'Native process + decode + public graph + encode; not isolated hot runtime, WASM or browser latency.','scope':'lipBrightness single operation plus lipColor/lipSaturation combinations. Cached parsing retained; no model rerun. Older four portraits lack ears/accessories. Geometry is not evaluated.','aesthetic_score':None}
(QA/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'cases':len(rows),'independent_originals':len(samples),'report':str(QA/'report.json')},indent=2))

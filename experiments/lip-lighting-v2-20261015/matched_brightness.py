"""Compare lip material at similar measured mean luminance, avoiding dose-only claims."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from check_skin_detail_quality import sha,run,comparison
QA=ROOT/'doc/qa/lip-lighting-v2-20261015'
OUT=HERE/'evidence/matched'
OUT.mkdir(exist_ok=True,parents=True)
report=json.loads((QA/'report.json').read_text())
baseline=HERE/'baseline-isolated-render'; expected=sha(baseline)
assert expected==report['binary_sha256']['baseline']
W=np.array([.2126,.7152,.0722])
def linear(rgb):
    v=rgb.astype(float)/255
    return np.where(v<=.04045,v/12.92,((v+.055)/1.055)**2.4)
def measures(original,output,support):
    a=original[:,:,:3][support];b=output[:,:,:3][support]
    la,lb=linear(a),linear(b)
    ya,yb=la@W,lb@W
    valid=la.sum(axis=1)>.005
    ca=la[valid]/la[valid].sum(axis=1,keepdims=True)
    cb=lb[valid]/np.maximum(lb[valid].sum(axis=1,keepdims=True),1e-9)
    chroma=np.abs(ca-cb).sum(axis=1)
    dark=ya<=np.quantile(ya,.25)
    middle=(ya>=np.quantile(ya,.40))&(ya<=np.quantile(ya,.60))
    return {'source_mean_Y_percent':float(ya.mean()*100),'output_mean_Y_percent':float(yb.mean()*100),'mean_delta_Y_percentage_points':float((yb-ya).mean()*100),'mean_delta_encoded_rec709':float(((b.astype(float)-a.astype(float))@W).mean()),'material_chromaticity_L1_median':float(np.median(chroma)),'material_chromaticity_L1_p95':float(np.percentile(chroma,95)),'shadow_quartile_mean_Y_percent':float(yb[dark].mean()*100),'shadow_to_midtone_ratio':float(yb[dark].mean()/max(yb[middle].mean(),1e-9))}
rows=[]
for row in report['rows']:
    if row['case'] not in ['brightness--100','brightness-100']:continue
    original_image=Image.open(ROOT/row['source']).convert('RGBA');o=np.asarray(original_image)
    base_image=Image.open(ROOT/row['baseline']['output']).convert('RGBA');b=np.asarray(base_image)
    cand_image=Image.open(ROOT/row['candidate']['output']).convert('RGBA');c=np.asarray(cand_image)
    support=np.any(o!=b,axis=2)|np.any(o!=c,axis=2)
    original_Y=(linear(o[:,:,:3])@W)
    target=float((linear(c[:,:,:3])@W)[support].mean())
    graph=json.loads((ROOT/row['graph']).read_text())
    lo,hi=(-100.,0.) if row['adjustments']['lipBrightness']<0 else (0.,100.)
    matched=None
    endpoint_Y=float((linear(b[:,:,:3])@W)[support].mean())
    original_mean_Y=float(original_Y[support].mean())
    reachable=min(endpoint_Y,original_mean_Y)-1e-9 <= target <= max(endpoint_Y,original_mean_Y)+1e-9
    for iteration in range(8 if reachable else 1):
        amount=(lo+hi)/2 if reachable else row['adjustments']['lipBrightness']
        graph['operations'][0]['params']['adjustments']['lipBrightness']=amount
        gp=OUT/f'{row["sample"]}-{row["case"]}-matched.json';gp.write_text(json.dumps(graph))
        op=OUT/f'{row["sample"]}-{row["case"]}-matched.png'
        _,result=run(baseline,expected,ROOT/row['source'],gp,op);assert result.returncode==0,result.stderr
        mi=Image.open(op).convert('RGBA');m=np.asarray(mi)
        mean=float((linear(m[:,:,:3])@W)[support].mean())
        if mean<target:lo=amount
        else:hi=amount
        matched=(mi,m,amount,mean,gp,op)
    mi,m,amount,mean,gp,op=matched
    points=graph['operations'][0]['params']['semanticMasks']['lips']['points'];width,height=original_image.size
    crop=(max(0,int(min(p['x'] for p in points)*width)-14),max(0,int(min(p['y'] for p in points)*height)-14),min(width,int(max(p['x'] for p in points)*width)+15),min(height,int(max(p['y'] for p in points)*height)+15))
    comparison([(original_image,'original'),(base_image,'old '+str(row['adjustments']['lipBrightness'])),(mi,f'old {amount:.1f} '+('matched Y' if reachable else 'dose limit')),(cand_image,'candidate '+str(row['adjustments']['lipBrightness']))],crop,QA/f'{row["sample"]}-{row["case"]}-matched.png')
    rows.append({'sample':row['sample'],'amount':row['adjustments']['lipBrightness'],'support_pixels':int(support.sum()),'support_definition':'union of actual baseline/candidate changed pixels, same support for every comparator; teeth/protection excluded by zero edits','baseline':measures(o,b,support),'candidate':measures(o,c,support),'matched_baseline':measures(o,m,support),'matched_baseline_amount':amount,'matched_absolute_Y_error_percentage_points':abs(mean-target)*100,'matching_status':('target outside baseline dose range' if not reachable else 'matched' if abs(mean-target)<.001 else 'quantization tolerance exceeded'),'matched_graph_sha256':sha(gp),'matched_output_sha256':sha(op)})
(QA/'matched-brightness.json').write_text(json.dumps({'binary_sha256':report['binary_sha256'],'metric':'linear-light relative luminance Y, sRGB transfer and Rec.709 weights; Y percentage points are 100*(Y_out-Y_in)','rows':rows,'limitation':'Quantitative color/brightness diagnostics are not aesthetic acceptance. Matching only mean luminance does not match spatial appearance.'},indent=2)+'\n')
for r in rows:
 print(r['sample'],r['amount'],'delta Y old/new',round(r['baseline']['mean_delta_Y_percentage_points'],3),round(r['candidate']['mean_delta_Y_percentage_points'],3),'old matched',round(r['matched_baseline_amount'],2),'match error',round(r['matched_absolute_Y_error_percentage_points'],4),'chroma median matched/new',round(r['matched_baseline']['material_chromaticity_L1_median'],4),round(r['candidate']['material_chromaticity_L1_median'],4),flush=True)

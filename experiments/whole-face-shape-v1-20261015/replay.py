"""Frozen source/graph replay for experimental whole-face width curves.

The compositor, physical alpha and existing attachment/feature constraints
are unchanged. This is not a deployed face preset or a learned aesthetic model.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,time
import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from compose_real import montage
from run_real import probe
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES={'user-face-contour':'test/results/user-face-contour-20261014/1024/canonical-source.png','portrait-woman':'experiments/face-anatomical-v22-20261015/inputs/portrait-woman-source.png','portrait-hair':'test/results/face-core-review/portrait-hair-source.png'}


def stage_metrics(path):
    trace=json.loads(path.read_text());origin=np.asarray(trace['origin']);across=np.asarray(trace['across']);down=np.asarray(trace['down']);l,t,r,b=trace['bounds']
    points=np.asarray(trace['source'])[:128];u=(points-origin)@across;v=((points-origin)@down-t)/(b-t)
    result={}
    for band,lo,hi in [('upper',.12,.32),('middle',.36,.62),('lower',.66,.88),('chin',.88,.985)]:
        result[band]={}
        for side,sign in [('left',-1),('right',1)]:
            domain=(v>=lo)&(v<hi)&(sign*u>.25*(r-l))
            info={'controls':int(domain.sum())}
            for stage in ['requested','feasible','actual']:
                q=np.asarray(trace[stage])[:128];signed_inward=np.sign(u)*((points-q)@across)
                info[stage+'InwardMedianPx']=float(np.median(signed_inward[domain])) if domain.any() else None
                info[stage+'InwardMaxPx']=float(signed_inward[domain].max()) if domain.any() else None
            result[band][side]=info
    return result


def main(name):
    out=HERE/'outputs'/name;out.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'.run/whole-face-shape-v1-target/release/image-render';frozen=out/('image-render-'+sha(binary));shutil.copy2(binary,frozen)
    src=ROOT/SOURCES[name];original=Image.open(src).convert('RGBA');original_array=np.asarray(original)
    base_graph=ROOT/f'experiments/face-source-curve-v36-20261015/outputs/{name}-0/graph.json'
    graph=json.loads(base_graph.read_text());params=graph['operations'][0]['params'];box=probe.face_crop(params,original.width,original.height)
    masks={k:probe.zim.coverage(v,original.width,original.height) for k,v in params['semanticMasks'].items()}
    fixed=np.maximum.reduce([masks[k] for k in ['ears','hair','accessories']])>=.5
    report={'accepted':False,'source':SOURCES[name],'sourceSha256':sha(src),'baseGraphSha256':sha(base_graph),'binarySha256':sha(frozen),'protocol':'faceShape width-only prototype; unchanged source masks/landmarks/compositor; no redetection','rows':[],'legacyCompatibility':[]}
    for amount in [-30,30]:
        p=ROOT/f'experiments/face-source-curve-v36-20261015/outputs/{name}-{amount}/graph.json';d=out/f'legacy-{amount}';d.mkdir()
        output=d/'candidate.png';result=subprocess.run([str(frozen),'--input',str(src),'--graph',str(p),'--output',str(output)],capture_output=True);(d/'render.log').write_bytes(result.stdout+result.stderr);result.check_returncode()
        reference=ROOT/f'experiments/face-source-residual-v39-20261015/outputs/{name}/{amount}/current.png'
        assert np.array_equal(np.asarray(Image.open(output).convert('RGBA')),np.asarray(Image.open(reference).convert('RGBA'))),'legacy output changed'
        report['legacyCompatibility'].append({'amount':amount,'graphSha256':sha(p),'reference':str(reference.relative_to(ROOT)),'referenceSha256':sha(reference),'candidateSha256':sha(output),'pixelExact':True})
    cases=[('natural',0),('natural',30),('natural',-30)]
    if name=='user-face-contour':cases += [('oval',30),('soft',30),('angular',30)]
    for preset,amount in cases:
        d=out/f'{preset}-{amount}';d.mkdir();g=copy.deepcopy(graph);p=g['operations'][0]['params'];p['adjustments']={};p['faceShape']={'preset':preset,'amount':amount};gp=d/'graph.json';gp.write_text(json.dumps(g,separators=(',',':')))
        start=time.perf_counter();result=subprocess.run([str(frozen),'--input',str(src),'--graph',str(gp),'--output',str(d/'candidate.png')],env={**os.environ,'WHOLE_FACE_DEBUG_DIR':str(d/'debug')},capture_output=True);elapsed=(time.perf_counter()-start)*1000
        (d/'render.log').write_bytes(result.stdout+result.stderr);result.check_returncode()
        edited=Image.open(d/'candidate.png').convert('RGBA');array=np.asarray(edited)
        assert np.array_equal(array[...,3],original_array[...,3]);assert np.array_equal(array[fixed],original_array[fixed])
        if amount==0:assert np.array_equal(array,original_array)
        reference=original if amount==0 else Image.open(out/f'legacy-{amount}/candidate.png') if preset=='natural' else Image.open(out/'natural-30/candidate.png')
        ref_label='Source' if amount==0 else f'Existing faceWidth {amount}' if preset=='natural' else 'Whole width natural 30'
        panels=[('face',box)]+([('left',[320,400,420,550]),('right',[590,400,700,550])] if name=='user-face-contour' else [])
        for label,crop in panels:montage(d/f'{label}-1x.png',[original,reference,edited],['Source',ref_label,f'Width curve {preset} {amount}'],crop)
        row={'preset':preset,'amount':amount,'elapsedMs':elapsed,'graphSha256':sha(gp),'outputSha256':sha(d/'candidate.png'),'changedPixels':int(np.any(array!=original_array,axis=2).sum()),'fixedChanged':0,'alphaChanged':0}
        if amount:row['stages']=stage_metrics(d/'debug/flow.json')
        report['rows'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'name':name,**{k:v for k,v in row.items() if k!='stages'}}),flush=True)
    # Requested curves are diagnostic overlays, clearly separate from images.
    presets=['natural','oval','soft','angular'] if name=='user-face-contour' else ['natural']
    x0,y0,x1,y1=box;cw,ch=x1-x0,y1-y0;canvas=Image.new('RGB',(cw*len(presets),ch+28),'#20242a');draw=ImageDraw.Draw(canvas)
    for i,preset in enumerate(presets):
        trace=json.loads((out/f'{preset}-30/debug/requested.json').read_text());panel=original.convert('RGB').crop(box);pen=ImageDraw.Draw(panel)
        for key,color in [('source',(250,220,60)),('target',(20,240,230))]:
            points=[(p[0]-x0,p[1]-y0) for p in trace[key]];pen.line(points+[points[0]],fill=color,width=1)
        canvas.paste(panel,(i*cw,28));draw.text((i*cw+3,8),f'{preset} 30: source yellow / target cyan',fill='white')
    canvas.save(out/'requested-curves-1x.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',choices=list(SOURCES),required=True);main(parser.parse_args().name)

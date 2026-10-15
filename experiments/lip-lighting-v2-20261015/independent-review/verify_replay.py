"""Review frozen replay evidence and measure actual lip material response.

No renderer, mask, strength, image or candidate source is modified.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from review_face_refinement import polygon_interior

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    r=json.loads(args.report.read_text());label=r['candidate'];rows=[];cache={}
    assert r['binary_sha256']['baseline']!=r['binary_sha256']['candidate'],'comparison used identical binaries'
    assert sha(HERE.parent/'baseline-isolated-render')==r['binary_sha256']['baseline']
    assert sha(HERE.parent/(label+'-render'))==r['binary_sha256']['candidate']
    for entry in r['rows']:
        name=entry['sample'];source_path=ROOT/entry['source'];graph_path=ROOT/entry['graph']
        assert sha(source_path)==entry['source_sha256'];assert sha(graph_path)==entry['graph_sha256']
        if name not in cache:
            source=np.array(Image.open(source_path).convert('RGBA'));h,w=source.shape[:2]
            p=json.loads(graph_path.read_text())['operations'][0]['params'];s=p['semanticMasks']
            lips=polygon_interior(s['lips'],(w,h))&~polygon_interior(s['teeth'],(w,h))
            rgb=source[...,:3].astype('float32');luma=rgb@np.array([.2126,.7152,.0722],dtype='float32')
            saturated=lips&(rgb.max(-1)==255)&(rgb.max(-1)-rgb.min(-1)>30)
            shadows=lips&(luma<=np.quantile(luma[lips],.2))
            highlights=lips&(luma>=235)&(rgb.max(-1)-rgb.min(-1)<20)
            cache[name]=(source,lips,saturated,shadows,highlights,luma)
        source,lips,saturated,shadows,highlights,original_luma=cache[name]
        variants={}
        for k in ['baseline','candidate']:
            path=ROOT/entry[k]['output'];assert sha(path)==entry[k]['output_sha256']
            variants[k]=np.array(Image.open(path).convert('RGBA'))
        old,new=variants['baseline'],variants['candidate']
        actual_delta=int(np.any(new!=old,axis=-1).sum())
        assert actual_delta==entry['candidate_vs_baseline_pixels']
        assert np.array_equal(new[...,3],source[...,3])
        case=entry['case']
        if case=='brightness-0':assert np.array_equal(new,source)
        if case in ['color-only','saturation-only','brightness-0']:assert np.array_equal(new,old)
        if case not in ['brightness--100','brightness--30','brightness-30','brightness-100']:continue
        facts={'sample':name,'case':case,'candidateVsBaselineChanged':actual_delta}
        for variant,array in variants.items():
            diff=array[...,:3].astype('float32')@np.array([.2126,.7152,.0722],dtype='float32')-original_luma
            info={}
            for band,domain in [('lipMaterial',lips),('saturatedColor',saturated),('shadowQuintile',shadows),('neutralHighlight',highlights)]:
                values=diff[domain]
                info[band]={'pixels':int(domain.sum()),'medianLumaDelta':float(np.median(values)) if values.size else None,'meanLumaDelta':float(values.mean()) if values.size else None,'positivePixels':int((values>.01).sum()),'negativePixels':int((values<-.01).sum()),'unchangedPixels':int((np.max(np.abs(array[...,:3].astype(int)-source[...,:3].astype(int)),axis=-1)[domain]==0).sum())}
            facts[variant]=info
        rows.append(facts)
    result={'reportSha256':sha(args.report),'label':label,'binaryHashes':r['binary_sha256'],'verifiedRows':len(r['rows']),'distinctComparedBinaries':True,'candidateModified':False,'note':'Luma metrics use independently computed polygon interiors; soft mask boundaries remain included. They are diagnostics, not aesthetic scores.','rows':rows}
    (HERE/f'{label}-verified-response.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'label':label,'verifiedRows':len(r['rows']),'diagnostics':str(HERE/f'{label}-verified-response.json')},indent=2))

if __name__=='__main__':main()

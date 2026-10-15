"""Fixed source/core/donor/domain ablation of the relative alpha cap."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw
from alpha import prepare,solve,VARIANTS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
EXP=ROOT/'experiments/hair-skin-decomposition-v1-20261015'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,v):Image.fromarray(np.floor(np.clip(v,0,1)*255+.5).astype('uint8')).save(path)
def panel(path,images,labels):
    h,w=images[0].shape[:2];im=Image.new('RGB',(w*len(images),h+28),'#20242a');draw=ImageDraw.Draw(im)
    for i,(v,label) in enumerate(zip(images,labels)):
        im.paste(Image.fromarray(np.floor(np.clip(v,0,1)*255+.5).astype('uint8')).convert('RGB'),(w*i,28));draw.text((w*i+3,7),label,fill='white')
    im.save(path)
def main():
    out=HERE/'source';out.mkdir(exist_ok=False)
    original_path=EXP/'source-decomposition-corrected-prompts/report.json';original=json.loads(original_path.read_text())
    report=dict(scope='Three existing originals; no inference. Fixed cores, donor support, solve domain and baseline trusted domain. Only alpha rule differs.',
        sourceReport=str(original_path.relative_to(ROOT)),sourceReportSha256=sha(original_path),
        scripts={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'alpha.py',HERE/'frozen/decompose.py']},rows=[])
    for row in original['rows']:
        if row['variant']!='source-fit-srgb':continue
        name=row['name'];dest=out/name;dest.mkdir()
        for key,digest in [('source','sourceSha256'),('graph','graphSha256'),('selectedAlpha','alphaSha256')]:assert sha(ROOT/row[key])==row[digest]
        x0,y0,x1,y1=row['crop'];rgb=np.asarray(Image.open(ROOT/row['source']).convert('RGB'),dtype='float64')[y0:y1,x0:x1]/255
        raw=np.load(ROOT/row['selectedAlpha'],mmap_mode='r')[row['officialIndex'],y0:y1,x0:x1].astype('float64')
        old_file=EXP/row['directory']/'layers.npz';old=np.load(old_file)
        domain=old['domain'];fixed_trusted=old['trusted'];prior_radius=max(1,int(np.ceil(row['parserCellPx'])))
        p=prepare(rgb,raw,old['hairCore'],old['skinCore'],domain,prior_radius)
        np.savez_compressed(dest/'frozen-inputs.npz',rawAlpha=raw.astype('float32'),hairCore=old['hairCore'],skinCore=old['skinCore'],domain=domain,
            trusted=fixed_trusted,hairPrior=p['hairPrior'].astype('float32'),skinPrior=p['skinPrior'].astype('float32'),
            intervalHalfWidth=p['intervalHalfWidth'].astype('float32'),intervalCompatible=p['intervalCompatible'],
            donorVariance=p['variance'].astype('float32'),separability=p['separability'].astype('float32'),rawPhotometric=p['fit'].astype('float32'))
        save(dest/'source.png',rgb);save(dest/'raw-alpha.png',raw);save(dest/'fixed-domain.png',domain)
        cap_active=domain&(np.abs(p['proposals']['uncapped']-raw)>.150001);save(dest/'cap-active.png',cap_active)
        panel_images=[rgb,np.repeat(raw[...,None],3,axis=-1)];panel_labels=['Source','Raw alpha']
        for variant in VARIANTS:
            folder=dest/variant;folder.mkdir();result=solve(rgb,p,variant)
            if variant=='bounded-015':
                errors={k:float(np.abs(result[k]-old[k]).max()) for k in ['alpha','hair','skin']}
                assert max(errors.values())<1e-7,errors
                assert np.array_equal(result['intrinsicTrusted'],old['trusted'])
            else:errors=None
            np.savez_compressed(folder/'layers.npz',hair=result['hair'].astype('float32'),skin=result['skin'].astype('float32'),alpha=result['alpha'].astype('float32'),
                domain=domain,trusted=fixed_trusted,intrinsicTrusted=result['intrinsicTrusted'],sourceResidual=result['residual'].astype('float32'),
                pureReclaim=old['pureReclaim'],hairCore=old['hairCore'],skinCore=old['skinCore'])
            save(folder/'alpha.png',result['alpha']);save(folder/'source-reconstructed.png',np.where(domain[...,None],result['sourceReconstructed'],rgb))
            save(folder/'source-residual-x16.png',result['residual']*16/255)
            save(folder/'alpha-change-x4.png',np.abs(result['alpha']-raw)*4)
            panel_images.append(np.repeat(result['alpha'][...,None],3,axis=-1));panel_labels.append(variant)
            stats=dict(**row)
            stats.update(space='srgb',directory=str(folder.relative_to(ROOT)),baselineLayerSha256=sha(old_file),
                frozenInputSha256=sha(dest/'frozen-inputs.npz'),layersSha256=sha(folder/'layers.npz'),baselineReproductionMaxErrors=errors,
                frozenTrustedPixels=int(fixed_trusted.sum()),trustedNowResidualOver4=int((fixed_trusted&~result['intrinsicTrusted']).sum()),
                **result['stats'])
            report['rows'].append(stats);print(json.dumps({k:stats[k] for k in ['name','variant','unknownPixels','sourceMeanAbsByte','capWouldBindPixels','trustedNowResidualOver4','alphaMaxChange']}),flush=True)
        panel(dest/'source-alpha-comparison-1x.png',panel_images,panel_labels)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for path,digest in report['scripts'].items():assert sha(ROOT/path)==digest

if __name__=='__main__':main()

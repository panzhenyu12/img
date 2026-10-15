"""Source-evidence diagnostic for all accessory alphas; never edits an alpha.

Low accessory probability is not skin evidence. Unknown holes remain unknown.
Intensity-tail checks are an automatic diagnostic, not metal ground truth.
"""
from pathlib import Path
import importlib.util,json
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('runner',HERE/'run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
ROOT,np,Image,probe,zim=run.ROOT,run.np,run.Image,run.probe,run.zim
cv2,ImageDraw=probe.cv2,probe.ImageDraw
from scipy.ndimage import binary_fill_holes,distance_transform_edt

def panel(path,items,crop):
    cw,ch=crop[2]-crop[0],crop[3]-crop[1]
    image=Image.new('RGB',(cw*len(items),ch+28),'#20242a');draw=ImageDraw.Draw(image)
    for i,(label,im) in enumerate(items):
        image.paste(im.convert('RGB').crop(crop),(cw*i,28));draw.text((cw*i+3,7),label,fill='white')
    image.save(path)
    image.resize((image.width*4,image.height*4),Image.Resampling.NEAREST).save(path.with_name(path.stem.replace('1x','4x')+'.png'))

def tint(rgb,layers):
    a=rgb.astype('float64')
    for mask,color in layers:a[mask]=.35*a[mask]+.65*np.asarray(color)
    return Image.fromarray(np.rint(a).clip(0,255).astype('uint8'))

def alpha_stats(a,mask):
    return dict(pixels=int(mask.sum()),meanAlpha=float(a[mask].mean()) if mask.any() else None,
        minimumAlpha=float(a[mask].min()) if mask.any() else None,
        retainedAt05Fraction=float((a[mask]>=.5).mean()) if mask.any() else None,
        retainedAt095Fraction=float((a[mask]>=.95).mean()) if mask.any() else None)

def main():
    out=HERE/'outputs';raw=json.loads((out/'report.json').read_text())
    source,graph=ROOT/raw['source'],ROOT/raw['graph']
    assert zim.sha(source)==raw['sourceSha256'] and zim.sha(graph)==raw['graphSha256']
    rgb=np.asarray(Image.open(source).convert('RGB'));h,w=rgb.shape[:2]
    params=json.loads(graph.read_text())['operations'][0]['params']
    masks={k:zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
    _,components=probe.prompts_from_components(masks['accessories'],masks['skin'],'accessories',max(p['y']*h for p in params['landmarks']))
    feature_names=['lips','teeth','leftBrow','rightBrow','leftEye','rightEye']
    features=np.maximum.reduce([masks[k] for k in feature_names])
    skin,acc,hair,ears=(masks[k] for k in ['skin','accessories','hair','ears'])
    anchors=(skin>=.9)&(np.maximum.reduce([acc,hair,ears,features])<.05)
    distance,nearest=distance_transform_edt(~anchors,return_indices=True)
    raster=params['semanticMasks']['accessories'];cell=max(w/raster['width'],h/raster['height'])
    face_width=(max(p['x'] for p in params['landmarks'])-min(p['x'] for p in params['landmarks']))*w
    radius=min(3*cell,.025*face_width)
    lab=cv2.cvtColor(rgb.astype('float32')/255,cv2.COLOR_RGB2LAB)
    kernel=2*int(np.ceil(cell))+1
    count=cv2.boxFilter(anchors.astype('float32'),-1,(kernel,kernel),normalize=False)
    means=np.stack([cv2.boxFilter(lab[...,i]*anchors,-1,(kernel,kernel),normalize=False)/np.maximum(count,1) for i in range(3)],axis=-1)
    delta=lab-means[nearest[0],nearest[1]]
    color_support=(np.linalg.norm(delta,axis=-1)<=8)&(np.abs(delta[...,0])<=6)&(np.linalg.norm(delta[...,1:],axis=-1)<=6)
    support=((skin>=.5)|color_support)&(distance<=radius)&(ears<.05)&(features<.05)
    report=dict(scope='Source-only observations, no alpha adjustment, no graph replacement, no warp. No ground truth. Every candidate retained.',
        rawReportSha256=zim.sha(out/'report.json'),scriptSha256=zim.sha(Path(__file__)),
        sourceEvidence=dict(anchor='skin>=.9; hair/accessories/ears/features<.05',
            evidence='skin>=.5 OR nearest local Lab mean DeltaE<=8, absDeltaL<=6, chroma distance<=6',
            maxAnchorDistance=radius,exclude='ears>=.05 or features>=.05',
            inheritedFrom='hair-accessory-matting-zim-v2-20261015/analyze.py; unchanged thresholds'),
        intensityDiagnostic='Highest 10% source sRGB channel mean within parser component; includes uncertainty and is NOT a metal/skin label.',
        holeDiagnostic='Binary fill holes on raw alpha>=.5; connectivity diagnostic only, never filled or relabelled.',
        physicalMatteAccepted=False,components=[],rows=[])
    for component in [2,1]:
        comp=components==component
        prompt=next(r['prompt'] for r in raw['rows'] if r['prompt']['component']==component)
        crop=next(r['crop'] for r in raw['rows'] if r['prompt']['component']==component)
        brightness=rgb.astype('float32').mean(axis=-1)
        threshold=float(np.quantile(brightness[comp],.9))
        bright=comp&(brightness>=threshold)
        parser_holes=binary_fill_holes(comp)&~comp
        folder=out/f'component-{component}-audit';folder.mkdir(exist_ok=True)
        probe.png(folder/'source-supported-skin.png',comp&support)
        probe.png(folder/'source-reference.png',comp)
        official_panels=[('Original',Image.fromarray(rgb)),('Parser',probe.cutout(rgb,acc))]
        component_report=dict(component=component,prompt=prompt,crop=crop,referencePixels=int(comp.sum()),
            sourceSupportedSkinInReference=int((comp&support).sum()),
            parserEnclosedHoles=int(parser_holes.sum()),brightThresholdMeanRGB=threshold,brightDiagnosticPixels=int(bright.sum()))
        report['components'].append(component_report)
        old=np.load(out/f'component-{component}-upstream-mixed/alphas.npy',mmap_mode='r')
        for row in [r for r in raw['rows'] if r['prompt']['component']==component]:
            alphas=np.load(out/row['directory']/'alphas.npy',mmap_mode='r')
            assert zim.sha(out/row['directory']/'alphas.npy')==row['alphaSha256']
            mode=row['mode'];official=row['officialScoreArgmax'];official_panels.append((mode,probe.cutout(rgb,alphas[official])))
            item=dict(component=component,mode=mode,officialScoreArgmax=official,candidates=[])
            for i,a in enumerate(alphas):
                excluded=comp&(a<=.05);eligible=excluded&support;unknown=excluded&~support
                holes=binary_fill_holes(a>=.5)&(a<.5)&comp
                case=dict(index=i,score=row['alphas'][i]['score'],positiveAlpha=row['alphas'][i]['positiveAlpha'],
                    referenceExcludedPixels=int(excluded.sum()),sourceSupportedExcludedPixels=int(eligible.sum()),
                    unknownExcludedPixels=int(unknown.sum()),enclosedHolesInReference=int(holes.sum()),
                    enclosedHolesWithSkinEvidence=int((holes&support).sum()),enclosedHolesWithoutSkinEvidence=int((holes&~support).sum()),
                    sourceSupportedSkin=alpha_stats(a,comp&support),brightSourceTail=alpha_stats(a,bright),
                    componentCore=alpha_stats(a,comp&(acc>=.9)),
                    alphaChangeFromSameIndexUpstream=dict(max=float(np.max(np.abs(a-old[i]))),mean=float(np.abs(a-old[i]).mean())))
                item['candidates'].append(case)
                if i==official:
                    probe.png(folder/f'{mode}-source-supported-excluded.png',eligible)
                    probe.png(folder/f'{mode}-unknown-excluded.png',unknown)
                    probe.png(folder/f'{mode}-holes-unknown.png',holes&~support)
                    panel(folder/f'{mode}-ownership-1x.png',[
                        ('Original',Image.fromarray(rgb)),('Parser',probe.cutout(rgb,acc)),
                        ('Official alpha',Image.fromarray(np.rint(a*255).clip(0,255).astype('uint8'))),
                        ('Official cutout',probe.cutout(rgb,a)),
                        ('Green supported',tint(rgb,[(unknown,(255,40,180)),(eligible,(40,255,80))]))],crop)
            report['rows'].append(item)
        panel(folder/'official-modes-1x.png',official_panels,crop)
    (out/'audit-report.json').write_text(json.dumps(report,indent=2)+'\n')
    for item in report['rows']:
        selected=item['candidates'][item['officialScoreArgmax']]
        print(json.dumps(dict(component=item['component'],mode=item['mode'],official=item['officialScoreArgmax'],
            excluded=selected['referenceExcludedPixels'],supported=selected['sourceSupportedExcludedPixels'],
            unknown=selected['unknownExcludedPixels'],holes=selected['enclosedHolesInReference'],
            brightMean=selected['brightSourceTail']['meanAlpha'],brightMin=selected['brightSourceTail']['minimumAlpha'])))

if __name__=='__main__':main()

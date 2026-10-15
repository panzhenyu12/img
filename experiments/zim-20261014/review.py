#!/usr/bin/env python3
"""Archive original-source ZIM alpha evidence; semantic clip is a separate adaptation."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.run/zim-deps'))
sys.path.insert(0, str(ROOT / '.run/inpaint4drag-deps'))
sys.path.insert(0, str(ROOT / 'scripts'))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from replay_user_face_contour import raster_coverage


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgba_cutout(source, alpha):
    value = np.rint(source.astype(np.float32)*alpha[...,None]+np.array([36,45,56],dtype=np.float32)*(1-alpha[...,None]))
    return Image.fromarray(np.clip(value,0,255).astype(np.uint8))


def panel(images, names, box, output, scale=1):
    w,h=box[2]-box[0],box[3]-box[1]
    result=Image.new('RGB',(w*len(images),h+25),'white')
    draw=ImageDraw.Draw(result)
    short_names={'ZIM raw argmax':'ZIM raw','ZIM semantic-excluded':'excluded',
                 'bisenet-face':'face','bisenet-movable-face':'movable','recorded-physical-alpha':'physical'}
    for i,(image,name) in enumerate(zip(images,names)):
        result.paste(image.crop(box).convert('RGB'),(w*i,25))
        draw.text((w*i+4,7),short_names.get(name,name) if w<320 else name,fill='black')
    if scale!=1:
        result=result.resize((int(result.width*scale),int(result.height*scale)),Image.Resampling.NEAREST)
    result.save(output)


def main():
    source_dir=ROOT/'test/results/zim-20261014'
    archive=ROOT/'doc/qa/zim-20261014'
    archive.mkdir(parents=True,exist_ok=False)
    source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
    source=np.array(Image.open(source_path).convert('RGB'))
    height,width=source.shape[:2]
    report=json.loads((source_dir/'report.json').read_text())
    assert sha(source_path)==report['source_sha256']
    graph_path=ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json'
    assert sha(graph_path)==report['graph_sha256']
    semantics=json.loads(graph_path.read_text())['operations'][0]['params']['semanticMasks']
    coverages={name:raster_coverage(semantics[name],width,height) for name in ['face','skin','faceContourMask','hair','ears','neck','accessories']}
    frozen=np.maximum.reduce([coverages[key] for key in ['hair','ears','neck','accessories']])
    raw=np.load(source_dir/'face-dense-signed-prompts/alpha-all-f32.npy',allow_pickle=False)[2]
    clipped=raw*(1-frozen)
    references={name:np.array(Image.open(source_dir/(name+'.png')).convert('L'),dtype=np.float32)/255
                for name in ['bisenet-face','bisenet-movable-face','recorded-physical-alpha']}
    values={'ZIM raw argmax':raw,'ZIM semantic-excluded':clipped,**references}
    alpha_images={name:Image.fromarray(np.rint(alpha*255).astype(np.uint8)) for name,alpha in values.items()}
    cutouts={name:rgba_cutout(source,alpha) for name,alpha in values.items()}
    source_image=Image.fromarray(source)
    for name,alpha in [('raw-alpha',raw),('semantic-excluded-alpha',clipped),('semantic-frozen-reference',frozen)]:
        np.save(archive/(name+'-f32.npy'),alpha,allow_pickle=False)
        Image.fromarray(np.rint(alpha*255).astype(np.uint8)).save(archive/(name+'.png'))
    names=list(values)
    panel([source_image]+[alpha_images[name] for name in names],['source']+names,(310,170,710,665),archive/'face-alpha-comparison.png')
    panel([source_image]+[cutouts[name] for name in names],['source']+names,(310,170,710,665),archive/'face-cutout-comparison.png')
    for name,box in [('right-ear',(623,294,712,447)),('right-cheek',(609,435,677,550)),('chin',(395,555,635,645)),('hairline',(355,176,654,265))]:
        panel([source_image]+[cutouts[key] for key in names],['source']+names,box,archive/(name+'-comparison-2x.png'),2)
    for case in [r['name'] for r in report['rows']]:
        shutil.copy2(source_dir/case/'four-mask-review.png',archive/(case+'-four-mask-review.png'))
    shutil.copy2(source_dir/'report.json',archive/'experiment-report.json')
    shutil.copy2(source_dir/'prompts.json',archive/'prompts.json')
    row=next(r for r in report['rows'] if r['name']=='face-dense-signed-prompts')
    metrics={'status':'RAW ZIM NOT ACCEPTED as movable-face alpha; semantic-excluded alpha is unverified adaptation',
        'experiment_report_sha256':sha(source_dir/'report.json'),'source_sha256':sha(source_path),
        'reviewer_sha256':sha(Path(__file__)),'raw_case':'face-dense-signed-prompts','raw_index':2,
        'selection':'official score argmax; not selection by visual/ground-truth accuracy',
        'adaptation':'raw alpha * (1-max(original hair, ears, neck, accessories coverage)); separate from official model',
        'measure_scope':'Source parser references are not ground-truth alpha. IoU/MAD are disagreement, not accuracy.',
        'raw_metrics':row['alphas'][2],'adapted_reference_disagreement':{},'adapted_frozen_region':{},'profiles':{}}
    for name,ref in references.items():
        a,b=clipped>=.5,ref>=.5
        metrics['adapted_reference_disagreement'][name]={'threshold_iou':float((a&b).sum()/max(1,(a|b).sum())),
            'mean_absolute_difference':float(np.mean(np.abs(clipped-ref)))}
    for name in ['hair','ears','neck','accessories']:
        region=coverages[name]>=.85
        metrics['adapted_frozen_region'][name]={'mean_alpha':float(clipped[region].mean()),
            'remaining_alpha_05_pixels':int((clipped[region]>=.5).sum()),'reference_pixels':int(region.sum())}
    # Fixed native pixel probes for true outline and ear-root review.
    for y,x_range in [(320,(635,696)),(370,(632,703)),(440,(621,686)),(480,(611,670)),(540,(580,645)),(600,(420,600))]:
        metrics['profiles'][str(y)]={'x':list(range(*x_range)),
            **{name:[round(float(x),6) for x in alpha[y,x_range[0]:x_range[1]]] for name,alpha in values.items()}}
    metrics['archived_files_sha256']={p.name:sha(p) for p in sorted(archive.iterdir()) if p.is_file()}
    (archive/'review.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps({'raw-ear-retained':row['alphas'][2]['regions']['ears'],
                      'adapted':metrics['adapted_reference_disagreement'],
                      'adapted-frozen':metrics['adapted_frozen_region']},indent=2))


if __name__=='__main__':
    main()

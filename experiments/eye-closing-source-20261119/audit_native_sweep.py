"""Check actual native parameter trajectories and keep compact visual evidence.

This verifies sampled geometry and parameter response, not material identity or
aesthetic quality. Run after eye-closure --save-fields --amounts=0,-1,...,-30.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from check_native import sheet


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--prune-generated',action='store_true')
    args=parser.parse_args()
    report=json.loads((args.directory/'report.json').read_text())
    rows=report['rows']
    if [r['amount'] for r in rows] != list(range(0,-31,-1)):
        raise ValueError('the fixed audit needs every consecutive integer from 0 to -30')
    source=np.asarray(Image.open(args.directory/'source.png').convert('RGBA'))
    h,w=source.shape[:2]
    previous=None;previous_step=None;image_previous=None;result=[]
    keep={0,-5,-15,-22,-30};to_delete=[]
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for row in rows:
        paths=[Path(row[key]) for key in ('field','stationary','output')]
        for key,path in zip(('fieldSHA256','stationarySHA256','outputSHA256'),paths):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row[key]:
                raise ValueError(f'native evidence changed: {path}')
        field=np.fromfile(paths[0],'<f4').reshape(h,w,2).astype(float)
        stationary=np.fromfile(paths[1],'u1').reshape(h,w).astype(bool)
        image=np.asarray(Image.open(paths[2]).convert('RGBA'))
        if not np.isfinite(field).all():raise ValueError('nonfinite inverse')
        across0=field[:-1,1:]-field[:-1,:-1];across1=field[1:,1:]-field[1:,:-1]
        down0=field[1:,:-1]-field[:-1,:-1];down1=field[1:,1:]-field[:-1,1:]
        minimum=min(float(cross(a,b).min()) for a,b in [(across0,down0),(across0,down1),(across1,down0),(across1,down1)])
        if minimum<=0:raise ValueError('folded bilinear sampling cell')
        yy,xx=np.indices((h,w));identity=(field[...,0]==xx)&(field[...,1]==yy)
        fixed=stationary|identity
        if np.any(image[fixed]!=source[fixed]):raise ValueError('fixed source pixels changed')
        step=None if previous is None else field-previous
        record={'amount':row['amount'],'minimumBilinearCellJacobian':minimum,
                'sourcePreparationRepeated':row['sourceRefitted'],
                'changedRGBPixels':int(np.any(image[...,:3]!=source[...,:3],axis=-1).sum()),
                'maxAdjacentFieldStepPx':None if step is None else float(np.linalg.norm(step,axis=-1).max()),
                'maxAdjacentStepDifferencePx':None if previous_step is None else float(np.linalg.norm(step-previous_step,axis=-1).max()),
                'pixelsChangedFromPreviousStep':None if image_previous is None else int(np.any(image!=image_previous,axis=-1).sum()),
                'integrationSteps':row['integrationSteps']}
        result.append(record);previous=field;previous_step=step;image_previous=image
        to_delete+=paths[:2]
        if row['amount'] not in keep:to_delete.append(paths[2])
    if result[0]['changedRGBPixels']!=0 or not all(r['changedRGBPixels']>0 for r in result[1:]):
        raise ValueError('zero identity or actual nonzero response failed')
    for g in report['geometries']:
        x0,y0,x1,y1=g['crop']
        panels=[]
        for amount in [0,-5,-15,-22,-30]:
            tag='zero' if amount==0 else f'minus{abs(amount)}'
            image=np.asarray(Image.open(args.directory/f'candidate-{tag}.png').convert('RGB'),float)
            panels.append((str(amount),image[y0:y1,x0:x1]))
        sheet(args.directory/f"{g['eye']}-parameter-sweep.png",panels)
    audit={'sourcePreparationCalls':report['sourcePreparationCalls'],'rows':result,
           'claim':'finite and positive sampled geometry, actual response, fixed pixels unchanged; visual/material review separate',
           'diagnosticCleanup':{'performed':args.prune_generated,'deletedBytes':0,'paths':[]}}
    if args.prune_generated:
        for path in to_delete:
            # Only exact generated artifacts whose bytes matched native hashes.
            if path.parent.resolve()!=args.directory.resolve():raise ValueError('artifact path leaves this native run')
            audit['diagnosticCleanup']['deletedBytes']+=path.stat().st_size
            path.unlink();audit['diagnosticCleanup']['paths'].append(str(path))
    (args.directory/'sweep-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'directory':str(args.directory),'steps':len(result),
        'minimumJacobian':min(r['minimumBilinearCellJacobian'] for r in result),
        'maximumAdjacentFieldStepPx':max(r['maxAdjacentFieldStepPx'] or 0 for r in result),
        'maximumAdjacentStepDifferencePx':max(r['maxAdjacentStepDifferencePx'] or 0 for r in result),
        'prunedBytes':audit['diagnosticCleanup']['deletedBytes']},indent=2))


if __name__=='__main__':main()

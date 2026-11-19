"""One frozen source-only anatomical eye-aperture localization experiment.

No target field, edited output, MEMatte exposure or underlay is read. Side-sclera
anchors need source luminance/chroma evidence; black iris/lash boundaries remain
unknown. Curves are geometric estimates, not alpha or permission to sample B.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'test/results/eye-height-layers-20261117'
OUT = BASE / 'source-curve-localization'
BAND, STEP, SMOOTH_LENGTH = 2., .25, 4.
EYES = {
    'leftEye': dict(upper=[33,246,161,160,159,158,157,173,133], lower=[33,7,163,144,145,153,154,155,133], iris=468, ring=[469,470,471,472]),
    'rightEye': dict(upper=[362,398,384,385,386,387,388,466,263], lower=[362,382,381,380,374,373,390,249,263], iris=473, ring=[474,475,476,477]),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def light(rgb):
    return np.asarray(rgb) @ np.array([.2126, .7152, .0722])


def redness(rgb):
    rgb = np.asarray(rgb)
    return (rgb[..., 0] - rgb[..., 1]) / (light(rgb) + 20.) * 100.


def read_rgb(rgb, xy):
    xy = np.asarray(xy)
    shape = xy.shape[:-1]; flat = xy.reshape(-1, 2)
    sampled = np.stack([map_coordinates(rgb[..., c], [flat[:, 1]-.5, flat[:, 0]-.5], order=1, mode='nearest', prefilter=False) for c in range(3)], axis=-1)
    return sampled.reshape(shape+(3,))


def frame(mesh, spec):
    a, b = mesh[spec['upper'][0]], mesh[spec['upper'][-1]]
    axis = b - a; width = float(np.linalg.norm(axis)); axis /= width
    normal = np.array([-axis[1], axis[0]])
    to_local = lambda p: (np.asarray(p) - a) @ np.stack([axis, normal], axis=1)
    to_image = lambda uv: a + np.asarray(uv)[..., 0, None] * axis + np.asarray(uv)[..., 1, None] * normal
    priors = {}
    for side in ['upper', 'lower']:
        points = to_local(mesh[spec[side]])
        if not np.all(np.diff(points[:, 0]) > 0):
            raise ValueError('ordered source eyelid points are not monotone along canthi')
        points[[0,-1], 0] = [0, width]; points[[0,-1], 1] = 0
        priors[side] = PchipInterpolator(points[:, 0], points[:, 1])
    grid = np.linspace(0, width, int(np.ceil(width)) + 1)
    if np.min(priors['lower'](grid) - priors['upper'](grid)) < -1e-6:
        raise ValueError('source anatomical upper/lower priors cross')
    center = to_local(mesh[spec['iris']]); ring = to_local(mesh[spec['ring']])
    radii = np.max(np.abs(ring-center), axis=0)
    return width, to_local, to_image, priors, grid, center, radii


def fit_curve(rgb, to_image, priors, grid, side, side_evidence, noise):
    prior = priors[side](grid); mid = (priors['upper'](grid)+priors['lower'](grid))/2
    inward = 1 if side == 'upper' else -1
    offsets = np.arange(-BAND, BAND+STEP*.5, STEP)
    records, anchors = [], []
    for i, u in enumerate(grid):
        reason = 'source does not confirm side sclera'
        record = dict(u=float(u), priorV=float(prior[i]), anchor=False, reason=reason)
        if i == 0 or i == len(grid)-1:
            record['reason'] = 'fixed source canthus'; records.append(record); continue
        if not side_evidence[i]:
            records.append(record); continue
        candidates = prior[i]+offsets
        feasible = candidates <= mid[i] if side == 'upper' else candidates >= mid[i]
        inner = read_rgb(rgb, to_image(np.stack([np.full(len(candidates), u), candidates+inward*.75], axis=-1)))
        outer = read_rgb(rgb, to_image(np.stack([np.full(len(candidates), u), candidates-inward*.75], axis=-1)))
        gradients = light(inner-outer)
        gradients[~feasible] = -np.inf
        best = int(np.argmax(gradients)); maximum = float(gradients[best])
        record.update(gradientMaximum=maximum, bestV=float(candidates[best]))
        high = np.where(gradients >= maximum*.8)[0]
        separated = len(high)>1 and np.any(np.diff(high)>1)
        if maximum <= 3*noise:
            record['reason'] = 'contrast does not exceed source noise'
        elif best in [0,len(offsets)-1]:
            record['reason'] = 'edge is not localized within frozen source band'
        elif separated:
            record['reason'] = 'multiple competing source edges'
        else:
            peak_width = max(STEP, (int(high[-1])-int(high[0])+1)*STEP)
            sigma = max(.5, peak_width/2)
            record.update(anchor=True, reason='side-sclera oriented source edge', localizationSigma=sigma)
            anchors.append((i, float(candidates[best]-prior[i]), sigma))
        records.append(record)
    n = len(grid); interior = n-2
    rows = [np.eye(interior)/BAND]; rhs = [np.zeros(interior)]
    du = float(grid[1]-grid[0])
    # Regularize the displacement relative to anatomy, not the anatomy itself.
    second = np.zeros((n-2, n))
    for i in range(n-2): second[i, i:i+3] = [1,-2,1]
    rows.append(second[:,1:-1]*(SMOOTH_LENGTH**2/BAND)/du**2); rhs.append(np.zeros(n-2))
    for i, delta, sigma in anchors:
        row = np.zeros(interior); row[i-1] = 1/sigma
        rows.append(row[None]); rhs.append(np.array([delta/sigma]))
    low = np.full(interior, -BAND); high = np.full(interior, BAND)
    if side == 'upper': high = np.minimum(high, mid[1:-1]-prior[1:-1])
    else: low = np.maximum(low, mid[1:-1]-prior[1:-1])
    solution = lsq_linear(np.vstack(rows), np.concatenate(rhs), bounds=(low, high), tol=1e-10, max_iter=100)
    if not solution.success: raise ValueError(solution.message)
    values = prior.copy(); values[1:-1] += solution.x
    curve = PchipInterpolator(grid, values)
    return curve, records, dict(anchors=len(anchors), evaluatedNodes=n, maximumDisplacement=float(np.max(np.abs(values-prior))), solverOptimality=float(solution.optimality))


def prepare_eye(name, rgb, mesh, eye, out):
    spec = EYES[eye]
    width, to_local, to_image, priors, grid, center, radii = frame(mesh, spec)
    points = mesh[spec['upper']+spec['lower']]
    context = max(12, int(np.ceil(width*.25)))
    box = [max(0,int(np.floor(points[:,0].min()))-context), max(0,int(np.floor(points[:,1].min()))-context),
           min(rgb.shape[1],int(np.ceil(points[:,0].max()))+context), min(rgb.shape[0],int(np.ceil(points[:,1].max()))+context)]
    x0,y0,x1,y1 = box; crop = rgb[y0:y1,x0:x1]
    # Both noise scales come from this frozen source, never from candidate output.
    lum = light(crop); chroma = redness(crop)
    noise = max(1., float(1.4826*np.median(np.abs(lum-gaussian_filter(lum,.6)))))
    chroma_noise = max(.5, float(1.4826*np.median(np.abs(chroma-gaussian_filter(chroma,.6)))))
    mid = (priors['upper'](grid)+priors['lower'](grid))/2
    external = np.median(np.stack([read_rgb(rgb,to_image(np.stack([grid,priors['upper'](grid)-4],axis=-1))),
                                  read_rgb(rgb,to_image(np.stack([grid,priors['lower'](grid)+4],axis=-1)))],axis=0),axis=0)
    candidate = read_rgb(rgb,to_image(np.stack([grid,mid],axis=-1)))
    iris_rgb = read_rgb(rgb,to_image(center))
    anatomical_side = (np.abs(grid-center[0])>radii[0]+1)&(grid>width*.12)&(grid<width*.88)&((priors['lower'](grid)-priors['upper'](grid))>2*BAND)
    # Position alone cannot establish sclera: require actual brighter, less-red
    # source material than adjacent observed exterior skin, and brighter than iris.
    side_evidence = anatomical_side & (light(candidate)>light(external)+3*noise) & (light(candidate)>light(iris_rgb)+3*noise) & (redness(candidate)<redness(external)-3*chroma_noise)
    curves, reports, records = {}, {}, {}
    for side in ['upper','lower']:
        curves[side], records[side], reports[side] = fit_curve(rgb,to_image,priors,grid,side,side_evidence,noise)
    dense = np.linspace(0,width,int(np.ceil(width*16))+1)
    gap = curves['lower'](dense)-curves['upper'](dense)
    if gap.min() < -1e-7: raise ValueError('fitted source curves cross between nodes; reject, never repair by clipping')
    yy,xx = np.mgrid[y0:y1,x0:x1]; uv = to_local(np.stack([xx+.5,yy+.5],axis=-1)); u,v=uv[...,0],uv[...,1]
    inside_u=(u>=0)&(u<=width); safeu=np.clip(u,0,width)
    top,bottom=curves['upper'](safeu),curves['lower'](safeu)
    interior=inside_u&(v>top+BAND)&(v<bottom-BAND)
    ellipse=((u-center[0])/radii[0])**2+((v-center[1])/radii[1])**2
    iris_core=interior&(ellipse<.25)
    ext=np.stack([np.interp(safeu,grid,external[:,c]) for c in range(3)],axis=-1)
    lateral=(np.abs(u-center[0])>radii[0]+1)&(u>width*.12)&(u<width*.88)
    sclera_core=interior&lateral&(light(crop)>light(ext)+3*noise)&(light(crop)>light(iris_rgb)+3*noise)&(redness(crop)<redness(ext)-3*chroma_noise)
    front_core=inside_u&((v<top-BAND)|(v>bottom+BAND))
    unknown=inside_u&~(iris_core|sclera_core|front_core)
    np.savez_compressed(out/'source-states.npz', irisCore=iris_core, scleraCore=sclera_core, frontCore=front_core, unknown=unknown,
                        grid=grid, upper=curves['upper'](grid), lower=curves['lower'](grid), sourceCrop=np.array(box))
    labels = np.zeros(crop.shape[:2],np.uint8);labels[front_core]=1;labels[unknown]=2;labels[iris_core]=3;labels[sclera_core]=4
    Image.fromarray(labels).save(out/'source-states.png')
    Image.fromarray(crop.astype(np.uint8)).save(out/'source.png')
    scale=8; sw,sh=(x1-x0)*scale,(y1-y0)*scale
    source_image=Image.fromarray(crop.astype(np.uint8)).resize((sw,sh),Image.Resampling.NEAREST)
    curve_image=source_image.copy();draw=ImageDraw.Draw(curve_image)
    def image_points(uvals,vvals):
        return [tuple(p) for p in (to_image(np.stack([uvals,vvals],axis=-1))-np.array([x0,y0]))*scale]
    for side in ['upper','lower']:
        draw.line(image_points(dense,priors[side](dense)),fill=(255,110,80),width=1)
        draw.line(image_points(dense,curves[side](dense)),fill=(60,170,255),width=2)
        for record in records[side]:
            if record['anchor']:
                px,py=image_points(np.array([record['u']]),np.array([record['bestV']]))[0]
                draw.ellipse((px-2,py-2,px+2,py+2),fill=(20,255,20))
    theta=np.linspace(0,2*np.pi,90);ringuv=center+np.stack([np.cos(theta)*radii[0],np.sin(theta)*radii[1]],axis=-1)
    draw.line([tuple(p) for p in (to_image(ringuv)-[x0,y0])*scale],fill=(255,215,0),width=1)
    for point in to_image(np.array([[0.,0.],[width,0.]])):
        px,py=(point-[x0,y0])*scale;draw.ellipse((px-3,py-3,px+3,py+3),fill='white')
    overlay=crop.copy()
    for mask,color in [(front_core,[245,150,40]),(unknown,[230,40,210]),(iris_core,[30,220,60]),(sclera_core,[20,220,230])]:
        overlay[mask]=np.rint(crop[mask]*.6+np.array(color)*.4)
    state_image=Image.fromarray(overlay.astype(np.uint8)).resize((sw,sh),Image.Resampling.NEAREST)
    original_path=ROOT/'test/images'/('user-face-contour-20261014.jpg' if name=='user' else 'portrait-dark.jpg')
    original=Image.open(original_path).convert('RGB');ratio=np.array(original.size)/[rgb.shape[1],rgb.shape[0]]
    original_box=tuple(np.rint(np.array(box)*np.tile(ratio,2)).astype(int));original_crop=original.crop(original_box).resize((sw,sh),Image.Resampling.NEAREST)
    sheet=Image.new('RGB',(sw*2,(sh+28)*2),'#222');d=ImageDraw.Draw(sheet)
    for col,row,label,panel in [(0,0,'Frozen source',source_image),(1,0,'Prior red / fitted blue / measured green',curve_image),
                                (0,1,'Iris green / sclera cyan / front orange / unknown magenta',state_image),(1,1,'Existing higher-resolution source (review only)',original_crop)]:
        sheet.paste(panel,(col*sw,row*(sh+28)+28));d.text((col*sw+3,row*(sh+28)+8),label,fill='white')
    sheet.save(out/'source-review.png')
    report=dict(sample=name,eye=eye,sourceCrop=box,canthiWidth=width,irisCenter=center.tolist(),irisRadii=radii.tolist(),
                sourceNoise=noise,sourceRednessNoise=chroma_noise,confirmedSideColumns=int(side_evidence.sum()),
                curves=reports,edgeRecords=records,minimumCurveGap=float(gap.min()),
                states={k:int(m.sum()) for k,m in [('irisCore',iris_core),('scleraCore',sclera_core),('frontCore',front_core),('unknown',unknown)]},
                higherResolutionSource=str(original_path),higherResolutionSHA256=sha(original_path),higherResolutionSize=original.size,
                claim='Source geometry and conservative material observations only; no alpha, ocular underlay or motion permission',
                status='requires independent source review')
    write_json(out/'report.json',report)
    print(name,eye,'side columns',report['confirmedSideColumns'],'anchors',{k:v['anchors'] for k,v in reports.items()},'states',report['states'])


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    cases=json.loads((BASE/'baseline/cases.json').read_text())
    protocol=dict(sourceOnly=True,usesEditedOutput=False,usesMEMatte=False,usesTargetField=False,
        localBandSourcePixels=BAND,candidateSpacingSourcePixels=STEP,displacementSmoothLengthSourcePixels=SMOOTH_LENGTH,
        canthiFixed=True,sourceSampling='Physical coordinates from normalized landmark*image-size; source pixel centers are x+.5,y+.5.',
        scleraRule='Lateral to observed iris ring by >1px; 12%-88% along canthi; prior aperture >4px; source brighter than local exterior and iris, less red than exterior by 3 source-noise scales.',
        irisRule='Inside inner half-radius ellipse and at least2px from fitted lids; observed iris landmark core only.',
        edgeRule='Only confirmed side-sclera columns. Inward luma difference across1.5px; one interior peak above3 source noise; competing peaks or frozen-band extrema remain unknown.',
        unknownRule='Unconfirmed material and black iris/lash boundaries stay unknown. A continuous blue curve through them is an anatomical estimate, not measured exposure.',
        solver='scipy bounded least squares on displacement relative to ordered anatomical PCHIP priors; midpoint bounds prevent crossing, dense validation rejects between-node crossing.',
        parametersFrozenBeforeEvaluation=True)
    write_json(OUT/'protocol.json',protocol)
    for name in ['user','dark']:
        case=next(c for c in cases if c['label']==name+':eyeHeight+0');source=Path(case['input'])
        assert sha(source)==case['sourceSHA256'], 'frozen source changed'
        rgb=np.asarray(Image.open(source).convert('RGB'),float);h,w=rgb.shape[:2]
        raw=BASE/'raw-source-observations'/name/'observations.json'
        obj=json.loads(raw.read_text());points=obj['rows'][0]['sourceLandmarks'][0]
        mesh=np.array([[p['x']*w,p['y']*h] for p in points])
        for eye in EYES:
            out=OUT/name/eye;out.mkdir(parents=True)
            write_json(out/'input.json',dict(source=str(source),sourceSHA256=sha(source),rawObservations=str(raw),rawSHA256=sha(raw),sourceDimensions=[w,h],landmarkIds=EYES[eye]))
            prepare_eye(name,rgb,mesh,eye,out)


if __name__=='__main__':
    main()

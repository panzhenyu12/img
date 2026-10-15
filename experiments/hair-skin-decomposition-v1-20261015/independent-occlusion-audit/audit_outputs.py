"""Read-only independent output/provenance audit, no algorithm mutation."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
from run_real import probe

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
rgba = lambda p: np.asarray(Image.open(p).convert('RGBA'))
mask = lambda p: np.asarray(Image.open(p)) > 0
report_path = HERE / 'frozen/report-before-unknown-fix.json'
report = read(report_path)
estimates_path = EXP / 'source-decomposition-corrected-prompts/report.json'
estimates = read(estimates_path)
assert sha(estimates_path) == report['sourceEstimatesSha256']
assert sha(EXP / 'decompose.py') == report['sourceColorSolverFrozenSha256']
assert sha(HERE / 'frozen/occlusion_hypotheses-before-unknown-fix.py') == report['scriptSha256']
result = {'scope': 'Independent saved-pixel audit; no aesthetic acceptance.',
          'frozenScriptSha256': report['scriptSha256'], 'rows': []}
for row in report['rows']:
    estimate = next(v for v in estimates['rows'] if v['name'] == row['name'] and v['variant'] == 'source-fit-srgb')
    source_path = ROOT / estimate['source']
    graph_path = ROOT / estimate['graph']
    assert sha(source_path) == row['sourceSha256'] == estimate['sourceSha256']
    assert sha(graph_path) == estimate['graphSha256']
    assert sha(ROOT / estimate['selectedAlpha']) == estimate['alphaSha256']
    npz_path = EXP / estimate['directory'] / 'layers.npz'
    assert sha(npz_path) == row['sourceLayerSha256']
    prefix = 'woman' if row['name'] == 'portrait-woman' else row['name']
    material = ROOT / f"experiments/face-material-fill-v1-20261015/{prefix}-{row['amount']}-full"
    layer = material / 'layers'
    assert sha(layer / 'inverse-source-xy-f32le.bin') == row['inverseSha256']
    state = read(layer / 'state.json')
    ox, oy = state['origin']; cw, ch = state['width'], state['height']
    source = rgba(source_path); h, w = source.shape[:2]
    area = np.s_[oy:oy+ch, ox:ox+cw]
    initial_path = material / 'candidate.png'
    if row['backing'].startswith('lama-'):
        size = row['backing'].split('-')[1]
        initial_path = ROOT / f'experiments/face-background-context-20261015/woman-100-local/{size}/candidate.png'
    initial = rgba(initial_path)
    params = read(graph_path)['operations'][0]['params']
    protected = np.maximum(*[probe.zim.coverage(params['semanticMasks'][k], w, h)[area] for k in ['ears','accessories']]) >= .05
    yy, xx = np.indices((ch,cw)); r = params['region']
    roi = ((xx+ox+.5)/w >= r['x']) & ((xx+ox+.5)/w <= r['x']+r['width']) & ((yy+oy+.5)/h >= r['y']) & ((yy+oy+.5)/h <= r['y']+r['height'])
    person = np.fromfile(layer / 'person-coverage-f32le.bin', dtype='<f4').reshape(ch,cw)
    xy = np.fromfile(layer / 'inverse-source-xy-f32le.bin', dtype='<f4').reshape(ch,cw,2)
    moved = np.hypot(xy[...,0]-xx, xy[...,1]-yy) > 1e-4
    core = np.zeros((ch,cw),bool)
    x0,y0,x1,y1 = estimate['crop']
    inferred = np.load(npz_path)
    core[y0-oy:y1-oy,x0-ox:x1-ox] = inferred['hairCore']
    # Independently sample the physical skin alpha using SciPy, not the
    # compositor's bilinear helper. This localizes what A freezes over skin.
    physical = np.fromfile(layer / 'source-alpha-f32le.bin',dtype='<f4').reshape(ch,cw).astype(float)
    local = np.s_[y0-oy:y1-oy,x0-ox:x1-ox]
    local_alpha = physical[local].copy()
    local_alpha[inferred['trusted']] = 1-inferred['alpha'][inferred['trusted']]
    local_alpha[inferred['pureReclaim']] = 1
    physical[local] = local_alpha
    new_skin = map_coordinates(physical,[xy[...,1],xy[...,0]],order=1,mode='grid-constant',cval=0,prefilter=False)
    dest = EXP / row['directory']
    cover = mask(dest / 'newly-covered-connected-hair.png')
    unknown = mask(dest / 'unknown-new-coverage.png')
    assert int(cover.sum()) == row['newlyCoveredConnectedHairPixels']
    assert int(unknown.sum()) == row['unknownCoveragePixels']
    a = rgba(dest / 'A-hair-front.png'); b = rgba(dest / 'B-skin-front.png')
    checked = []
    for label, output in [('A', a), ('B', b)]:
        writes = mask(dest / f'{label}-write.png')
        assert int(writes.sum()) == row[f'{label}WritePixels']
        changed = np.any(output != initial, axis=2)
        full_write = np.zeros((h,w),bool); full_write[area] = writes
        forbidden = protected | ~roi | (person < .5) | ~moved
        visible_core = core if label == 'A' else core & ~cover
        assert not (changed & ~full_write).any()
        assert not (writes & forbidden).any()
        assert np.array_equal(output[...,3], source[...,3])
        assert np.array_equal(output[area][visible_core], source[area][visible_core])
        checked.append({'hypothesis':label, 'changedPixelsVsInitial':int(changed.sum()),
                        'outsideWriteChanged':int((changed & ~full_write).sum()),
                        'forbiddenWritePixels':int((writes & forbidden).sum()),
                        'unknownWritePixels':int((writes & unknown).sum()),
                        'hairCoreChangedVsSource':int((core & np.any(output[area]!=source[area],axis=2)).sum()),
                        'uncoveredHairCoreChangedVsSource':int((visible_core & np.any(output[area]!=source[area],axis=2)).sum()),
                        'outputSha256':sha(dest / ('A-hair-front.png' if label=='A' else 'B-skin-front.png'))})
    core_occluded = core & cover
    core_contrast = np.abs(a[area][core_occluded,:3].astype(float)-b[area][core_occluded,:3].astype(float))
    result['rows'].append({'name':row['name'],'amount':row['amount'],'backing':row['backing'],
        'initialSha256':sha(initial_path),'graphSha256':sha(graph_path),'selectedAlphaSha256':estimate['alphaSha256'],
        'ABDifferentPixels':int(np.any(a!=b,axis=2).sum()),'BHairCoreAllowedOcclusionPixels':int(core_occluded.sum()),
        'opaqueMovedSkinSuppressedByAFrozenHairCore':int((core_occluded & (new_skin>.95)).sum()),
        'ABMeanAbsByteAtOccludedCore':float(core_contrast.mean()) if core_contrast.size else None,
        'inheritedEarAccessoryDifferentPixels':int((protected & np.any(initial[area]!=source[area],axis=2)).sum()),
        'unknownPixels':int(unknown.sum()),'nonSemanticMaskKeysPresent':[k for k in params if 'mask' in k.lower() and k!='semanticMasks'],
        'checks':checked})

# Audit historical mixed outputs and the exact downstream corrected input.
fixed_root = ROOT / 'experiments/zim-combined-prompts-20261015/outputs'
fixed = read(fixed_root / 'report.json')
assert sha(fixed_root / 'report.json') == estimates['correctedPromptReportSha256']
historical_root = ROOT / 'experiments/hair-accessory-matting-zim-v2-20261015/outputs'
result['historicalMixedControls'] = []
for image in fixed['images']:
    for row in image['rows']:
        if row['mode'] != 'upstream-mixed': continue
        old = historical_root / image['name'] / ('hair-1-' + row['prompt']['protocol']) / 'alphas.npy'
        new = fixed_root / row['directory'] / 'alphas.npy'
        assert sha(new) == row['alphaSha256']
        old_array = np.load(old); new_array = np.load(new)
        assert np.array_equal(old_array,new_array)
        result['historicalMixedControls'].append({'name':image['name'],'protocol':row['prompt']['protocol'],
            'floatExact':True, 'oldSha256':sha(old), 'newSha256':sha(new)})
result['limitations'] = [
    'Unknown coverage is empty in all real cases; see independent synthetic counterexample.',
    'No generic protectMask, selection or separate-person identity contract is exercised.',
    'Stationary texture coordinates do not require constant hair visibility.',
    'A/B are counterfactual layer-order interpretations, not automatic layer-order predictions.',
]
(HERE / 'report.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'rows':len(result['rows']),'historicalFloatExactControls':len(result['historicalMixedControls']),
                  'allSavedOutputDomainChecksPassed':True},indent=2))

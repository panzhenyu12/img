"""Behavior checks of the explicit, unknown-order occlusion experiment.

Tests do not choose a depth order or compare to self-generated golden pixels.
They assert geometric coverage eligibility and immutable source provenance.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from occlusion_hypotheses import connected_coverage, write_domains
from run_real import HERE, ROOT, probe, sha


def coverage_cases():
    shape = (32, 40)
    old = np.zeros(shape)
    new = np.zeros(shape)
    hair = np.ones(shape)
    edge = np.zeros(shape, bool)
    edge[8:24, 10] = True
    allowed = np.ones(shape, bool)
    new[8:24, 9:13] = 1
    new[10:14, 26:30] = 1
    connected, unknown, _, _ = connected_coverage(new, old, hair, edge, allowed)
    assert connected[8:24, 9:13].all()
    assert not connected[10:14, 26:30].any()
    assert unknown[10:14, 26:30].all()
    assert not (connected & unknown).any()
    # Excluding a protected separator must not allow coverage to flood through.
    new[:] = 0
    new[12:16, 9:20] = 1
    allowed[:, 13] = False
    connected, unknown, _, _ = connected_coverage(new, old, hair, edge, allowed)
    assert connected[12:16, 9:13].all()
    assert not connected[:, 13:].any()
    assert unknown[12:16, 14:20].all()
    assert not (connected & ~allowed).any()
    # Already occupied/receding skin and nonhair targets cannot become new hair coverage.
    old[:] = 1
    connected, unknown, _, _ = connected_coverage(new, old, hair, edge, allowed)
    assert not connected.any() and not unknown.any()
    old[:] = 0
    hair[:] = 0
    connected, unknown, _, _ = connected_coverage(new, old, hair, edge, allowed)
    assert not connected.any() and not unknown.any()
    # Counterexample: a transported mixed edge lands on a disconnected unknown
    # new-coverage island. Neither edge route may bypass the unknown refusal.
    transported = np.zeros(shape,bool); transported[10:14,26:30] = True
    unknown = transported.copy()
    zero = np.zeros(shape,bool)
    write_a,write_b = write_domains(transported,zero,zero,zero,unknown,zero)
    assert not write_a.any() and not write_b.any()
    return {'connectedBoundaryAccepted': True, 'disconnectedIslandUnknown': True,
            'protectedSeparatorBlocksFlood': True, 'oldCoverageAndNonhairRejected': True,
            'transportedEdgeCannotBypassUnknown':True}


def validate_real():
    report_path = HERE / 'occlusion-corrected-prompts/report.json'
    report = json.loads(report_path.read_text())
    estimates_path = HERE / 'source-decomposition-corrected-prompts/report.json'
    estimates = json.loads(estimates_path.read_text())
    assert sha(HERE / 'occlusion_hypotheses.py') == report['scriptSha256']
    assert sha(estimates_path) == report['sourceEstimatesSha256']
    assert sha(HERE / 'decompose.py') == report['sourceColorSolverFrozenSha256']
    result = []
    for row in report['rows']:
        estimate = next(r for r in estimates['rows'] if r['name'] == row['name'] and r['variant'] == 'source-fit-srgb')
        source_path = ROOT / estimate['source']
        alpha_path = ROOT / estimate['selectedAlpha']
        assert sha(source_path) == row['sourceSha256'] == estimate['sourceSha256']
        assert sha(alpha_path) == estimate['alphaSha256']
        assert sha(ROOT / estimate['graph']) == estimate['graphSha256']
        assert sha(HERE / estimate['directory'] / 'layers.npz') == row['sourceLayerSha256']
        prefix = 'woman' if row['name'] == 'portrait-woman' else row['name']
        material = ROOT / f"experiments/face-material-fill-v1-20261015/{prefix}-{row['amount']}-full"
        layers = material / 'layers'
        assert sha(layers / 'inverse-source-xy-f32le.bin') == row['inverseSha256']
        state = json.loads((layers / 'state.json').read_text())
        ox, oy = state['origin']; cw, ch = state['width'], state['height']
        source = np.asarray(Image.open(source_path).convert('RGBA'))
        h, w = source.shape[:2]
        initial_path = material / 'candidate.png'
        if row['backing'].startswith('lama-'):
            size = row['backing'].split('-')[1]
            initial_path = ROOT / f'experiments/face-background-context-20261015/woman-100-local/{size}/candidate.png'
        initial = np.asarray(Image.open(initial_path).convert('RGBA'))
        dest = HERE / row['directory']
        params = json.loads((ROOT / estimate['graph']).read_text())['operations'][0]['params']
        protected = np.maximum(*[probe.zim.coverage(params['semanticMasks'][k],w,h)[oy:oy+ch,ox:ox+cw] for k in ['ears','accessories']]) >= .05
        data = np.load(HERE / estimate['directory'] / 'layers.npz')
        x0,y0,x1,y1 = estimate['crop']
        core = np.zeros((ch,cw),bool); core[y0-oy:y1-oy,x0-ox:x1-ox] = data['hairCore']
        coverage = np.asarray(Image.open(dest/'newly-covered-connected-hair.png')) > 0
        for label, suffix in [('A', 'hair-front'), ('B', 'skin-front')]:
            rendered = np.asarray(Image.open(dest / f'{label}-{suffix}.png').convert('RGBA'))
            writes = np.asarray(Image.open(dest / f'{label}-write.png')) > 0
            changed = np.any(rendered != initial,axis=2)
            full = np.zeros((h,w),bool);full[oy:oy+ch,ox:ox+cw] = writes
            assert not (changed & ~full).any()
            assert not (writes & protected).any()
            assert np.array_equal(rendered[...,3],source[...,3])
            remaining = core if label == 'A' else core & ~coverage
            assert np.array_equal(rendered[oy:oy+ch,ox:ox+cw][remaining], source[oy:oy+ch,ox:ox+cw][remaining])
            if row['name'] == 'user-face-contour':
                assert not changed.any(), 'Hair-only experiment must not claim accessory repair'
            result.append({'name':row['name'],'amount':row['amount'],'backing':row['backing'],'hypothesis':label,
                           'changedPixels':int(changed.sum()),'outsidePermittedWritesChanged':0,
                           'protectedNewlyChanged':0,'uncoveredHairCoreChanged':0,
                           'imageSha256':sha(dest / f'{label}-{suffix}.png'),
                           'initialSha256':sha(initial_path)})
        assert row['layerOrder'].startswith('UNKNOWN')
    return result


if __name__ == '__main__':
    result = {'behaviorChecks':coverage_cases(),'realChecks':validate_real(),
              'note':'Coverage and byte invariants only. No automatic depth inference or aesthetics acceptance.'}
    (HERE/'occlusion-corrected-prompts/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'coverageChecks':len(result['behaviorChecks']),'validatedImages':len(result['realChecks']),'status':'passed'}))

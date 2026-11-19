"""Apply a source-only parser revision locally, keeping optical inputs separate.

Unknown evidence retains the original runtime policy. This is deliberately a
mixed-revision experiment outside the declared valid domain, not a new matte
or a claim that semantic confidence measures physical opacity.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter

ROLES = ('hair', 'skin', 'neck', 'ears', 'accessories')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_roles(graph, evidence):
    if evidence.get('physicalAlpha') is not False or evidence.get('manualEvidence') is not False:
        raise ValueError('Automatic semantic evidence required; optical alpha is independent')
    if evidence.get('appliedToSource') is not False or set(evidence['masks']) != set(ROLES):
        raise ValueError('Exactly one unapplied shared role revision required')
    validity = evidence['sourceValidityMask']
    width, height = validity['width'], validity['height']
    valid_bytes = np.asarray(validity['data'])
    if (validity.get('shape') != 'raster' or width < 1 or height < 1
            or valid_bytes.shape != (width * height,) or not np.isin(valid_bytes, [0, 255]).all()):
        raise ValueError('Binary source validity raster required')
    valid = valid_bytes == 255
    # Every bilinear cell touched by an updated node must have four observed
    # corners. Holding only null nodes would still change samples in a cell
    # straddling known/unknown evidence. Nearest padding matches clamp sampling.
    selected = minimum_filter(valid.reshape(height, width), size=3, mode='nearest').ravel()
    result = copy.deepcopy(graph)
    masks = result['operations'][0]['params']['semanticMasks']
    rows = {}
    for role in ROLES:
        proposed = evidence['masks'][role]
        values = proposed['data']
        if (proposed.get('shape') != 'nullable-raster-evidence'
                or (proposed['width'], proposed['height']) != (width, height)
                or len(values) != width * height
                or not np.array_equal([v is not None for v in values], valid)):
            raise ValueError(f'{role}: evidence revisions/domains disagree')
        if any(type(v) is not int or not 0 <= v <= 255 for v in values if v is not None):
            raise ValueError(f'{role}: invalid semantic byte')
        existing = masks.get(role)
        if existing is not None:
            if (existing.get('shape') != 'raster'
                    or (existing['width'], existing['height']) != (width, height)
                    or len(existing['data']) != width * height):
                raise ValueError(f'{role}: original grid differs; do not silently resample')
            before = np.asarray(existing['data'])
        else:
            # Missing roles already mean no class support in the legacy
            # consumer. Retain that behavior, explicitly NOT verified absence.
            before = np.zeros(width * height, dtype=np.uint8)
            existing = {'shape': 'raster', 'width': width, 'height': height}
        after = before.copy()
        after[selected] = [values[i] for i in np.flatnonzero(selected)]
        assert np.array_equal(after[~selected], before[~selected])
        masks[role] = {**existing, 'data': after.tolist()}
        rows[role] = {'changedNodes': int(np.count_nonzero(before != after)),
                      'unknownNodesRetained': int(np.count_nonzero(~valid)),
                      'validBoundaryNodesRetained': int(np.count_nonzero(valid & ~selected)),
                      'originalRolePresent': role in graph['operations'][0]['params']['semanticMasks'],
                      'outsideValidDomain': 'original values' if role in graph['operations'][0]['params']['semanticMasks'] else 'legacy missing-role zero support; not confirmed absence'}
    unchanged = copy.deepcopy(result)
    unchanged['operations'][0]['params']['semanticMasks'] = copy.deepcopy(graph['operations'][0]['params']['semanticMasks'])
    assert unchanged == graph
    return result, {'roles': rows, 'appliedToSource': True, 'manualEvidence': False,
                    'selectionRule': 'all adjacent bilinear cells have observed corners; one shared role domain',
                    'selectedNodes': int(np.count_nonzero(selected)),
                    'wholeImageSingleRevision': False, 'opticalInputsDirectlyChanged': False,
                    'opticalLayersMustBeReestimated': True,
                    'sourcePreparation': 'must rebuild trace, blocked/owned/support and source layer cache',
                    'driverPolicy': 'original contour preserved; revised driver supplied separately to geometry'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['graph', 'source', 'sidecar', 'out']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text())
    evidence = json.loads(args.sidecar.read_text())
    provenance = evidence['provenance']
    if provenance.get('manualEvidence') is not False or provenance.get('sourceGraphOrigin') != 'automatic-detection':
        raise ValueError('Automatic source graph lineage required')
    if provenance['sourceGraphSHA256'] != sha(args.graph) or provenance['sourceSHA256'] != sha(args.source):
        raise ValueError('Source/graph hash mismatch')
    width, height = Image.open(args.source).size
    original = copy.deepcopy(graph)
    dimensions = {'width': width, 'height': height}
    if 'source' in graph and graph['source'] != dimensions:
        raise ValueError('Source dimensions mismatch')
    graph['source'] = dimensions
    candidate, report = apply_roles(graph, evidence)
    report.update(sourceSHA256=sha(args.source), originalGraphSHA256=sha(args.graph),
                  sidecarSHA256=sha(args.sidecar), sourceDimensionsAdded='source' not in original,
                  provenance=provenance, implementationSHA256=sha(__file__))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'baseline-graph.json').write_text(json.dumps(graph, separators=(',', ':')))
    (args.out/'candidate-graph.json').write_text(json.dumps(candidate, separators=(',', ':')))
    (args.out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'provenance'}))


if __name__ == '__main__':
    main()

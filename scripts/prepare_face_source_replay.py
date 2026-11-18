"""Prepare hash-bound source layers for native face experiments, without rendering.

Requires NumPy and Pillow. --output-dir must not exist. The public graph is never
edited. If its encoded hash differs from the cache's graph, --cache-graph must
identify the exact cache-bound graph; only omission of private keys is allowed
from the public graph. Its two supported source permissions are exported unchanged
for candidate replay, unless --omit-source-permissions requests an ablation.
Unsupported private keys are rejected. No target image or generated motion field
is read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


SOURCE_PERMISSION_FIELDS = {
    '/operations/0/params/__sourceMotionInterface': 'sourceMotionInterface',
    '/operations/0/params/__sourceMovingAttachmentSupport': 'sourceMovingAttachmentSupport',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                   separators=(',', ':')) + '\n')


def public_view(value, path=''):
    """Compare graph contracts; never save a silently sanitized input graph."""
    if isinstance(value, dict):
        public, private = {}, []
        for key, child in value.items():
            location = path + '/' + key
            if key.startswith('__') or key.lower().startswith('experimental'):
                private.append(location)
            else:
                public[key], nested = public_view(child, location)
                private.extend(nested)
        return public, private
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        public, private = [], []
        for i, child in enumerate(value):
            item, nested = public_view(child, path + '/' + str(i))
            public.append(item)
            private.extend(nested)
        return public, private
    return value, []


def checked_box(box, width, height, label):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(v) is not int for v in box)):
        raise ValueError(f'{label} must contain four integer coordinates')
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f'{label} ROI is outside the source canvas or empty')
    return left, top, right, bottom


def replay_box(cache, width, height):
    view = cache['replayView']
    values = [*view['origin'], view['width'], view['height']]
    if len(values) != 4 or any(type(v) is not int for v in values):
        raise ValueError('replayView must contain integer origin and dimensions')
    x, y, w, h = values
    return checked_box((x, y, x + w, y + h), width, height, 'replayView')


def read_array(path, dtype, shape):
    dtype = np.dtype(dtype)
    if Path(path).stat().st_size != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError(f'ROI dimensions disagree with byte count: {path}')
    return np.fromfile(path, dtype).reshape(shape)


def validate_layers(source, alpha, background):
    if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError('source alpha must be finite in [0,1]')
    if np.any(source[..., 3] != 255) or np.any(background[..., 3] != 255):
        raise ValueError('source and paired background must be opaque RGBA')
    if not np.array_equal(source[alpha == 0], background[alpha == 0]):
        raise ValueError('zero-alpha paired background differs from observed source')
    color, scene = source[..., :3].astype(np.float64), background[..., :3].astype(np.float64)
    premultiplied = color - scene + alpha[..., None].astype(np.float64) * scene
    if np.any(premultiplied < -1e-4) or np.any(premultiplied > 255 * alpha[..., None] + 1e-4):
        raise ValueError('source alpha/background pair is outside bounded premultiplied RGB')


def alpha_raster(alpha, box):
    height, width = alpha.shape
    left, top, right, bottom = box
    return {
        'shape': 'raster', 'width': width, 'height': height,
        'data': (alpha * np.float32(255)).ravel().tolist(),
        'sourceRegion': {'x': left / width, 'y': top / height,
                         'width': (right - left) / width, 'height': (bottom - top) / height},
    }


def prepare(source, graph, cache_path, output_dir, case_id, cache_graph=None,
            omit_source_permissions=False):
    source, graph, cache_path, output_dir = map(Path, (source, graph, cache_path, output_dir))
    source, graph, cache_path = (p.resolve() for p in (source, graph, cache_path))
    # Do not resolve the output before rejecting it: even a dangling symlink
    # is pre-existing evidence and must not be followed or replaced.
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f'output directory already exists: {output_dir}')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', case_id):
        raise ValueError('id must be a filename-safe alphanumeric identifier')
    cache = read_json(cache_path)
    if sha(source) != cache['sourceSHA256']:
        raise ValueError('source PNG SHA256 does not match cache')
    with Image.open(source) as image:
        if image.format != 'PNG':
            raise ValueError('source must be a PNG with frozen orientation/size')
        if image.getexif().get(274, 1) != 1:
            raise ValueError('source orientation must be frozen before preparing layers')
        observed = np.asarray(image.convert('RGBA'))
    height, width = observed.shape[:2]
    public_graph = read_json(graph)
    _, private = public_view(public_graph)
    if private:
        raise ValueError('input graph must already be public; private keys: ' + ', '.join(private))
    if public_graph.get('source') != {'width': width, 'height': height}:
        raise ValueError('public graph source dimensions do not match source PNG')
    operations = public_graph.get('operations', [])
    if len(operations) != 1 or operations[0].get('type') != 'face.adjust':
        raise ValueError('replay input requires one face.adjust operation')
    binding = {'publicGraphSHA256': sha(graph), 'cacheGraphSHA256': cache['graphSHA256'],
               'privateFieldsAbsentFromPublicGraph': [], 'publicGraphModified': False}
    permissions = {}
    if cache_graph is None and sha(graph) != cache['graphSHA256']:
        raise ValueError('graph SHA256 differs from cache; supply --cache-graph for explicit binding')
    if cache_graph is not None:
        cache_graph = Path(cache_graph).resolve()
        if sha(cache_graph) != cache['graphSHA256']:
            raise ValueError('--cache-graph SHA256 does not match cache')
        cached_graph = read_json(cache_graph)
        cache_public, absent = public_view(cached_graph)
        unsupported = [path for path in absent if path not in SOURCE_PERMISSION_FIELDS]
        if unsupported:
            raise ValueError('unsupported private fields in cache graph: ' + ', '.join(unsupported))
        if cache_public != public_graph:
            raise ValueError('public graph differs from cache in nonprivate fields')
        for path in absent:
            permissions[path] = cached_graph['operations'][0]['params'][path.rsplit('/', 1)[1]]
        # Validate exact JSON export before creating output files. Do not cast,
        # rescale, or reshape these cached source permission values.
        json.dumps(permissions, allow_nan=False)
        binding.update(cacheGraph=str(cache_graph), privateFieldsAbsentFromPublicGraph=absent,
                       difference='Only supported source permissions and JSON serialization may differ')
    else:
        binding['difference'] = 'Exact encoded graph hash'
    permission_provenance = {
        'applied': [] if omit_source_permissions else list(permissions),
        'omitted': list(permissions) if omit_source_permissions else [],
        'omissionRequested': bool(omit_source_permissions),
    }

    replay = replay_box(cache, width, height)
    rx, ry, rr, rb = replay
    rw, rh = rr - rx, rb - ry
    expected_files = {
        'sourceInputSHA256': 'input/source.rgba',
        'sourceFrozenSHA256': 'input/frozen.u8',
        'finalAlphaSHA256': 'layers/final-alpha.f32',
        'sourceBackgroundSHA256': 'layers/source-background.rgba',
    }
    for key, relative in expected_files.items():
        if sha(cache_path.parent / relative) != cache[key]:
            raise ValueError(f'cache file SHA256 mismatch: {relative}')
    replay_source = read_array(cache_path.parent / 'input/source.rgba', 'u1', (rh, rw, 4))
    replay_alpha = read_array(cache_path.parent / 'layers/final-alpha.f32', '<f4', (rh, rw))
    replay_background = read_array(cache_path.parent / 'layers/source-background.rgba', 'u1', (rh, rw, 4))
    read_array(cache_path.parent / 'input/frozen.u8', 'u1', (rh, rw))
    replay_region = np.s_[ry:rb, rx:rr]
    if not np.array_equal(replay_source, observed[replay_region]):
        raise ValueError('cache source RGBA differs from source PNG crop')
    validate_layers(replay_source, replay_alpha, replay_background)

    complete = cache_path.parent.parent / 'complete-optical-domain'
    if complete.exists():
        box = checked_box(cache['sourceObservationBox'], width, height, 'sourceObservationBox')
        layers = complete
        domain_kind = 'complete-optical-domain'
    else:
        # A plain source cache may only define its replay view. Do not pretend
        # it contains a complete head or extend alpha to unobserved pixels.
        box, layers, domain_kind = replay, cache_path.parent, 'cache-replay-view-only'
    left, top, right, bottom = box
    region = np.s_[top:bottom, left:right]
    dw, dh = right - left, bottom - top
    domain_source = read_array(layers / 'input/source.rgba', 'u1', (dh, dw, 4))
    domain_alpha = read_array(layers / 'layers/final-alpha.f32', '<f4', (dh, dw))
    domain_background = read_array(layers / 'layers/source-background.rgba', 'u1', (dh, dw, 4))
    if not np.array_equal(domain_source, observed[region]):
        raise ValueError('complete source RGBA differs from source PNG crop')
    validate_layers(domain_source, domain_alpha, domain_background)
    alpha = np.zeros((height, width), '<f4')
    background = observed.copy()
    weights = np.zeros((height, width), '<f4')
    alpha[region], background[region], weights[region] = domain_alpha, domain_background, 1
    if (not np.array_equal(alpha[replay_region], replay_alpha)
            or not np.array_equal(background[replay_region], replay_background)):
        raise ValueError('complete optical domain does not reproduce the cache replay view')

    # Finish validation before creating any output; never reuse an old folder.
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=False)
    alpha_path = output_dir / f'{case_id}-physical-alpha.json'
    background_path = output_dir / f'{case_id}-source-background.png'
    weights_path = output_dir / f'{case_id}-scene-weights.f32'
    descriptor_path = output_dir / f'{case_id}-scene-descriptor.json'
    write_json(alpha_path, alpha_raster(alpha, box))
    # Read the saved file and emulate PreparedMask's documented numeric decode.
    encoded = read_json(alpha_path)
    decoded = np.asarray(encoded['data'], np.float32).reshape(height, width) / np.float32(255)
    domain_error = float(np.max(np.abs(decoded[region] - domain_alpha)))
    replay_error = float(np.max(np.abs(decoded[replay_region] - replay_alpha)))
    if max(domain_error, replay_error) > 1e-7:
        raise ValueError('serialized alpha fails renderer decode/cache round-trip')
    if not (np.array_equal(decoded == 0, alpha == 0) and np.array_equal(decoded == 1, alpha == 1)):
        raise ValueError('serialized alpha changed exact zero/opaque source support')
    Image.fromarray(background).save(background_path)
    weights.tofile(weights_path)
    write_json(descriptor_path, {'source': str(source), 'sourceSha256': sha(source),
                               'image': str(background_path), 'weights': str(weights_path),
                               'coveragePolicy': 'scene-lower-bound'})
    case = {'id': case_id, 'source': str(source), 'graph': str(graph),
            'physicalAlpha': str(alpha_path), 'sourcePlatePrior': str(descriptor_path)}
    permission_paths = []
    if not omit_source_permissions:
        for path, value in permissions.items():
            field = SOURCE_PERMISSION_FIELDS[path]
            exported = output_dir / f'{case_id}-{field}.json'
            write_json(exported, value)
            case[field] = str(exported)
            permission_paths.append(exported)
    write_json(output_dir / 'replay-case.json', case)
    provenance = {
        'source': str(source), 'sourceSHA256': sha(source), 'sourceSize': [width, height],
        'graph': str(graph), 'graphBinding': binding,
        'sourcePermissions': permission_provenance,
        'cache': str(cache_path), 'cacheSHA256': sha(cache_path),
        'sourceLayerOrigin': str(layers), 'sourceDomainKind': domain_kind,
        'sourceObservationBox': list(box), 'cacheReplayBox': list(replay),
        'sourceRGBAEqualsCache': True, 'completeLayersReproduceReplayView': True,
        'sourceLayerSHA256': {name: sha(layers / name) for name in
                              ('input/source.rgba', 'layers/final-alpha.f32', 'layers/source-background.rgba')},
        'sourceAlphaDerivation': cache.get('sourceAlphaDerivation'),
        'sourceBackgroundDerivation': cache.get('sourceBackgroundDerivation'),
        'physicalAlphaRendererDecodeAudit': {
            'encoding': 'float32 alpha*255 without integer rounding',
            'decoder': 'JSON -> float32 / float32(255), PreparedMask raster contract',
            'sourceDomainMaxAbsError': domain_error, 'cacheReplayViewMaxAbsError': replay_error,
            'absoluteTolerance': 1e-7, 'exactZeroAndOnePreserved': True,
        },
        'weightsRule': 'one inside the observed source domain; zero outside',
        'sourceOnly': True, 'renderOrModelRun': False, 'manualAnnotationsAdded': False,
        'inheritedManualEvidence': 'Not established by this adapter; inspect upstream cache provenance',
        'limitations': ['Source alpha and hidden background are estimates, not ground truth.',
                        'Supported cached source permissions are forwarded unchanged by default; explicit omission is an ablation.',
                        'Prepared input correctness does not establish rendered image quality.'],
        'outputSHA256': {p.name: sha(p) for p in
                         (alpha_path, background_path, weights_path, descriptor_path,
                          output_dir / 'replay-case.json', *permission_paths)},
        'preparationScriptSHA256': sha(__file__),
    }
    write_json(output_dir / 'provenance.json', provenance)
    return case, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'graph', 'cache', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--id', required=True, dest='case_id')
    parser.add_argument('--cache-graph', type=Path,
                        help='Exact cache-bound graph, required if the supplied public graph has a different hash')
    parser.add_argument('--omit-source-permissions', action='store_true',
                        help='Ablation only: omit supported cached source permissions from candidate replay')
    args = parser.parse_args()
    try:
        case, _ = prepare(args.source, args.graph, args.cache, args.output_dir, args.case_id,
                          args.cache_graph, args.omit_source_permissions)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Invalid source replay input: {error}\n')
    print(json.dumps(case, ensure_ascii=False))


if __name__ == '__main__':
    main()

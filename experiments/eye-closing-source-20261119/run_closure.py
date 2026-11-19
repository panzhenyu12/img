"""Replay experimental eye closing on a new original and fresh observation.

This is an algorithm experiment, not a public feature.adjust implementation.
An explicitly selected source estimator is called once per eye, before fields
or edited images are read. No historical source state, graph or field is used.
The supplied existing native executable is invoked without rebuilding it.
"""
import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from source_geometry import (ROOT, EYES, NATIVE_COORDINATE_ADAPTER, load_observation,
                             make_graph, prepare_geometry, sha, write_json)
from source_estimator import prepare_sampling_support
from source_material import apply_materials, crop_materials, load_materials, validate_material_transport
from layers import compose, sample

STRENGTHS = (0, -15, -30)
FIELD_TEST = 'geometry::tests::photographic_brow_flow_diagnostics'


def png(path, array):
    array = np.asarray(array)
    if not np.isfinite(array).all() or np.any((array < -1e-6) | (array > 255+1e-6)):
        raise ValueError('cannot encode out-of-range image; no RGB clipping permitted')
    Image.fromarray(np.rint(array).astype(np.uint8)).save(path)


def sheet(path, panels, scale=4):
    h, w = panels[0][1].shape[:2]
    out = Image.new('RGB', (w*scale*len(panels), h*scale+26), '#222')
    draw = ImageDraw.Draw(out)
    for index, (label, array) in enumerate(panels):
        if array.shape[:2] != (h, w):
            raise ValueError('comparison panels must use the identical source crop')
        panel = Image.fromarray(np.rint(array).astype(np.uint8)).convert('RGB')
        out.paste(panel.resize((w*scale, h*scale), Image.Resampling.NEAREST), (index*w*scale, 26))
        draw.text((index*w*scale+3, 7), label, fill='white')
    out.save(path)


def load_estimator(spec):
    parts = spec.split(':')
    if len(parts) != 2:
        raise ValueError('--estimator must be an explicit module:function')
    module = importlib.import_module(parts[0])
    function = getattr(module, parts[1])
    if not callable(function):
        raise ValueError('selected estimator is not callable')
    return function, Path(inspect.getsourcefile(function)).resolve()


def local_function_dependencies(function):
    """Record imported experimental Python functions as well as the entry file."""
    pending, paths = [function], set()
    while pending:
        current = pending.pop()
        path = Path(inspect.getsourcefile(current)).resolve()
        if path in paths or not path.is_relative_to(ROOT/'experiments'):
            continue
        paths.add(path)
        for value in current.__globals__.values():
            if inspect.isfunction(value) and inspect.getsourcefile(value):
                imported_path = Path(inspect.getsourcefile(value)).resolve()
                if imported_path.is_relative_to(ROOT/'experiments'):
                    pending.append(value)
    return sorted(paths)


def guards(diagnostics, amount):
    if amount == 0:
        if diagnostics.get('identityRGBAExact') is not True:
            raise ValueError('native zero did not establish RGBA identity in its decoder domain')
        return
    for key in ('fixedDonorImports', 'foldPixels'):
        if diagnostics.get(key) != 0:
            raise ValueError(f'native field fails {key}: {diagnostics.get(key)}')
    for key in ('maxInverseFlowReverseErrorPx', 'maxEyeEnvelopeDeviationFromOtherEditsPx'):
        value = diagnostics.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value >= .05:
            raise ValueError(f'native field fails {key}: {value}')


def tag(amount):
    return 'zero' if amount == 0 else f'minus{abs(amount)}'


def read_stationary(path, height, width):
    # The frozen native code deliberately uses Vec::new() when no stationary
    # hair/ear/accessory masks exist. Its lookup treats absent entries as false.
    byte_count = path.stat().st_size
    if byte_count == 0:
        return np.zeros((height, width), bool), 'empty native vector; no stationary semantic masks'
    if byte_count != height*width:
        raise ValueError('native stationary mask has an incorrect byte count')
    return (np.fromfile(path, 'u1').reshape(height, width) != 0,
            'full native stationary raster')


def run(args, report):
    source_path, observations = args.source.resolve(), args.observations.resolve()
    output, field_runner = args.output.resolve(), args.field_runner.resolve()
    report.update(source=str(source_path), sourceSHA256=sha(source_path),
                  observations=str(observations), observationsSHA256=sha(observations),
                  fieldRunner=str(field_runner), fieldRunnerSHA256=sha(field_runner),
                  sourcePreparedOnce=True, sourcePreparationCalls=0, rows=[])
    estimator, estimator_path = load_estimator(args.estimator)
    dependencies = [Path(__file__).resolve(), Path(__file__).with_name('source_geometry.py'),
                    Path(__file__).with_name('source_estimator.py'), estimator_path,
                    ROOT/'experiments/eye-height-layers-20261117/source_curves.py',
                    ROOT/'experiments/eye-height-layers-20261117/layers.py']
    dependencies.extend(local_function_dependencies(estimator))
    if args.materials:
        dependencies.extend(local_function_dependencies(apply_materials))
    code_hashes = {str(path): sha(path) for path in dependencies}
    report.update(estimator=args.estimator, codeSHA256=code_hashes,
                  publicFeatureEnabled=False, opticalTruth=False)
    image = Image.open(source_path)
    if 'A' in image.getbands() or 'transparency' in image.info:
        if image.convert('RGBA').getchannel('A').getextrema() != (255, 255):
            raise ValueError('this experiment supports opaque RGB only; transparent source is not silently flattened')
    orientation = image.getexif().get(274, 1)
    if orientation != 1:
        raise ValueError('source has an EXIF transform; normalize source and regenerate its observation explicitly')
    source = np.asarray(image.convert('RGB'), dtype=float)
    height, width = source.shape[:2]
    mesh, normalized, observation_report = load_observation(
        source_path, observations, (width, height), args.face_index)
    report.update(dimensions=[width, height], exifOrientation=orientation, opaqueSource=True,
                  observationValidation=observation_report,
                  nativeCoordinateAdapter=NATIVE_COORDINATE_ADAPTER,
                  sourceDecoder='Pillow RGB; every source layer and composed output uses this same decode')
    png(output/'source.png', source)
    report['decodedSourceSHA256'] = sha(output/'source.png')
    material_eyes = {}
    if args.materials:
        material_eyes, material_report = load_materials(args.materials, report['sourceSHA256'], (width, height))
        report['materials'] = material_report
        (output/'materials.json').write_bytes(args.materials.read_bytes())
    exposure = np.zeros((height, width))
    ocular = np.zeros_like(source)
    support = np.zeros((height, width), bool)
    core = np.zeros((height, width), bool)
    material_unknown = np.zeros((height, width), bool)
    automatic_layers = ({'exposure': np.zeros((height, width)), 'ocular': np.zeros_like(source),
                         'support': np.zeros((height, width), bool)} if args.materials else None)
    eyes = []
    report['sourceEyes'] = eyes
    write_json(output/'report.json', report)

    # Complete source preparation before generating any requested field. The
    # same prepared arrays are reused for every strength, including zero.
    for eye in EYES:
        directory = output/eye
        directory.mkdir()
        geometry = prepare_geometry(source, mesh, eye, directory)
        report['sourcePreparationCalls'] += 1
        prepared = estimator(geometry['crop'], geometry['u'], geometry['v'],
                             geometry['upper'], geometry['lower'], geometry['to_pixel'],
                             geometry['width'], geometry['iris_center'], geometry['iris_radii'])
        prepared = prepare_sampling_support(geometry['crop'], prepared)
        x0, y0, x1, y1 = geometry['box']
        shape = (y1-y0, x1-x0)
        if automatic_layers is not None:
            automatic_active = prepared['support']
            if np.any(automatic_layers['support'][y0:y1, x0:x1] & automatic_active):
                raise ValueError('automatic ocular supports overlap; no silent last-writer ownership')
            for key in automatic_layers:
                automatic_layers[key][y0:y1, x0:x1][automatic_active] = prepared[key][automatic_active]
        if eye in material_eyes:
            labels = crop_materials(material_eyes[eye], geometry['box'])
            u, v = geometry['u'], geometry['v']
            safe = np.clip(u, 0, geometry['width'])
            domain = ((u >= 0) & (u <= geometry['width'])
                      & (v > geometry['upper'](safe)) & (v < geometry['lower'](safe)))
            prepared = apply_materials(geometry['crop'], prepared, labels, domain,
                                       noise=geometry['report']['sourceNoise'])
            material_unknown[y0:y1, x0:x1] |= prepared['materialUnknown']
            overlay = geometry['crop'].copy()
            for label, color in [('ocularAnchors', [20, 220, 255]),
                                 ('movingLashSupport', [255, 190, 20]), ('unknown', [230, 30, 200])]:
                mask = labels[label]
                overlay[mask] = .65*overlay[mask]+.35*np.array(color)
                png(directory/f'material-{label}.png', mask*255)
            sheet(directory/'material-source-review.png', [('unmodified source', geometry['crop']),
                  ('source labels: ocular cyan / moving gold / unknown magenta', overlay)])
        for key in ('exposure', 'support', 'core'):
            if np.asarray(prepared[key]).shape != shape:
                raise ValueError(f'{eye}: estimator {key} dimensions differ')
        if np.asarray(prepared['ocular']).shape != shape+(3,):
            raise ValueError(f'{eye}: estimator ocular dimensions differ')
        active = np.asarray(prepared['support'], bool)
        unresolved = prepared.get('materialUnknown', np.zeros(shape, bool))
        if np.any(prepared['exposure'][~active & ~unresolved] != 0):
            raise ValueError(f'{eye}: nonzero exposure has no declared source support')
        if np.any(support[y0:y1, x0:x1] & active):
            raise ValueError('the two estimated ocular supports overlap; no silent last-writer ownership')
        exposure[y0:y1, x0:x1][active] = prepared['exposure'][active]
        ocular[y0:y1, x0:x1][active] = prepared['ocular'][active]
        support[y0:y1, x0:x1] |= active
        core[y0:y1, x0:x1] |= prepared['core']
        np.savez_compressed(directory/'candidate-source.npz',
                            **{k: v for k, v in prepared.items() if isinstance(v, np.ndarray)})
        write_json(directory/'source-report.json', prepared['report'])
        png(directory/'source.png', geometry['crop'])
        for key in ('exposure', 'support', 'core', 'observedIris', 'lashVeto'):
            if key in prepared:
                png(directory/f'{key}.png', np.asarray(prepared[key])*255)
        png(directory/'ocular.png', prepared['ocular'])
        eyes.append({'eye': eye, 'crop': geometry['box'], 'preparationCalls': 1,
                     'geometrySHA256': sha(directory/'source-curves.npz'),
                     'sourceLayersSHA256': sha(directory/'candidate-source.npz'),
                     'sourceReport': prepared['report']})
        write_json(output/'report.json', report)
    report['sourcePreparationCompletedBeforeFields'] = True

    manifest = []
    for amount in STRENGTHS:
        graph_path = output/f'graph-{tag(amount)}.json'
        write_json(graph_path, make_graph(normalized, (width, height), amount))
        manifest.append({'input': str(source_path), 'graph': str(graph_path),
                         'output': str(output/f'native-{tag(amount)}.png'),
                         'diagnostics': str(output/f'field-{tag(amount)}.json'),
                         'label': f'fresh-source:eyeHeight{amount:+d}', 'value': amount,
                         'sourceSHA256': report['sourceSHA256'], 'graphSHA256': sha(graph_path)})
    manifest_path = output/'cases.json'
    write_json(manifest_path, manifest)
    command = [str(field_runner), FIELD_TEST, '--exact', '--ignored', '--nocapture']
    start = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT,
                               env={**os.environ, 'BROW_FLOW_CASES': str(manifest_path)},
                               capture_output=True, timeout=args.field_timeout)
    (output/'native.log').write_bytes(completed.stdout+completed.stderr)
    report['nativeRun'] = {'command': command, 'returnCode': completed.returncode,
                           'seconds': time.monotonic()-start, 'manifestSHA256': sha(manifest_path),
                           'recompiled': False}
    yy, xx = np.indices((height, width))
    identity = np.stack([xx, yy], axis=-1).astype(float)
    for case in manifest:
        amount = case['value']
        row = {'amount': amount, 'sourceRefitted': False, 'fieldModified': False,
               'status': 'pending', 'graphSHA256': case['graphSHA256']}
        report['rows'].append(row)
        try:
            diagnostic_path = Path(case['diagnostics'])
            diagnostics = json.loads(diagnostic_path.read_text())
            row['nativeDiagnostics'] = diagnostics
            guards(diagnostics, amount)
            native = np.asarray(Image.open(case['output']).convert('RGB'), dtype=float)
            if native.shape != source.shape:
                raise ValueError('native image dimensions differ from source')
            if amount == 0:
                inverse = identity
                stationary = np.ones((height, width), bool)
                row.update(fieldOrigin='explicit identity after native RGBA identity assertion; no native zero field file claimed',
                           nativeVersusPythonSourceMaxRGBDifference=float(abs(native-source).max()),
                           nativeVersusPythonSourceChangedPixels=int(np.any(native != source, axis=-1).sum()))
            else:
                field_path = Path(str(diagnostic_path)+'.inverse.f32')
                stationary_path = Path(str(diagnostic_path)+'.stationary.u8')
                if field_path.stat().st_size != height*width*2*4:
                    raise ValueError('native inverse field has an incorrect byte count')
                inverse = np.fromfile(field_path, '<f4').reshape(height, width, 2).astype(float)
                stationary, stationary_contract = read_stationary(stationary_path, height, width)
                row.update(fieldOrigin='fresh native absolute source pixel-index XY',
                           fieldSHA256=sha(field_path), stationarySHA256=sha(stationary_path),
                           stationaryContract=stationary_contract)
            automatic = None
            if automatic_layers is not None:
                automatic, automatic_metrics = compose(source, automatic_layers['exposure'],
                    automatic_layers['ocular'], automatic_layers['support'], inverse, stationary)
                automatic_path = output/f'automatic-{tag(amount)}.png'
                png(automatic_path, automatic)
                row.update(automaticComposition=automatic_metrics, automaticOutput=str(automatic_path),
                           automaticOutputSHA256=sha(automatic_path), manualAndAutomaticUseIdenticalField=True)
            row['materialTransport'] = validate_material_transport(material_unknown, inverse, stationary)
            rendered, metrics = compose(source, exposure, ocular, support, inverse, stationary)
            target_exposure = sample(exposure, inverse)
            retained = core & (target_exposure == 1)
            path = output/f'candidate-{tag(amount)}.png'
            png(path, rendered)
            row.update(status='rendered; independent visual review required', composition=metrics,
                       output=str(path), outputSHA256=sha(path),
                       retainedOcularPixels=int(retained.sum()),
                       retainedOcularMaxRGBChange=float(abs(rendered[retained]-source[retained]).max()) if retained.any() else None)
            if amount == 0 and not np.array_equal(rendered, source):
                raise ValueError('zero composition changed the Python decoded source')
            for eye in eyes:
                x0, y0, x1, y1 = eye['crop']
                crop = rendered[y0:y1, x0:x1]
                directory = output/eye['eye']
                png(directory/f'candidate-{tag(amount)}.png', crop)
                panels = [('source', source[y0:y1, x0:x1])]
                if automatic is None:
                    panels += [(f'layered {amount}', crop),
                               ('native single-plane (own decode)', native[y0:y1, x0:x1])]
                else:
                    png(directory/f'automatic-{tag(amount)}.png', automatic[y0:y1, x0:x1])
                    panels += [(f'automatic {amount}', automatic[y0:y1, x0:x1]),
                               (f'manual material correction {amount}', crop)]
                sheet(directory/f'comparison-{tag(amount)}.png', panels)
        except (ValueError, OSError, KeyError) as error:
            row.update(status='rejected', reason=f'{type(error).__name__}: {error}')
        write_json(output/'report.json', report)

    # No result is attributed to a mixed estimator or to changing observations.
    unchanged = all(sha(path) == digest for path, digest in code_hashes.items())
    unchanged &= sha(source_path) == report['sourceSHA256']
    unchanged &= sha(observations) == report['observationsSHA256']
    unchanged &= sha(field_runner) == report['fieldRunnerSHA256']
    if args.materials:
        unchanged &= sha(args.materials) == report['materials']['sha256']
    report['inputsAndCodeUnchangedThroughRun'] = bool(unchanged)
    if not unchanged:
        raise ValueError('source, observation, field executable or estimator code changed during this replay')
    rejected = [row for row in report['rows'] if row['status'] == 'rejected']
    if completed.returncode != 0:
        report['status'] = 'rejected: native validation process failed; see native.log and per-strength evidence'
        return 1
    report['status'] = 'rejected: one or more strengths failed' if rejected else 'rendered; independent visual review required'
    return int(bool(rejected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--field-runner', type=Path, required=True)
    parser.add_argument('--estimator', required=True, help='explicit module:function; no stable default is claimed')
    parser.add_argument('--materials', type=Path, help='explicit original-source-bound categorical material JSON')
    parser.add_argument('--face-index', type=int)
    parser.add_argument('--field-timeout', type=float, default=180)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'preparing', 'experiment': 'fresh-source-eye-closing',
              'strengths': list(STRENGTHS), 'publicFeatureEnabled': False,
              'claim': 'experimental closing only; successful rendering is not aesthetic or general usability acceptance',
              'sourceThresholdsNotTunedToOutput': True, 'readsHistoricalSampleState': False}
    try:
        status = run(args, report)
    except Exception as error:
        report.update(status='rejected', reason=f'{type(error).__name__}: {error}')
        (args.output/'failure.log').write_text(traceback.format_exc())
        status = 1
    write_json(args.output/'report.json', report)
    print(json.dumps({'output': str(args.output.resolve()), 'status': report['status'],
                      'reason': report.get('reason'), 'rows': [
                          {k: row[k] for k in ('amount', 'status', 'reason') if k in row}
                          for row in report.get('rows', [])]}, indent=2))
    return status


if __name__ == '__main__':
    sys.exit(main())

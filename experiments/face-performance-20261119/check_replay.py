"""Compare ordinary face replay outputs in decoded RGBA, not PNG encoding.

This checks numerical preservation after a performance change. It does not
establish aesthetic quality or extrapolate native timings to a browser.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from PIL import Image


def case(path, identifier):
    report = json.loads(path.read_text())
    matches = [item for item in report['cases'] if item['id'] == identifier]
    if len(matches) != 1 or 'outputs' not in matches[0]:
        raise ValueError(f'{path}: exactly one successful input case required')
    if any(item['status'] != 'ok' for item in matches[0]['outputs']):
        raise ValueError(f'{path}: a requested output failed')
    return matches[0]


def key(item):
    return json.dumps(item['adjustments'], sort_keys=True, allow_nan=False)


def rgba(path):
    with Image.open(path) as image:
        return image.size, image.convert('RGBA').tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    baseline = case(args.baseline, args.case)
    candidate = case(args.candidate, args.case)
    if baseline['input'] != candidate['input']:
        raise ValueError('input provenance differs; compare the same frozen evidence')
    originals = {}
    for item in baseline['outputs']:
        originals.setdefault(key(item), []).append(item)
    rows = []
    for item in candidate['outputs']:
        matches = originals.get(key(item))
        if not matches:
            raise ValueError(f'no baseline for requested adjustments: {item["adjustments"]}')
        expected_size, expected = rgba(matches[0]['output'])
        actual_size, actual = rgba(item['output'])
        rows.append({
            'adjustments': item['adjustments'],
            'baseline': matches[0]['output'], 'candidate': item['output'],
            'sameDimensions': actual_size == expected_size,
            'decodedRgbaEqual': actual_size == expected_size and actual == expected,
            'sha256DecodedRgba': hashlib.sha256(actual).hexdigest(),
            'baselineMedianMilliseconds': statistics.median(x['milliseconds'] for x in matches),
            'candidateMilliseconds': item['milliseconds'],
        })
    passed = bool(rows) and all(row['decodedRgbaEqual'] for row in rows)
    result = {'allOutputsEqual': passed, 'case': args.case,
              'timingLimit': 'native algorithm only; no browser or aesthetic acceptance',
              'outputs': rows}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'allOutputsEqual': passed, 'outputs': len(rows),
                      'report': str(args.output.resolve())}))
    if not passed:
        raise SystemExit('decoded image mismatch')


if __name__ == '__main__':
    main()

"""Source-frozen RGB crossing measurements for a visible garment edge.

This measures image-space horizontal displacement at declared scanlines, not
hidden waist width, material opacity, anatomical truth or aesthetic quality.
The source fixes the color projection and search intervals before a candidate
is inspected. Multiple crossings are reported as ambiguous, never selected by
the requested edit direction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float64)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crossings(rgb, line):
    y, left, right = line['y'], line['left'], line['right']
    colors = np.median(rgb[y-1:y+2, left:right+1], axis=0)
    low, vector = np.asarray(line['outsideRGB']), np.asarray(line['colorVector'])
    values = (colors-low) @ vector / (vector @ vector)
    result = []
    for i in range(len(values)-1):
        a, b = float(values[i]), float(values[i+1])
        if a <= .5 < b:
            result.append(left+i+(.5-a)/(b-a))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'measure'])
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--probes', help='JSON [[y, approximate source edge x], ...]')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rgb = read(args.image)
    height, width = rgb.shape[:2]
    if args.mode == 'freeze':
        if args.protocol.exists():
            raise FileExistsError('frozen source protocol must not be overwritten')
        probes = json.loads(args.probes or '[]')
        if not probes:
            raise ValueError('source-visible scanline probes required')
        lines = []
        for y, x in probes:
            if (not isinstance(y, int) or not isinstance(x, int)
                    or not 1 <= y < height-1 or not 18 <= x < width-18):
                raise ValueError('probe or full measurement support outside source')
            outside = np.median(rgb[y-1:y+2, x-8:x-4].reshape(-1,3), axis=0)
            inside = np.median(rgb[y-1:y+2, x+4:x+8].reshape(-1,3), axis=0)
            vector = inside-outside
            if np.linalg.norm(vector) < 20:
                raise ValueError('source probe lacks an independently distinguishable color edge')
            line = {'y': y, 'approximateSourceX': x, 'left': x-18, 'right': x+18,
                    'outsideRGB': outside.tolist(), 'colorVector': vector.tolist()}
            found = crossings(rgb, line)
            if len(found) != 1:
                raise ValueError(f'ambiguous source crossing at y={y}: {found}')
            line['sourceX'] = found[0]
            lines.append(line)
        protocol = {'source': str(args.image.resolve()), 'sourceSHA256': sha(args.image),
                    'dimensions': [width,height], 'lines': lines,
                    'meaning': 'visible RGB edge horizontal response; not hidden anatomy',
                    'candidateUsedToSelectMeasurement': False}
        args.protocol.write_text(json.dumps(protocol,indent=2)+'\n')
        print(json.dumps(protocol,indent=2))
        return
    protocol = json.loads(args.protocol.read_text())
    if (sha(Path(protocol['source'])) != protocol['sourceSHA256']
            or protocol['dimensions'] != [width,height]):
        raise ValueError('source identity or candidate dimensions changed')
    rows = []
    for line in protocol['lines']:
        found = crossings(rgb, line)
        rows.append({'y': line['y'], 'sourceX': line['sourceX'], 'crossings': found,
                     'horizontalShift': found[0]-line['sourceX'] if len(found)==1 else None,
                     'status': 'measured' if len(found)==1 else 'ambiguous-or-absent'})
    result = {'candidate': str(args.image.resolve()), 'candidateSHA256': sha(args.image),
              'protocolSHA256': sha(args.protocol), 'measurements': rows,
              'aestheticAcceptanceEstablished': False}
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

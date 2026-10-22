"""Conservatively merge native/reference protection before a new source solve."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from resolution_field import shared_fixed_support
from run_portable_jaw import read, save, sha


def prepare(reference, native, output):
    if output.exists():
        raise FileExistsError('refusing to overwrite shared protection')
    reports = [read(folder/'report.json') for folder in [reference, native]]
    sources = [Path(report['source']) for report in reports]
    masks = [np.asarray(Image.open(folder/'fixed-source.png')) > 0 for folder in [reference, native]]
    for report, source, mask in zip(reports, sources, masks):
        if (report['inputs_sha256'].get(str(source)) != sha(source)
                or tuple(reversed(Image.open(source).size)) != mask.shape):
            raise ValueError('source identity or fixed mask dimensions differ')
    common = reports[0]['inputs_sha256'].keys() & reports[1]['inputs_sha256'].keys()
    originals = [p for p in common if Path(p).suffix.lower() in ['.jpg', '.jpeg']
                 and reports[0]['inputs_sha256'][p] == reports[1]['inputs_sha256'][p] == sha(p)]
    if len(originals) != 1:
        raise ValueError('one common unchanged original JPEG is required')
    result = shared_fixed_support(*masks)
    paths = [*sources, Path(originals[0]), Path(__file__).resolve(),
             Path(__file__).with_name('resolution_field.py')]
    for folder in [reference, native]:
        paths += [folder/'report.json', folder/'fixed-source.png']
    hashes = {str(p): sha(p) for p in paths}
    output.mkdir(parents=True)
    mask_path = output/'fixed-reference.png'
    Image.fromarray(result['mask'].astype('uint8')*255).save(mask_path)
    save(output/'protocol.json', {
        'operation': 'union_of_reference_and_native_fixed_support',
        'reference_source_sha256': sha(sources[0]), 'mask': str(mask_path),
        'mask_sha256': sha(mask_path), 'inputs_sha256': hashes, 'support': result['report'],
        'scope': 'New source constraints before solving; neither solved-field masking nor output pixel replacement.'})
    return result['report']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['reference', 'native', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.reference.resolve(), args.native.resolve(), args.output.resolve()))

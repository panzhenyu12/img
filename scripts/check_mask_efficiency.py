"""Measure a mask-only optimization using isolated, immutable engine snapshots.

The snapshots must otherwise have identical source; this does not benchmark
subsequent skin/face changes or claim controlled performance acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, default=ROOT/'.run/mask-perf-before-bench')
    parser.add_argument('--after', type=Path, default=ROOT/'.run/mask-perf-after-bench')
    args = parser.parse_args()
    binaries = {'before':args.before.resolve(), 'after':args.after.resolve()}
    hashes = {k:sha(v) for k,v in binaries.items()}
    out = ROOT/'test/results/mask-efficiency';out.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = [('portrait-makeup','whiten'),('portrait-makeup','smooth-strong'),('portrait-dark','whiten'),('portrait-hair','smooth-strong')]
    for index,(sid,case) in enumerate(cases):
        source = ROOT/f'test/results/native-face-contour/{sid}-source.png'
        graph = ROOT/f'test/results/skin-detail/{sid}-{case}.json'
        row = {'sample':sid, 'case':case, 'source_sha256':sha(source), 'graph_sha256':sha(graph)}
        order = ['before','after'] if index%2==0 else ['after','before']
        for variant in order:
            path = out/f'{sid}-{case}-{variant}.png'
            assert sha(binaries[variant])==hashes[variant]
            result=subprocess.run([str(binaries[variant]),str(source),str(graph),str(path)],capture_output=True,text=True,check=True)
            assert sha(binaries[variant])==hashes[variant]
            row[variant]=json.loads(result.stdout)
        before=np.asarray(Image.open(out/f'{sid}-{case}-before.png'))
        after=np.asarray(Image.open(out/f'{sid}-{case}-after.png'))
        row['different_channels']=int(np.count_nonzero(before!=after))
        if row['different_channels']:raise RuntimeError(f'Optimization changed pixels: {row}')
        row['hot_ratio']=row['after']['hot_median_ms']/row['before']['hot_median_ms']
        rows.append(row);print(json.dumps(row),flush=True)
    assert all(sha(v)==hashes[k] for k,v in binaries.items())
    report={'binaries_sha256':hashes,'binary_checks':'before and after each subprocess and final hash',
            'scope':'mask.rs-only difference, identical other modules; interleaved processes, 1 first + 5 hot runs per case',
            'limitations':'Concurrent builds/model experiments. Not a controlled performance acceptance, detection/I/O excluded.',
            'rows':rows}
    (ROOT/'doc/qa/mask-efficiency.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

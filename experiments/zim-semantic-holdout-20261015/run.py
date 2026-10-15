"""Two independent source photographs for frozen ZIM semantic selection.

Calls the existing v2 automatic component prompt runner unchanged. All four raw
alphas and official argmax remain evidence. No model, prompt or eligibility
threshold is tuned on these images.
"""
from pathlib import Path
import importlib.util
import json
import sys
import hashlib

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUNNER=ROOT/'experiments/hair-accessory-matting-zim-v2-20261015/run.py'
spec=importlib.util.spec_from_file_location('frozen_zim_v2',RUNNER)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.SOURCES={
 'portrait-makeup':('experiments/face-anatomical-v22-20261015/inputs/portrait-makeup-source.png',
                    'experiments/face-anatomical-v22-20261015/inputs/portrait-makeup-graph.json'),
 'portrait-glasses-street':('test/images/portrait-glasses-street.jpg',
                    'test/results/face-core-fresh/portrait-glasses-street-faceWidth-60.json'),
}
if __name__=='__main__':
 output=HERE/'raw'
 sys.argv=[str(RUNNER),'--names',*runner.SOURCES,'--out',str(output)]
 runner.main()
 manifest={'scope':'Two independent source images unused in the ZIM selection development triplet; not an external benchmark.',
   'protocol':'Unmodified existing automatic prompts and four official outputs; selection applied later with exact frozen choose function.',
   'originalResolution':{'portrait-makeup':[1280,1128],'portrait-glasses-street':[1280,853]},
   'detectionResolution':{'portrait-makeup':[1280,1128],'portrait-glasses-street':[768,512]},
   'detectionNote':'Glasses source-only graph is normalized from its earlier 768x512 decode; mapped to original 1280x853, not re-detected from any edited output. Parser masks remain confidence references, not alpha truth.',
   'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),RUNNER,HERE/'selector-snapshot.py']}}
 (HERE/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')

"""Copy MediaPipe runtime and download optional local detection models."""
from pathlib import Path
import shutil
import urllib.request
import hashlib

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'apps/web/public'
runtime = ROOT / 'apps/web/node_modules/@mediapipe/tasks-vision/wasm'
(PUBLIC / 'vision').mkdir(parents=True, exist_ok=True)
for source in runtime.iterdir():
    if source.is_file():
        shutil.copyfile(source, PUBLIC / 'vision' / source.name)
ort_runtime = ROOT / 'apps/web/node_modules/onnxruntime-web/dist'
(PUBLIC / 'ort').mkdir(parents=True, exist_ok=True)
for filename in ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']:
    shutil.copyfile(ort_runtime / filename, PUBLIC / 'ort' / filename)
models = {
    'face_landmarker.task': 'face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    'pose_landmarker_lite.task': 'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
}
(PUBLIC / 'models').mkdir(parents=True, exist_ok=True)
for filename, path in models.items():
    target = PUBLIC / 'models' / filename
    if target.exists() and target.stat().st_size > 100_000:
        print(f'{filename}: available')
        continue
    partial = target.with_suffix('.download')
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/' + path, partial)
    partial.replace(target)
    print(f'{filename}: downloaded')

# Open BiSeNet face parser, ONNX inference is optional and entirely local.
parser = PUBLIC / 'models/faceparser.onnx'
parser_sha256 = '1d5c459e3d4691eb6120994fd36a8d360fe3732997d1bbf245278145cdd11d20'
if not parser.exists() or hashlib.sha256(parser.read_bytes()).hexdigest() != parser_sha256:
    partial = parser.with_suffix('.download')
    urllib.request.urlretrieve(
        'https://huggingface.co/bluefoxcreation/Face_parsing_onnx/resolve/main/faceparser_sim.onnx', partial)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != parser_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError('face parser checksum mismatch')
    partial.replace(parser)
print('faceparser.onnx: checksum verified')

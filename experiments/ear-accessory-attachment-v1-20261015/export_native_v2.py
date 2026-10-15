from pathlib import Path
import json,os,subprocess,hashlib
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    native=ROOT/'experiments/joint-skin-matting-v1-20261015/native-v2-user';old=ROOT/'experiments/face-material-fill-v1-20261015/user-face-contour-100-full/provenance.json';prov=json.loads(old.read_text());binary=Path(prov['binary']);assert sha(binary)==prov['binarySha256'];source=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';out=HERE/'native-v2-layers';out.mkdir(exist_ok=False);rows=[]
    for amount in [100,-100,30,-30]:
        d=out/str(amount);d.mkdir();graph=native/str(amount)/'graph.json';(d/'graph.json').write_bytes(graph.read_bytes());proc=subprocess.run([str(binary),'--input',str(source),'--graph',str(graph),'--output',str(d/'candidate.png')],env={**os.environ,'FACE_LAYERS_DEBUG_DIR':str(d/'layers')},capture_output=True);(d/'render.log').write_bytes(proc.stdout+proc.stderr);proc.check_returncode()
        exact=np.array_equal(np.asarray(Image.open(d/'candidate.png')),np.asarray(Image.open(native/str(amount)/'candidate.png')));assert exact
        row={'amount':amount,'nativeRgbaExact':bool(exact),'graphSha256':sha(graph),'binarySha256':sha(binary),'sourceSha256':sha(source),'layers':{p.name:sha(p) for p in (d/'layers').iterdir()}};rows.append(row);print(json.dumps({'amount':amount,'nativeRgbaExact':bool(exact)}),flush=True)
        (out/'report.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
if __name__=='__main__':main()

"""Pinned official FASHN checkpoint -> checked browser ONNX, optional parity.

Use the dedicated Python 3.11 environment described in FUNC-012. This script
does not install packages, contact an LLM, or download images.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REVISION = '1f80c34dbab321c5730dda5c3fea279fd3e97498'
BASE = f'https://huggingface.co/fashn-ai/fashn-human-parser/resolve/{REVISION}'
WEIGHTS_SHA = 'e43c8c8a9b04f28798f0a4630cf18caa2cdb27a0d454fae43a5716e6f7078244'
CONFIG_SHA = '87bd5b66419dbfa7c02cfbeb94e292454863b1606c5df29526b19a6c1de70d9f'
ONNX_SHA = '3b4925715aecbf4996fea5de5e7ce30b173c20f751f21cffe826b5db46ed7729'

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()

def download(directory, name, expected):
    path = directory / name
    if path.is_file() and sha(path) == expected: return path
    partial = path.with_suffix(path.suffix + '.part')
    with urllib.request.urlopen(BASE + '/' + name, timeout=60) as response, partial.open('wb') as output:
        while chunk := response.read(1024 * 1024): output.write(chunk)
    if sha(partial) != expected: raise RuntimeError(f'{name} checksum mismatch')
    partial.replace(path)
    return path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,default=Path('.models/body-parser'))
    parser.add_argument('--output',type=Path,default=Path('apps/web/public/models/bodyparser.onnx'))
    parser.add_argument('--verify-image',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();args.model_dir.mkdir(parents=True,exist_ok=True)
    weights=download(args.model_dir,'model.safetensors',WEIGHTS_SHA)
    config=download(args.model_dir,'config.json',CONFIG_SHA)
    import torch
    from safetensors.torch import load_file
    from transformers import SegformerConfig,SegformerForSemanticSegmentation
    import onnx
    torch.set_num_threads(2)
    model=SegformerForSemanticSegmentation(SegformerConfig.from_json_file(str(config)))
    model.load_state_dict(load_file(str(weights)));model.eval()
    class Wrapper(torch.nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,pixel_values):return self.model(pixel_values).logits
    if not args.output.is_file():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        partial=args.output.with_suffix('.part.onnx')
        torch.onnx.export(Wrapper(model).eval(),torch.zeros(1,3,576,384),str(partial),opset_version=17,input_names=['pixel_values'],output_names=['logits'],do_constant_folding=True)
        onnx.checker.check_model(str(partial));partial.replace(args.output)
    exported_sha=sha(args.output)
    if exported_sha != ONNX_SHA:raise RuntimeError(f'ONNX export differs from pinned artifact: {exported_sha}; compare dependency versions and outputs before adopting it')
    report={'revision':REVISION,'weightsSHA256':sha(weights),'configSHA256':sha(config),'onnxSHA256':exported_sha,'onnxBytes':args.output.stat().st_size,'input':[1,3,576,384],'output':[1,18,144,96],'opset':17}
    if args.verify_image:
        import cv2
        import numpy as np
        from PIL import Image
        import onnxruntime as ort
        rgb=np.array(Image.open(args.verify_image).convert('RGB'))
        resized=cv2.resize(rgb,(384,576),interpolation=cv2.INTER_AREA)
        values=((resized.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None].copy()
        started=time.monotonic()
        with torch.inference_mode():reference=model(torch.from_numpy(values)).logits.numpy()
        torch_ms=(time.monotonic()-started)*1000
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(args.output),options,providers=['CPUExecutionProvider'])
        started=time.monotonic();actual=session.run(None,{'pixel_values':values})[0];ort_ms=(time.monotonic()-started)*1000
        error=np.abs(reference-actual)
        report['parity']={'source':str(args.verify_image),'torchMs':round(torch_ms,2),'ortMs':round(ort_ms,2),'maxLogitError':float(error.max()),'meanLogitError':float(error.mean()),'argmaxMismatchPixels':int(np.count_nonzero(reference.argmax(1)!=actual.argmax(1)))}
        if error.max()>.005:raise RuntimeError('ONNX reference parity failed')
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()

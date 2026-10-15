"""Source-only accessory alpha probe, reusing frozen official ZIM code/weights.

No Rust graph or edited image is changed. All four raw model outputs are kept.
"""
from pathlib import Path
import importlib.util
import json
import time
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('existing_zim_probe', ROOT/'experiments/zim-20261014/run.py')
zim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zim)
np, cv2, Image, ImageDraw = zim.np, zim.cv2, zim.Image, zim.ImageDraw


def main():
    out = ROOT/'test/results/accessory-matting-zim-v1-20261014'
    out.mkdir(exist_ok=False)
    source = ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
    graph_path = source.with_name('face-graph.json')
    rgb = np.asarray(Image.open(source).convert('RGB'))
    h,w = rgb.shape[:2]
    graph = json.loads(graph_path.read_text())
    params = graph['operations'][0]['params']
    confidence = zim.coverage(params['semanticMasks']['accessories'],w,h)
    chin = max(p['y']*h for p in params['landmarks'])
    count, labels, stats, centers = cv2.connectedComponentsWithStats((confidence>=.5).astype('uint8'))
    components = [i for i in range(1,count) if centers[i][1] < chin and stats[i,4] > 30]
    components.sort(key=lambda i: int(stats[i,4]),reverse=True)
    prompts=[]
    for i in components[:2]:
        x,y,cw,ch,area = map(int,stats[i])
        box=[max(0,x-4),max(0,y-4),min(w-1,x+cw+3),min(h-1,y+ch+3)]
        distance=cv2.distanceTransform((labels==i).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        py,px=np.unravel_index(np.argmax(distance),distance.shape)
        prompts.append(dict(component=i,box=box,point=[int(px),int(py)],sourceReferenceArea=area))
    Encoder,Decoder,Zim,Predictor,vendor=zim.load_vendor()
    weights=ROOT/'.models/zim-20261014'
    digests={'encoder.onnx':'6123ea10722dd43a0f8bbb96cd6a0532d8a65e2b1e8339f0cd6593d494807809',
             'decoder.onnx':'a99d14bac3dc28b17809020b1711b1991bd5b7eb18f678c3624ce508748cfa11'}
    for file,digest in digests.items(): assert zim.sha(weights/file)==digest
    zim.torch.set_num_threads(3)
    start=time.perf_counter()
    predictor=Predictor(Zim(Encoder(str(weights/'encoder.onnx'),num_threads=3),Decoder(str(weights/'decoder.onnx'),num_threads=3)))
    load_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter();predictor.set_image(rgb);encode_ms=(time.perf_counter()-start)*1000
    report=dict(scope='One original photo; automatically derived accessory component boxes and interior points. Raw model alphas, no semantic clipping, no edited-output detection, no graph replacement, no alpha ground truth.',
                sourceSha256=zim.sha(source),graphSha256=zim.sha(graph_path),runnerSha256=zim.sha(Path(__file__)),
                vendorCommit=vendor['commit'],weights=digests,loadMs=load_ms,encodeMs=encode_ms,rows=[])
    for prompt in prompts:
        start=time.perf_counter()
        alphas,scores,_=predictor.predict(box=np.array(prompt['box'],dtype=np.float32),
            point_coords=np.array([prompt['point']],dtype=np.float32),point_labels=np.array([1],dtype=np.float32),multimask_output=True)
        elapsed=(time.perf_counter()-start)*1000
        folder=out/f"component-{prompt['component']}";folder.mkdir()
        np.save(folder/'alphas.npy',alphas,allow_pickle=False)
        x0,y0,x1,y1=prompt['box'];crop=(max(0,x0-12),max(0,y0-12),min(w,x1+13),min(h,y1+13))
        panelw,panelh=crop[2]-crop[0],crop[3]-crop[1]
        sheet=Image.new('RGB',(panelw*6,panelh+28),'#20242a');draw=ImageDraw.Draw(sheet)
        sheet.paste(Image.fromarray(rgb).crop(crop),(0,28));draw.text((4,8),'Original',fill='white')
        old=rgb*confidence[...,None]+np.array([45,80,60])*(1-confidence[...,None])
        sheet.paste(Image.fromarray(old.clip(0,255).astype('uint8')).crop(crop),(panelw,28));draw.text((panelw+4,8),'Parser reference',fill='white')
        summaries=[]
        for index,alpha in enumerate(alphas):
            Image.fromarray(np.rint(alpha*255).astype('uint8')).save(folder/f'alpha-{index}.png')
            composite=np.rint(rgb*alpha[...,None]+np.array([45,80,60])*(1-alpha[...,None])).clip(0,255).astype('uint8')
            sheet.paste(Image.fromarray(composite).crop(crop),((index+2)*panelw,28));draw.text(((index+2)*panelw+4,8),f'{index}: {scores[index]:.3f}',fill='white')
            summaries.append(dict(index=index,score=float(scores[index]),foregroundPixels=int((alpha>=.5).sum()),
                promptAlpha=float(alpha[prompt['point'][1],prompt['point'][0]])))
        sheet.save(folder/'comparison-1x.png')
        sheet.resize((sheet.width*2,sheet.height*2),Image.Resampling.NEAREST).save(folder/'comparison-2x.png')
        report['rows'].append(dict(prompt=prompt,decodeMs=elapsed,officialScoreArgmax=int(np.argmax(scores)),alphas=summaries,crop=crop))
        print(json.dumps(report['rows'][-1]),flush=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()

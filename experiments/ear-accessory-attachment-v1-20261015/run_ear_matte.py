"""Independent neural ear matte from source-only ear/lobe contact prompts.

The semantic ear mask stops before the visible jewelry attachment. A geometric
hook constraint has no effect if that source tissue is absent from the ear
material. Raw candidates and official score choice are retained without
selecting a mask based on the desired edited image.
"""
from pathlib import Path
import sys,json,importlib.util,time
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt,label
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('earprobe',ROOT/'experiments/joint-skin-matting-v1-20261015/run.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
zim=probe.zim


def deepest(mask):
    if not np.any(mask):return None
    d=distance_transform_edt(mask);y,x=np.unravel_index(np.argmax(d),d.shape);return [int(x),int(y)]


def main():
    out=HERE/'ear-matte-v6';out.mkdir(exist_ok=False)
    source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';graph_path=ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json'
    source=np.asarray(Image.open(source_path).convert('RGB'));h,w=source.shape[:2];params=json.loads(graph_path.read_text())['operations'][0]['params'];masks={k:zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()};D=np.load(ROOT/'experiments/joint-skin-matting-v1-20261015/prompts-v2/user-face-contour/crop-point/alphas-full.npy')[2]
    previous=json.loads((HERE/'outputs-pivot-v3/report.json').read_text());J=np.load(HERE/'native-pivot-v4/source-attachments.npz')['jewelryAlpha'];Y,X=np.mgrid[:h,:w]
    Encoder,Decoder,Zim,Predictor,vendor=zim.load_vendor();weights=ROOT/'.models/zim-20261014';zim.torch.set_num_threads(3);decoder=Decoder(str(weights/'decoder.onnx'),num_threads=3);model=Zim(Encoder(str(weights/'encoder.onnx'),num_threads=3),decoder);predictor=Predictor(model)
    report={'scope':__doc__,'sourceSha256':zim.sha(source_path),'graphSha256':zim.sha(graph_path),'scriptSha256':zim.sha(Path(__file__)),'vendorCommit':vendor['commit'],'accepted':False,'rows':[]}
    for c in previous['components']:
        side=c['side'];bb=c['semanticEarBounds'];earheight=bb[3]-bb[1];earwidth=bb[2]-bb[0];hx,hy=c['hookSource'];half=X<w/2 if side=='left' else X>=w/2
        box=[max(0,int(bb[0]-.45*earheight)),max(0,int(bb[1]-.12*earheight)),min(w,int(bb[2]+.45*earheight)),min(h,int(max(hy,bb[3])+.2*earheight))]
        x0,y0,x1,y1=box;roi=(X>=x0)&(X<x1)&(Y>=y0)&(Y<y1)
        ear=(masks['ears']>=.9)&half
        positive=[deepest(ear),deepest(ear&(Y>bb[1]+.68*earheight))]
        supported=(D>=.95)&(J<.01)&(masks['hair']<.01)&roi&((X-hx)**2+(Y-hy)**2<(.18*earheight)**2)
        cost=np.where(supported,(X-hx)**2+(Y-hy)**2,np.inf);yy,xx=np.unravel_index(np.argmin(cost),cost.shape);positive.append([int(xx),int(yy)])
        positive=[p for p in positive if p is not None]
        away=distance_transform_edt(masks['ears']<.01)
        face=(masks['skin']>=.95)&(masks['ears']<.01)&(masks['hair']<.01)&(J<.01)&roi&(away>.16*earheight)
        occupied=np.maximum.reduce([masks[k] for k in ['skin','ears','hair','accessories','neck']])
        negative=[deepest(face&(Y<hy)),deepest((J>=.95)&roi),deepest((masks['hair']>=.95)&roi),deepest((occupied<.01)&roi&(D<.001))]
        negative=[p for p in negative if p is not None]
        start=time.perf_counter();predictor.set_image(source[y0:y1,x0:x1]);encode=(time.perf_counter()-start)*1000
        coords=np.asarray(positive+negative,dtype=np.float32)-np.array([x0,y0],dtype=np.float32)
        localbox=np.array([bb[0]-x0,bb[1]-y0,bb[2]-x0,max(bb[3],hy+1)-y0],dtype=np.float32)
        for attention in ['point','box']:
            model.decoder=probe.adapter.CombinedPromptDecoder(decoder,attention=attention,bbox_attention=model.bbox_attn_mask)
            start=time.perf_counter();alphas,scores,_=predictor.predict(point_coords=coords,point_labels=np.array([1]*len(positive)+[0]*len(negative),dtype=np.float32),box=localbox,multimask_output=True);decode=(time.perf_counter()-start)*1000
            mapped=np.zeros((len(alphas),h,w),dtype=np.float32);mapped[:,y0:y1,x0:x1]=alphas;official=int(np.argmax(scores));d=out/f'{side}-{attention}';d.mkdir();np.save(d/'alphas-full.npy',mapped,allow_pickle=False)
            items=[('Source',Image.fromarray(source)),('Semantic ears',probe.probe.cutout(source,masks['ears']))]
            candidates=[]
            for i,a in enumerate(mapped):
                probe.probe.png(d/f'alpha-{i}.png',a);items.append((f'{i}: {float(scores[i]):.3f}'+(' official' if i==official else ''),probe.probe.cutout(source,a)))
                candidates.append({'index':i,'score':float(scores[i]),'positive':[float(a[y,x]) for x,y in positive],'negative':[float(a[y,x]) for x,y in negative],'hookAlpha':float(a[round(hy),round(hx)])})
            panelbox=[box[0],box[1],box[2],min(h,box[3]+10)];pw=panelbox[2]-panelbox[0];ph=panelbox[3]-panelbox[1];panel=Image.new('RGB',(pw*len(items),ph+25),'#171b20');draw=ImageDraw.Draw(panel)
            for i,(title,im) in enumerate(items):panel.paste(im.crop(panelbox),(i*pw,25));draw.text((i*pw+3,6),title,fill='white')
            panel.save(d/'comparison-1x.png')
            entry={'side':side,'attention':attention,'box':box,'positive':positive,'negative':negative,'officialScoreArgmax':official,'candidates':candidates,'encodeMs':encode,'decodeMs':decode,'alphaSha256':zim.sha(d/'alphas-full.npy')};report['rows'].append(entry);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'side':side,'attention':attention,'scores':scores.tolist(),'positive':positive,'negative':negative}),flush=True)


if __name__=='__main__':main()

"""ZIM whole-person source matte for joint head/neck material coverage.

Automatic prompts from original parsing; no edited-image detection, and no
quality claim from the decoder's score. All four candidates are retained.
"""
from pathlib import Path
import importlib.util
import json
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('direct_skin',ROOT/'experiments/joint-skin-matting-v1-20261015/run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
zim,np,Image,ImageDraw=base.zim,base.np,base.Image,base.ImageDraw


def main():
    out=HERE/'outputs';out.mkdir(exist_ok=False)
    source=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
    graph=ROOT/'experiments/whole-face-shape-v2-20261015/outputs-scoped-owner/user-face-contour/natural-30/graph.json'
    rgb=np.asarray(Image.open(source).convert('RGB'));h,w=rgb.shape[:2]
    params=json.loads(graph.read_text())['operations'][0]['params']
    masks={k:zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
    person=zim.coverage(params['subjectMask'],w,h)
    positive=[]
    for key in ['skin','hair','ears','accessories','neck']:
        positive+=base.component_points(masks[key]>.95,3)
    # Explicitly prompt connected clothing too: the target is the whole person,
    # not skin alone. Low face probability must not create a neck/cloth cut.
    head=np.maximum.reduce([masks[k] for k in ['face','skin','hair','ears','accessories','neck']])
    positive+=base.component_points((person>.95)&(head<.01),2)
    occupied=np.maximum(person,head)
    negative=base.component_points(occupied<.005,4)
    yy,xx=np.nonzero(occupied>.5)
    box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
    enc,dec,Zim,Predictor,vendor=zim.load_vendor();zim.torch.set_num_threads(3)
    weights=ROOT/'.models/zim-20261014';decoder=dec(str(weights/'decoder.onnx'),num_threads=3)
    model=Zim(enc(str(weights/'encoder.onnx'),num_threads=3),decoder);predictor=Predictor(model)
    start=time.perf_counter();predictor.set_image(rgb);encode_ms=(time.perf_counter()-start)*1000
    report={'accepted':False,'purpose':__doc__,'sourceSha256':zim.sha(source),'graphSha256':zim.sha(graph),
            'scriptSha256':zim.sha(Path(__file__)),'vendorCommit':vendor['commit'],
            'weights':{p.name:zim.sha(p) for p in weights.glob('*.onnx')},
            'positive':positive,'negative':negative,'box':box,'encodeMs':encode_ms,'rows':[]}
    for mode in ['points','combined-point']:
        args={'point_coords':np.asarray(positive+negative,dtype='float32'),
              'point_labels':np.asarray([1]*len(positive)+[0]*len(negative),dtype='float32'),
              'multimask_output':True}
        if mode=='combined-point':
            model.decoder=base.adapter.CombinedPromptDecoder(decoder,attention='point',bbox_attention=model.bbox_attn_mask)
            args['box']=np.asarray(box,dtype='float32')
        else:model.decoder=decoder
        start=time.perf_counter();alphas,scores,_=predictor.predict(**args);elapsed=(time.perf_counter()-start)*1000
        dest=out/mode;dest.mkdir();np.save(dest/'alphas.npy',alphas,allow_pickle=False)
        rows=[];official=int(np.argmax(scores))
        for i,a in enumerate(alphas):
            base.probe.png(dest/f'alpha-{i}.png',a)
            rows.append({'index':i,'score':float(scores[i]),'coreRecall':{k:float((a[masks[k]>.95]>.95).mean()) for k in ['skin','hair','ears','accessories']},
                         'negativeAlpha':[float(a[y,x]) for x,y in negative]})
        for name,box2 in [('head',[287,0,730,709]),('left',[305,285,430,565]),('right',[590,285,715,565])]:
            x0,y0,x1,y1=box2;pw,ph=x1-x0,y1-y0
            panel=Image.new('RGB',(pw*6,ph+28),'#20242a');draw=ImageDraw.Draw(panel)
            items=[('Source',Image.fromarray(rgb)),('Parser person',base.probe.cutout(rgb,person))]+[(f'{i}: {scores[i]:.3f}'+(' official' if i==official else ''),base.probe.cutout(rgb,a)) for i,a in enumerate(alphas)]
            for i,(title,im) in enumerate(items):panel.paste(im.crop(box2),(i*pw,28));draw.text((i*pw+3,6),title,fill='white')
            panel.save(dest/f'{name}-1x.png')
        entry={'mode':mode,'officialScoreArgmax':official,'decodeMs':elapsed,'alphaSha256':zim.sha(dest/'alphas.npy'),'candidates':rows}
        if mode=='combined-point':entry['actualDecoderPrompt']=model.decoder.last_prompt
        report['rows'].append(entry);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(entry),flush=True)

if __name__=='__main__':main()

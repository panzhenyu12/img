"""Two unchanged automatic accessory prompts, five decoder modes, all raw alphas.

Original 1024px RGB only; single encoder pass; no graph or production changes.
No candidate selection beyond reporting official argmax. B=1 only.
"""
from pathlib import Path
import hashlib, importlib.util, json, time

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
probe=module('accessory_probe',ROOT/'experiments/hair-accessory-matting-zim-v2-20261015/run.py')
adapter=module('accessory_adapter',HERE/'adapter-snapshot.py')
zim,np,Image=probe.zim,probe.np,probe.Image
sha=zim.sha
EXPECTED={'encoder.onnx':'6123ea10722dd43a0f8bbb96cd6a0532d8a65e2b1e8339f0cd6593d494807809',
          'decoder.onnx':'a99d14bac3dc28b17809020b1711b1991bd5b7eb18f678c3624ce508748cfa11'}
MODES=['upstream-mixed','box-only','points-only','combined-point','combined-box']

class RecordingSession:
    def __init__(self,session):self.session=session;self.feeds=[]
    def __getattr__(self,name):return getattr(self.session,name)
    def run(self,outputs,inputs,*args,**kw):
        c,l,a=inputs['point_coords'],inputs['point_labels'],inputs['attn_mask']
        assert c.shape[0]==l.shape[0]==a.shape[0]==1
        assert c.shape[:2]==l.shape and c.shape[2]==2
        self.feeds.append(dict(pointCoords=c.tolist(),pointLabels=l.tolist(),pointCoordsDtype=str(c.dtype),
            pointLabelsDtype=str(l.dtype),attentionShape=list(a.shape),attentionDtype=str(a.dtype),
            attentionSum=float(a.sum()),attentionSha256=hashlib.sha256(a.tobytes()).hexdigest(),
            promptSha256=hashlib.sha256(c.tobytes()+l.tobytes()).hexdigest()))
        return self.session.run(outputs,inputs,*args,**kw)

def main():
    out=HERE/'outputs';out.mkdir(exist_ok=False)
    source=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'
    graph=ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json'
    prompt_file=ROOT/'experiments/hair-accessory-matting-zim-v2-20261015/outputs/user-face-contour/prompts.json'
    assert sha(source)=='5f15e92b14c44a468d924af5ac985a7cc7dc1c87d8ab235f2c94b2e88b173c73'
    frozen=[Path(__file__),HERE/'adapter-snapshot.py',source,graph,prompt_file,Path(probe.__file__),Path(zim.__file__)]
    script_hashes={str(p.relative_to(ROOT)):sha(p) for p in frozen}
    rgb=np.asarray(Image.open(source).convert('RGB'));h,w=rgb.shape[:2]
    params=json.loads(graph.read_text())['operations'][0]['params']
    masks={k:zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
    prompts=[p for p in json.loads(prompt_file.read_text()) if p['kind']=='accessories']
    derived,components=probe.prompts_from_components(masks['accessories'],masks['skin'],'accessories',max(p['y']*h for p in params['landmarks']))
    assert prompts==derived and len(prompts)==2
    (out/'prompts.json').write_text(json.dumps(prompts,indent=2)+'\n')
    for k in ['accessories','skin','hair','ears']:probe.png(out/('source-'+k+'.png'),masks[k])
    for name,digest in EXPECTED.items():assert sha(ROOT/'.models/zim-20261014'/name)==digest
    Encoder,Decoder,Zim,Predictor,vendor=zim.load_vendor();zim.torch.set_num_threads(3)
    decoder=Decoder(str(ROOT/'.models/zim-20261014/decoder.onnx'),num_threads=3)
    decoder.ort_session=RecordingSession(decoder.ort_session)
    model=Zim(Encoder(str(ROOT/'.models/zim-20261014/encoder.onnx'),num_threads=3),decoder)
    predictor=Predictor(model)
    adapters={k:adapter.CombinedPromptDecoder(decoder,attention=k,bbox_attention=model.bbox_attn_mask) for k in ['point','box']}
    start=time.perf_counter();predictor.set_image(rgb);encode_ms=(time.perf_counter()-start)*1000
    report=dict(scope='Two existing automatic earring components; source-only; no graph, warp, or production edits. No alpha ground truth. Scores are not semantic correctness.',
        source=str(source.relative_to(ROOT)),sourceSha256=sha(source),graph=str(graph.relative_to(ROOT)),graphSha256=sha(graph),
        size=[w,h],sourceHasAIWatermark=True,promptFile=str(prompt_file.relative_to(ROOT)),promptFileSha256=sha(prompt_file),
        weights=EXPECTED,scriptHashes=script_hashes,vendorCommit=vendor['commit'],encodeMs=encode_ms,
        featureReuse='One encoder invocation; same original-image features for all 10 cases',batchLimitation='B=1 only',
        onnxInputs=[dict(name=i.name,shape=i.shape,type=i.type) for i in decoder.ort_session.get_inputs()],rows=[])
    for prompt in prompts:
        comp=components==prompt['component'];core=comp&(masks['accessories']>=.9)
        assert int(comp.sum())==prompt['sourceReferenceArea'] and comp[prompt['positive'][1],prompt['positive'][0]]
        box=prompt['box'];crop=[max(0,box[0]-16),max(0,box[1]-16),min(w,box[2]+17),min(h,box[3]+17)]
        for mode in MODES:
            dest=out/f"component-{prompt['component']}-{mode}";dest.mkdir()
            args=dict(multimask_output=True)
            if mode!='points-only':args['box']=np.asarray(box,dtype=np.float32)
            if mode!='box-only':
                args['point_coords']=np.asarray([prompt['positive']]+prompt['negative'],dtype=np.float32)
                args['point_labels']=np.asarray([1]+[0]*len(prompt['negative']),dtype=np.float32)
            model.decoder=adapters[mode.removeprefix('combined-')] if mode.startswith('combined-') else decoder
            start=time.perf_counter();alphas,scores,_=predictor.predict(**args);elapsed=(time.perf_counter()-start)*1000
            assert alphas.shape==(4,h,w) and np.isfinite(alphas).all()
            feed=decoder.ort_session.feeds[-1];labels=feed['pointLabels'][0]
            expected=[2.,3.] if mode in ['upstream-mixed','box-only'] else [1.,-1.] if mode=='points-only' else [1.,2.,3.]
            assert labels==expected
            np.save(dest/'alphas.npy',alphas,allow_pickle=False)
            summaries=[]
            skin=(masks['skin']>=.9)&(masks['hair']<.05)&(masks['accessories']<.05)
            for i,a in enumerate(alphas):
                probe.png(dest/f'alpha-{i}.png',a)
                summaries.append(dict(index=i,score=float(scores[i]),positiveAlpha=float(a[prompt['positive'][1],prompt['positive'][0]]),
                    negativeAlpha=[],componentCorePixels=int(core.sum()),componentCoreRetainedFraction=float((a[core]>=.5).mean()) if core.any() else None,
                    skinCoreMeanAlpha=float(a[skin].mean()),skinCoreRetainedFraction=float((a[skin]>=.5).mean()),foregroundPixels=int((a>=.5).sum())))
            probe.make_panel(dest/'component-candidates-1x.png',rgb,masks,alphas,scores,prompt,crop)
            Image.open(dest/'component-candidates-1x.png').resize(((crop[2]-crop[0])*6*4,(crop[3]-crop[1]+28)*4),Image.Resampling.NEAREST).save(dest/'component-candidates-4x.png')
            row=dict(mode=mode,prompt=prompt,crop=crop,directory=str(dest.relative_to(out)),decodeMs=elapsed,
                officialScoreArgmax=int(np.argmax(scores)),alphas=summaries,alphaSha256=sha(dest/'alphas.npy'),recordedActualDecoderFeed=feed)
            report['rows'].append(row)
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(dict(component=prompt['component'],mode=mode,official=row['officialScoreArgmax'],decodeMs=elapsed)),flush=True)
    assert len(report['rows'])==len(decoder.ort_session.feeds)==10
    for name,digest in script_hashes.items():assert sha(ROOT/name)==digest
    report['hashesUnchangedAfterRun']=True
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

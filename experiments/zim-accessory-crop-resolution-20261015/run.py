"""Resolution/context ablation for two frozen automatic earring prompts.

Each 2x/4x centered context crop is encoded once and decoded with corrected
combined-point and combined-box adapters. All four local raw candidates are
mapped back to the original 1024px frame. No graph/output redetection or
production change; the crop changes both input resolution and context.
"""
from pathlib import Path
import hashlib, importlib.util, json, time
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
probe=load('crop_probe',ROOT/'experiments/hair-accessory-matting-zim-v2-20261015/run.py')
adapter=load('crop_adapter',ROOT/'experiments/zim-accessory-combined-20261015/adapter-snapshot.py')
zim,np,Image=probe.zim,probe.np,probe.Image
sha=zim.sha
WEIGHTS={'encoder.onnx':'6123ea10722dd43a0f8bbb96cd6a0532d8a65e2b1e8339f0cd6593d494807809','decoder.onnx':'a99d14bac3dc28b17809020b1711b1991bd5b7eb18f678c3624ce508748cfa11'}
SCALES=[2,4];MODES=['combined-point','combined-box']

class RecordingSession:
    def __init__(self,session):self.session=session;self.feeds=[]
    def __getattr__(self,name):return getattr(self.session,name)
    def run(self,outputs,inputs,*args,**kw):
        c,l,a=inputs['point_coords'],inputs['point_labels'],inputs['attn_mask']
        assert c.shape[0]==l.shape[0]==a.shape[0]==1 and c.shape[:2]==l.shape
        self.feeds.append(dict(pointCoords=c.tolist(),pointLabels=l.tolist(),pointCoordsDtype=str(c.dtype),pointLabelsDtype=str(l.dtype),attentionShape=list(a.shape),attentionDtype=str(a.dtype),attentionSum=float(a.sum()),attentionSha256=hashlib.sha256(a.tobytes()).hexdigest(),promptSha256=hashlib.sha256(c.tobytes()+l.tobytes()).hexdigest()))
        return self.session.run(outputs,inputs,*args,**kw)

def centered_crop(box,scale,w,h):
    bw,bh=box[2]-box[0]+1,box[3]-box[1]+1
    cw,ch=round(scale*bw),round(scale*bh);cx=(box[0]+box[2]+1)/2;cy=(box[1]+box[3]+1)/2
    x0=int(round(cx-cw/2));y0=int(round(cy-ch/2));x0=max(0,min(w-cw,x0));y0=max(0,min(h-ch,y0))
    return [x0,y0,x0+cw,y0+ch]

def localize(prompt,origin):
    x0,y0=origin;p=dict(prompt);p['box']=[prompt['box'][0]-x0,prompt['box'][1]-y0,prompt['box'][2]-x0,prompt['box'][3]-y0];p['positive']=[prompt['positive'][0]-x0,prompt['positive'][1]-y0];p['negative']=[[x-x0,y-y0] for x,y in prompt['negative']];return p

def panel(path,items,crop):
    x0,y0,x1,y1=crop;cw,ch=x1-x0,y1-y0;out=Image.new('RGB',(cw*len(items),ch+28),'#20242a');draw=ImageDraw.Draw(out)
    for i,(label,image) in enumerate(items):out.paste(image.convert('RGB').crop(crop),(i*cw,28));draw.text((i*cw+3,7),label,fill='white')
    out.save(path);out.resize((out.width*4,out.height*4),Image.Resampling.NEAREST).save(path.with_name(path.stem.replace('1x','4x')+'.png'))

def main():
    out=HERE/'outputs';out.mkdir(exist_ok=False)
    source=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';graph=ROOT/'test/results/user-face-contour-20261014/1024/face-graph.json';prompt_file=ROOT/'experiments/hair-accessory-matting-zim-v2-20261015/outputs/user-face-contour/prompts.json';full_report=ROOT/'experiments/zim-accessory-combined-20261015/outputs/report.json'
    assert sha(source)=='5f15e92b14c44a468d924af5ac985a7cc7dc1c87d8ab235f2c94b2e88b173c73'
    rgb=np.asarray(Image.open(source).convert('RGB'));h,w=rgb.shape[:2];params=json.loads(graph.read_text())['operations'][0]['params'];masks={k:zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()};prompts=[p for p in json.loads(prompt_file.read_text()) if p['kind']=='accessories'];assert len(prompts)==2
    full=json.loads(full_report.read_text());base={(r['prompt']['component'],r['mode']):r for r in full['rows'] if r['mode'] in MODES}
    for k in WEIGHTS:assert sha(ROOT/'.models/zim-20261014'/k)==WEIGHTS[k]
    paths=[Path(__file__),HERE/'adapter-snapshot.py',source,graph,prompt_file,full_report,Path(probe.__file__),Path(zim.__file__)];script_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    Encoder,Decoder,Zim,Predictor,vendor=zim.load_vendor();zim.torch.set_num_threads(3);decoder=Decoder(str(ROOT/'.models/zim-20261014/decoder.onnx'),num_threads=3);decoder.ort_session=RecordingSession(decoder.ort_session);model=Zim(Encoder(str(ROOT/'.models/zim-20261014/encoder.onnx'),num_threads=3),decoder);predictor=Predictor(model)
    report=dict(scope='Fixed source/graph/prompts; local crop resolution/context ablation only. No output redetection, warp, graph or production changes.',source=str(source.relative_to(ROOT)),sourceSha256=sha(source),graph=str(graph.relative_to(ROOT)),graphSha256=sha(graph),size=[w,h],promptFile=str(prompt_file.relative_to(ROOT)),promptFileSha256=sha(prompt_file),fullBaseline=str(full_report.relative_to(ROOT)),fullBaselineSha256=sha(full_report),sourceHasAIWatermark=True,weights=WEIGHTS,vendorCommit=vendor['commit'],scripts=script_hashes,scales=SCALES,modes=MODES,encoderInvocations=0,decoderInvocations=0,batchLimitation='B=1 only',rows=[])
    for prompt in prompts:
      for scale in SCALES:
        crop=centered_crop(prompt['box'],scale,w,h);x0,y0,x1,y1=crop;local=localize(prompt,(x0,y0));crop_rgb=rgb[y0:y1,x0:x1];start=time.perf_counter();predictor.set_image(crop_rgb);encode_ms=(time.perf_counter()-start)*1000;report['encoderInvocations']+=1;folder=out/f'component-{prompt["component"]}-scale-{scale}x';folder.mkdir();np.save(folder/'crop-rgb.npy',crop_rgb);rowbase=dict(component=prompt['component'],scale=scale,crop=crop,cropSize=[x1-x0,y1-y0],promptOriginal=prompt,promptLocal=local,encodeMs=encode_ms,rows=[])
        for mode in MODES:
            model.decoder=adapter.CombinedPromptDecoder(decoder,attention=mode.removeprefix('combined-'),bbox_attention=model.bbox_attn_mask);args={'box':np.asarray(local['box'],dtype=np.float32),'point_coords':np.asarray([local['positive']]+local['negative'],dtype=np.float32),'point_labels':np.asarray([1]+[0]*len(local['negative']),dtype=np.float32),'multimask_output':True};start=time.perf_counter();alphas,scores,_=predictor.predict(**args);elapsed=(time.perf_counter()-start)*1000;report['decoderInvocations']+=1;assert alphas.shape==(4,y1-y0,x1-x0) and np.isfinite(alphas).all();feed=decoder.ort_session.feeds[-1];assert feed['pointLabels'][0]==[1.,2.,3.];dest=folder/mode;dest.mkdir();np.save(dest/'alphas-local.npy',alphas,allow_pickle=False);mapped=np.zeros((4,h,w),dtype=np.float32)
            for i,a in enumerate(alphas):mapped[i,y0:y1,x0:x1]=a;probe.png(dest/f'alpha-{i}-full.png',mapped[i]);probe.png(dest/f'alpha-{i}-local.png',a)
            np.save(dest/'alphas-full.npy',mapped,allow_pickle=False);baseline=base[(prompt['component'],mode)];baseline_alphas=np.load(ROOT/'experiments/zim-accessory-combined-20261015/outputs'/baseline['directory']/'alphas.npy',mmap_mode='r');comp=masks['accessories']>=.9;skin=(masks['skin']>=.9)&(masks['hair']<.05)&(masks['accessories']<.05);summaries=[]
            for i,a in enumerate(alphas):
                fulla=mapped[i];ba=baseline_alphas[i];overlap=np.zeros((h,w),bool);overlap[y0:y1,x0:x1]=True;diff=np.abs(fulla.astype('float64')-ba.astype('float64'));summaries.append(dict(index=i,score=float(scores[i]),positiveAlpha=float(a[local['positive'][1],local['positive'][0]]),componentCoreRetainedFraction=float((fulla[comp]>=.5).mean()),skinCoreRetainedFraction=float((fulla[skin]>=.5).mean()),componentMeanAlpha=float(fulla[comp].mean()),skinMeanAlpha=float(fulla[skin].mean()),sameCropMeanAbsAlphaDiff=float(diff[overlap].mean()),sameCropMaxAbsAlphaDiff=float(diff[overlap].max()),candidateForegroundPixels=int((fulla>=.5).sum())))
            crop_panel=[('Original',Image.fromarray(rgb)),('Full baseline',probe.cutout(rgb,baseline_alphas[int(baseline['officialScoreArgmax'])]))]+[(f'Crop {i} {scores[i]:.3f}',probe.cutout(rgb,mapped[i])) for i in range(4)];panel(dest/'candidates-1x.png',crop_panel,crop)
            item=dict(mode=mode,directory=str(dest.relative_to(out)),decodeMs=elapsed,officialScoreArgmax=int(np.argmax(scores)),alphas=summaries,alphaLocalSha256=sha(dest/'alphas-local.npy'),alphaFullSha256=sha(dest/'alphas-full.npy'),recordedActualDecoderFeed=feed,baselineDirectory=baseline['directory'],baselineOfficialScoreArgmax=baseline['officialScoreArgmax'],transform=dict(fullToCrop='subtract crop origin',cropToFull='paste at crop origin',origin=[x0,y0]));rowbase['rows'].append(item);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(component=prompt['component'],scale=scale,mode=mode,official=item['officialScoreArgmax'],encodeMs=encode_ms,decodeMs=elapsed)),flush=True)
        report['rows'].append(rowbase);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert len(report['rows'])==4 and sum(len(r['rows']) for r in report['rows'])==8 and report['encoderInvocations']==4 and report['decoderInvocations']==8
    for name,digest in script_hashes.items():assert sha(ROOT/name)==digest
    report['hashesUnchangedAfterRun']=True;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

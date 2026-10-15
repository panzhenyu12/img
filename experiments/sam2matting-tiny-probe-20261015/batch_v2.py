"""Same source-only v2 crop-point mask protocol for all three photographs."""
from pathlib import Path
import sys,json,time,hashlib,resource,gc
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE/'deps'),str(HERE/'vendor')]
import numpy as np
from PIL import Image,ImageDraw
import torch
from sam2.build_sam import build_sam2matting
from sam2.sam2matting_image_predictor import SAM2MattingImagePredictor

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def montage(path,arrays,labels,box):
 x0,y0,x1,y1=box;cw=x1-x0;ch=y1-y0;c=Image.new('RGB',(cw*len(arrays),ch+28),'#20242a');d=ImageDraw.Draw(c)
 for i,(a,l) in enumerate(zip(arrays,labels)):c.paste(Image.fromarray(a).crop(box),(i*cw,28));d.text((i*cw+2,8),l,fill='white')
 c.save(path)
def main():
 out=HERE/'outputs-v2-crop-seed';out.mkdir(exist_ok=False);torch.set_num_threads(3);torch.set_num_interop_threads(1)
 wp=ROOT/'.models/sam2matting-20261015/SAM2Matting-SAM2.1Tiny.pt';start=time.perf_counter();model=build_sam2matting('configs/sam2matting-sam2.1tiny.yaml',str(wp),device='cpu');load=time.perf_counter()-start;predictor=SAM2MattingImagePredictor(model)
 report={'status':'running','accepted':False,'vendorCommit':'73dd721d77b56749248aefe5e8824d7f61b9d13c','modelSha256':sha(wp),'torch':torch.__version__,'device':'cpu','dtype':'float32','threads':3,'modelLoadSeconds':load,'protocol':'All three: source-only prompts-v2/crop-point official index2 > .5 binary mask, full source RGB. No semantic clipping, no alpha clamping, no production output.','rows':[]}
 for name in ['user-face-contour','portrait-woman','portrait-hair']:
  sd=ROOT/'experiments/joint-skin-matting-v1-20261015/prompts-v2'/name;meta=json.loads((sd/'report.json').read_text());sp=ROOT/meta['source'];cp=sd/'crop-point/alphas-full.npy';fp=sd/'full-point/alphas-full.npy';zim=np.load(cp)[2];full=np.load(fp)[2];mask=zim>.5;source=Image.open(sp).convert('RGB');arr=np.asarray(source);d=out/name;d.mkdir();Image.fromarray(np.uint8(mask*255)).save(d/'input-mask.png')
  assert sha(sp)==meta['sourceSha256'];assert zim.shape==arr.shape[:2]
  with torch.inference_mode():
   start=time.perf_counter();img=predictor.set_image(source);encode=time.perf_counter()-start
   raw_mask=torch.from_numpy(mask);mask_input=torch.nn.functional.interpolate((raw_mask.float()*20-10)[None,None],size=(256,256),mode='bilinear',align_corners=False)
   start=time.perf_counter();_,alpha,_=predictor.predict(img=img,raw_mask=raw_mask,mask_input=mask_input,multimask_output=False);decode=time.perf_counter()-start
  alpha=np.asarray(alpha).squeeze();assert alpha.shape==zim.shape and np.isfinite(alpha).all();np.save(d/'alpha-raw.npy',alpha);Image.fromarray(np.floor(np.clip(alpha,0,1)*255+.5).astype('uint8')).save(d/'alpha.png')
  green=np.array([120,255,155]);preview=[arr,np.uint8(np.clip(arr*zim[...,None]+green*(1-zim[...,None]),0,255)),np.uint8(np.clip(arr*alpha[...,None]+green*(1-alpha[...,None]),0,255))];box=meta['box'];montage(d/'face-1x.png',preview,['Source','ZIM v2 crop alpha preview','SAM2Matting v2 crop preview'],box)
  if name=='user-face-contour':
   for title,b in [('left',[326,409,397,548]),('right',[620,414,692,551])]:montage(d/(title+'-1x.png'),preview,['Source','ZIM','SAM2'],b)
  diff=np.abs(alpha-zim);consensus=(zim>=.99)&(full>=.99);agreed_background=(zim<.01)&(full<.01)
  row={'name':name,'source':meta['source'],'sourceSha256':sha(sp),'graphSha256':meta['graphSha256'],'seedAlphaSha256':sha(cp),'fullAlphaSha256':sha(fp),'inputMaskSha256':sha(d/'input-mask.png'),'seedCrop':box,'seedBinaryPixels':int(mask.sum()),'v2FullVsCropBinaryDiff':int(((full>.5)!=(zim>.5)).sum()),'encodeSeconds':encode,'decodeSeconds':decode,'peakRSSBytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rawAlphaSha256':sha(d/'alpha-raw.npy'),'alphaMin':float(alpha.min()),'alphaMax':float(alpha.max()),'consensusOpaquePixels':int(consensus.sum()),'consensusOpaqueLostBelow09':int((consensus&(alpha<.9)).sum()),'consensusOpaqueLostBelow05':int((consensus&(alpha<.5)).sum()),'agreedBackgroundLeakedOver005':int((agreed_background&(alpha>.05)).sum()),'alphaChangedOver01':int((diff>.1).sum()),'directory':str(d.relative_to(HERE))}
  if name=='user-face-contour':
   old=np.load(ROOT/'experiments/joint-skin-matting-v1-20261015/outputs/full-point/alphas-full.npy')[2];row['v1FullVsV2CropBinaryDiff']=int(((old>.5)!=mask).sum())
  report['rows'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True);del img,raw_mask,mask_input,alpha;predictor.reset_predictor();gc.collect()
 report['status']='inference_completed';(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()

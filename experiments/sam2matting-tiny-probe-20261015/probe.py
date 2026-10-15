"""Frozen SAM2Matting Tiny image inference; CPU, no TensorRT/compilation.

Only alpha output is evaluated. Green-background fgr previews are never treated
as foreground color estimates. The vendor source and checkpoint stay unchanged.
"""
from pathlib import Path
import sys,json,time,hashlib,resource,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE/'deps'),str(HERE/'vendor')]
import numpy as np
from PIL import Image,ImageDraw
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=HERE/'outputs-cpu';out.mkdir(exist_ok=False)
 report={'vendorCommit':'73dd721d77b56749248aefe5e8824d7f61b9d13c','torch':torch.__version__,'device':'cpu','dtype':'float32','threads':3,'autocast':False,'compile':False,'accepted':False,'status':'loading'}
 try:
  torch.set_num_threads(3);torch.set_num_interop_threads(1)
  from sam2.build_sam import build_sam2matting
  from sam2.sam2matting_image_predictor import SAM2MattingImagePredictor
  sp=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';ap=ROOT/'experiments/joint-skin-matting-v1-20261015/outputs/full-point/alphas-full.npy';wp=ROOT/'.models/sam2matting-20261015/SAM2Matting-SAM2.1Tiny.pt'
  report.update(sourceSha256=sha(sp),alphaInputSha256=sha(ap),weightsSha256=sha(wp),maskProtocol='direct ZIM full-point official index2 > .5; one fixed binary mask; no accessory subtraction or semantic postfilter')
  start=time.perf_counter();model=build_sam2matting('configs/sam2matting-sam2.1tiny.yaml',str(wp),device='cpu');report['modelLoadSeconds']=time.perf_counter()-start;print('modelLoaded',report['modelLoadSeconds'],flush=True)
  predictor=SAM2MattingImagePredictor(model);source=Image.open(sp).convert('RGB');zim=np.load(ap)[2];mask=zim>.5;Image.fromarray((mask*255).astype('uint8')).save(out/'input-mask.png')
  with torch.inference_mode():
   t=time.perf_counter();img=predictor.set_image(source);report['encodeSeconds']=time.perf_counter()-t;print('encoded',report['encodeSeconds'],flush=True)
   raw_mask=torch.from_numpy(mask);mask_input=raw_mask.float()*20-10;mask_input=torch.nn.functional.interpolate(mask_input[None,None],size=(256,256),mode='bilinear',align_corners=False)
   t=time.perf_counter();result=predictor.predict(img=img,raw_mask=raw_mask,mask_input=mask_input,multimask_output=False);report['decodeSeconds']=time.perf_counter()-t
  _,alpha,_=result;alpha=np.asarray(alpha).squeeze();assert alpha.shape==(1024,1024);assert np.isfinite(alpha).all();np.save(out/'alpha-raw.npy',alpha);Image.fromarray(np.floor(np.clip(alpha,0,1)*255+.5).astype('uint8')).save(out/'alpha.png')
  source_arr=np.asarray(source);green=np.array([120,255,155]);images=[source_arr,np.floor(source_arr*zim[...,None]+green*(1-zim[...,None])+.5).astype('uint8'),np.floor(source_arr*alpha[...,None]+green*(1-alpha[...,None])+.5).astype('uint8')]
  for name,crop in [('face',[287,197,730,709]),('left',[326,409,397,548]),('right',[620,414,692,551])]:
   x0,y0,x1,y1=crop;cw=x1-x0;ch=y1-y0;canvas=Image.new('RGB',(cw*3,ch+28),'#20242a');draw=ImageDraw.Draw(canvas)
   for i,(arr,label) in enumerate(zip(images,['Source','ZIM raw alpha preview','SAM2Matting alpha preview'])):canvas.paste(Image.fromarray(arr).crop(crop),(i*cw,28));draw.text((i*cw+2,8),label,fill='white')
   canvas.save(out/(name+'-1x.png'))
  report.update(status='inference_completed',peakRSSBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,alphaMin=float(alpha.min()),alphaMax=float(alpha.max()),alphaSha256=sha(out/'alpha-raw.npy'),visualAssessment='pending original-resolution inspection')
 except Exception as e:
  report.update(status='failed',error=repr(e),traceback=traceback.format_exc());print(report['traceback'],flush=True)
 finally:
  report['peakRSSBytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

"""Independent V37 read-only layer instrumentation; verify identical native PNG."""
from pathlib import Path
import os,json,hashlib,subprocess,shutil
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'qa/v37-trace-inverse'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
binary=ROOT/'.run/feature-eye-v37-trace-target/release/image-render';native=ROOT/'experiments/face-direct-physical-v37-20261015/outputs-delta/user';source=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';frozen=OUT/f'image-render-{sha(binary)}';shutil.copy2(binary,frozen);rows=[]
for amount in [100,-100]:
 d=OUT/str(amount);d.mkdir(exist_ok=False);graph=ROOT/f'experiments/joint-skin-matting-v1-20261015/native-v2-user/{amount}/graph.json'
 p=subprocess.run([str(frozen),'--input',str(source),'--graph',str(graph),'--output',str(d/'candidate.png')],env={**os.environ,'FACE_V37_SOURCE_RESIDUAL':'1','FACE_V37_PHYSICAL_ALPHA':str(native/'physical-alpha-f32le.bin'),'FACE_V37_LAYER_DEBUG_DIR':str(d/'layers'),'FACE_V36_DEBUG_DIR':str(d/'debug'),'FACE_V21_DEBUG_DIR':str(d/'debug')},capture_output=True);(d/'render.log').write_bytes(p.stdout+p.stderr);p.check_returncode()
 image=np.asarray(Image.open(d/'candidate.png').convert('RGBA'));reference=np.asarray(Image.open(native/str(amount)/'candidate.png').convert('RGBA'));mismatch=int(np.any(image!=reference,axis=2).sum());assert mismatch==0,mismatch
 layers=d/'layers';state=json.loads((layers/'state.json').read_text());ox,oy=state['origin'];w,h=state['width'],state['height'];a=np.fromfile(layers/'source-alpha-f32le.bin',dtype='<f4').reshape(h,w);t=np.fromfile(layers/'target-alpha-f32le.bin',dtype='<f4').reshape(h,w);s=np.fromfile(layers/'samples-f32le.bin',dtype='<f4').reshape(h,w,5);c=np.asarray(Image.open(layers/'source-crop.png').convert('RGB')).astype(float);f=np.asarray(Image.open(layers/'foreground.png').convert('RGB')).astype(float);b=np.asarray(Image.open(layers/'background.png').convert('RGB')).astype(float);changed=np.fromfile(layers/'changed.bin',dtype='uint8').reshape(h,w)>0;fixed=np.fromfile(layers/'frozen.bin',dtype='uint8').reshape(h,w)>0;r=c-a[...,None]*f-(1-a[...,None])*b
 # Native branch is sequential f32; reproduce operation precision/rounding.
 cf,ff,bf=c.astype('float32'),f.astype('float32'),b.astype('float32');rf=(cf+s[...,:3])-ff*a[...,None];rf=rf+(a-t)[...,None]*bf;rounded=np.floor(np.clip(rf,0,255)+.5).astype('uint8');actual=image[oy:oy+h,ox:ox+w,:3];replay=int(np.any(rounded[changed]!=actual[changed],axis=1).sum())
 points=[]
 for x,y in [(630,503),(631,501),(652,430),(374,440),(375,450),(376,460),(378,470),(379,480),(643,440),(641,450),(640,460),(639,470),(638,480),(635,490)]:
  xx,yy=x-ox,y-oy;points.append({'xy':[x,y],'sourceRGB':list(map(int,c[yy,xx])),'foregroundRGB':list(map(int,f[yy,xx])),'backgroundRGB':list(map(int,b[yy,xx])),'residualRGBByte':list(map(float,r[yy,xx])),'sourceAlpha':float(a[yy,xx]),'targetAlpha':float(t[yy,xx]),'samplePremultRGB':list(map(float,s[yy,xx,:3])),'changed':bool(changed[yy,xx]),'frozen':bool(fixed[yy,xx]),'outputRGB':list(map(int,actual[yy,xx]))})
 row={'amount':amount,'outputSha256':sha(d/'candidate.png'),'referenceSha256':sha(native/str(amount)/'candidate.png'),'decodedPNGChangedPixels':mismatch,'nativeF32ReplayMismatches':replay,'geometryMatches':sha(d/'debug/flow-geometry.json')==sha(native/str(amount)/'debug/flow-geometry.json'),'points':points};rows.append(row)
 # Show true byte F/B/residual directly, no re-estimation or output detection.
 for name,box in [('left',(355,425,400,530)),('right',(615,425,670,530))]:
  x0,y0,x1,y1=box;sl=np.s_[y0-oy:y1-oy,x0-ox:x1-ox];ims=[c[sl],actual[sl],f[sl],b[sl],np.abs(r[sl])*5];labels=['Source','Output','F layer','B layer','Resid x5'];cw,ch=x1-x0,y1-y0;sheet=Image.new('RGB',(cw*5,ch+24),'#20242a');draw=ImageDraw.Draw(sheet)
  for i,(im,label) in enumerate(zip(ims,labels)):sheet.paste(Image.fromarray(np.floor(np.clip(im,0,255)+.5).astype('uint8')),(cw*i,24));draw.text((cw*i+1,5),label,fill='white')
  sheet.save(d/f'{name}-layers-1x.png');sheet.resize((sheet.width*5,sheet.height*5),Image.Resampling.NEAREST).save(d/f'{name}-layers-5x.png')
report={'scope':'Instrumented original V37 only, source and geometry frozen, explicit FACE_V37_SOURCE_RESIDUAL=1. No production edits.','sourceSha256':sha(source),'beforeSourceSha256':sha(OUT/'face-before.rs'),'instrumentedSourceSha256':sha(ROOT/'experiments/face-direct-physical-v37-20261015/src/face.rs'),'binarySha256':sha(frozen),'rows':rows};(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

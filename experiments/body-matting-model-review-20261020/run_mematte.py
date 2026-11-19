"""One official MEMatte ViT-S mixed-data source-matte comparison, CPU float32.

The full source PNG and pre-existing source trimap are frozen. No prompt,
threshold, scale, geometric field or compositor is changed by this program.
Both the network output and official trimap-clamped output are retained.
"""
from pathlib import Path
import sys,json,time,hashlib,resource,traceback

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
VENDOR=ROOT/'.run/mematte-20261020/vendor'
sys.path[:0]=[str(ROOT/'.run/mematte-20261020-deps'),str(ROOT/'experiments/sam2matting-tiny-probe-20261015/deps'),str(VENDOR)]
import numpy as np
from PIL import Image,ImageDraw
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_prediction(rgb,tri,wp,config_path=None):
    """Shared unmodified ViT-S configuration, one CPU float32 inference."""
    torch.set_num_threads(3);torch.set_num_interop_threads(1)
    from detectron2.config import LazyConfig,instantiate
    cfg=LazyConfig.load(str(config_path or VENDOR/'configs/common/model.py'))
    cfg.model.teacher_backbone=None
    cfg.model.backbone.topk=.25
    cfg.model.backbone.window_block_indexes=[0,1,3,4,6,7,9,10]
    cfg.model.backbone.multi_score=True
    cfg.model.backbone.max_number_token=18500
    t=time.perf_counter();model=instantiate(cfg.model).cpu().eval()
    checkpoint=torch.load(wp,map_location='cpu');state=checkpoint.get('model',checkpoint)
    match=model.load_state_dict(state,strict=True)
    evidence={'missing_keys':list(match.missing_keys),'unexpected_keys':list(match.unexpected_keys),
        'checkpoint_tensors':len(state),'load_seconds':time.perf_counter()-t}
    assert rgb.shape[:2]==tri.shape and set(np.unique(tri))=={0,128,255}
    inputs={'image':torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()/255,
            'trimap':torch.from_numpy(tri.copy())[None,None].float()/255}
    t=time.perf_counter()
    with torch.inference_mode():result,_,_=model(inputs,patch_decoder=True)
    evidence['inference_seconds']=time.perf_counter()-t
    raw=result['phas'].squeeze().cpu().numpy()
    assert raw.shape==tri.shape and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
    official=raw.copy();official[tri==0]=0;official[tri==255]=1
    return raw,official,evidence


def main():
    out=ROOT/'test/results/body-matting-model-review-20261020/mematte-aim500-vits-cpu-v1'
    out.mkdir(parents=True,exist_ok=False)
    sp=ROOT/'test/results/body-parts-v14/fashion-standing/source.png'
    tp=ROOT/'test/results/body-shared-garment-v1-20261020/source-instance-v2/trimap-proposal.png'
    wp=ROOT/'.models/mematte-20261020/MEMatte-AIM500-ViTS.pth'
    report={'status':'initializing','accepted':False,'source':str(sp),'source_sha256':sha(sp),'trimap':str(tp),'trimap_sha256':sha(tp),
        'weights':str(wp),'weights_sha256':sha(wp),'weights_bytes':wp.stat().st_size,
        'official_revision':'3c887e2a517b38f936b97f27d933c5897c6d47a1','torch':torch.__version__,'device':'cpu','dtype':'float32','threads':3,
        'settings':{'variant':'MEMatte ViT-S mixed-data AIM-500 checkpoint','input_resize':None,'max_number_token':18500,'patch_decoder':True,'topk':.25,'multi_score':True,
            'window_block_indexes':[0,1,3,4,6,7,9,10],'input_trimap_encoding':'official 0/255, 128/255, 255/255','autocast':False},
        'network_raw_output':'alpha-raw.npy','official_output':'alpha-official.npy','output_contract':'Official inference clamps network alpha to 0/1 only at the supplied source trimap 0/255 locations; raw alpha retained separately.',
        'source_prompts_modified':False,'source_trimap_modified':False,'other_model_or_scale_trials':False}
    try:
        rgb=np.array(Image.open(sp).convert('RGB'));tri=np.array(Image.open(tp).convert('L'))
        assert rgb.shape[:2]==tri.shape and set(np.unique(tri))=={0,128,255}
        report['source_dimensions']={'width':int(rgb.shape[1]),'height':int(rgb.shape[0])}
        report['trimap_counts']={str(v):int((tri==v).sum()) for v in [0,128,255]}
        raw,official,evidence=official_prediction(rgb,tri,wp);report.update(evidence)
        np.save(out/'alpha-raw.npy',raw);np.save(out/'alpha-official.npy',official)
        Image.fromarray(np.floor(raw*255+.5).astype('uint8')).save(out/'alpha-raw.png')
        Image.fromarray(np.floor(official*255+.5).astype('uint8')).save(out/'alpha-official.png')
        zim=np.load(ROOT/'test/results/body-shared-garment-v1-20261020/source-instance-v2/alphas-raw.npy')[2]
        report['alpha_raw_range']=[float(raw.min()),float(raw.max())]
        report['known_trimap_network_disagreement']={'foreground_below_point9':int(((tri==255)&(raw<.9)).sum()),'background_above_point1':int(((tri==0)&(raw>.1)).sum())}
        report['clamped_pixels']=int((official!=raw).sum())
        report['probes']=[]
        for x,y in [(350,462),(351,463),(352,464),(360,519)]:
            report['probes'].append({'xy':[x,y],'rgb':rgb[y,x].tolist(),'trimap':int(tri[y,x]),'zim_alpha':float(zim[y,x]),'raw_alpha':float(raw[y,x]),'official_alpha':float(official[y,x])})
        report['unknown_band_regions']={}
        for label,box in [('left-waist',(125,385,210,535)),('right-waist',(325,390,410,535)),('left-sleeve-junction',(100,350,240,445)),('right-sleeve',(340,350,430,535))]:
            x0,y0,x1,y1=box;m=tri[y0:y1,x0:x1]==128;a=official[y0:y1,x0:x1];b=zim[y0:y1,x0:x1]
            report['unknown_band_regions'][label]={'box':box,'pixels':int(m.sum()),'mean_abs_alpha_difference':float(abs(a-b)[m].mean()),'signed_alpha_area_change':float((a-b)[m].sum()),
                'new_above_half':int(((b<.5)&(a>=.5)&m).sum()),'new_below_half':int(((b>=.5)&(a<.5)&m).sum())}
            pw,ph=(x1-x0)*3,(y1-y0)*3;sheet=Image.new('RGB',(pw*3,ph+24),'#191919');d=ImageDraw.Draw(sheet)
            green=np.array([120,255,155]);zimpreview=np.floor(rgb*zim[...,None]+green*(1-zim[...,None])+.5).astype('uint8');mempreview=np.floor(rgb*official[...,None]+green*(1-official[...,None])+.5).astype('uint8')
            for col,(title,im) in enumerate([('Source',rgb),('ZIM alpha preview',zimpreview),('MEMatte official alpha preview',mempreview)]):
                sheet.paste(Image.fromarray(im).crop(box).resize((pw,ph),Image.Resampling.NEAREST),(pw*col,24));d.text((pw*col+3,5),title,fill='white')
            sheet.save(out/(label+'-3x.png'))
        report['status']='completed_unreviewed'
        report['outputs']={name:sha(out/name) for name in ['alpha-raw.npy','alpha-official.npy']}
        report['limitations']=['Existing source trimap is a model-derived proposal, not alpha ground truth.','A higher predicted alpha or smaller compositor residual does not establish true foreground color or natural warp.','Green previews multiply observed RGB by alpha; they are not recovered foreground RGB.','Single source photograph, no generalization claim.']
        assert sha(sp)==report['source_sha256'] and sha(tp)==report['trimap_sha256']
    except Exception as error:
        report['status']='failed';report['error']=repr(error);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
    finally:
        report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if report['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()

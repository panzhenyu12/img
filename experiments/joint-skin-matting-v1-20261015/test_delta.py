"""Independent behavior audit of residual-preserving differential composition.

Runs known-layer synthetic counterexamples and exact saved photographic replay.
Only writes qa/delta-audit. Never edits a candidate, geometry, or model output.
Acceptance failures are real failures, not expectedFailure or score substitutes.
"""
from pathlib import Path
import sys,json,hashlib,unittest
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'qa/delta-audit'
sys.path.insert(0,str(ROOT/'experiments/hair-skin-decomposition-v1-20261015'))
from compose_real import sample
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def image_save(path,a):
    Image.fromarray(np.floor(np.clip(a,0,1)*255+.5).astype('uint8')).save(path)


def layers(alpha,foreground,background,jewelry,metal):
    return jewelry[...,None]*metal+(1-jewelry[...,None])*(alpha[...,None]*foreground+(1-alpha[...,None])*background)


def move(alpha,foreground,background,jewelry,metal,xy):
    # Oracle uses independently known full layers. No residual or source image.
    pm=sample(alpha[...,None]*foreground,xy);am=sample(alpha,xy)[...,0]
    return jewelry[...,None]*metal+(1-jewelry[...,None])*(pm+(1-am[...,None])*background)


def differential(source,alpha,foreground,background,jewelry,xy,freeze_near_opaque=False):
    pm=sample(alpha[...,None]*foreground,xy);am=sample(alpha,xy)[...,0]
    result=source+(1-jewelry[...,None])*(pm-alpha[...,None]*foreground+(alpha-am)[...,None]*background)
    if freeze_near_opaque:result=np.where((jewelry>=.98)[...,None],source,result)
    return result


def solve_source_colors(source,alpha,background,jewelry,skin_prior,metal_prior):
    # Same source-color inverse problem as candidate. Its clipping is part of
    # the tested algorithm; synthetic oracle never uses these estimates.
    beneath=alpha[...,None]*skin_prior+(1-alpha[...,None])*background
    metal=np.clip((source-(1-jewelry[...,None])*beneath)/np.maximum(jewelry[...,None],1e-6),0,1)
    metal[jewelry<.01]=metal_prior[jewelry<.01]
    metal[jewelry>.98]=source[jewelry>.98]
    clean=np.clip((source-jewelry[...,None]*metal)/np.maximum(1-jewelry[...,None],1e-6),0,1)
    clean[jewelry>.98]=beneath[jewelry>.98]
    fg=np.clip((clean-(1-alpha[...,None])*background)/np.maximum(alpha[...,None],1e-6),0,1)
    fg[alpha<.01]=skin_prior[alpha<.01]
    return fg,metal


def panel(path,images,labels,scale=3):
    h,w=images[0].shape[:2];im=Image.new('RGB',(len(images)*w,h+24),'#20242a');d=ImageDraw.Draw(im)
    for i,(a,s) in enumerate(zip(images,labels)):
        im.paste(Image.fromarray(np.floor(np.clip(a,0,1)*255+.5).astype('uint8')),(i*w,24));d.text((i*w+3,5),s,fill='white')
    im.save(path);im.resize((im.width*scale,im.height*scale),Image.Resampling.NEAREST).save(path.with_name(path.stem+'-3x.png'))


def synthetic_cases():
    h,w=112,168;yy,xx=np.indices((h,w));edge=91+3*np.sin(yy/18);alpha=np.clip((edge-xx+.75)/1.5,0,1)
    fg=np.zeros((h,w,3))+np.array([.78,.52,.39]);fg+=.012*np.sin(yy[...,None]/8)*np.array([1,.7,.5])
    bg=np.zeros_like(fg)+np.array([.035,.045,.055]);metal=np.zeros_like(fg)+np.array([.93,.9,.85]);jzero=np.zeros((h,w));xy=np.stack([xx+12,yy],axis=-1).astype(float);identity=np.stack([xx,yy],axis=-1).astype(float)
    edge_error=(alpha>.01)&(alpha<.99);under=np.maximum(0,alpha-.02*edge_error)
    valid=(yy>=8)&(yy<h-8)&(xx>=8)&(xx<w-20)
    old_exposed=(alpha>.01)&(sample(alpha,xy)[...,0]<1e-8)&valid
    source=layers(alpha,fg,bg,jzero,metal);oracle=move(alpha,fg,bg,jzero,metal,xy)
    cases=[]
    def record(name,src,true_out,estimate,source_model,ahat,jhat,required,reason,region=None):
        diff=np.abs(estimate-true_out)*255;reg=valid if region is None else valid&region
        old=old_exposed&reg;res=(src-source_model)*255
        row={'name':name,'criterion':required,'reason':reason,'maxErrorByte':float(diff[reg].max()),'meanErrorByte':float(diff[reg].mean()),'oldExposedMaxErrorByte':float(diff[old].max()) if old.any() else 0,'oldExposedMeanErrorByte':float(diff[old].mean()) if old.any() else 0,'oldExposedPixelsWithErrorOver1Byte':int((old&(diff.max(axis=2)>1)).sum()),'sourceModelMaxResidualByte':float(np.abs(res)[reg].max()),'maxAlphaError':float(np.abs(ahat-alpha).max()),'jewelryRange':[float(jhat.min()),float(jhat.max())]}
        row['passesAcceptance']=row['maxErrorByte']<=required
        folder=OUT/'synthetic'/name;folder.mkdir(parents=True,exist_ok=True)
        image_save(folder/'source.png',src);image_save(folder/'oracle.png',true_out);image_save(folder/'candidate.png',estimate);image_save(folder/'error-amplified-x10.png',np.abs(estimate-true_out)*10)
        panel(folder/'comparison.png',[src,true_out,estimate,np.abs(estimate-true_out)*10],['Source','Known-layer oracle','Residual delta','Absolute error x10'])
        np.savez_compressed(folder/'known-layers.npz',trueAlpha=alpha,estimatedAlpha=ahat,trueOutput=true_out,candidate=estimate,source=src,sourceModel=source_model,valid=reg,oldExposed=old)
        cases.append(row)
        return row
    result=differential(source,alpha,fg,bg,jzero,xy)
    record('exact_layers',source,oracle,result,source,alpha,jzero,1e-8,'Correct layers must equal an independently rendered three-layer oracle.')
    # Source identity is not evidence that unknown layer ownership is correct.
    wrongfg=np.clip(fg+.12,0,1);wrongbg=np.clip(bg+.2,0,1)
    idresult=differential(source,under,wrongfg,wrongbg,jzero,identity)
    record('zero_motion_wrong_layers',source,source,idresult,layers(under,wrongfg,wrongbg,jzero,metal),under,jzero,1e-8,'Zero displacement must preserve original RGB even when source layers are deliberately wrong.')
    result=differential(source,under,fg,bg,jzero,xy)
    record('independent_colors_alpha_low_by_002',source,oracle,result,layers(under,fg,bg,jzero,metal),under,jzero,1.0,'A 0.02 source-edge alpha error with independent correct RGB must not leave a visible stationary old-skin line.')
    decoded,decodedmetal=solve_source_colors(source,under,bg,jzero,fg,metal)
    result=differential(source,under,decoded,bg,jzero,xy)
    record('inverse_colors_alpha_error_no_clipping',source,oracle,result,layers(under,decoded,bg,jzero,decodedmetal),under,jzero,1e-8,'Control: exact color inversion with constant background can cancel an alpha error; alpha error does not always imply ghosts.')
    bright=fg.copy();bright[...,0]=1
    srcbright=layers(alpha,bright,bg,jzero,metal);brightoracle=move(alpha,bright,bg,jzero,metal,xy)
    decoded,decodedmetal=solve_source_colors(srcbright,under,bg,jzero,bright,metal)
    result=differential(srcbright,under,decoded,bg,jzero,xy)
    record('inverse_colors_alpha_error_clipped_highlight',srcbright,brightoracle,result,layers(under,decoded,bg,jzero,decodedmetal),under,jzero,1.0,'Actual clipped color inversion plus only 0.02 alpha error must not leave an old high-light skin line.')
    # A thin stationary metal ring crosses both the source and moved edge.
    radial=np.sqrt(((xx-88)/17)**2+((yy-56)/29)**2);ring=np.clip(1.8-np.abs(radial-1)*13,0,1)
    jewelry=ring*.75;jewelry[(abs(xx-87)<=2)&(yy>29)&(yy<85)]=1
    srcmetal=layers(alpha,fg,bg,jewelry,metal);metaloracle=move(alpha,fg,bg,jewelry,metal,xy)
    result=differential(srcmetal,alpha,fg,bg,jewelry,xy)
    record('correct_soft_and_opaque_metal',srcmetal,metaloracle,result,srcmetal,alpha,jewelry,1e-8,'Correct translucent metal must reveal changed backing; truly opaque metal must stay exact.')
    opaque=jewelry==1;opaque_max=float(np.abs(result-srcmetal)[opaque].max()*255)
    assert opaque_max<1e-8
    # The script uses .98 as opaque although a known physical .985 still has
    # observable background contribution. This is a quality counterexample.
    almost=np.zeros((h,w));almost[(xx>78)&(xx<97)&(yy>20)&(yy<92)]=.985
    srcalmost=layers(alpha,fg,bg,almost,metal);oraclealmost=move(alpha,fg,bg,almost,metal,xy)
    result=differential(srcalmost,alpha,fg,bg,almost,xy,freeze_near_opaque=True)
    record('near_opaque_metal_frozen_at_098',srcalmost,oraclealmost,result,srcalmost,alpha,almost,1.0,'A known 0.985 physical metal alpha must not be treated as exactly opaque when the skin below moves.',region=almost>0)
    # Conversely, an actually opaque object estimated at .97 is not protected
    # by the hard threshold. Source residual preservation does not fix it.
    jtrue=np.zeros((h,w));jtrue[(xx>78)&(xx<97)&(yy>20)&(yy<92)]=1;jhat=jtrue*.97
    srcopaque=layers(alpha,fg,bg,jtrue,metal);oracleopaque=move(alpha,fg,bg,jtrue,metal,xy)
    decoded,decodedmetal=solve_source_colors(srcopaque,alpha,bg,jhat,fg,metal)
    result=differential(srcopaque,alpha,decoded,bg,jhat,xy,freeze_near_opaque=True)
    record('opaque_metal_alpha_low_by_003',srcopaque,oracleopaque,result,layers(alpha,decoded,bg,jhat,decodedmetal),alpha,jhat,1.0,'Known opaque metal must stay byte-exact; a small jewelry alpha error must not move skin color into it.',region=jtrue==1)
    return cases


def real_replay():
    candidate=HERE/'composite-delta-v2';source_path=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png';source=np.asarray(Image.open(source_path).convert('RGBA'));rgb=source[...,:3].astype(float)/255
    z=np.load(candidate/'source-layers.npz');data={k:z[k] for k in z.files};a,f,b,j,m,local=[data[k] for k in ['skinAlpha','skin','background','jewelry','metal','local']]
    fixed=np.asarray(Image.open(candidate/'fixed.png'))>0;weight=np.clip(distance_transform_edt(local)/4,0,1);weight=weight*weight*(3-2*weight)
    rows=[];dest=OUT/'real';dest.mkdir(parents=True,exist_ok=True)
    for amount in [100,-100]:
        mat=ROOT/f'experiments/face-material-fill-v1-20261015/user-face-contour-{amount}-full';layersdir=mat/'layers';state=json.loads((layersdir/'state.json').read_text());ox,oy=state['origin'];cw,ch=state['width'],state['height'];sl=np.s_[oy:oy+ch,ox:ox+cw]
        xy=np.fromfile(layersdir/'inverse-source-xy-f32le.bin',dtype='<f4').reshape(ch,cw,2);physical=np.fromfile(layersdir/'source-alpha-f32le.bin',dtype='<f4').reshape(ch,cw).astype(float);v36fg=np.asarray(Image.open(layersdir/'foreground.png').convert('RGB')).astype(float)/255
        blendalpha=physical*(1-weight[sl])+a[sl]*weight[sl];blendfg=v36fg*(1-weight[sl][...,None])+f[sl]*weight[sl][...,None]
        result=differential(rgb[sl],blendalpha,blendfg,b[sl],j[sl],xy)
        source_model=layers(blendalpha,blendfg,b[sl],j[sl],m[sl]);target_model=move(blendalpha,blendfg,b[sl],j[sl],m[sl],xy);residual=rgb[sl]-source_model
        transport=sample(weight[sl],xy)[...,0];write=np.maximum(weight[sl],transport);write[fixed[sl]|(j[sl]>=.98)]=0
        base=np.asarray(Image.open(mat/'candidate.png').convert('RGBA'));replay=base.copy();patch=base[sl][...,:3].astype(float)/255;patch=patch*(1-write[...,None])+result*write[...,None];replay[sl][...,:3]=np.floor(np.clip(patch,0,1)*255+.5).astype('uint8')
        actual=np.asarray(Image.open(candidate/str(amount)/'candidate.png').convert('RGBA'))
        mismatch=int(np.any(replay!=actual,axis=2).sum());assert mismatch==0,mismatch
        identity_error=float(np.max(np.abs(result-target_model-residual))*255);assert identity_error<1e-8,identity_error
        tested=write>0;applied=np.abs(residual)*write[...,None]*255;peak=applied.max(axis=2);coords=np.argwhere(tested);ranking=np.argsort(peak[tested])[-12:][::-1];top=[]
        for idx in ranking:
            yy,xx=coords[idx];top.append({'xy':[int(xx+ox),int(yy+oy)],'appliedResidualRGBByte':list(map(float,residual[yy,xx]*write[yy,xx]*255)),'sourceRGB':list(map(int,source[yy+oy,xx+ox,:3])),'directSkinAlpha':float(a[yy+oy,xx+ox]),'blendSkinAlpha':float(blendalpha[yy,xx]),'jewelryAlpha':float(j[yy+oy,xx+ox]),'writeWeight':float(write[yy,xx]),'sourceMapXY':[float(xy[yy,xx,0]+ox),float(xy[yy,xx,1]+oy)]})
        stats={'amount':amount,'savedCandidateSha256':sha(candidate/str(amount)/'candidate.png'),'decodedReplayMismatchPixels':mismatch,'residualIdentityMaxErrorByte':identity_error,'writePixels':int(tested.sum()),'appliedStationaryResidualMaxByte':float(peak.max()),'appliedStationaryResidualPixelsOver1Byte':int((peak>1).sum()),'appliedStationaryResidualPixelsOver8Byte':int((peak>8).sum()),'topResidualLocations':top}
        residual_img=np.zeros_like(rgb);residual_img[sl]=np.abs(residual)*write[...,None]*5;image_save(dest/f'{amount}-absolute-stationary-residual-x5.png',residual_img)
        for side,box in [('left',(320,400,420,550)),('right',(590,400,700,550))]:
            x0,y0,x1,y1=box;s=np.s_[y0:y1,x0:x1];panel(dest/f'{amount}-{side}.png',[rgb[s],actual[s][...,:3]/255,residual_img[s]],['Source','Delta v2','Stationary residual x5'])
        rows.append(stats)
    return {'sourceSha256':sha(source_path),'candidateScriptSha256':sha(HERE/'compose_delta.py'),'sourceLayersSha256':sha(candidate/'source-layers.npz'),'rows':rows,'interpretation':'These are stationary residual magnitudes, not ground-truth errors. Nonzero residual may belong to metal, skin, or background; source reconstruction alone cannot decide ownership.'}


class DeltaAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True,exist_ok=True);cls.cases=synthetic_cases();cls.lookup={r['name']:r for r in cls.cases};cls.real=real_replay();cls.report={'algorithmAccepted':all(r['passesAcceptance'] for r in cls.cases),'scope':'Independent known-layer synthetic oracle + exact real candidate replay, no photo quality score. Failures are not suppressed.','scriptSha256':sha(Path(__file__)),'synthetic':cls.cases,'real':cls.real};(OUT/'report.json').write_text(json.dumps(cls.report,indent=2)+'\n')
    def check(self,name):
        r=self.lookup[name];self.assertLessEqual(r['maxErrorByte'],r['criterion'],f"{name}: {r['reason']} Measured max={r['maxErrorByte']:.6f}, old-exposed max={r['oldExposedMaxErrorByte']:.6f} byte")
    def test_exact_layers_match_oracle(self):self.check('exact_layers')
    def test_zero_motion_exact_with_wrong_model(self):self.check('zero_motion_wrong_layers')
    def test_independent_color_alpha_error_has_no_old_skin_line(self):self.check('independent_colors_alpha_low_by_002')
    def test_no_clip_inverse_control(self):self.check('inverse_colors_alpha_error_no_clipping')
    def test_clipped_inverse_has_no_old_highlight_line(self):self.check('inverse_colors_alpha_error_clipped_highlight')
    def test_soft_and_opaque_metal_with_correct_layers(self):self.check('correct_soft_and_opaque_metal')
    def test_near_opaque_metal_must_reveal_backing(self):self.check('near_opaque_metal_frozen_at_098')
    def test_small_metal_alpha_error_preserves_true_opaque_metal(self):self.check('opaque_metal_alpha_low_by_003')
    def test_saved_candidate_replay(self):
        for r in self.real['rows']:self.assertEqual(r['decodedReplayMismatchPixels'],0)


if __name__=='__main__':unittest.main(verbosity=2)

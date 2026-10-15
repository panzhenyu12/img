"""Hypothesis B, isolated clamped anatomical boundary experiment.

No source-alpha, source F, backdrop, semantic masks, or production file changes.
The extra map is a compact C2 horizontal post-deformation, whose side boundary
starts with zero displacement and zero derivatives at observed earlobe contact.
It deliberately tests the price in achieved width and the remaining appearance.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from replay_user_face_contour import raster_coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
MAT=ROOT/'experiments/face-material-fill-v1-20261015'
V36=ROOT/'experiments/face-source-curve-v36-20261015/outputs'
SOURCE=ROOT/'test/results/user-face-contour-20261014/1024/canonical-source.png'

def bilinear(a,x,y):
    h,w=a.shape[:2]; x0=np.floor(x).astype(int);y0=np.floor(y).astype(int); tx=x-x0;ty=y-y0
    out=np.zeros(x.shape+a.shape[2:],np.float64)
    for dx,dy,weight in [(0,0,(1-tx)*(1-ty)),(1,0,tx*(1-ty)),(0,1,(1-tx)*ty),(1,1,tx*ty)]:
        xx=x0+dx;yy=y0+dy;valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        out+=a[np.clip(yy,0,h-1),np.clip(xx,0,w-1)]*(weight*valid)[(...,)+(None,)*(a.ndim-2)]
    return out

def basis(t):
    r=np.minimum(np.abs(t),1);return (1-r)**4*(4*r+1)
def smooth5(t):
    t=np.clip(t,0,1);return t*t*t*(10+t*(-15+6*t))
def forward(inv,p):
    # Solve inverse(destination)=source for measured source boundary probes.
    q=p.copy()
    for _ in range(24):
        error=p-bilinear(inv,q[...,0],q[...,1]);q+=error*.85
    residual=np.linalg.norm(bilinear(inv,q[...,0],q[...,1])-p,axis=-1)
    return q,residual

def make_baseline(amount):
    p=MAT/f'user-face-contour-{amount}-full'
    if p.exists():return p
    p=HERE/'baseline'/str(amount)
    if p.exists():return p
    p.mkdir(parents=True,exist_ok=False)
    g=V36/f'user-face-contour-{amount}/graph.json';(p/'graph.json').write_bytes(g.read_bytes())
    provenance=json.loads((MAT/'user-face-contour-100-full/provenance.json').read_text());binary=Path(provenance['binary'])
    assert sha(binary)==provenance['binarySha256']
    run=subprocess.run([str(binary),'--input',str(SOURCE),'--graph',str(p/'graph.json'),'--output',str(p/'candidate.png')],env={**os.environ,'FACE_LAYERS_DEBUG_DIR':str(p/'layers')},capture_output=True)
    (p/'render.log').write_bytes(run.stdout+run.stderr);run.check_returncode()
    assert np.array_equal(np.asarray(Image.open(p/'candidate.png')),np.asarray(Image.open(V36/f'user-face-contour-{amount}/candidate.png')))
    return p

def panel(images,labels,path,box=None,scale=1):
    ims=[im.crop(box) if box else im for im in images];w,h=ims[0].size
    out=Image.new('RGB',(w*len(ims)*scale,h*scale+24),'#16191e');dr=ImageDraw.Draw(out)
    for i,(im,label) in enumerate(zip(ims,labels)):
        out.paste(im.convert('RGB').resize((w*scale,h*scale)),(i*w*scale,24));dr.text((i*w*scale+5,6),label,fill='white')
    out.save(path)

def curvature(x):
    slope=np.gradient(x);return np.gradient(slope)/(1+slope*slope)**1.5

def main():
    src=np.asarray(Image.open(SOURCE).convert('RGBA'));h,w=src.shape[:2]
    graph=json.loads((V36/'user-face-contour-100/graph.json').read_text());masks=graph['operations'][0]['params']['semanticMasks']
    semantics={k:raster_coverage(masks[k],w,h) for k in ['ears','accessories','hair']}
    geo=json.loads((V36/'user-face-contour-100/debug/flow-geometry.json').read_text())
    lm=np.array([[p['x']*w,p['y']*h] for p in graph['operations'][0]['params']['landmarks']])
    face_width=float(lm[:,0].max()-lm[:,0].min());face_height=float(lm[:,1].max()-lm[:,1].min());mid=float(lm[:,0].mean());jaw=float(lm[:,1].max())
    # V36's source-alpha row probes are frozen observed isoline samples, not
    # semantic contour vertices. Their valid face portion stops above the chin.
    widths=json.loads((V36/'user-face-contour-100/debug/actual-source-width.json').read_text())['rows']
    yy=np.arange(h,dtype=float);side_source=[];nodes=[]
    for side in range(2):
        ps=np.array([r['points'][side]['source'] for r in widths if r['points'][side]['valid']]);order=np.argsort(ps[:,1]);ps=ps[order]
        sx=np.interp(yy,ps[:,1],ps[:,0]);side_source.append(sx)
        ear=semantics['ears']>=.5
        xx=np.arange(w)[None,:];side_mask=(xx<mid) if side==0 else (xx>=mid)
        near=np.abs(xx-sx[:,None])<=face_width*.065
        hits=np.flatnonzero(np.any(ear&side_mask&near,axis=1))
        if len(hits)<4:raise RuntimeError('abstain: unsupported ear-face contact')
        root=float(hits.min());lobe=float(hits.max());ix=int(lobe)
        nodes.append({'side':['left','right'][side],'root':[float(sx[int(root)]),root],'lobe':[float(sx[ix]),lobe],
            'contactRows':int(len(hits)),'confidence':'semantic contact only; no anatomical ground truth','jawStopY':jaw})
    report={'scope':'Hypothesis B: geometry-only post-deformation with clamped earlobe attachment. All source layers/background frozen. Not a production fix.',
        'sourceSha256':sha(SOURCE),'graphSha256':sha(V36/'user-face-contour-100/graph.json'),'nodes':nodes,
        'accepted':False,'blackWedgeMetric':'proxy: new luminance < 50 where source > 65 near earlobe; not semantic ground truth','rows':[]}
    for amount in [30,-30,100,-100]:
        mat=make_baseline(amount);layers=mat/'layers';state=json.loads((layers/'state.json').read_text());ox,oy=state['origin'];cw,ch=state['width'],state['height'];cs=np.s_[oy:oy+ch,ox:ox+cw]
        inv=np.fromfile(layers/'inverse-source-xy-f32le.bin','<f4').reshape(ch,cw,2).astype(float)
        alpha=np.fromfile(layers/'source-alpha-f32le.bin','<f4').reshape(ch,cw)
        fg=np.asarray(Image.open(layers/'foreground.png').convert('RGBA')).astype(float);bg=np.asarray(Image.open(layers/'background.png').convert('RGBA')).astype(float)
        frozen=np.fromfile(layers/'frozen-u8.bin',np.uint8).reshape(ch,cw)>0
        baseline=np.asarray(Image.open(mat/'candidate.png').convert('RGBA')); Y,X=np.mgrid[:ch,:cw].astype(float);gy=Y+oy;gx=X+ox
        curves=[]
        for sx in side_source:
            p=np.stack([sx-ox,yy-oy],axis=-1);q,res=forward(inv,p);q+=np.array([ox,oy]);valid=(yy>oy+4)&(yy<oy+ch-4)&(res<.1)
            # Forward y drift is retained; row interpolation expresses x(y).
            b=np.interp(yy,q[valid,1],q[valid,0]);curves.append((b,float(res[(yy>min(n['root'][1] for n in nodes))&(yy<jaw)].max())))
        # Independent widths test includes all motion-relevant cheek/jaw rows,
        # preventing a chosen favorable jaw row from hiding lost reshaping.
        for frac in [.08,.12,.16]:
            dest=HERE/f'clamp-{frac:.2f}'/str(amount);dest.mkdir(parents=True,exist_ok=True)
            corrections=[];curve_rows=[];roi=np.zeros((h,w),bool);desired=[]
            for si,(b,maxres) in enumerate(curves):
                s=side_source[si];y0=nodes[si]['lobe'][1];span=face_height*frac
                g=smooth5((yy-y0)/span); d=(b-s)*g;target=s+d;correction=target-b
                # Below the lower jaw and above observed root: unchanged.
                active=(yy>=nodes[si]['root'][1])&(yy<=jaw-face_height*.04)
                correction=np.where(active,correction,0);target=b+correction;desired.append(target)
                corrections.append(correction)
                region=(yy>=y0)&(yy<y0+span*1.4)
                measurable=(yy>=y0)&(yy<=jaw-face_height*.08)&(np.abs(b-s)>.25)
                k0=curvature(s);kb=curvature(b);kc=curvature(target)
                errb=kb-k0;errc=kc-k0
                ratio=float(np.abs(target-s)[measurable].sum()/max(np.abs(b-s)[measurable].sum(),1e-8))
                curve_rows.append({'side':nodes[si]['side'],'widthDisplacementRatio':ratio,'attachmentSpanPx':span,'maxInverseSolveResidualPx':maxres,
                    'baselineAddedCurvatureRms':float(np.sqrt(np.mean(errb[region]**2))),'candidateAddedCurvatureRms':float(np.sqrt(np.mean(errc[region]**2))),
                    'baselineMaxAbsCurvature':float(np.abs(kb[region]).max()),'candidateMaxAbsCurvature':float(np.abs(kc[region]).max()),
                    'baselineLobeDisplacementPx':float(b[int(y0)]-s[int(y0)]),'candidateLobeDisplacementPx':float(target[int(y0)]-s[int(y0)])})
                roi|=(np.arange(h)[:,None]>=y0-5)&(np.arange(h)[:,None]<=y0+span*1.4)&(np.abs(np.arange(w)[None,:]-s[:,None])<face_width*.12)
            # Invert a compact horizontal C2 post-map, then compose with frozen
            # V36 inverse. This changes one geometry authority without new fills.
            oldx=X.copy();radius=face_width*.30
            for _ in range(18):
                delta=np.zeros_like(X)
                for si,(b,_) in enumerate(curves):
                    delta+=corrections[si][gy.astype(int)]*basis((oldx+ox-b[gy.astype(int)])/radius)
                oldx=X-delta
            c_inv=bilinear(inv,oldx,Y)
            premul=np.concatenate([fg[...,:3]*(fg[...,3:4]/255)*alpha[...,None],fg[...,3:4]/255*alpha[...,None]],axis=2)
            pixel=bilinear(premul,c_inv[...,0],c_inv[...,1]);membership=bilinear(alpha,c_inv[...,0],c_inv[...,1])
            bw=bg[...,3]/255*(1-membership);a=pixel[...,3]+bw
            rgb=(pixel[...,:3]+bg[...,:3]*bw[...,None])/np.maximum(a[...,None],1e-8)
            computed=np.clip(np.floor(np.concatenate([rgb,a[...,None]*255],axis=2)+.5),0,255).astype(np.uint8)
            # No unrelated re-rounding changes: only where the new map differs.
            altered=(np.linalg.norm(c_inv-inv,axis=2)>1e-4)&~frozen&((alpha>1e-6)|(membership>1e-6))
            out=baseline.copy();crop=out[cs];crop[altered]=computed[altered]
            full_changed=np.any(out!=baseline,axis=2);source_changed=np.any(out!=src,axis=2)
            lum=lambda a:np.dot(a[...,:3],[.2126,.7152,.0722])
            wedge_b=(lum(baseline)<50)&(lum(src)>65)&roi;wedge_c=(lum(out)<50)&(lum(src)>65)&roi
            core=(semantics['accessories']>=.95);soft=(semantics['accessories']>1/255)&~core
            # Jacobian finite differences of complete inverse, and postmap monotonicity.
            dx=np.gradient(c_inv,axis=1);dy=np.gradient(c_inv,axis=0);jac=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
            postjac=np.gradient(oldx,axis=1)
            Image.fromarray(out).save(dest/'candidate.png');Image.fromarray((roi*255).astype('uint8')).save(dest/'ear-roi.png')
            c_inv.astype('<f4').tofile(dest/'inverse-source-xy-f32le.bin')
            panel([Image.fromarray(src),Image.fromarray(baseline),Image.fromarray(out)],['Source',f'V36 {amount}',f'Clamped {amount} / {frac}'],dest/'face-1x.png',tuple(map(int,[mid-face_width*.75,lm[:,1].min()-face_height*.2,mid+face_width*.75,jaw+face_height*.12])))
            for si,node in enumerate(nodes):
                px,py=node['lobe'];box=tuple(map(int,[px-face_width*.16,py-face_height*.10,px+face_width*.16,py+face_height*.30]))
                panel([Image.fromarray(src),Image.fromarray(baseline),Image.fromarray(out)],['Source','V36','Clamped'],dest/(node['side']+'-2x.png'),box,2)
            row={'amount':amount,'spanFraction':frac,'sides':curve_rows,'minimumWidthRatio':min(r['widthDisplacementRatio'] for r in curve_rows),
                'baselineBlackWedgeProxyPixels':int(wedge_b.sum()),'candidateBlackWedgeProxyPixels':int(wedge_c.sum()),
                'newAccessoryCoreChangesVsBaseline':int((full_changed&core).sum()),'newAccessorySoftChangesVsBaseline':int((full_changed&soft).sum()),
                'accessoryCoreChangesVsSource':int((source_changed&core).sum()),'alphaChangedPixelsVsSource':int((out[...,3]!=src[...,3]).sum()),
                'sourceAlphaSha256':sha(layers/'source-alpha-f32le.bin'),'backgroundSha256':sha(layers/'background.png'),'foregroundSha256':sha(layers/'foreground.png'),
                'sourceAlphaReusedUnmodified':True,'backgroundReusedUnmodified':True,'changedPixelsVsBaseline':int(full_changed.sum()),
                'inverseMinJacobianInterior':float(jac[2:-2,2:-2].min()),'inverseFoldPixels':int((jac[2:-2,2:-2]<=0).sum()),
                'postmapInverseMinDx':float(postjac.min()),'outputSha256':sha(dest/'candidate.png'),'directory':str(dest.relative_to(HERE))}
            report['rows'].append(row);print(json.dumps({k:row[k] for k in ['amount','spanFraction','minimumWidthRatio','baselineBlackWedgeProxyPixels','candidateBlackWedgeProxyPixels','newAccessoryCoreChangesVsBaseline','newAccessorySoftChangesVsBaseline','inverseFoldPixels']}),flush=True)
    report['scriptSha256']=sha(Path(__file__));(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

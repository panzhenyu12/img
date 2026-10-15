"""Read-only public CLI checks of the frozen scoped-owner candidate.

Synthetic RGB and class masks are generated independently of Rust. The head
bridge models touching hair of adjacent people in a semantic (non-instance)
head parsing mask; selected faceContour remains person A's skin contour.
"""
from pathlib import Path
import copy, hashlib, json, os, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
BIN = HERE / 'frozen-image-render'
W, H = 160, 320

def raster(a):
    return {'shape':'raster','width':W,'height':H,'data':a.astype('uint8').ravel().tolist()}

def raster_outline(a):
    """Marching-squares .5 isoline, same convention as frontend alphaOutline."""
    edges={}
    def connect(p,q):
        p=tuple(np.round(p,6));q=tuple(np.round(q,6))
        if p==q:return
        edges.setdefault(p,[]).append(q);edges.setdefault(q,[]).append(p)
    def at(x,y):return float(a[y,x]) if 0<=x<W and 0<=y<H else 0
    for y in range(-1,H):
        for x in range(-1,W):
            v=[at(x,y),at(x+1,y),at(x+1,y+1),at(x,y+1)]
            c=[np.array(p) for p in [(x+.5,y+.5),(x+1.5,y+.5),(x+1.5,y+1.5),(x+.5,y+1.5)]]
            cross={}
            for i in range(4):
                j=(i+1)%4
                if (v[i]>=127.5)!=(v[j]>=127.5):cross[i]=c[i]+(c[j]-c[i])*(127.5-v[i])/(v[j]-v[i])
            if len(cross)==2:connect(*cross.values())
            if len(cross)==4:
                pairs=[(0,3),(1,2)] if (v[0]>=127.5)!=(sum(v)/4>=127.5) else [(0,1),(2,3)]
                for i,j in pairs:connect(cross[i],cross[j])
    visited=set();loops=[]
    for start in edges:
        if start in visited:continue
        loop=[];cur=start;previous=None
        while cur not in visited:
            visited.add(cur);loop.append(cur)
            nxt=next((k for k in edges[cur] if k!=previous),None)
            if nxt is None:break
            previous,cur=cur,nxt
            if cur==start:loops.append(np.array(loop)/[W,H]);break
    area=lambda p:abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))
    loop=max(loops,key=area)
    def simplify(p,eps):
        if len(p)<=3:return p
        v=p[-1]-p[0];t=np.clip((p-p[0])@v/max(v@v,1e-20),0,1)
        d=np.linalg.norm(p-(p[0]+t[:,None]*v),axis=1);i=int(d.argmax())
        return np.concatenate([simplify(p[:i+1],eps)[:-1],simplify(p[i:],eps)]) if d[i]>eps else p[[0,-1]]
    eps=.3/max(W,H);mid=len(loop)//2
    for _ in range(20):
        p=np.concatenate([simplify(loop[:mid+1],eps)[:-1],simplify(np.concatenate([loop[mid:],loop[:1]]),eps)[:-1]])
        if len(p)<=256:break
        eps*=1.25
    return {'shape':'polygon','points':[{'x':float(x),'y':float(y)} for x,y in p]}

def portrait():
    y,x = np.mgrid[:H,:W]
    a = np.clip(.5+(1-np.hypot((x+.5-80)/38,(y+.5-112)/80))*38/3,0,1)
    n = ((x>=69)&(x<91)&(y>=170)).astype(float)
    skin=np.maximum(a,n)
    bg=np.stack([20+x%16,45+y%16,np.full_like(x,130)],axis=-1)
    rgb=np.floor(skin[...,None]*np.array([220,155,100])+(1-skin[...,None])*bg+.5).astype('uint8')
    rgba=np.concatenate([rgb,np.full((H,W,1),255,dtype='uint8')],axis=-1)
    theta=np.arange(64)*2*np.pi/64
    outline={'shape':'polygon','points':[{'x':float((80+38*np.cos(t))/W),'y':float((112+80*np.sin(t))/H)} for t in theta]}
    face=raster(np.floor(a*255+.5));skin=raster(np.floor(skin*255+.5))
    p={'region':{'x':0,'y':0,'width':1,'height':1},'semanticMasks':{'face':face,'faceOutline':outline,'faceContour':copy.deepcopy(outline),'faceContourMask':copy.deepcopy(face),'skin':skin,'neck':raster(np.where((n>0)&(a<.5),255,0))},'subjectMask':copy.deepcopy(skin),'adjustments':{}}
    return rgba,p

def case(name):
    src,p=portrait();y,x=np.mgrid[:H,:W];other=(x>=123)&(x<145)&(y>=88)&(y<136)
    masks=p['semanticMasks'];head=np.array(masks['face']['data']).reshape(H,W);person=np.array(p['subjectMask']['data']).reshape(H,W)
    hair=np.zeros((H,W),dtype='uint8')
    if name in ['disconnected','connected_hair','connected_hair_protected','disconnected_consistent','connected_hair_consistent']:
        head[other]=255;person[other]=255;hair[other]=255
        src[other,:3]=np.stack([30+(x[other]%9)*20,60+(y[other]%7)*20,np.full(other.sum(),170)],axis=-1)
        if not name.startswith('disconnected'):
            bridge=(x>=112)&(x<124)&(y>=110)&(y<114)
            head[bridge]=255;person[bridge]=255;hair[bridge]=255
            src[bridge,:3]=[40,32,30]
            if name.endswith('_consistent'):
                skin=np.array(masks['faceContourMask']['data']).reshape(H,W)
                hair[bridge]=255-skin[bridge]
                alpha=skin[bridge,None]/255
                src[bridge,:3]=np.floor(alpha*np.array([220,155,100])+(1-alpha)*np.array([40,32,30])+.5).astype('uint8')
        if name.endswith('protected'):
            p['protectMask']=raster(other*255)
    elif name=='own_ear_outside_head':
        other=(x>=115)&(x<126)&(y>=97)&(y<128)
        masks['ears']=raster(other*255);person[other]=255
        src[other,:3]=[20,235,230]
    masks['face']=raster(head);masks['hair']=raster(hair);p['subjectMask']=raster(person)
    if name.endswith('_consistent'):
        masks['faceOutline']=raster_outline(head)
        masks['faceContour']=raster_outline(np.array(masks['faceContourMask']['data']).reshape(H,W))
    return src,p,other

def render(name,amount):
    src,p,domain=case(name);p['faceShape']={'preset':'natural','amount':amount}
    d=HERE/name/str(amount);d.mkdir(parents=True,exist_ok=True)
    Image.fromarray(src).save(d/'source.png');Image.fromarray((domain*255).astype('uint8')).save(d/'audit-domain.png')
    graph={'operations':[{'type':'face.adjust','params':p}]};gp=d/'graph.json';gp.write_text(json.dumps(graph,separators=(',',':')))
    result=subprocess.run([str(BIN),'--input',str(d/'source.png'),'--graph',str(gp),'--output',str(d/'candidate.png')],env={**os.environ,'WHOLE_FACE_DEBUG_DIR':str(d/'debug')},capture_output=True)
    (d/'render.log').write_bytes(result.stdout+result.stderr)
    if result.returncode:
        return {'case':name,'amount':amount,'returncode':result.returncode,'error':result.stderr.decode()}
    out=np.asarray(Image.open(d/'candidate.png').convert('RGBA'));delta=np.max(np.abs(out.astype(int)-src.astype(int)),axis=-1);changed=delta>0
    row={'case':name,'amount':amount,'sourceSha256':hashlib.sha256((d/'source.png').read_bytes()).hexdigest(),'graphSha256':hashlib.sha256(gp.read_bytes()).hexdigest(),'domainPixels':int(domain.sum()),'domainChanged':int((domain&changed).sum()),'domainDeltaMax':int(delta[domain].max()),'domainDeltaMean':float(delta[domain].mean()),'allChanged':int(changed.sum()),'alphaChanged':int((out[...,3]!=src[...,3]).sum())}
    if amount:
        layers=json.loads((d/'debug/layers.json').read_text());left,top,cw,ch=layers['crop'];local=domain[top:top+ch,left:left+cw]
        row['layers']=layers
        for key in ['head-component','owner-support','frozen','moving-physical']:
            a=np.asarray(Image.open(d/f'debug/{key}.png'))
            row[key]={'positiveInDomain':int(((a>0)&local).sum()),'meanInDomain':float(a[local].mean())}
    panel=Image.new('RGB',(W*3,H+24),'#20242a');draw=ImageDraw.Draw(panel)
    for i,(a,label) in enumerate([(src,'Source'),(out,f'Natural {amount}'),(np.repeat(np.minimum(delta*8,255)[...,None],3,axis=-1).astype('uint8'),'Difference x8')]):
        panel.paste(Image.fromarray(a).convert('RGB'),(i*W,24));draw.text((i*W+2,6),label,fill='white')
    panel.save(d/'comparison-native-1x.png');(d/'metrics.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True);return row

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--consistent-only',action='store_true');args=parser.parse_args()
    names=['disconnected_consistent','connected_hair_consistent'] if args.consistent_only else ['disconnected','connected_hair','connected_hair_protected','own_ear_outside_head']
    rows=[render(name,amount) for name in names for amount in [0,100,-100]]
    (HERE/('report-consistent.json' if args.consistent_only else 'report.json')).write_text(json.dumps({'binarySha256':hashlib.sha256(BIN.read_bytes()).hexdigest(),'candidateModified':False,'rows':rows},indent=2)+'\n')

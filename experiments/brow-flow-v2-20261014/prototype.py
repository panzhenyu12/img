from pathlib import Path
import sys,json,numpy as np,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from audit_brow_eye_field import frame,mapping,inside
root=Path(__file__).resolve().parents[2]
graph=json.load(open(root/'test/results/user-face-contour-20261014/1024/face-graph.json'))
p={k:np.array([[z['x']*1024,z['y']*1024] for z in v['points']]) for k,v in graph['operations'][0]['params']['semanticMasks'].items() if v['shape']=='polygon'}
def flow(p,key,a,x,y,n=32):
    def v(x,y):
        sx,sy=mapping(p,key,a,x,y,True)
        return sx-x,sy-y
    h=1/n; sx,sy=x.copy(),y.copy()
    for _ in range(n):
        k1=v(sx,sy);k2=v(sx+h*.5*k1[0],sy+h*.5*k1[1]);k3=v(sx+h*.5*k2[0],sy+h*.5*k2[1]);k4=v(sx+h*k3[0],sy+h*k3[1])
        sx+=h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]);sy+=h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    return sx,sy
if __name__ == "__main__":
    rows=[]
    y,x=np.mgrid[255:430,300:725].astype(float)
    for key in ['browPeak','browTail']:
     for a in [-1,1]:
        t=time.time(); sx,sy=flow(p,key,a,x,y)
        eps=.05
        xp,yp=flow(p,key,a,x+eps,y);xm,ym=flow(p,key,a,x-eps,y);xt,yt=flow(p,key,a,x,y+eps);xb,yb=flow(p,key,a,x,y-eps)
        det=((xp-xm)*(yt-yb)-(yp-ym)*(xt-xb))/(4*eps*eps)
        rx,ry=flow(p,key,-a,sx,sy)
        eye=inside(p['leftEye'],x+.5,y+.5)|inside(p['rightEye'],x+.5,y+.5)
        moved=np.hypot(sx-x,sy-y)
        donor=np.zeros(x.shape,bool); bx,by=np.floor(sx),np.floor(sy); tx,ty=sx-bx,sy-by
        for dx,dy,w in [(0,0,(1-tx)*(1-ty)),(1,0,tx*(1-ty)),(0,1,(1-tx)*ty),(1,1,tx*ty)]:
            donor|=(w>1e-6)&(inside(p['leftEye'],bx+dx+.5,by+dy+.5)|inside(p['rightEye'],bx+dx+.5,by+dy+.5))
        r={'key':key,'amount':a*100,'minJac':float(det.min()),'foldPixels':int((det<=0).sum()),'maxEyeMove':float(moved[eye].max()),'eyeDonorImport':int((donor&~eye&(moved>.1)).sum()),'maxMove':float(moved.max()),'maxReverseError':float(np.hypot(rx-x,ry-y).max()),'seconds':time.time()-t}
        rows.append(r);print(json.dumps(r),flush=True)
    Path(__file__).with_name('prototype-report.json').write_text(json.dumps(rows,indent=2))

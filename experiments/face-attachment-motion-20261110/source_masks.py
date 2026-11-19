"""Exact source-mask sampling shared by source preparation and experiments.

No field solver, native binary, model or historical experiment is imported.
The f32 raster operations are lifted unchanged from the native-mask adapter.
"""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _raster_mask(m,w,h):
    """Sample valid rasters in the exact PreparedMask f32 operation order.

    Native callers normalize source pixel centres before scaling by the mask
    size. Algebraically equivalent float64/scipy coordinates can cross a hard
    material threshold. Keep every intermediate f32, including decoded bytes.
    Raster inversion is supported; painted strokes need the native brush
    compositor and are explicitly unsupported by this isolated adapter.
    """
    if m.get('shape',m.get('kind'))!='raster':
        raise ValueError('isolated mask adapter supports polygons and rasters only')
    if m.get('strokes'):
        raise ValueError('raster strokes are unsupported by the isolated native-sampling adapter')
    mw,mh=m.get('width'),m.get('height')
    if any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) or v<1 for v in [w,h,mw,mh]):
        raise ValueError('source and raster dimensions must be positive integers')
    data=np.asarray(m['data'],dtype=np.float32)
    if data.ndim!=1 or data.size!=mw*mh or not np.isfinite(data).all():
        raise ValueError('raster requires one finite value per node')
    data=data.reshape(mh,mw)/np.float32(255.)
    half,one,zero=np.float32(.5),np.float32(1.),np.float32(0.)
    nx=(np.arange(w,dtype=np.float32)+half)/np.float32(w)
    ny=(np.arange(h,dtype=np.float32)+half)/np.float32(h)
    px=np.clip(nx*np.float32(mw)-half,zero,np.float32(mw-1))
    py=np.clip(ny*np.float32(mh)-half,zero,np.float32(mh-1))
    x0=np.floor(px).astype(np.intp);y0=np.floor(py).astype(np.intp)
    x1=np.minimum(x0+1,mw-1);y1=np.minimum(y0+1,mh-1)
    tx=(px-x0.astype(np.float32))[None,:];ty=(py-y0.astype(np.float32))[:,None]
    top=data[y0[:,None],x0[None,:]]*(one-tx)+data[y0[:,None],x1[None,:]]*tx
    bottom=data[y1[:,None],x0[None,:]]*(one-tx)+data[y1[:,None],x1[None,:]]*tx
    coverage=np.clip(top*(one-ty)+bottom*ty,zero,one)
    return one-coverage if m.get('invert') is True else coverage

def mask(m,w,h):
    if m.get('shape')=='polygon':
        im=Image.new('L',(w,h));ImageDraw.Draw(im).polygon([(p['x']*w-.5,p['y']*h-.5) for p in m['points']],fill=255)
        return np.asarray(im,dtype=float)/255
    return _raster_mask(m,w,h)

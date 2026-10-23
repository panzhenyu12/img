"""Exact untapered prefix energy for original piecewise-linear 1D controls.

No new window, taper, budget factor or target is introduced. The fixed outer
scale remains 12 px for the three different prefix endpoints in the caller.
"""
import numpy as np
from scipy.linalg import null_space, solve_triangular


def prefix_energy_operator(nodes,prefix_px,outer_scale_px):
    nodes=np.asarray(nodes,float);end=float(prefix_px);scale=float(outer_scale_px)
    if (nodes.ndim!=1 or len(nodes)<2 or not np.isfinite(nodes).all() or nodes[0]!=0.
            or np.any(np.diff(nodes)<=0.) or not np.isfinite([end,scale]).all()
            or not 0.<end<=nodes[-1] or scale<=0.):
        raise ValueError('original increasing nodes from zero, in-domain positive prefix and outer scale required')
    # On a partially covered source edge the PL derivative is unchanged;
    # only the integration measure is shortened, never the denominator h.
    ids=np.flatnonzero(nodes[:-1]<end)
    lengths=np.minimum(nodes[ids+1],end)-nodes[ids]
    coefficients=scale*np.sqrt(lengths)/(nodes[ids+1]-nodes[ids])
    operator=np.zeros((len(ids),len(nodes)));rows=np.arange(len(ids))
    operator[rows,ids]=-coefficients;operator[rows,ids+1]=coefficients
    return operator


def reduce_fixed_area(hessian,linear,area,wanted,lower,upper,normalization):
    """Exact-coordinate elimination; no ridge, changed objective or area."""
    H=np.asarray(hessian,float);l=np.asarray(linear,float);area=np.asarray(area,float)
    lower,upper=np.asarray(lower,float),np.asarray(upper,float)
    n=len(l)
    if H.shape!=(n,n) or area.shape!=(n,) or lower.shape!=(n,) or upper.shape!=(n,):
        raise ValueError('matching original objective, area and bounds required')
    if not np.isfinite(H).all() or not np.isfinite(l).all() or not np.isfinite(area).all() or not np.isfinite([wanted,normalization]).all() or normalization<=0.:
        raise ValueError('finite original objective, area and positive normalization required')
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower>upper):
        raise ValueError('consistent original bounds required')
    fixed=lower==upper;free=~fixed
    if not np.isfinite(lower[fixed]).all() or not free.any():
        raise ValueError('finite fixed coordinates and a nonempty free subspace required')
    base=np.zeros(n);base[fixed]=lower[fixed]
    a=area[free]
    if a@a==0.:raise ValueError('original area has no free support')
    base[free]=a*((wanted-area@base)/(a@a))
    null=null_space(a[None,:]);Hf=H[np.ix_(free,free)]
    if null.shape[1]:
        factor=np.linalg.cholesky((null.T@Hf@null)/normalization)
        whiten=solve_triangular(factor.T,np.eye(null.shape[1]),lower=False)
        transform=np.zeros((n,null.shape[1]));transform[free]=null@whiten
    else:transform=np.zeros((n,0))
    q=2*transform.T@(H@base+l)/normalization
    return {'base':base,'transform':transform,'q':q,'fixed':fixed,'free':free}

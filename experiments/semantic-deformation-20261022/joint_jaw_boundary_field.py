"""Isolated joint jaw/interface source-field feasibility and curve diagnostics.

One scalar bilinear source field supplies every observation. Independent source
curves are never joined, snapped or assigned an anatomical role. This module
does not authorize rendering or export a runnable target profile.
"""
import time
import hashlib
import json
from types import SimpleNamespace
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix, csr_matrix, diags, vstack, hstack, block_diag as sparse_block_diag

from portable_field import point_operators, source_geometry
from probe_jaw_plateau import OriginalProblem, PlateauProblem
from jaw_curve_qp import _basis
from jaw_bilateral import build_bilateral_coupling
from source_boundary_soft_targets import sample_boundary_soft_targets
from forward import cell_jacobians
from interval_polynomial_cones import nonnegative_cubic_block
from measure_jaw_preference_start import measure_preference_start


def qp_fingerprint(p,q,a,b,cone_description):
    """Hash the exact supplied sparse conic QP, excluding solver tolerances."""
    digest=hashlib.sha256()
    for matrix in [p,a]:
        matrix=matrix.tocsc()
        for value in [np.asarray(matrix.shape,dtype=np.int64),matrix.indptr,matrix.indices,matrix.data]:
            digest.update(np.ascontiguousarray(value).tobytes())
    for value in [q,b]:digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(json.dumps(cone_description,sort_keys=True).encode())
    return digest.hexdigest()


def rank_one_inverse_soc_slacks(d,b):
    """Exact rank-one inverse sym [.52,1.8], skew<=.24; no physical tolerance."""
    d,b=np.broadcast_arrays(np.asarray(d,float),np.asarray(b,float))
    return np.stack([1.-np.hypot(2*.52*d-1.,np.sqrt(.52/.48)*b),
        2*1.8*d-1.-np.hypot(np.ones_like(d),np.sqrt(1.8/.8)*b),.48*d-abs(b)],axis=-1)


def corner_gradient_rows(problem,indices,across,down):
    """One-sided original bilinear cell-corner gradients in the face frame."""
    grid=problem['grid'];mapping=np.asarray(problem['mapping']);size=problem['hessian'].shape[0]
    corners,y,x=np.asarray(indices,int).T;w=grid['width'];n=len(x)
    def edge_rows(first,last,spacing):
        columns=np.r_[mapping[first],mapping[last]];rows=np.tile(np.arange(n),2)
        values=np.r_[np.full(n,-1./spacing),np.full(n,1./spacing)];valid=columns>=0
        return csr_matrix((values[valid],(rows[valid],columns[valid])),shape=(n,size))
    row_y=y+corners//2;column_x=x+corners%2
    dx=edge_rows(row_y*w+x,row_y*w+x+1,grid['spacing'][0])
    dy=edge_rows(y*w+column_x,(y+1)*w+column_x,grid['spacing'][1])
    return across[0]*dx+across[1]*dy,down[0]*dx+down[1]*dy


def all_corner_inverse_check(problem,z,across):
    """Independent direct inverse/eigenvalue oracle over all source corners."""
    grid=problem['grid'];mapping=np.asarray(problem['mapping']);scalar=np.zeros(len(mapping))
    spacing=np.asarray(grid['spacing'],float)
    if spacing.shape!=(2,) or not np.isfinite(spacing).all() or spacing[0]<=0 or spacing[0]!=spacing[1]:
        raise ValueError('source differential oracle requires positive isotropic grid spacing')
    valid=mapping>=0;scalar[valid]=z[mapping[valid]]
    field=(scalar[:,None]*across).reshape(grid['height'],grid['width'],2)
    f=cell_jacobians(field/np.asarray(grid['spacing']));det=np.linalg.det(f)
    positive=det>0.;inv=np.zeros_like(f);inv[positive]=np.linalg.inv(f[positive])
    eigen=np.linalg.eigvalsh((inv+inv.swapaxes(-1,-2))*.5)
    skew=abs(inv[...,1,0]-inv[...,0,1])*.5
    bad=(~positive)|(eigen[...,0]<.52)|(eigen[...,1]>1.8)|(skew>.24)
    report={'positive_source_determinant':bool(positive.all()),'minimum_source_determinant':float(det.min()),
        'inverse_symmetric_min':float(eigen[...,0][positive].min()) if positive.any() else None,
        'inverse_symmetric_max':float(eigen[...,1][positive].max()) if positive.any() else None,
        'inverse_absolute_skew_max':float(skew[positive].max()) if positive.any() else None,
        'violating_source_cell_corners':int(bad.sum()),'checked_corners':int(det.size),
        'accepted':bool(not bad.any()),'oracle':'direct inverse/eigvalsh at every one-sided source cell corner',
        'limits':'Exact ideal continuous inverse differential only; final reconstructed f32 inverse remains unchecked.'}
    return report,np.argwhere(bad)


def polynomial_minima(values):
    """Minimum of each sampled degree<=3 polynomial on normalized [0,1]."""
    values=np.asarray(values,float)
    if values.ndim==1:values=values[:,None]
    degree=len(values)-1
    coefficients=np.linalg.solve(np.vander(np.linspace(0.,1.,degree+1),increasing=True),values)
    locations=[];minima=[]
    for c in coefficients.T:
        roots=np.polynomial.polynomial.polyroots(np.arange(1,len(c))*c[1:])
        candidates=[0.,1.,*[float(r.real) for r in roots if abs(r.imag)<1e-9 and 0.<r.real<1.]]
        evaluations=np.polynomial.polynomial.polyval(candidates,c);i=int(np.argmin(evaluations))
        locations.append(candidates[i]);minima.append(evaluations[i])
    return np.asarray(locations),np.asarray(minima)


class GridCurve:
    def __init__(self, points, problem, across, down, *, source_knots=None):
        self.points=np.asarray(points,float);self.problem=problem
        self.across=np.asarray(across,float);self.down=np.asarray(down,float)
        if self.points.ndim!=2 or self.points.shape[1]!=2 or len(self.points)<2 or not np.isfinite(self.points).all():
            raise ValueError('finite original source polyline required')
        steps=np.linalg.norm(np.diff(self.points,axis=0),axis=1)
        if np.any(steps<=0):raise ValueError('positive source edge lengths required')
        self.knots=np.r_[0.,steps.cumsum()]
        if source_knots is not None:
            knots=np.asarray(source_knots,float)
            if (knots.shape!=(len(self.points),) or not np.isfinite(knots).all()
                    or knots[0]!=0. or np.any(np.diff(knots)<=0.)):
                raise ValueError('original source arclength knots must be finite and strictly increasing from zero')
            # Original contour cumulative distances and coordinates extracted
            # from them differ by roundoff when re-summed. Keep the original
            # parameter; no source positions, requested strengths or bounds
            # are clipped. This tolerance is only a representational check.
            error_bound=128*np.finfo(float).eps*max(1.,float(abs(self.points).max()),float(knots[-1]))
            if np.any(abs(np.diff(knots)-steps)>error_bound):
                raise ValueError('original source knots disagree with geometric arclength')
            self.knots=knots.copy()
        self.length=float(self.knots[-1])
        self.size=problem['hessian'].shape[0]
        grid=problem['grid'];cuts=list(self.knots)
        for i,(a,b) in enumerate(zip(self.points[:-1],self.points[1:])):
            for axis,count in enumerate([grid['width'],grid['height']]):
                if a[axis]==b[axis]:continue
                lines=grid['origin'][axis]+grid['spacing'][axis]*np.arange(count)
                t=(lines-a[axis])/(b[axis]-a[axis]);t=t[(t>0)&(t<1)]
                cuts.extend(self.knots[i]+(self.knots[i+1]-self.knots[i])*t)
        self.breaks=np.unique(cuts)

    def source(self,s):
        s=np.asarray(s,float)
        if s.ndim!=1 or not np.isfinite(s).all() or np.any(s<0.) or np.any(s>self.length):
            raise ValueError('source arclength must be a finite vector within [0, curve length]')
        return np.column_stack([np.interp(s,self.knots,self.points[:,i]) for i in range(2)])

    def op(self,s):
        points=self.source(s);grid=self.problem['grid']
        uv=(points-grid['origin'])/grid['spacing']
        if np.any(uv < 0.) or np.any(uv > np.array([grid['width']-1,grid['height']-1])):
            raise ValueError('reviewed source curve must lie inside the unchanged source crop')
        return point_operators(points,grid,self.problem['mapping'],self.size)[0].tocsr()

    def area_row(self):
        # Within source-edge/grid-cell cuts, B(source(s)) is quadratic and dv/ds
        # constant. Two Gauss points integrate u dv exactly up to roundoff.
        lo,hi=self.breaks[:-1],self.breaks[1:]
        g,w=np.polynomial.legendre.leggauss(2)
        s=((lo+hi)[:,None]/2+(hi-lo)[:,None]*g/2).ravel()
        delta=self.source(hi)-self.source(lo)
        dv=self.across[0]*delta[:,1]-self.across[1]*delta[:,0]
        return csr_matrix(((dv[:,None]*w/2).ravel()@self.op(s)).reshape(1,-1))

    def chord(self,s,radius):
        s=np.asarray(s,float)
        if (not np.isfinite(radius) or radius<=0. or self.length<=2*radius
                or s.ndim!=1 or not np.isfinite(s).all()
                or np.any(s<radius) or np.any(s>self.length-radius)):
            raise ValueError('chord center arclength must fit the original source interval')
        shifted=[s-radius,s,s+radius]
        # Exact interval endpoints are symbolic: (L-r)+r means L even if
        # floating subtraction/addition is not an exact inverse. No near-site
        # snapping or out-of-domain center is permitted.
        shifted[2]=np.where(s==self.length-radius,self.length,shifted[2])
        a,b,c=[self.source(p) for p in shifted]
        u=[p@self.across for p in [a,b,c]];v=[p@self.down for p in [a,b,c]]
        h0,h1=v[1]-v[0],v[2]-v[1]
        ns=(u[2]-u[1])*h0-(u[1]-u[0])*h1
        b0,b1,b2=[self.op(p) for p in shifted]
        rows=diags(h0)@(b2-b1)-diags(h1)@(b1-b0)
        return ns,rows.tocsr(),h0*h1

    def chord_partition(self,radius):
        if radius<=0 or self.length<=2*radius:raise ValueError('positive chord radius must fit source curve')
        return np.unique(np.r_[radius,self.length-radius,*[
            (self.breaks+d)[(self.breaks+d>radius)&(self.breaks+d<self.length-radius)]
            for d in [-radius,0.,radius]]])

    def source_turn_regions(self,radius,threshold=1e-8):
        """Both source signs; near-zero source turn remains unassigned."""
        if np.any(np.diff(self.points@self.down)<=0):
            raise ValueError('full visible interface requires strictly height-monotone source points')
        cuts=self.chord_partition(radius);regions=[]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            if hi-lo<1e-10:continue
            ns,_,_=self.chord(lo+(hi-lo)*np.array([0.,.5,1.]),radius)
            c=np.linalg.solve(np.vander([0.,.5,1.],increasing=True),ns)
            roots=[]
            for level in [-threshold,threshold]:
                shifted=c.copy();shifted[0]-=level
                roots.extend(float(r.real) for r in np.polynomial.polynomial.polyroots(shifted)
                    if abs(r.imag)<1e-9 and 0.<r.real<1.)
            ts=sorted(set([0.,*roots,1.]))
            for a,b in zip(ts[:-1],ts[1:]):
                value=np.polynomial.polynomial.polyval((a+b)/2,c)
                if abs(value)>threshold:regions.append([lo+(hi-lo)*a,lo+(hi-lo)*b,float(np.sign(value))])
        return np.asarray(regions).reshape(-1,3)

    def continuous_chord_check(self,z,regions,radius,rho):
        if not len(regions):return {'minimum':None,'violations':0},np.array([]),np.array([])
        lo,hi,signs=np.asarray(regions).T
        ts=np.linspace(0.,1.,4);s=(lo[None,:]+(hi-lo)[None,:]*ts[:,None]).ravel()
        ns,rows,_=self.chord(s,radius)
        values=((1-rho)*ns+rows@z).reshape(4,-1)*signs
        t,_=polynomial_minima(values);centers=lo+(hi-lo)*t
        ns,rows,_=self.chord(centers,radius);margin=signs*((1-rho)*ns+rows@z)
        bad=margin < -1e-8
        return {'minimum':float(margin.min()),'violations':int(bad.sum()),'violating_piece_indices':np.flatnonzero(bad).tolist(),
                'worst_source_xy':self.source(centers[[int(np.argmin(margin))]])[0].tolist()},centers[bad],signs[bad]

    def chord_polynomial(self,region,radius,rho):
        """Affine power coefficients on one original edge/grid chord piece."""
        lo,hi,sign=region;t=np.linspace(0.,1.,4)
        s=(1.-t)*lo+t*hi
        ns,rows,_=self.chord(s,radius)
        transform=np.linalg.inv(np.vander(t,increasing=True))
        return sign*(csr_matrix(transform)@rows),sign*(1.-rho)*(transform@ns)

    def sign_polynomial(self,region):
        """Exact quadratic source-piece coefficients embedded as a cubic."""
        lo,hi,sign=region
        rows=self.op([lo,(lo+hi)/2.,hi])
        transform=csr_matrix([[1.,0.,0.],[-3.,4.,-1.],[2.,-4.,2.],[0.,0.,0.]])
        return sign*(transform@rows),np.zeros(4)

    def continuous_sign_check(self,z,sign,regions=None):
        if regions is None:regions=np.column_stack([self.breaks[:-1],self.breaks[1:],np.full(len(self.breaks)-1,sign)])
        lo,hi,signs=np.asarray(regions).T
        s=(lo[None,:]+(hi-lo)[None,:]*np.array([0.,.5,1.])[:,None]).ravel()
        t,_=polynomial_minima((self.op(s)@z).reshape(3,-1)*signs)
        centers=lo+(hi-lo)*t;values=signs*(self.op(centers)@z);bad=values < -1e-10
        report={'minimum':float(values.min()),'violations':int(bad.sum()),'violating_piece_indices':np.flatnonzero(bad).tolist()}
        return report,centers[bad]


def source_sign_regions(curve,origin,midline,amount):
    radius=(curve.points-np.asarray(origin))@curve.across-midline
    zeros=list(curve.knots[radius==0.])
    for i,(a,b) in enumerate(zip(radius[:-1],radius[1:])):
        if a*b<0:zeros.append(curve.knots[i]+(curve.knots[i+1]-curve.knots[i])*(-a)/(b-a))
    cuts=np.unique(np.r_[curve.breaks,zeros]);lo,hi=cuts[:-1],cuts[1:]
    r=(curve.source((lo+hi)/2)-origin)@curve.across-midline
    signs=-np.sign(amount*r)
    # Beta also vanishes on an entire source segment lying on the midline.
    # Three evaluations enforce the quadratic field to be zero on that piece.
    flat=signs==0
    required_zero=np.r_[0.,curve.length,zeros,lo[flat],(lo[flat]+hi[flat])/2,hi[flat]]
    return np.column_stack([lo,hi,signs]),np.unique(required_zero)


def hard_zero_sign_piece_rows(curve,region,required_zero):
    """Exact nonnegativity after factoring a known zero of a quadratic.

    The caller must retain the original hard q(endpoint)=0 equalities. For
    q(0)=0, q=t*(q'(0)+(q(1)-q'(0))*t), so nonnegativity is exactly the
    nonnegativity of that linear factor at both endpoints. The t=1 case is
    analogous. Only exact original required_zero endpoints qualify.
    """
    lo,hi,_=region;required_zero=np.asarray(required_zero)
    at_start=bool(np.any(required_zero==lo));at_end=bool(np.any(required_zero==hi))
    if not (at_start or at_end):return None
    coefficients,_=curve.sign_polynomial(region)
    if at_start:
        return vstack([coefficients.getrow(1),coefficients.sum(axis=0)],format='csr')
    return vstack([-coefficients.getrow(1)-2*coefficients.getrow(2),coefficients.getrow(0)],format='csr')


def _refine_regions(regions,cuts,sign):
    result=[]
    for lo,hi in regions:
        partition=np.unique(np.r_[lo,cuts[(cuts>lo)&(cuts<hi)],hi])
        result.extend([a,b,sign] for a,b in zip(partition[:-1],partition[1:]) if b-a>1e-10)
    return np.asarray(result).reshape(-1,3)


def interface_chord_conditions(name,curve,radius,band=None):
    """Optional source-relative bend magnitudes; never expand known domains.

    N is the finite-chord numerator, not the turning angle. For source sign s,
    the lower condition is s*(Ntarget-low*Nsource)>=0 and the upper condition
    is -s*(Ntarget-high*Nsource)>=0. Both reuse the exact cubic interval engine.
    """
    regions=curve.source_turn_regions(radius)
    if band is None:return [(name,curve,regions,radius,0.)]
    band=np.asarray(band,float)
    if band.shape!=(2,) or not np.isfinite(band).all() or not (0.<band[0]<=1.<=band[1]):
        raise ValueError('source-relative interface bend band must be finite with 0 < lower <= 1 <= upper')
    upper=regions.copy();upper[:,2]*=-1.
    return [(name+':source-bend-lower',curve,regions,radius,float(band[0])),
            (name+':source-bend-upper',curve,upper,radius,float(band[1]))]


def interface_displacement_bending(curve,density,smoothing_length_px,node_spacing_px):
    """Jaw-convention discrete bending of displacement on original arclength.

    For uniform original s nodes, R = D2 B with D2 = [1,-2,1]/h^2.
    H = density * length_scale^4 * h * R.T R. The solver passes P=2H,
    so its added objective is z.T H z (no extra 1/2). Affine u(s) is
    unpenalized; this does not claim arbitrary spatial-affine invariance.
    No source curvature, zero-order displacement or endpoint slope is fitted.
    Like the existing jaw term, interior-node quadrature spans L-h, not L.
    """
    parameters=np.asarray([density,smoothing_length_px,node_spacing_px],float)
    if not np.isfinite(parameters).all() or density<0 or smoothing_length_px<0 or node_spacing_px<=0:
        raise ValueError('finite nonnegative bending density/length and positive node spacing required')
    count=max(2,int(np.ceil(curve.length/node_spacing_px)))+1
    nodes=np.linspace(0.,curve.length,count);h=curve.length/(count-1)
    d2=diags([np.ones(count-2),np.full(count-2,-2.),np.ones(count-2)],
             [0,1,2],shape=(count-2,count),format='csr')/h**2
    operator=(d2@curve.op(nodes)).tocsr();weight=float(density*smoothing_length_px**4*h)
    return {'operator':operator,'weight':weight,'nodes':nodes,
            'hessian':(weight*(operator.T@operator)).tocsc(),
            'report':{'node_count':count,'source_length_px':curve.length,'actual_spacing_px':h,
                      'quadrature_span_px':(count-2)*h,'density':float(density),
                      'smoothing_length_px':float(smoothing_length_px),
                      'requested_node_spacing_px':float(node_spacing_px)}}


def interface_displacement_bending_energy(regularizers,z):
    """Measure the exact added objective using residuals, avoiding cancellation."""
    curves=[]
    for name,regularizer in regularizers:
        derivative=regularizer['operator']@z
        energy=float(regularizer['weight']*(derivative@derivative))
        curves.append({'id':name,**regularizer['report'],'energy_px3':energy,
                       'maximum_abs_discrete_second_derivative_per_px':float(abs(derivative).max(initial=0.))})
    return {'total_energy_px3':sum(c['energy_px3'] for c in curves),'curves':curves}


def jaw_start_displacement_regularizer(curve,length_px):
    """Exact weighted first-derivative energy on a known source jaw prefix.

    u(source(s)) is quadratic inside each original source-edge/grid-cell
    piece. Its derivative is linear, and w(s/length) is degree five, so
    four-point Gauss integrates w*u'^2 exactly in real arithmetic. The
    spatial gradient is evaluated only at interior quadrature sites and
    contracted with the original source-segment tangent. No unknown upper
    continuation or endpoint derivative equality is invented.
    """
    if not np.isfinite(length_px) or length_px<=0 or length_px>curve.length:
        raise ValueError('positive jaw-start window must fit the known original source arc')
    cuts=np.unique(np.r_[0.,curve.breaks[(curve.breaks>0.)&(curve.breaks<length_px)],length_px])
    lo,hi=cuts[:-1],cuts[1:];g,weights=np.polynomial.legendre.leggauss(4)
    positions=((lo+hi)[:,None]/2+(hi-lo)[:,None]*g/2).ravel()
    points=curve.source(positions)
    source_edges=np.searchsorted(curve.knots,positions,side='right')-1
    tangent=(curve.points[source_edges+1]-curve.points[source_edges])/(
        curve.knots[source_edges+1]-curve.knots[source_edges])[:,None]
    _,dx,dy=point_operators(points,curve.problem['grid'],curve.problem['mapping'],curve.size)
    operator=(diags(tangent[:,0])@dx+diags(tangent[:,1])@dy).tocsr()
    t=positions/length_px
    # Algebraically 1-10t^3+15t^4-6t^5; factored to retain positivity near 1.
    taper=(1.-t)**3*(1.+3.*t+6.*t**2)
    ds_weights=((hi-lo)[:,None]*weights/2).ravel()
    quadrature_weights=length_px**2*ds_weights*taper
    return {'curve':curve,'operator':operator,'weights':quadrature_weights,'positions':positions,'ds_weights':ds_weights,
            'hessian':(operator.T@diags(quadrature_weights)@operator).tocsc(),
            'report':{'window_end_px':float(length_px),'source_grid_piece_count':len(lo),
                      'quadrature_points':len(positions),'quadrature_degree_exact':7,
                      'original_source_window':[0.,float(length_px)]}}


def curve_window_motion(curve,z,end):
    """Exact displacement extrema of each original quadratic source piece."""
    if not np.isfinite(end) or end<=0 or end>curve.length:
        raise ValueError('motion window must fit the known original source arc')
    cuts=np.unique(np.r_[0.,curve.breaks[(curve.breaks>0.)&(curve.breaks<end)],end])
    lo,hi=cuts[:-1],cuts[1:]
    positions=(lo[None,:]+(hi-lo)[None,:]*np.array([0.,.5,1.])[:,None]).ravel()
    values=(curve.op(positions)@z).reshape(3,-1);candidates=[]
    for sign in [1.,-1.]:
        t,_=polynomial_minima(sign*values);where=lo+(hi-lo)*t
        motion=curve.op(where)@z;i=int(np.argmax(abs(motion)))
        candidates.append((float(abs(motion[i])),float(where[i])))
    magnitude,where=max(candidates)
    return {'window_end_source_arclength_px':float(end),'at_window_end_px':float((curve.op([end])@z)[0]),
            'maximum_abs_motion_px':magnitude,'maximum_at_source_arclength_px':where,
            'method':'exact quadratic extrema on original source/grid pieces, not endpoint-only sampling'}


def jaw_start_displacement_energy(regularizers,z):
    curves=[]
    for name,regularizer in regularizers:
        derivative=regularizer['operator']@z
        curves.append({'id':name,**regularizer['report'],
            'energy_px3':float(regularizer['weights']@(derivative**2)),
            'prefix_motion':[curve_window_motion(regularizer['curve'],z,end) for end in [2.,6.]
                             if end<=regularizer['curve'].length]})
    return {'total_energy_px3':sum(c['energy_px3'] for c in curves),'curves':curves}


def jaw_prefix_intent_block(curve,prefix_px,scale_px,energy_cap):
    """One exact unweighted known-prefix Dirichlet SOC, never a soft weight."""
    if not np.isfinite([scale_px,energy_cap]).all() or scale_px<=0 or energy_cap<0:
        raise ValueError('positive fixed physical scale and nonnegative original-intent budget required')
    quadrature=jaw_start_displacement_regularizer(curve,prefix_px)
    operator=(diags(scale_px*np.sqrt(quadrature['ds_weights']))@quadrature['operator']).tocsr()
    zero=csr_matrix((1,curve.size))
    return {'curve':curve,'operator':operator,'energy_cap':float(energy_cap),
            'prefix_px':float(prefix_px),'scale_px':float(scale_px),
            'A':vstack([zero,-operator],format='csr'),
            'b':np.r_[np.sqrt(energy_cap),np.zeros(operator.shape[0])],
            'report':{'prefix_px':float(prefix_px),'outer_scale_px':float(scale_px),
                      'energy_cap_px3':float(energy_cap),'soc_radius':float(np.sqrt(energy_cap)),
                      'cauchy_motion_bound_at_prefix_end_px':float(np.sqrt(prefix_px*energy_cap)/scale_px),
                      'source_grid_piece_count':quadrature['report']['source_grid_piece_count'],
                      'quadrature_points':len(quadrature['positions'])}}


def jaw_prefix_intent_blocks(preference,jaws,amount,fraction,scale_px,node_spacing_px):
    """Six separate budgets from existing unsolved Plateau beta, not fitted fields."""
    prefixes=[float(node_spacing_px),float(scale_px/2.),float(scale_px)]
    if not (0.<prefixes[0]<prefixes[1]<prefixes[2]):
        raise ValueError('three distinct nested original physical prefix scales required')
    result=[]
    for arc,curve in zip(preference.arcs,jaws):
        for prefix in prefixes:
            measured=measure_preference_start(preference,arc,amount,fraction,prefix,'equal_area_plateau')
            # The measurement uses prefix^2; convert only that dimensional
            # prefactor to the same scale^2 as every field constraint.
            cap=measured['unweighted_energy_px3']*(scale_px/prefix)**2
            block=jaw_prefix_intent_block(curve,prefix,scale_px,cap)
            block['report'].update(id=arc['id'],intent='existing equal-area PlateauProblem.beta',
                amount=float(amount),max_radius_fraction=float(fraction),
                preference_scale=float(preference.scale),
                preference_integral_method='analytic radius/envelope derivative, degree<=10, nine-point Gauss on original polynomial pieces',
                preference_antiderivative_difference_px3=measured['unweighted_quadrature_antiderivative_difference_px3']*(scale_px/prefix)**2)
            result.append(block)
    return result


def check_jaw_prefix_intent(blocks,z):
    rows=[]
    for block in blocks:
        residual=block['operator']@z;energy=float(residual@residual);cap=block['energy_cap']
        rows.append({**block['report'],'actual_energy_px3':energy,'energy_margin_px3':cap-energy,
                     'accepted':bool(energy<=cap),
                     'actual_motion':curve_window_motion(block['curve'],z,block['prefix_px'])})
    count=sum(not row['accepted'] for row in rows)
    return {'accepted':count==0,'violations':count,'constraints':rows,
            'gate':'each independent exact physical energy <= its original Plateau budget, with no added tolerance'}


def _independent_equalities(rows,targets,tolerance=1e-10):
    """Small equality set only: normalized modified Gram-Schmidt, keep RHS."""
    vectors=[];rhs=[];redundant=0
    for row,value in zip(rows,targets):
        vector=np.asarray(row.toarray()).ravel();norm=np.linalg.norm(vector)
        if norm>0:vector/=norm;value/=norm
        for _ in range(2):
            for q,b in zip(vectors,rhs):
                scale=vector@q;vector-=scale*q;value-=scale*b
        norm=np.linalg.norm(vector)
        if norm<=tolerance:
            if abs(value)>tolerance:raise ValueError('inconsistent joint equality or zero-support nonzero area')
            redundant+=1;continue
        vectors.append(vector/norm);rhs.append(value/norm)
    return csr_matrix(np.asarray(vectors).reshape(-1,rows[0].shape[1])),np.asarray(rhs),redundant


def source_field_geometry(problem,z,across):
    """Existing independent whole-grid geometry oracle, without inverse/render."""
    grid=problem['grid'];mapping=np.asarray(problem['mapping']);valid=mapping>=0
    spacing=np.asarray(grid['spacing'],float)
    if spacing.shape!=(2,) or not np.isfinite(spacing).all() or spacing[0]<=0 or spacing[0]!=spacing[1]:
        raise ValueError('source differential oracle requires positive isotropic grid spacing')
    scalar=np.zeros(len(mapping));scalar[valid]=np.asarray(z)[mapping[valid]]
    field=(scalar[:,None]*np.asarray(across)).reshape(grid['height'],grid['width'],2)
    return source_geometry(field/np.asarray(grid['spacing']))


def continuous_nonjaw_motion(full,old,z):
    endpoints=[s for a in old.arcs for s in [a['start'],(a['start']+a['direction']*a['length'])%full.length]]
    cuts=np.unique(np.r_[full.breaks,endpoints]);lo,hi=cuts[:-1],cuts[1:];middle=(lo+hi)/2
    outside=np.ones(len(middle),bool)
    for a in old.arcs:
        local=((middle-a['start'])*a['direction'])%full.length
        outside&=local>a['length']
    lo,hi=lo[outside],hi[outside]
    if not len(lo):return {'maximum_abs_across_motion_px':0.,'source_xy':None}
    s=(lo[None,:]+(hi-lo)[None,:]*np.array([0.,.5,1.])[:,None]).ravel()
    values=(full.op(s)@z).reshape(3,-1);candidates=[]
    for sign in [1.,-1.]:
        t,_=polynomial_minima(sign*values);where=lo+(hi-lo)*t
        motion=full.op(where)@z;i=int(np.argmax(abs(motion)))
        candidates.append((float(abs(motion[i])),float(where[i])))
    magnitude,where=max(candidates)
    return {'maximum_abs_across_motion_px':magnitude,'source_arclength_px':where,
            'source_xy':full.source([where])[0].tolist(),
            'method':'degree-two extrema on every original nonjaw source-edge/grid-cell piece; endpoints included',
            'zero_total_nonjaw_area_does_not_mean_zero_motion':True}


def solve_joint_field(problem,geometry,amount,interface_segments,neck_segments,*,
        smoothing_length_px=12.,jaw_radius_px=16.,interface_radius_px=12.,
        node_spacing_px=2.,neck_density=.05,neck_fade_px=16.,max_rounds=8,
        interface_bend_band=None,allow_strict_polish=True,regularize_interface_displacement=False,
        regularize_jaw_start=False,constrain_jaw_start_intent=False,reuse_first_round=None):
    """Bounded-budget new target distribution; diagnostic source field only.

    Jaw preference, discrete smoothing and bilateral coupling equal the prior
    node-based plateau QP. Their energy is assembled through the same spatial
    field as the visible interface. The portable input's Hessian must contain
    only its existing grid bending/membrane energy (zero-weight dummy target).
    """
    import clarabel
    started=time.monotonic()
    prior_elapsed=0. if reuse_first_round is None else float(reuse_first_round['prior_elapsed_seconds'])
    if not np.isfinite(prior_elapsed) or prior_elapsed<0.:
        raise ValueError('finite nonnegative prior source solve budget debit required')
    if not np.isfinite(amount) or abs(amount)>100:raise ValueError('amount must be finite within ±100')
    if np.any(problem['target_weights']!=0):raise ValueError('joint grid must have no already-frozen target objective')
    frame=geometry['frame'];across,down=[np.asarray(frame[k],float) for k in ['across','down']]
    if not np.allclose(np.array([across,down])@np.array([across,down]).T,np.eye(2),atol=1e-10,rtol=0.):
        raise ValueError('orthonormal source frame required')
    old=OriginalProblem(geometry,node_spacing_px,jaw_radius_px)
    preference=PlateauProblem(geometry,node_spacing_px,jaw_radius_px)
    jaws=[GridCurve(old.source(a,a['knots']),problem,across,down,source_knots=a['knots']) for a in old.arcs]
    sign_domains=[source_sign_regions(c,old.origin,old.midline,amount) for c in jaws]
    interfaces=[GridCurve(s['source_points'],problem,across,down) for s in interface_segments]
    full_points=np.asarray(geometry['contour']);full=GridCurve(np.vstack([full_points,full_points[0]]),problem,across,down)
    jaw_area=sum((a['direction']*c.area_row() for a,c in zip(old.arcs,jaws)),csr_matrix((1,full.size)))
    full_area=full.area_row();nonjaw_area=full_area-jaw_area
    wanted=old.beta_area(amount,.13)
    endpoint_rows=[];endpoint_keys=set()
    for c,(_,zeros) in zip(jaws,sign_domains):
        positions=np.unique(np.r_[0.,c.length,zeros])
        for s,p in zip(positions,c.source(positions)):
            if tuple(p) not in endpoint_keys:
                endpoint_keys.add(tuple(p));endpoint_rows.append(c.op([s]))
    raw_eq=[jaw_area,full_area,*endpoint_rows];raw_rhs=[wanted,wanted,*[0.]*len(endpoint_rows)]
    report={'schema':'joint-jaw-visible-interface-source-diagnostic-v1',
        'amount':float(amount),'requested_jaw_and_full_contour_area_px2':float(wanted),
        'source_continuous_accepted':False,'executable':False,'image_approval':False,
        'source_continuous_accepted_scope':'jaw/interface chord conditions, original radius sign, zero sites, both area equalities and all original source-cell ideal inverse Jacobian conditions; final f32 inverse/render excluded',
        'geometry_controller':'lazy exact rank-one inverse SOC constraints, direct all-corner inverse oracle',
        'curve_controller':'exact interval polynomial PSD2/SOC certificates on actually violating original source/grid pieces; known hard-zero quadratic sign pieces factor exactly to linear inequalities; original point cuts, hard equalities and continuous extrema oracle retained',
        'new_target_distribution':True,'source_curves_unmodified':True,'unknown_curves_joined':False,
        'anatomical_interface_assignment':'unknown','rho':.5,'interface_radius_px':interface_radius_px,
        'interface_source_relative_bend_band':None if interface_bend_band is None else list(map(float,interface_bend_band)),
        'interface_shape_constraint_change':interface_bend_band is not None,
        'interface_bend_band_scope':'signed target/source finite-chord numerator ratio on unchanged known source-turn pieces only; not an angle ratio or absolute slope bound',
        'interface_displacement_bending':{'enabled':bool(regularize_interface_displacement),
            'density':float(neck_density),'smoothing_length_px':float(smoothing_length_px),
            'requested_node_spacing_px':float(node_spacing_px),
            'formula':'density * smoothing_length_px^4 * h * sum((D2 B z)^2); P=2H, no additional half',
            'scope':'displacement only on uniform original arclength across each complete observed interface; no source-curvature fitting, endpoint slope constraint, new hard domain or joined anatomy',
            'units':'px^3; matching the existing jaw displacement objective',
            'limits':'discrete second derivative at interior nodes; quadrature spans L-h; affine u(s) has zero energy, not all spatial affine fields'},
        'jaw_start_displacement_regularization':{'enabled':bool(regularize_jaw_start),
            'length_px':float(smoothing_length_px),'additional_weight':1.,
            'formula':'length_px^2 * integral_0^length_px w(s/length_px) * (du/ds)^2 ds; P=2H, no additional half',
            'taper':'w(t)=1-10t^3+15t^4-6t^5',
            'scope':'known original jaw prefix only; no continuation above the jaw, endpoint tangent equality, area/support/band modification or weight scan',
            'units':'px^3; matching the existing jaw displacement objective',
            'quadrature':'four Gauss points per original source-edge/grid-cell piece split at the exact window end; polynomial degree seven'},
        'jaw_start_intent_constraints':{'enabled':bool(constrain_jaw_start_intent),
            'new_hard_shape_intent_constraint':bool(constrain_jaw_start_intent),'feasibility_assumed':False,
            'formula':'each side and prefix: smoothing_length^2 * integral_0^prefix (du/ds)^2 <= same integral of original PlateauProblem.beta derivative',
            'prefix_definition':'[node_spacing, smoothing_length/2, smoothing_length] on the original known source jaw',
            'budget_source':'existing unsolved equal-area Plateau beta; no adjusted total strength, ratio, side pooling or weight scan',
            'field_integral':'four-Gauss exact degree-two integrand on original source/grid pieces; no taper',
            'preference_integral':'nine-Gauss exact degree-ten integrand on original radius/quintic pieces; no taper'},
        'jaw_radius_px':jaw_radius_px,'source_sign_threshold_px2':1e-8,
        'nonjaw_area_row_norm':float(np.linalg.norm(nonjaw_area.data)),
        'objective':'existing node-sampled plateau preference + same discrete jaw smoothing and bilateral1; portable grid energy; existing discrete neck soft0 quadrature',
        'maximum_refinement_rounds':int(max_rounds),'refinement':[],
        'intermediate_almost_solved_policy':'finite AlmostSolved may only provide new original geometry/curve/sign constraints; never accepted; no-new-constraint and other unsuccessful status stop',
        'reused_first_round':None if reuse_first_round is None else {
            'source_report':reuse_first_round['source_report'],'subproblem_sha256':reuse_first_round['subproblem_sha256'],
            'prior_elapsed_seconds':prior_elapsed,'counted_as_round':0,'new_round_budget_not_reset':True},
        'total_solve_budget_seconds':60.,'maximum_round_seconds':15.,
        'numerical_solver_tolerances':{'primal_feasibility':1e-12,'absolute_objective_gap':1e-10,'relative_objective_gap':1e-10},
        'same_subproblem_strict_polish':{'maximum_attempts':int(bool(allow_strict_polish)),'only_after_no_new_constraint_and_remaining_geometry_violation':True,
            'primal_feasibility':1e-13,'absolute_objective_gap':1e-12,'relative_objective_gap':1e-12},
        'jaw_sign_definition':'sign(-amount * original source radius); source midline crossings partitioned; zero-radius sites constrained to zero, unknown anatomy unchanged',
        'source_parameterization':'original jaw cumulative arclength knots retained exactly; chord right endpoint uses symbolic (L-r)+r=L only for an exactly equal center; out-of-range source/center inputs rejected',
        'limits':'Source C0 bilinear field only. No f32 inverse, rendering, beauty or executable-profile approval.'}
    try:eq,eq_rhs,redundant=_independent_equalities(raw_eq,raw_rhs)
    except ValueError as error:
        report['rejection']=str(error);return {'report':report,'diagnostic_solution':None}
    report['equality_count']=len(eq_rhs);report['redundant_equalities']=redundant
    report['nonjaw_area_requirement']='full_contour_area == jaw_area == original beta-cubed request; no fixed pixels added'
    operators=[];blocks=[];weights=[];beta=[]
    for arc,curve in zip(preference.arcs,jaws):
        n=len(arc['nodes']);h=arc['length']/(n-1);w=np.full(n,h);w[[0,-1]]*=.5
        d2=np.diff(np.eye(n),n=2,axis=0)/h**2
        blocks.append(np.diag(w)+smoothing_length_px**4*h*(d2.T@d2))
        operators.append(curve.op(arc['nodes']));weights.append(w)
        beta.append(preference.beta(arc,arc['nodes'],amount,.13))
    j=vstack(operators,format='csr');jaw_h=block_diag(*blocks)
    jaw_h+=build_bilateral_coupling(preference,_basis).hessian
    hessian=(problem['hessian']+j.T@csc_matrix(jaw_h)@j).tocsc()
    linear=-np.asarray(j.T@(np.concatenate(weights)*np.concatenate(beta))).ravel()
    soft=sample_boundary_soft_targets([s['source_points'] for s in neck_segments],
        spacing=2.,density=neck_density,fade_length=neck_fade_px,
        uncertainty_px=[s['uncertainty_radius_px'] for s in neck_segments])
    neck_op=point_operators(soft['source'],problem['grid'],problem['mapping'],full.size)[0]
    hessian=(hessian+neck_op.T@diags(soft['weights'])@neck_op).tocsc()
    report['neck_soft0']=soft['report']
    interface_regularizers=[(segment['id'],interface_displacement_bending(
        curve,neck_density,smoothing_length_px,node_spacing_px))
        for segment,curve in zip(interface_segments,interfaces)]
    if regularize_interface_displacement:
        for _,regularizer in interface_regularizers:
            hessian=(hessian+regularizer['hessian']).tocsc()
        report['objective']+='; full observed interface displacement bending on original arclength'
    jaw_start_regularizers=[]
    if regularize_jaw_start:
        jaw_start_regularizers=[(arc['id'],jaw_start_displacement_regularizer(curve,smoothing_length_px))
                                for arc,curve in zip(old.arcs,jaws)]
        for _,regularizer in jaw_start_regularizers:
            hessian=(hessian+regularizer['hessian']).tocsc()
        report['objective']+='; known jaw-start finite-window displacement derivative energy'
    start_intent_blocks=jaw_prefix_intent_blocks(preference,jaws,amount,.13,smoothing_length_px,node_spacing_px) if constrain_jaw_start_intent else []
    report['jaw_start_intent_constraints']['budgets']=[block['report'] for block in start_intent_blocks]
    if constrain_jaw_start_intent:
        report['source_continuous_accepted_scope']+='; all six independent original-Plateau prefix-energy budgets'
    reports=[]
    for arc,curve in zip(old.arcs,jaws):
        regions=_refine_regions(old.convex_regions(arc),curve.chord_partition(jaw_radius_px),arc['side'])
        reports.append((arc['id'],curve,regions,jaw_radius_px,.5))
    for segment,curve in zip(interface_segments,interfaces):
        reports.extend(interface_chord_conditions(segment['id'],curve,interface_radius_px,interface_bend_band))
    matrix_rows=[];thresholds=[];seen_planes=set()
    def add_rows(matrix,lower):
        matrix=matrix.tocsr();keep=[]
        for index,value in enumerate(lower):
            row=matrix.getrow(index);scale=max(float(np.linalg.norm(row.data)),1.)
            signature=(tuple(row.indices),tuple(np.round(row.data/scale,12)),round(float(value)/scale,12))
            if signature not in seen_planes:
                seen_planes.add(signature);keep.append(index)
        if keep:
            matrix_rows.append(matrix[keep]);thresholds.append(np.asarray(lower)[keep])
    def add_chord(curve,positions,signs,radius,rho):
        if not len(positions):return
        ns,op,_=curve.chord(positions,radius)
        add_rows(diags(signs)@op,-signs*(1-rho)*ns)
    def add_sign(curve,positions,sign):
        signs=np.broadcast_to(sign,len(positions));add_rows(diags(signs)@curve.op(positions),np.zeros(len(positions)))
    for _,curve,regions,radius,rho in reports:
        if len(regions):
            s=np.concatenate([regions[:,0],regions[:,:2].mean(1),regions[:,1]])
            signs=np.tile(regions[:,2],3);add_chord(curve,s,signs,radius,rho)
    for curve,(regions,_) in zip(jaws,sign_domains):
        positions=np.concatenate([regions[:,0],regions[:,:2].mean(1),regions[:,1]])
        add_sign(curve,positions,np.tile(regions[:,2],3))
    z=None;solver_ok=False;diagnostic_solver_state=None;diagnostic_qp=None
    strict_pending=False;strict_used=False
    geometry_rows=[];geometry_rhs=[];geometry_cones=[];seen_corners=set()
    intent_rows=[block['A'] for block in start_intent_blocks]
    intent_rhs=[block['b'] for block in start_intent_blocks]
    intent_cones=[clarabel.SecondOrderConeT(len(block['b'])) for block in start_intent_blocks]
    polynomial_blocks=[];polynomial_keys=[];seen_pieces=set();factorized_sign_pieces=set()
    def add_polynomial(key,coefficients):
        if key in seen_pieces:return 0
        seen_pieces.add(key);polynomial_keys.append(key);polynomial_blocks.append(coefficients)
        return 1
    def add_geometry(indices):
        fresh=[tuple(map(int,index)) for index in indices if tuple(map(int,index)) not in seen_corners]
        if not fresh:return 0
        p_rows,b_rows=corner_gradient_rows(problem,fresh,across,down)
        zero=csr_matrix((1,full.size))
        for index,p,b in zip(fresh,p_rows,b_rows):
            # Clarabel s=b-Az. The first coordinate is each SOC radius.
            geometry_rows.extend([vstack([zero,-1.04*p,-np.sqrt(.52/.48)*b],format='csr'),
                                  vstack([-3.6*p,zero,-np.sqrt(1.8/.8)*b],format='csr'),
                                  vstack([-.48*p-b,-.48*p+b],format='csr')])
            geometry_rhs.extend([np.array([1.,.04,0.]),np.array([2.6,1.,0.]),np.array([.48,.48])])
            geometry_cones.extend([clarabel.SecondOrderConeT(3),clarabel.SecondOrderConeT(3),clarabel.NonnegativeConeT(2)])
            seen_corners.add(index)
        return len(fresh)
    report['assembly_seconds']=time.monotonic()-started
    for iteration in range(max_rounds+1):
        is_strict=strict_pending;strict_pending=False
        if iteration>=max_rounds and not is_strict:break
        remaining=60.-prior_elapsed-(time.monotonic()-started)
        if remaining<=0:report['rejection']='TimedOut: total source solve budget exhausted';break
        if not is_strict:
            matrix=vstack(matrix_rows,format='csr');lower=np.concatenate(thresholds)
            rownorm=np.sqrt(np.asarray(matrix.multiply(matrix).sum(1)).ravel())
            zero=rownorm<=1e-15
            if np.any(zero & (lower>1e-10)):
                report['rejection']='Fixed spatial support conflicts with a source-curve constraint';break
            matrix=matrix[~zero];lower=lower[~zero];rownorm=rownorm[~zero]
            scale=np.maximum(rownorm,1.)
            matrix=diags(1/scale)@matrix;lower=lower/scale
            extra_variables=6*len(polynomial_blocks);total_variables=full.size+extra_variables
            base=vstack([eq,-matrix,*intent_rows,*geometry_rows],format='csc')
            a=hstack([base,csr_matrix((base.shape[0],extra_variables))],format='csc')
            b=np.concatenate([eq_rhs,-lower,*intent_rhs,*geometry_rhs])
            cones=[clarabel.ZeroConeT(len(eq_rhs)),clarabel.NonnegativeConeT(len(lower)),*intent_cones,*geometry_cones]
            blocks=[nonnegative_cubic_block(coefficients,constant,
                        auxiliary_start=full.size+6*index,total_variables=total_variables)
                    for index,(coefficients,constant) in enumerate(polynomial_blocks)]
            if blocks:
                a=vstack([a,*[block['A'] for block in blocks]],format='csc')
                b=np.concatenate([b,*[block['b'] for block in blocks]])
                cones.extend(cone for block in blocks for cone in block['cones'])
            solver_h=sparse_block_diag([2*hessian,csc_matrix((extra_variables,extra_variables))],format='csc')
            solver_q=np.r_[2*linear,np.zeros(extra_variables)]
            cone_description=[{'type':type(cone).__name__,'dim':cone.dim} for cone in cones]
            subproblem_hash=qp_fingerprint(solver_h,solver_q,a,b,cone_description)
        else:
            strict_used=True
            if qp_fingerprint(solver_h,solver_q,a,b,cone_description)!=subproblem_hash:
                raise AssertionError('strict numerical polish must use the unchanged conic QP')
        remaining=60.-prior_elapsed-(time.monotonic()-started)
        if remaining<=0:report['rejection']='TimedOut: total source solve budget exhausted';break
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=160
        settings.time_limit=min(15.,remaining)
        settings.tol_gap_abs=1e-12 if is_strict else 1e-10
        settings.tol_gap_rel=1e-12 if is_strict else 1e-10
        settings.tol_feas=1e-13 if is_strict else 1e-12
        diagnostic_qp={'P':solver_h.copy(),'q':solver_q.copy(),'A':a.copy(),'b':b.copy(),
            'cones':cone_description,'sha256':subproblem_hash,'scalar_variables':full.size,
            'settings':{name:getattr(settings,name) for name in dir(settings)
                        if not name.startswith('_') and not callable(getattr(settings,name))}}
        round_started=time.monotonic()
        reused=iteration==0 and reuse_first_round is not None
        if reused:
            if subproblem_hash!=reuse_first_round['subproblem_sha256']:
                raise ValueError('reused first-round QP fingerprint must exactly match the rebuilt unchanged numerical problem')
            state=reuse_first_round['state'];old_row=reuse_first_round['row']
            if (np.shape(state['primal_vector'])!=(total_variables,)
                or np.shape(state['primal_slack'])!=(a.shape[0],)
                or np.shape(state['dual_vector'])!=(a.shape[0],)
                or old_row['solver_status']!='AlmostSolved'
                or any(not np.isfinite(value).all() for value in state.values())):
                raise ValueError('finite complete same-dimension AlmostSolved first-round state required')
            solved=SimpleNamespace(status=clarabel.SolverStatus.AlmostSolved,x=state['primal_vector'],
                s=state['primal_slack'],z=state['dual_vector'],iterations=old_row['solver_iterations'],
                r_prim=old_row['solver_primal_residual'],r_dual=old_row['solver_dual_residual'])
        else:solved=clarabel.DefaultSolver(solver_h,solver_q,a,b,cones,settings).solve()
        solver_ok=solved.status==clarabel.SolverStatus.Solved
        finite_state=all(np.isfinite(value).all() for value in [solved.x,solved.s,solved.z])
        z=np.asarray(solved.x[:full.size])
        if np.isfinite(solved.x).all():
            diagnostic_solver_state={'primal_vector':np.asarray(solved.x),
                'primal_slack':np.asarray(solved.s),'dual_vector':np.asarray(solved.z)}
        row={'round':iteration,'solver_status':str(solved.status),'constraint_rows':len(lower),'curves':[],'sign':[],
             'same_subproblem_strict_polish':is_strict,'subproblem_sha256':subproblem_hash,
             'solver_seconds':old_row['solver_seconds'] if reused else time.monotonic()-round_started,
             'solver_state_reused_without_resolve':reused,'round_budget_seconds':settings.time_limit,
             'geometry_constrained_corners':len(seen_corners),
             'interval_polynomial_certificates':len(polynomial_blocks),'auxiliary_variables':extra_variables,
             'interval_certificate_piece_keys':[list(key) for key in polynomial_keys],
             'hard_zero_factorized_sign_pieces':len(factorized_sign_pieces),
             'solver_iterations':solved.iterations,
             'solver_primal_residual':float(solved.r_prim) if np.isfinite(solved.r_prim) else None,
             'solver_dual_residual':float(solved.r_dual) if np.isfinite(solved.r_dual) else None,
             'solver_absolute_objective_gap':old_row['solver_absolute_objective_gap'] if reused else (
                 float(abs(solved.obj_val-solved.obj_val_dual)) if np.isfinite([solved.obj_val,solved.obj_val_dual]).all() else None)}
        if not finite_state:
            report['refinement'].append(row);report['rejection']='TimedOut' if str(solved.status)=='MaxTime' else 'Solver did not return Solved within fixed budget';break
        count=0
        previous_planes=len(seen_planes);previous_pieces=len(seen_pieces);previous_polynomials=len(polynomial_blocks)
        for name,curve,regions,radius,rho in reports:
            checked,positions,signs=curve.continuous_chord_check(z,regions,radius,rho)
            row['curves'].append({'id':name,**checked});count+=checked['violations']
            add_chord(curve,positions,signs,radius,rho)
            for index in checked.get('violating_piece_indices',[]):
                add_polynomial(('chord',name,index),curve.chord_polynomial(regions[index],radius,rho))
        for arc,curve,(regions,zeros) in zip(old.arcs,jaws,sign_domains):
            checked,positions=curve.continuous_sign_check(z,None,regions)
            row['sign'].append({'id':arc['id'],**checked});count+=checked['violations']
            if len(positions):
                signs=-np.sign(amount*((curve.source(positions)-old.origin)@across-old.midline))
                add_sign(curve,positions,signs)
            for index in checked['violating_piece_indices']:
                key=('sign',arc['id'],index)
                factored=hard_zero_sign_piece_rows(curve,regions[index],zeros)
                if factored is not None:
                    if key not in seen_pieces:
                        add_rows(factored,np.zeros(factored.shape[0]))
                        seen_pieces.add(key);factorized_sign_pieces.add(key)
                else:add_polynomial(key,curve.sign_polynomial(regions[index]))
        geometry_check,bad_corners=all_corner_inverse_check(problem,z,across)
        row['ideal_inverse_source_geometry']=geometry_check
        row['new_geometry_corners']=add_geometry(bad_corners)
        count+=geometry_check['violating_source_cell_corners']
        row['jaw_start_intent']=check_jaw_prefix_intent(start_intent_blocks,z)
        count+=row['jaw_start_intent']['violations']
        row['jaw_area_error_px2']=float((jaw_area@z)[0]-wanted)
        row['full_contour_area_error_px2']=float((full_area@z)[0]-wanted)
        row['endpoint_max_px']=max(abs(float((r@z)[0])) for r in endpoint_rows)
        report['refinement'].append(row)
        row['new_distinct_cut_planes']=len(seen_planes)-previous_planes
        row['new_interval_polynomial_certificates']=len(polynomial_blocks)-previous_polynomials
        row['new_exact_interval_constraints']=len(seen_pieces)-previous_pieces
        if not solver_ok:
            new_constraints=(row['new_distinct_cut_planes']>0 or row['new_geometry_corners']>0 or row['new_exact_interval_constraints']>0)
            if solved.status==clarabel.SolverStatus.AlmostSolved and finite_state and new_constraints and not is_strict:
                row['intermediate_almost_solved_separation_only']=True
                row['next_action']='Append newly discovered original geometry/curve/sign constraints and continue; this state cannot be accepted'
                continue
            report['rejection']='TimedOut' if str(solved.status)=='MaxTime' else 'Solver did not return Solved within fixed budget'
            row['physical_checks_are_failed_solver_diagnostics_only']=True
            break
        area_ok=max(abs(row['jaw_area_error_px2']),abs(row['full_contour_area_error_px2']))<=max(1e-8,abs(wanted)*1e-10)
        if count==0:
            report['source_continuous_accepted']=bool(area_ok and row['endpoint_max_px']<=1e-10)
            break
        if is_strict:
            report['rejection']='Same-subproblem strict numerical polish failed original physical constraints'
            break
        if row['new_distinct_cut_planes']==0 and row['new_geometry_corners']==0 and row['new_exact_interval_constraints']==0:
            if allow_strict_polish and not strict_used and geometry_check['violating_source_cell_corners']>0:
                strict_pending=True
                row['next_action']='One strict numerical solve of exactly this conic subproblem; no changed objective or constraint'
                continue
            report['rejection']='Stagnant continuous violation: no new distinct cut plane, interval certificate or source corner';break
    if not report['source_continuous_accepted'] and 'rejection' not in report:
        report['rejection']='Continuous physical constraints or area/endpoints failed within fixed refinement budget'
    if z is not None and np.isfinite(z).all():
        report['jaw_area_px2']=float((jaw_area@z)[0]);report['full_contour_area_px2']=float((full_area@z)[0])
        report['nonjaw_area_px2']=float((nonjaw_area@z)[0])
        report['neck_motion_max_px']=float(abs(neck_op@z).max())
        report['neck_weighted_squared_motion']=float(soft['weights']@((neck_op@z)**2))
        report['interface_displacement_bending'].update(interface_displacement_bending_energy(interface_regularizers,z))
        report['jaw_start_displacement_regularization'].update(jaw_start_displacement_energy(jaw_start_regularizers,z))
        report['jaw_start_intent_constraints']['final_check']=check_jaw_prefix_intent(start_intent_blocks,z)
        report['source_field_geometry']=source_field_geometry(problem,z,across)
        report['continuous_nonjaw_motion']=continuous_nonjaw_motion(full,old,z)
    report['current_source_solve_seconds']=time.monotonic()-started
    report['prior_source_solve_seconds']=prior_elapsed
    report['total_source_solve_seconds']=prior_elapsed+report['current_source_solve_seconds']
    # Even a feasible source solution is diagnostic-only. Caller may save raw
    # arrays for independent review, never candidate-profile/executable output.
    return {'report':report,'diagnostic_solution':z if z is not None and np.isfinite(z).all() else None,
            'diagnostic_solver_state':diagnostic_solver_state,
            'diagnostic_qp':diagnostic_qp,
            'interface_displacement_regularizers':interface_regularizers,
            'jaw_start_regularizers':jaw_start_regularizers,
            'jaw_start_intent_blocks':start_intent_blocks,
            'curves':jaws+interfaces,'full_curve':full,'jaw_problem':old}

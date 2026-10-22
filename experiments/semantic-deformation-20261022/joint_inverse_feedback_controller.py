"""Bounded nonlinear inverse-field correction of a complete frozen source QP.

Source cuts are exact and accumulate. Inverse tangents are local and must be
replaced after relocating samples at each accepted iterate. No physical gate,
edit area, original objective or source amplitude is relaxed by this module.
"""
from pathlib import Path

import clarabel
import numpy as np
from scipy.sparse import csr_matrix, eye, hstack, load_npz, vstack

from joint_jaw_boundary_field import qp_fingerprint
from probe_joint_inverse_feedback import read, sha


def load_frozen(reference_path,interface_path):
    """Verify and load exact source/QP evidence; no solve or file mutation."""
    reference_path,interface_path=map(lambda p:Path(p).resolve(),[reference_path,interface_path])
    reference,interface=read(reference_path),read(interface_path)
    folder=reference_path.parent/'diagnostic-qp'
    state_path=reference_path.parent/'diagnostic-source-state.npz'
    hashes={str(reference_path):sha(reference_path),str(interface_path):reference['inputs_sha256'][str(interface_path)],
        str(state_path):reference['diagnostic_source_state_sha256'],
        str(folder/'metadata.json'):reference['diagnostic_qp_metadata_sha256']}
    if any(sha(path)!=digest for path,digest in hashes.items()):
        raise ValueError('frozen source/QP evidence changed')
    metadata=read(folder/'metadata.json')
    hashes.update({str(folder/name):digest for name,digest in metadata['files_sha256'].items()})
    if any(sha(path)!=digest for path,digest in hashes.items()):
        raise ValueError('frozen QP data changed')
    p,a=load_npz(folder/'P.npz'),load_npz(folder/'A.npz')
    with np.load(folder/'vectors.npz') as vectors:q,b=vectors['q'],vectors['b']
    if qp_fingerprint(p,q,a,b,metadata['cones'])!=metadata['sha256']:
        raise ValueError('frozen QP fingerprint mismatch')
    with np.load(state_path) as state:z0,mapping=state['scalar_solution'],state['mapping']
    size=metadata['scalar_variables'];total=p.shape[0]
    if (p.shape!=(total,total) or a.shape[1]!=total or q.shape!=(total,) or b.shape!=(a.shape[0],)
            or z0.shape!=(size,) or total<size or not all(np.isfinite(v).all() for v in [p.data,a.data,q,b,z0])):
        raise ValueError('finite complete original source/auxiliary QP dimensions required')
    cones=[getattr(clarabel,c['type'])(c['dim']) for c in metadata['cones']]
    if sum(c.dim for c in cones)!=len(b):raise ValueError('frozen cones do not exhaust all constraints')
    return {'reference':reference,'interface':interface,'metadata':metadata,'P':p,'q':q,
        'A':a,'b':b,'cones':cones,'z0':z0,'mapping':mapping,'hashes':hashes,
        'reference_path':reference_path,'interface_path':interface_path}


def _atomic_blocks(block):
    """Nonnegative cones may be split into rows; SOCs must stay intact."""
    a=block['A'].tocsr();b=np.asarray(block['b']);offset=0
    for cone in block['cones']:
        stop=offset+cone.dim
        if type(cone).__name__=='NonnegativeConeT':
            for i in range(offset,stop):yield a[i:i+1],b[i:i+1],clarabel.NonnegativeConeT(1)
        else:yield a[offset:stop],b[offset:stop],cone
        offset=stop
    if offset!=a.shape[0] or b.shape!=(offset,):raise ValueError('complete cone block required')


def _constraint_key(matrix,rhs,cone):
    matrix=matrix.tocsr(copy=True);matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
    return (type(cone).__name__,cone.dim,matrix.shape,tuple(matrix.indptr),tuple(matrix.indices),
            matrix.data.tobytes(),np.asarray(rhs,dtype=float).tobytes())


class ExactCutLedger:
    """Prevent frozen constraints and repeated cuts from masquerading as progress."""
    def __init__(self,base):
        self.variables=base['A'].shape[1];self.blocks=[]
        self.seen={_constraint_key(a,b,c) for a,b,c in _atomic_blocks(base)}

    def add(self,candidate):
        if candidate['A'].shape[1]!=self.variables:raise ValueError('original total variable count changed')
        rows=[];rhs=[];cones=[];duplicate=0
        for a,b,c in _atomic_blocks(candidate):
            key=_constraint_key(a,b,c)
            if key in self.seen:duplicate+=1;continue
            self.seen.add(key);rows.append(a);rhs.append(b);cones.append(c)
        if rows:self.blocks.append({'A':vstack(rows,format='csc'),'b':np.concatenate(rhs),'cones':cones})
        return {'new_cone_blocks':len(cones),'new_rows':sum(c.dim for c in cones),
                'duplicate_cone_blocks':duplicate,
                'oracle_reported_constraints':candidate.get('new_constraints')}


def assemble_trial(base,source_blocks,feedback,z0,anchor,radius):
    """Keep original prefix verbatim; only this round's tangent is appended."""
    z0,anchor=np.asarray(z0,float),np.asarray(anchor,float)
    n=len(z0);total=base['A'].shape[1]
    if anchor.shape!=z0.shape or not 0.<radius<=.25:raise ValueError('bounded original source trust radius required')
    # Intersection of the original experiment's total box and the current
    # accepted-iterate trust box. Auxiliary certificate variables are unboxed.
    lower=np.maximum(z0-.25,anchor-radius);upper=np.minimum(z0+.25,anchor+radius)
    if np.any(lower>upper):raise ValueError('source trust boxes do not intersect')
    trust=hstack([eye(n,format='csr'),csr_matrix((n,total-n))],format='csr')
    blocks=[base,*source_blocks,feedback]
    return {'A':vstack([*(b['A'] for b in blocks),trust,-trust],format='csc'),
        'b':np.concatenate([*(b['b'] for b in blocks),upper,-lower]),
        'cones':[*(c for b in blocks for c in b['cones']),clarabel.NonnegativeConeT(2*n)]}


def geometry_merit(geometry):
    """Maximum original final-Jacobian violation, without arbitrary weights."""
    values=np.array([.52-geometry['symmetric_eigenvalue_min'],
        geometry['symmetric_eigenvalue_max']-1.8,geometry['skew_absolute_max']-.24,
        -geometry['jacobian_determinant_min'],0.])
    return float(values.max()) if np.isfinite(values).all() else float('inf')


class FeedbackState:
    """Small tested acceptance machine; it never evaluates a frozen-cell model."""
    def __init__(self,z0,initial_actual,initial_corners):
        self.z0=np.asarray(z0,float).copy();self.anchor=self.z0.copy()
        if self.z0.ndim!=1 or not np.isfinite(self.z0).all():raise ValueError('finite source coefficients required')
        self.merit=geometry_merit(initial_actual['geometry'])
        self.corners={tuple(map(int,p)) for p in initial_corners}
        self.radius=.25;self.attempts=0;self.accepted=False

    def observe(self,status,z,source,actual,bad,new_source_blocks):
        """Only fresh full-domain audits may be supplied as `actual`."""
        self.attempts+=1;z=np.asarray(z,float)
        old=len(self.corners);self.corners.update(tuple(map(int,p)) for p in bad)
        row={'solver_status':str(status),'attempt':self.attempts,'radius_px':self.radius,
             'new_actual_corners':len(self.corners)-old,'tracked_actual_corners':len(self.corners),
             'new_source_cone_blocks':int(new_source_blocks),'anchor_merit':self.merit}
        def end(action,reason):
            row.update(action=action,reason=reason,next_radius_px=self.radius);return row
        if z.shape!=self.anchor.shape or not np.isfinite(z).all():return end('stop','nonfinite_trial')
        step=float(abs(z-self.anchor).max(initial=0.));total=float(abs(z-self.z0).max(initial=0.))
        row.update(maximum_source_step_px=step,maximum_source_total_change_px=total)
        if str(status) not in ['Solved','AlmostSolved']:return end('stop','terminal_solver_status')
        if step>self.radius or total>.25:return end('stop','trust_box_failure')
        if source is None:return end('stop','missing_full_source_audit')
        if not source['accepted']:
            return end('exchange_source','new_exact_source_constraints') if new_source_blocks else end('stop','source_failure_without_new_exact_constraint')
        if actual is None:
            self.radius*=.5;return end('shrink','fresh_inverse_failed')
        fixed=actual['fixed'];geometry=actual['geometry']
        if fixed['changed'] or fixed['inverse_max_px']!=0. or geometry['boundary_max_displacement']!=0.:
            return end('stop','fixed_identity_failure')
        merit=geometry_merit(geometry);row['actual_merit']=merit
        curves=actual['sampled_curves']['sampled_passed'] and actual['curve_response']['passed']
        if geometry['passed_strict']:
            if not curves:return end('stop','curve_failure_without_geometry_direction')
            if str(status)!='Solved':return end('stop','solver_not_solved_without_geometry_direction')
            self.anchor=z.copy();self.merit=merit;self.accepted=True
            return end('accept','all_numerical_gates_passed')
        if merit==0.:return end('stop','failed_contract_without_positive_merit')
        if merit<self.merit:
            self.anchor=z.copy();self.merit=merit
            return end('advance','fresh_global_geometry_strictly_improved')
        self.radius*=.5
        return end('shrink','fresh_global_geometry_did_not_improve')

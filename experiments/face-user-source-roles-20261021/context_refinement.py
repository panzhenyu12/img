"""Preserve the old guaranteed continuous context zero region on a fine grid.

Only source constraint membership is inspected. Solved coefficient values are
never used to decide which context constraints to keep.
"""
import hashlib
import numpy as np


def expanded_mask(raw):
    h,w=raw.shape;pad=np.pad(raw,1,constant_values=False);out=np.zeros_like(raw)
    for y in range(3):
        for x in range(3):out |= pad[y:y+h,x:x+w]
    return out


def guaranteed_zero_patches(coefficients):
    """Open cardinal B3 patches whose every positive basis coefficient is zero.

    Coefficients outside the finite rectangle are structurally zero extension.
    A patch (x,y) occupies [x,x+1] by [y,y+1]. Its stencil is x/y-1..+2.
    """
    h,w=coefficients.shape;pad=np.pad(coefficients,((1,2),(1,2)),constant_values=True)
    patches=np.ones((h-1,w-1),bool)
    for y in range(4):
        for x in range(4):patches &= pad[y:y+h-1,x:x+w-1]
    return patches


def dependency_violations(old_zero,fine_required):
    """Every nonzero two-scale mask dependency must be structurally zero."""
    h,w=old_zero.shape;yy,xx=np.where(fine_required);bad=[];links=0
    for by in [-1,0,1]:
        cy=yy//2+by;valid_y=abs(yy-2*cy)<=2
        for bx in [-1,0,1]:
            cx=xx//2+bx;positive=valid_y&(abs(xx-2*cx)<=2)
            inside=positive&(cy>=0)&(cy<h)&(cx>=0)&(cx<w);links+=int(inside.sum())
            okay=np.ones(len(xx),bool);okay[inside]=old_zero[cy[inside],cx[inside]]
            for i in np.flatnonzero(~okay):bad.append({'fine':[int(xx[i]),int(yy[i])],'old_parent':[int(cx[i]),int(cy[i])]})
    return bad,links


def preserve_context(grid,old_context_nodes):
    w,h=grid['width'],grid['height']
    if not isinstance(w,int) or not isinstance(h,int) or min(w,h)<5:
        raise ValueError('at least five nodes per original grid dimension required')
    ids=np.asarray(old_context_nodes)
    if ids.ndim!=1 or (ids.size and not np.issubdtype(ids.dtype,np.integer)) or np.any(ids<0) or np.any(ids>=w*h):
        raise ValueError('valid original context node indices required')
    raw=np.zeros((h,w),bool);raw.ravel()[ids.astype(np.intp)]=True
    # Native automatically prescribes the boundary and expands one coefficient
    # ring. Include that existing guarantee before deriving zero patches.
    raw[[0,-1],:]=True;raw[:,[0,-1]]=True
    old_zero=expanded_mask(raw);old_patches=guaranteed_zero_patches(old_zero)
    fine_patches=np.repeat(np.repeat(old_patches,2,axis=0),2,axis=1)
    fh,fw=2*h-1,2*w-1;fine_raw=np.zeros((fh,fw),bool)
    yy,xx=np.where(fine_patches)
    for y in [0,1]:
        for x in [0,1]:fine_raw[yy+y,xx+x]=True
    fine_raw[[0,-1],:]=True;fine_raw[:,[0,-1]]=True
    fine_zero=expanded_mask(fine_raw);actual_fine_patches=guaranteed_zero_patches(fine_zero)
    missing=fine_patches&~actual_fine_patches
    bad,links=dependency_violations(old_zero,fine_zero)
    if np.any(missing) or bad:
        raise ValueError(f'context refinement lost {int(missing.sum())} patches or introduced {len(bad)} unsupported zero dependencies')
    return {'raw_nodes':np.flatnonzero(fine_raw).tolist(),
            'expanded_nodes':np.flatnonzero(fine_zero).tolist(),
            'old_zero_patch_mask':old_patches,'required_fine_zero_patch_mask':fine_patches,
            'actual_fine_zero_patch_mask':actual_fine_patches,'old_structural_zero_coefficients':old_zero,
            'proof':{'old_grid':[w,h],'fine_grid':[fw,fh],
                'old_expanded_context_plus_boundary_coefficients':int(old_zero.sum()),
                'old_guaranteed_zero_patches':int(old_patches.sum()),
                'fine_required_zero_patches':int(fine_patches.sum()),
                'fine_actual_zero_patches':int(actual_fine_patches.sum()),
                'lost_old_region_fine_patches':int(missing.sum()),
                'additional_fine_zero_patches':int((actual_fine_patches&~fine_patches).sum()),
                'fine_raw_context_and_boundary_count':int(fine_raw.sum()),
                'fine_expanded_context_and_boundary_count':int(fine_zero.sum()),
                'checked_nonzero_refinement_parent_links':links,
                'unsupported_refinement_parent_links':len(bad),
                'old_zero_patch_mask_sha256':hashlib.sha256(old_patches.tobytes()).hexdigest(),
                'fine_required_patch_mask_sha256':hashlib.sha256(fine_patches.tobytes()).hexdigest(),
                'uses_actual_coefficient_values':False,
                'zero_extension_and_original_two_boundary_rings_retained':True}}

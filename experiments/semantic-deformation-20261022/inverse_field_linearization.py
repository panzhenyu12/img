"""Local source-coefficient derivative of exact inverse point samples.

This is a diagnostic linearization, not an accepted edited image or a global
convex constraint on the nonlinear inverse. No output field is clipped.
"""
import numpy as np
from scipy.sparse import diags

from portable_field import point_operators
from forward import inverse_points, source_geometry


def linearize_inverse_samples(grid, mapping, z, across, destinations):
    """Linearize the scalar inverse motion v(q), where s=q+across*v(q)."""
    z=np.asarray(z,float);mapping=np.asarray(mapping);across=np.asarray(across,float)
    destinations=np.asarray(destinations,float);spacing=np.asarray(grid['spacing'],float)
    if (spacing.shape!=(2,) or not np.isfinite(spacing).all() or spacing[0]<=0 or spacing[0]!=spacing[1]
            or across.shape!=(2,) or not np.isfinite(across).all()
            or abs(np.linalg.norm(across)-1.)>1e-12
            or z.ndim!=1 or not np.isfinite(z).all()
            or mapping.shape!=(grid['width']*grid['height'],) or mapping.dtype.kind not in 'iu'
            or np.any(mapping < -1) or np.any(mapping>=len(z))
            or destinations.ndim!=2 or destinations.shape[1]!=2 or not np.isfinite(destinations).all()):
        raise ValueError('finite matching scalar source field, unit axis and isotropic grid required')
    scalar=np.zeros(len(mapping));valid=mapping>=0;scalar[valid]=z[mapping[valid]]
    field=(scalar[:,None]*across).reshape(grid['height'],grid['width'],2)
    geometry=source_geometry(field/spacing)
    if geometry['strong_monotonicity_lower_bound']<=0 or not geometry['boundary_zero']:
        raise ValueError('strongly monotone source field with zero boundary required')
    q=(destinations-grid['origin'])/spacing
    source,closure=inverse_points(field/spacing,q)
    source=source*spacing+grid['origin']
    v=(source-destinations)@across
    basis,dx,dy=point_operators(source,grid,mapping,len(z))
    directional_jacobian=1.+across[0]*(dx@z)+across[1]*(dy@z)
    if np.any(directional_jacobian<=0.):
        raise ValueError('positive directional Jacobian required for inverse differentiation')
    # v + u_z(q+a*v) = 0, hence dv/dz = -B(s)/(1+a.grad(u(s))).
    rows=(-diags(1./directional_jacobian)@basis).tocsr()
    uv=(source-grid['origin'])/spacing
    boundary_distance=np.min(np.minimum(uv%1.,1.-uv%1.),axis=1)*spacing[0]
    return {'source':source,'inverse_scalar':v,'rows':rows,'constant':v-rows@z,
            'directional_jacobian':directional_jacobian,
            'source_cell_boundary_distance_px':boundary_distance,
            'closure_grid_units':closure,
            'closure_max_source_px':closure['closure_max_px']*spacing[0],
            'linearization_only':True,
            'limits':'One-sided source-cell derivative before float32 rounding. '
                     'Every trial needs fresh inversion, curve checks and actual stored-field geometry checks.'}

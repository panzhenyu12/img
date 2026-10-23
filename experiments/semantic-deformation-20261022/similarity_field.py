"""Analytical probe of exact Q1 local-similarity strain energy.

This module is not a photograph solver or an aesthetic quality claim.

The proposed energy is .5*integral(|grad d|^2)-integral(det(grad d)).
For a continuous field whose two displacement components vanish on the whole
outer boundary, the determinant integral vanishes. Consequently this energy
is exactly half the integrated Q1 membrane energy on the current zero-collar
domain, including with interior fixed nodes. Coupled terms cancel globally;
this is not a corotational/ARAP replacement. No photograph solver is provided
because a field experiment would conflate this no-op in model class with a
change of membrane coefficient and discretization.
"""
import numpy as np
from scipy.sparse import coo_matrix


def similarity_residuals(width, height, spacing, mapping=None):
    """Return A for .5 integral[(u_x-v_y)^2+(u_y+v_x)^2] = ||Az||^2.

    Displacements are packed as all x components, then all y components.
    A mapping entry of -1 fixes the original grid node to zero displacement.
    """
    if type(width) is not int or type(height) is not int or min(width, height) < 2:
        raise ValueError('at least two integer grid nodes per axis required')
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('finite positive physical xy spacing required')
    mapping = np.arange(width*height) if mapping is None else np.asarray(mapping)
    if (mapping.shape != (width*height,) or mapping.dtype.kind not in 'iu'
            or np.any(mapping < -1)):
        raise ValueError('integer node mapping with -1 fixed nodes required')
    free = mapping[mapping >= 0]
    n = len(free)
    if not np.array_equal(np.sort(free), np.arange(n)):
        raise ValueError('one contiguous variable per free node required')
    hx, hy = spacing
    lower = (np.arange(height-1)[:, None]*width + np.arange(width-1)).ravel()
    corners = mapping[lower[:, None] + [0, 1, width, width+1]]
    valid = corners >= 0
    cells = np.arange(len(lower))[:, None]
    rows, columns, values = [], [], []
    # 2x2 Gauss is exact: each squared bilinear derivative is degree <=2
    # on either axis. Physical area/4 and the requested .5 enter once here.
    scale = np.sqrt(hx*hy/8.)
    gauss = .5 + np.array([-1., 1.])/(2*np.sqrt(3.))
    for j, ty in enumerate(gauss):
        for i, tx in enumerate(gauss):
            dx = np.array([-(1-ty), 1-ty, -ty, ty])/hx
            dy = np.array([-(1-tx), -tx, 1-tx, tx])/hy
            for component, derivatives in enumerate(((dx, -dy), (dy, dx))):
                row = np.broadcast_to(8*cells+2*(2*j+i)+component, corners.shape)
                for vector_component, derivative in enumerate(derivatives):
                    rows.append(row[valid])
                    columns.append(corners[valid]+vector_component*n)
                    values.append(np.broadcast_to(scale*derivative, corners.shape)[valid])
    return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                      shape=(8*len(lower), 2*n)).tocsr()

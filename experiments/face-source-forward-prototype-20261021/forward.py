"""Small source-domain mechanism probe, not the face editing contract.

Source displacement is a C0 bilinear field. No target-domain material sweep,
coefficient clipping, image repainting or inferred image boundaries are used.
"""
import numpy as np


def sample(displacement, points):
    """Evaluate the declared bilinear source map and its one-sided Jacobian.

    Coordinates outside the source rectangle are rejected, never clamped.
    At the upper/right edge the adjacent interior cell is used.
    """
    u = np.asarray(displacement, float); p = np.asarray(points, float)
    h, w = u.shape[:2]
    if u.shape != (h, w, 2) or min(h, w) < 2 or p.ndim != 2 or p.shape[1] != 2:
        raise ValueError('expected HxWx2 source field and Nx2 points')
    if not np.isfinite(u).all() or not np.isfinite(p).all():
        raise ValueError('nonfinite source field or query')
    if np.any(p < 0) or np.any(p > [w-1, h-1]):
        raise ValueError('source query outside rectangle; clamping is forbidden')
    xy = np.minimum(np.floor(p).astype(int), [w-2, h-2]); t = p-xy
    x, y = xy.T; tx, ty = t[:, 0, None], t[:, 1, None]
    a, b, c, d = u[y, x], u[y, x+1], u[y+1, x], u[y+1, x+1]
    value = a*(1-tx)*(1-ty)+b*tx*(1-ty)+c*(1-tx)*ty+d*tx*ty
    dx = (b-a)*(1-ty)+(d-c)*ty
    dy = (c-a)*(1-tx)+(d-b)*tx
    jac = np.stack([dx, dy], axis=-1)+np.eye(2)
    return p+value, jac


def cell_jacobians(displacement):
    """All four one-sided corners; their convex hull covers each source cell."""
    u = np.asarray(displacement, float)
    dx, dy = np.diff(u, axis=1), np.diff(u, axis=0)
    return np.stack([np.stack([x, y], axis=-1)+np.eye(2) for x, y in
        [(dx[:-1], dy[:, :-1]), (dx[:-1], dy[:, 1:]),
         (dx[1:], dy[:, :-1]), (dx[1:], dy[:, 1:])]])


def source_geometry(displacement):
    f = cell_jacobians(displacement); s = (f+f.swapaxes(-1, -2))*.5
    eigen = np.linalg.eigvalsh(s); skew = abs(f[..., 1, 0]-f[..., 0, 1])*.5
    schur = s-.52*(f.swapaxes(-1, -2)@f)
    u = np.asarray(displacement, float)
    boundary = np.concatenate([u[0], u[-1], u[1:-1, 0], u[1:-1, -1]])
    low, high = float(eigen[..., 0].min()), float(eigen[..., 1].max())
    omega, margin = float(skew.max()), float(np.linalg.eigvalsh(schur)[..., 0].min())
    return {'strong_monotonicity_lower_bound': low, 'symmetric_eigenvalue_max': high,
        'skew_absolute_max': omega, 'determinant_min': float(np.linalg.det(f).min()),
        'boundary_zero': bool(np.all(boundary == 0.)), 'source_cell_corners': len(f.reshape(-1, 2, 2)),
        'convex_sufficient_original_inverse_contract': bool(low >= 1/1.8 and margin >= 0 and omega <= .24/(1.8**2)),
        'schur_min_eigenvalue': margin,
        'limitation': 'Positive source symmetric part plus zero boundary certifies injectivity on the rectangle. The convex sufficient inverse subset is conservative. Final stored inverse cells require their own original-contract check.'}


def inverse_points(displacement, destination, tolerance=1e-10, max_iterations=40):
    """Return source positions and a measurable closure report."""
    u = np.asarray(displacement, float); q = np.asarray(destination, float)
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError('positive closure tolerance and iteration limit required')
    geometry = source_geometry(u); m = geometry['strong_monotonicity_lower_bound']
    if not np.isfinite(m) or m <= 0:
        raise ValueError('source field is not certified strongly monotone')
    source = q.copy(); h, w = u.shape[:2]; accepted_backtracks = 0
    for iteration in range(max_iterations+1):
        actual, jac = sample(u, source); residual = actual-q
        error = np.linalg.norm(residual, axis=1); active = error > tolerance
        if not np.any(active):
            return source, {'iterations': iteration, 'closure_max_px': float(error.max(initial=0)),
                'inverse_error_bound_px': float(error.max(initial=0)/m),
                'strong_monotonicity_lower_bound': m, 'backtracked_points': accepted_backtracks,
                'coordinate_clamp_used': False}
        if iteration == max_iterations:
            raise ValueError(f'inverse did not converge: max closure {error.max():.6g}px')
        ids = np.flatnonzero(active)
        step = np.linalg.solve(jac[ids], residual[ids, :, None])[..., 0]
        factor = np.ones(len(ids)); pending = np.ones(len(ids), bool)
        for _ in range(40):
            local = np.flatnonzero(pending); indices = ids[local]
            proposal = source[indices]-factor[local, None]*step[local]
            inside = np.all((proposal >= 0) & (proposal <= [w-1, h-1]), axis=1)
            trial_error = np.full(len(local), np.inf)
            if np.any(inside):
                trial, _ = sample(u, proposal[inside])
                trial_error[inside] = np.linalg.norm(trial-q[indices[inside]], axis=1)
            better = (trial_error <= tolerance) | (trial_error < error[indices])
            source[indices[better]] = proposal[better]
            accepted_backtracks += int(np.count_nonzero(better & (factor[local] < 1)))
            pending[local[better]] = False
            if not np.any(pending):
                break
            factor[local[~better]] *= .5
        if np.any(pending):
            raise ValueError('inverse line search failed; no result exported')


def source_patch_field(shape, moving_mask, fixed_mask, translation):
    """One scalar biharmonic solve with exact source material constraints.

    This is a small mechanism probe, not a face-target optimizer. The ±1px
    closed square region is fixed through every bilinear cell vertex. Hard
    constraint conflicts are errors, not mask exclusions or weaker weights.
    """
    from scipy.ndimage import binary_dilation
    from scipy.sparse import diags, eye, kron
    from scipy.sparse.linalg import spsolve
    h, w = shape; moving = np.asarray(moving_mask, bool); fixed = np.asarray(fixed_mask, bool)
    if moving.shape != (h, w) or fixed.shape != (h, w) or not moving.any():
        raise ValueError('invalid source material masks')
    t = np.asarray(translation, float)
    if t.shape != (2,) or not np.isfinite(t).all():
        raise ValueError('invalid source translation')
    moving_support = binary_dilation(moving, structure=np.ones((3, 3)))
    fixed_support = binary_dilation(fixed, structure=np.ones((3, 3)))
    fixed_support[:2] = fixed_support[-2:] = True
    fixed_support[:, :2] = fixed_support[:, -2:] = True
    if np.any(moving_support & fixed_support):
        raise ValueError('moving full source fringe conflicts with fixed material or boundary')
    tx = diags([-np.ones(w-1), 2*np.ones(w), -np.ones(w-1)], [-1, 0, 1])
    ty = diags([-np.ones(h-1), 2*np.ones(h), -np.ones(h-1)], [-1, 0, 1])
    laplacian = (kron(eye(h), tx)+kron(ty, eye(w))).tocsc()
    energy = (laplacian.T@laplacian+.05*laplacian).tocsc()
    locked = (moving_support | fixed_support).ravel(); free = np.flatnonzero(~locked)
    scalar = np.zeros(h*w); scalar[moving_support.ravel()] = 1.
    rhs = -(energy@scalar)[free]
    scalar[free] = spsolve(energy[free][:, free], rhs)
    if not np.isfinite(scalar).all():
        raise ValueError('nonfinite source solve')
    residual = float(abs((energy@scalar)[free]).max(initial=0.))
    field = scalar.reshape(h, w, 1)*t
    return field, {'moving_source_pixels': int(moving.sum()), 'fixed_source_pixels': int(fixed.sum()),
        'moving_full_square_support_nodes': int(moving_support.sum()),
        'fixed_full_square_and_boundary_nodes': int(fixed_support.sum()), 'free_scalar_variables': len(free),
        'linear_stationarity_residual_max': residual, 'translation': t.tolist(),
        'basis': 'unit source pixel grid, continuous piecewise bilinear', 'coefficient_clipping': False,
        'source_support_depends_on_translation': False, 'source_square_fringe_px': 1.}

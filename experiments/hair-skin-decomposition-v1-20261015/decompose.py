"""Source-only regularized hair/skin RGB decomposition of a learned soft alpha.

This is a small inverse problem built on frozen ZIM alpha, not a newly trained
model or a claim that alpha/hidden RGB are identifiable from one photograph.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def linearize(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def encode(linear):
    v = np.clip(linear, 0, 1)
    return np.where(v <= .0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - .055)


def local_prior(colors, core, radius=2):
    if not core.any():
        raise ValueError('Missing confident source donor layer')
    distance, nearest = distance_transform_edt(~core, return_indices=True)
    size = radius * 2 + 1
    mass = uniform_filter(core.astype('float64'), size=size, mode='nearest')
    mean = np.stack([uniform_filter(colors[..., c] * core, size=size, mode='nearest')
                     / np.maximum(mass, 1e-10) for c in range(3)], axis=-1)
    prior = mean[nearest[0], nearest[1]]
    prior[core] = colors[core]
    return prior, distance


def solve_layers(colors, alpha, domain, hair_prior, skin_prior, smooth=.12):
    """Joint sparse quadratic RGB solve in the observed mixed-edge band.

    Data + local source-color priors + neighboring-layer smoothness. No global
    average color, image-dependent constants, hand windows or hidden RGB donor.
    """
    h, w = alpha.shape
    yy, xx = np.nonzero(domain)
    n = len(xx)
    if not n:
        return hair_prior.copy(), skin_prior.copy(), {'unknownPixels': 0}
    index = np.full((h, w), -1, dtype='int32')
    index[yy, xx] = np.arange(n)
    a = alpha[domain].astype('float64')
    b = 1 - a
    # The less visible layer stays closer to its actual nearby source donors.
    prior_h = .01 + .15 * b * b
    prior_s = .01 + .15 * a * a
    diag_h, diag_s = a * a + prior_h, b * b + prior_s
    rhs_h = a[:, None] * colors[domain] + prior_h[:, None] * hair_prior[domain]
    rhs_s = b[:, None] * colors[domain] + prior_s[:, None] * skin_prior[domain]
    rows, cols, vals = [], [], []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = np.clip(xx + dx, 0, w - 1), np.clip(yy + dy, 0, h - 1)
        neighbor = index[ny, nx]
        weight = smooth * (.01 + np.abs(a - alpha[ny, nx]))
        diag_h += weight
        diag_s += weight
        internal = neighbor >= 0
        ii = np.nonzero(internal)[0]
        jj = neighbor[internal]
        rows.extend([ii, ii + n]); cols.extend([jj, jj + n]); vals.extend([-weight[internal], -weight[internal]])
        external = ~internal
        rhs_h[external] += weight[external, None] * hair_prior[ny[external], nx[external]]
        rhs_s[external] += weight[external, None] * skin_prior[ny[external], nx[external]]
    identity = np.arange(n)
    rows.extend([identity, identity + n, identity, identity + n])
    cols.extend([identity, identity + n, identity + n, identity])
    vals.extend([diag_h, diag_s, a * b, a * b])
    matrix = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n)).tocsr()
    values = spsolve(matrix, np.concatenate([rhs_h, rhs_s], axis=0))
    hair, skin = hair_prior.copy(), skin_prior.copy()
    hair[domain], skin[domain] = np.clip(values[:n], 0, 1), np.clip(values[n:], 0, 1)
    return hair, skin, {'unknownPixels': n, 'matrixNonzero': int(matrix.nnz),
                        'unclippedOutOfGamutChannels': int(((values < 0) | (values > 1)).sum())}


def decompose(rgb, alpha, hair_core, skin_core, candidate, radius,
              space='linear', refine_alpha=False, prior_radius=2):
    rgb = np.asarray(rgb, dtype='float64')
    original_alpha = np.asarray(alpha, dtype='float64')
    a = original_alpha.copy()
    colors = linearize(rgb) if space == 'linear' else rgb
    hair_prior, dh = local_prior(colors, hair_core, prior_radius)
    skin_prior, ds = local_prior(colors, skin_core, prior_radius)
    domain = candidate & (dh <= radius) & (ds <= radius) & (a > .001) & (a < .995)
    if refine_alpha:
        # First estimate alpha against independently observed source donors.
        # Fitting unconstrained layer colors first can absorb alpha error into
        # hidden colors and falsely obtain a good source reconstruction.
        direction = hair_prior - skin_prior
        lam = .0025
        fitted = (((colors - skin_prior) * direction).sum(axis=-1) + lam * original_alpha) / ((direction * direction).sum(axis=-1) + lam)
        fitted = np.clip(fitted, np.maximum(.001, original_alpha - .15), np.minimum(.995, original_alpha + .15))
        a[domain] = fitted[domain]
    hair, skin, stats = solve_layers(colors, a, domain, hair_prior, skin_prior)
    reconstruction = a[..., None] * hair + (1 - a[..., None]) * skin
    display = encode(reconstruction) if space == 'linear' else np.clip(reconstruction, 0, 1)
    residual = np.max(np.abs(display - rgb), axis=-1) * 255
    # A rejection gate, never proof of correct hidden RGB. Do not paste source
    # residual over the new backing: doing so can recreate the old skin ghost.
    trusted = domain & (residual <= 4)
    stats.update(space=space, refinedAlpha=refine_alpha, trustedPixels=int(trusted.sum()),
                 reconstructionMeanAbsByte=float(np.abs(display[domain] - rgb[domain]).mean() * 255) if domain.any() else None,
                 reconstructionMaxAbsByte=float(residual[domain].max()) if domain.any() else None,
                 alphaMaxChange=float(np.abs(a - original_alpha).max()))
    return dict(hair=hair, skin=skin, alpha=a, sourceReconstructed=display, residual=residual,
                domain=domain, trusted=trusted, hairDistance=dh, skinDistance=ds,
                space=space, stats=stats)


def recompose(result, replacement_underlayer):
    rgb = np.asarray(replacement_underlayer, dtype='float64')
    under = linearize(rgb) if result['space'] == 'linear' else rgb
    mixture = result['alpha'][..., None] * result['hair'] + (1 - result['alpha'][..., None]) * under
    return encode(mixture) if result['space'] == 'linear' else np.clip(mixture, 0, 1)

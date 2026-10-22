"""Sparse exact interval nonnegativity representation for cubic polynomials.

For real degree <= 3, the odd-degree Markov-Lukacs interval representation is
    p(t) = t*[1,t] Q [1,t]^T + (1-t)*[1,t] R [1,t]^T,  Q,R >= 0.
It is necessary and sufficient for p >= 0 on [0,1], including boundary and
interior zeros. This is an exact conic representation, not a sampled test or
a positive safety offset. Callers must still verify physical constraints after
the numerical solve; a solver status is not a roundoff-free certificate.
"""
import numpy as np
from scipy.sparse import block_diag, csr_matrix, hstack, vstack


def nonnegative_cubic_block(coefficient_rows, constant, *, auxiliary_start, total_variables):
    """Return Clarabel A, b, cones for p_coefficients = constant + C @ z.

Coefficients are ascending powers of the piece's normalized t in [0,1]. C is
4 x n_original; those original variables occupy columns [0,n_original).
The six new variables are q00,q01,q11,r00,r01,r11 at auxiliary_start. Other
auxiliary columns are zero here, so multiple blocks can share one padded QP.

The block has four coefficient equalities followed by two 3D second-order
cones, under Clarabel's A*x+s=b convention. The caller supplies zero objective
coefficients for the new variables and keeps every original hard equality.
"""
    import clarabel

    coefficients = csr_matrix(coefficient_rows, dtype=float)
    constant = np.asarray(constant, float)
    if (coefficients.shape[0] != 4 or constant.shape != (4,)
            or not np.isfinite(coefficients.data).all() or not np.isfinite(constant).all()
            or type(auxiliary_start) is not int or type(total_variables) is not int
            or auxiliary_start < coefficients.shape[1] or total_variables < auxiliary_start+6):
        raise ValueError('finite four ascending coefficients, original columns and six distinct auxiliary variables required')
    # L*[q00,q01,q11,r00,r01,r11] gives the expanded polynomial.
    expansion = csr_matrix([[0., 0., 0., 1., 0., 0.],
                            [1., 0., 0., -1., 2., 0.],
                            [0., 2., 0., 0., -2., 1.],
                            [0., 0., 1., 0., 0., -1.]])
    equalities = hstack([coefficients, csr_matrix((4, auxiliary_start-coefficients.shape[1])),
                        -expansion, csr_matrix((4, total_variables-auxiliary_start-6))], format='csr')
    # PSD([[a,b],[b,c]]) iff ||(2b,a-c)|| <= a+c. These are the
    # cone slack coordinates, so A contains their negatives when b_SOC=0.
    psd = csr_matrix([[1., 0., 1.], [0., 2., 0.], [1., 0., -1.]])
    conic_rows = hstack([csr_matrix((6, auxiliary_start)), -block_diag([psd, psd], format='csr'),
                        csr_matrix((6, total_variables-auxiliary_start-6))], format='csr')
    return {'A': vstack([equalities, conic_rows], format='csc'),
            'b': np.r_[-constant, np.zeros(6)],
            'cones': [clarabel.ZeroConeT(4), clarabel.SecondOrderConeT(3), clarabel.SecondOrderConeT(3)],
            'auxiliary_indices': np.arange(auxiliary_start, auxiliary_start+6),
            'auxiliary_order': ['q00', 'q01', 'q11', 'r00', 'r01', 'r11'],
            'coefficient_order': ['1', 't', 't^2', 't^3'],
            'interval': [0., 1.]}

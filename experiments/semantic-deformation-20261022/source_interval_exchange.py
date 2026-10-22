"""Close newly violated original curve pieces with exact interval constraints.

Point cuts remain valid but can chase a moving minimum indefinitely. A piece
is registered once, using its unchanged source parameterization. Original
auxiliary columns stay in place; newly required PSD variables append after
them. This adds no new physical condition and does not accept solver output.
"""
import clarabel
import numpy as np
from scipy.sparse import block_diag, csr_matrix, hstack, vstack

from interval_polynomial_cones import nonnegative_cubic_block
from joint_jaw_boundary_field import hard_zero_sign_piece_rows


class SourceIntervalExchange:
    def __init__(self, audit, frozen_keys):
        self.audit = audit; self.polynomials = []; self.factored = []; self.keys = []
        valid = {('chord', name, i) for name, _, regions, _, _ in audit.specifications for i in range(len(regions))}
        valid.update(('sign', arc['id'], i) for arc, (regions, _) in zip(audit.old.arcs, audit.domains)
                     for i in range(len(regions)))
        self.seen = set(map(tuple, frozen_keys))
        if not self.seen <= valid:
            raise ValueError('frozen certificates must refer to original source pieces')

    def discover(self, z):
        z = np.asarray(z, float)
        if z.shape != (self.audit.size,) or not np.isfinite(z).all():
            raise ValueError('finite original scalar source state required')
        before = len(self.polynomials); before_factored = len(self.factored)
        for name, curve, regions, radius, rho in self.audit.specifications:
            checked, _, _ = curve.continuous_chord_check(z, regions, radius, rho)
            for i in checked.get('violating_piece_indices', []):
                key = ('chord', name, i)
                if key not in self.seen:
                    self.polynomials.append(curve.chord_polynomial(regions[i], radius, rho))
                    self.keys.append(key); self.seen.add(key)
        for arc, curve, (regions, zeros) in zip(self.audit.old.arcs, self.audit.jaws, self.audit.domains):
            checked, _ = curve.continuous_sign_check(z, None, regions)
            for i in checked['violating_piece_indices']:
                key = ('sign', arc['id'], i)
                if key in self.seen: continue
                factored = hard_zero_sign_piece_rows(curve, regions[i], zeros)
                if factored is None: self.polynomials.append(curve.sign_polynomial(regions[i]))
                else: self.factored.append(factored)
                self.keys.append(key); self.seen.add(key)
        return {'new_certificates': len(self.polynomials)-before,
                'new_factorized_pieces': len(self.factored)-before_factored,
                'total_added_certificates': len(self.polynomials),
                'total_added_factorized_pieces': len(self.factored)}

    def augment(self, base):
        old = base['P'].shape[0]; extra = 6*len(self.polynomials); total = old+extra
        if (base['P'].shape != (old, old) or old < self.audit.size
                or base['q'].shape != (old,) or base['A'].shape[1] != old
                or base['b'].shape != (base['A'].shape[0],)
                or sum(cone.dim for cone in base['cones']) != len(base['b'])):
            raise ValueError('complete original source/auxiliary QP required')
        blocks = [{'A': hstack([base['A'], csr_matrix((len(base['b']), extra))], format='csc'),
                   'b': base['b'], 'cones': list(base['cones'])}]
        for i, (rows, constant) in enumerate(self.polynomials):
            blocks.append(nonnegative_cubic_block(rows, constant,
                auxiliary_start=old+6*i, total_variables=total))
        for rows in self.factored:
            blocks.append({'A': hstack([-rows, csr_matrix((rows.shape[0], total-self.audit.size))], format='csc'),
                'b': np.zeros(rows.shape[0]), 'cones': [clarabel.NonnegativeConeT(rows.shape[0])]})
        return {'P': block_diag([base['P'], csr_matrix((extra, extra))], format='csc'),
                'q': np.r_[base['q'], np.zeros(extra)],
                'A': vstack([block['A'] for block in blocks], format='csc'),
                'b': np.concatenate([block['b'] for block in blocks]),
                'cones': [cone for block in blocks for cone in block['cones']]}

import unittest

import numpy as np
from scipy.integrate import quad

from measure_jaw_preference_start import preference_polynomial,measure_preference_start
from probe_jaw_plateau import OriginalProblem,PlateauProblem
from test_jaw_curve_qp import curved


class ExistingPreferenceEnergyTests(unittest.TestCase):
    def test_piece_polynomials_equal_the_unchanged_existing_profiles(self):
        for kind,problem in [('original_beta_cubed',OriginalProblem(curved(),2.,16.)),
                             ('equal_area_plateau',PlateauProblem(curved(),2.,16.))]:
            for arc in problem.arcs:
                cuts=np.unique(np.r_[arc['knots'],.3*arc['length'],.7*arc['length']])
                for lo,hi in zip(cuts[:-1],cuts[1:]):
                    polynomial=preference_polynomial(problem,arc,lo,hi,100.,.13,kind)
                    q=np.array([0.,.13,.51,.93,1.]);positions=lo+(hi-lo)*q
                    np.testing.assert_allclose(polynomial(q),problem.beta(arc,positions,100.,.13),atol=3e-12,rtol=1e-11)

    def test_exact_energy_matches_independent_closed_derivative_integral(self):
        for kind,problem in [('original_beta_cubed',OriginalProblem(curved(),2.,16.)),
                             ('equal_area_plateau',PlateauProblem(curved(),2.,16.))]:
            for arc in problem.arcs:
                radius=(problem.source(arc,arc['knots'])-problem.origin)@problem.across-problem.midline
                slopes=np.diff(radius)/np.diff(arc['knots']);length=.91*arc['length']
                def derivative(s):
                    t=s/arc['length'];index=min(np.searchsorted(arc['knots'],s,side='right')-1,len(slopes)-1)
                    r=np.interp(s,arc['knots'],radius);rp=slopes[index]
                    if kind=='original_beta_cubed':
                        envelope=64.*(t*(1.-t))**3
                        slope=192.*t**2*(1.-t)**2*(1.-2.*t)/arc['length'];scale=1.
                    else:
                        if t<.3:q=t/.3;sign=1.
                        elif t>.7:q=(1.-t)/.3;sign=-1.
                        else:q=1.;sign=0.
                        envelope=q**3*(10.-15.*q+6.*q**2)
                        slope=sign*30.*q**2*(1.-q)**2/(.3*arc['length']);scale=problem.scale
                    return -.13*scale*(rp*envelope+r*slope)
                def integrand(s):
                    t=s/length
                    return length**2*(1.-t)**3*(1.+3.*t+6.*t**2)*derivative(s)**2
                cuts=np.unique(np.r_[0.,arc['knots'],.3*arc['length'],.7*arc['length'],length])
                cuts=cuts[(cuts>=0.)&(cuts<=length)]
                expected=sum(quad(integrand,lo,hi,epsabs=1e-11,epsrel=1e-11)[0] for lo,hi in zip(cuts[:-1],cuts[1:]))
                expected0=sum(quad(lambda s:length**2*derivative(s)**2,lo,hi,epsabs=1e-11,epsrel=1e-11)[0]
                              for lo,hi in zip(cuts[:-1],cuts[1:]))
                report=measure_preference_start(problem,arc,100.,.13,length,kind)
                self.assertAlmostEqual(report['energy_px3'],expected,places=9)
                self.assertAlmostEqual(report['polynomial_antiderivative_energy_px3'],expected,places=9)
                self.assertAlmostEqual(report['unweighted_energy_px3'],expected0,places=8)
                self.assertAlmostEqual(report['unweighted_polynomial_antiderivative_energy_px3'],expected0,places=8)
                negative=measure_preference_start(problem,arc,-100.,.13,length,kind)
                self.assertAlmostEqual(negative['energy_px3'],report['energy_px3'],places=12)

    def test_no_polynomial_may_cross_an_original_radius_or_plateau_branch(self):
        problem=PlateauProblem(curved(),2.,16.);arc=problem.arcs[0]
        with self.assertRaisesRegex(ValueError,'inside polynomial piece'):
            preference_polynomial(problem,arc,0.,arc['length'],100.,.13,'equal_area_plateau')


if __name__=='__main__':unittest.main()

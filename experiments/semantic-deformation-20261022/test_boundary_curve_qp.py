import unittest

import numpy as np
from scipy.interpolate import PchipInterpolator

from boundary_curve_qp import solve_boundary_curve


FRAME = {'across': [1., 0.], 'down': [0., 1.]}


class BoundaryCurveTests(unittest.TestCase):
    def test_local_chord_custom_limit_is_sign_and_rigid_frame_invariant(self):
        y=np.arange(0.,121.,2.)
        source=np.column_stack([8*np.sin((y-30.)/35.),y])
        preferred=5*np.sin(np.pi*y/120.)**2
        options=dict(preserve_through_down=30.,neck_start_down=60.,neck_end_down=112.,
            controller='local_chord_motion_fairing',local_slope_change_max=.008,spacing_px=.05)
        positive=solve_boundary_curve(source,preferred,FRAME,**options)
        across=np.array([np.cos(.71),np.sin(.71)]);down=np.array([-across[1],across[0]])
        translation=np.array([300.,400.]);offset=float(translation@down)
        rotated=solve_boundary_curve(source@np.array([across,down])+translation,
            -preferred,{'across':across.tolist(),'down':down.tolist()},
            **{**options,'preserve_through_down':30.+offset,
               'neck_start_down':60.+offset,'neck_end_down':112.+offset})
        for result in [positive,rotated]:
            self.assertTrue(result['report']['accepted_numerically'],result['report'])
            self.assertEqual(result['report']['local_slope_change_max'],.008)
            for budget in result['report']['chord_budgets']:
                self.assertLessEqual(budget['actual_max_abs_local_slope_change'],.008+1e-8)
        s=np.linspace(0.,min(positive['arclength_px'][-1],rotated['arclength_px'][-1]),2003)
        plus=np.interp(s,positive['arclength_px'],(positive['target']-positive['source'])[:,0])
        minus=np.interp(s,rotated['arclength_px'],(rotated['target']-rotated['source'])@across)
        self.assertLess(abs(plus+minus).max(),2e-6)
        self.assertLess(abs((rotated['target']-rotated['source'])@down).max(),1e-10)

    def test_local_chord_budget_limits_pointwise_change_without_zeroing_motion(self):
        y=np.arange(0.,121.,2.)
        source=np.column_stack([8*np.sin((y-30.)/35.),y])
        preferred=-5*np.sin(np.pi*y/120.)**2
        knots=np.r_[0.,np.linalg.norm(np.diff(source,axis=0),axis=1).cumsum()]
        reference=PchipInterpolator(knots,preferred)
        options=dict(preserve_through_down=30.,neck_start_down=60.,
            neck_end_down=112.,spacing_px=.03)
        global_result=solve_boundary_curve(source,preferred,FRAME,
            controller='chord_bounded_motion_fairing',**options)
        self.assertTrue(global_result['report']['accepted_numerically'])
        def measure(result):
            s=result['arclength_px'];u=(result['target']-result['source'])[:,0]
            measured={}
            for radius in [4.,8.,16.]:
                centers=np.arange(radius+.013,knots[-1]-radius,.037)
                a,b,c=[np.interp(centers+d,knots,y) for d in [-radius,0.,radius]]
                h0,h1=b-a,c-b
                values=[np.interp(centers+d,s,u) for d in [-radius,0.,radius]]
                before=[reference(centers+d) for d in [-radius,0.,radius]]
                delta=[v-p for v,p in zip(values,before)]
                change=(delta[2]-delta[1])/h1-(delta[1]-delta[0])/h0
                original=(before[2]-before[1])*h0-(before[1]-before[0])*h1
                actual=(values[2]-values[1])*h0-(values[1]-values[0])*h1
                measured[radius]=float(abs(change).max())
                self.assertGreaterEqual(float(actual.min()),min(0.,float(original.min()))-2e-4)
                self.assertLessEqual(float(actual.max()),max(0.,float(original.max()))+2e-4)
            return measured
        # A signed global budget alone permits much larger local shape changes.
        self.assertGreater(max(measure(global_result).values()),.03)
        result=solve_boundary_curve(source,preferred,FRAME,
            controller='local_chord_motion_fairing',**options)
        self.assertTrue(result['report']['accepted_numerically'],result['report'])
        measured=measure(result)
        self.assertLessEqual(max(measured.values()),.01+2e-6)
        s=result['arclength_px'];u=(result['target']-result['source'])[:,0]
        upper=result['source'][:,1]<=30.
        self.assertLess(abs(u[upper]-reference(s[upper])).max(),1e-10)
        neck=(result['source'][:,1]>=65.)&(result['source'][:,1]<=100.)
        self.assertLess(np.mean(abs(u[neck])),np.mean(abs(reference(s[neck])))-.005)
        for budget in result['report']['chord_budgets']:
            self.assertEqual(budget['local_slope_change_max'],.01)
            self.assertAlmostEqual(budget['actual_max_abs_local_slope_change'],
                measured[budget['radius_px']],delta=2e-6)

    def test_chord_fairing_is_sign_symmetric_and_rigid_frame_invariant(self):
        y=np.arange(0.,121.,2.)
        source=np.column_stack([8*np.sin((y-30.)/35.),y])
        preferred=5*np.sin(np.pi*y/120.)**2
        options=dict(preserve_through_down=30.,neck_start_down=60.,
            neck_end_down=112.,controller='chord_bounded_motion_fairing',spacing_px=.2)
        positive=solve_boundary_curve(source,preferred,FRAME,**options)
        negative=solve_boundary_curve(source,-preferred,FRAME,**options)
        across=np.array([np.cos(.71),np.sin(.71)])
        down=np.array([-across[1],across[0]])
        translation=np.array([300.,400.]);height_offset=float(translation@down)
        rotated_source=source@np.array([across,down])+translation
        rotated=solve_boundary_curve(rotated_source,preferred,
            {'across':across.tolist(),'down':down.tolist()},
            **{**options,'preserve_through_down':30.+height_offset,
               'neck_start_down':60.+height_offset,'neck_end_down':112.+height_offset})
        # This curved source previously failed SLSQP after seven refinement
        # rounds, while its rigidly rotated copy passed the same constraints.
        for name,result in [('positive',positive),('negative',negative),('rotated',rotated)]:
            self.assertTrue(result['report']['accepted_numerically'],
                (name,result['report'].get('rejection'),result['report']['refinement']))
        samples=np.linspace(0.,min(result['arclength_px'][-1]
            for result in [positive,negative,rotated]),2003)
        def displacement(result,direction):
            values=(result['target']-result['source'])@np.asarray(direction)
            return np.interp(samples,result['arclength_px'],values)
        plus=displacement(positive,[1.,0.]);minus=displacement(negative,[1.,0.])
        self.assertLess(abs(plus+minus).max(),2e-6)
        self.assertLess(abs(plus-displacement(rotated,across)).max(),2e-6)
        self.assertLess(abs((rotated['target']-rotated['source'])@down).max(),1e-10)

    def test_neck_recovery_does_not_amplify_multiscale_chord_turns(self):
        y=np.arange(0.,121.,2.)
        source=np.column_stack([np.zeros_like(y),y])
        preferred=5*np.sin(np.pi*y/120.)**2
        result=solve_boundary_curve(source,preferred,FRAME,
            preserve_through_down=30.,neck_start_down=60.,neck_end_down=112.,
            controller='chord_bounded_motion_fairing',spacing_px=.05)
        self.assertTrue(result['report']['accepted_numerically'],result['report'])
        s=result['arclength_px'];u=(result['target']-result['source'])[:,0]
        reference=PchipInterpolator(y,preferred)
        for radius in [4.,8.,16.]:
            centers=np.arange(radius+.013,120.-radius,.037)
            before=radius*(reference(centers+radius)-2*reference(centers)+reference(centers-radius))
            after=radius*(np.interp(centers+radius,s,u)-2*np.interp(centers,s,u)+np.interp(centers-radius,s,u))
            self.assertGreaterEqual(float(after.min()),min(0.,float(before.min()))-2e-4)
            self.assertLessEqual(float(after.max()),max(0.,float(before.max()))+2e-4)
        self.assertLess(abs(u[s<=30.]-reference(s[s<=30.])).max(),1e-10)
        neck=(s>=65.)&(s<=100.)
        self.assertLess(np.mean(abs(u[neck])),np.mean(abs(reference(s[neck])))-.02)

    def test_return_is_spread_without_weakening_preserved_upper_motion(self):
        y = np.arange(0., 121., 2.)
        source = np.column_stack([np.zeros_like(y), y])
        t = np.clip((y-60.)/12., 0., 1.)
        preferred = 6.*(1.-t*t*(3.-2.*t))
        result = solve_boundary_curve(source, preferred, FRAME,
            preserve_through_down=30., preserve_suffix_px=8.,
            neck_start_down=60., neck_end_down=112.,
            smoothing_length_px=16., neck_weight=4., spacing_px=.5)
        self.assertTrue(result['report']['accepted_numerically'], result['report'])
        sample = result['source']; displacement = result['target']-sample
        self.assertTrue(np.array_equal(displacement[:, 1], np.zeros(len(sample))))
        self.assertLess(np.max(abs(displacement[sample[:, 1] <= 30., 0]-6.)), 1e-10)
        self.assertLess(result['report']['max_abs_displacement_gradient'], .6)
        self.assertLess(result['report']['max_abs_displacement_gradient'],
                        .8*result['report']['preferred_max_abs_displacement_gradient'])
        # Recovery starts before the old narrow return at y=60; this specifically
        # catches a replacement that only weights the known neck below y=60.
        self.assertLess(displacement[np.argmin(abs(sample[:, 1]-50.)), 0], 5.8)
        self.assertLess(np.max(abs(displacement[sample[:, 1] >= 112., 0])), 1e-10)
        self.assertTrue(np.all(result['weights'] > 0.))

    def test_source_turn_sign_is_preserved_without_imposing_one_global_sign(self):
        y = np.arange(0., 121., 2.)
        # A sinusoidal interface has both positive and negative source turns.
        # No rule is allowed to reinterpret the whole curve as convex anatomy.
        source = np.column_stack([8.*np.sin((y-30.)/35.), y])
        preferred = np.full_like(y, 3.)
        result = solve_boundary_curve(source, preferred, FRAME,
            preserve_through_down=24., preserve_suffix_px=8.,
            neck_start_down=60., neck_end_down=112.,
            smoothing_length_px=16., neck_weight=4., spacing_px=.25)
        self.assertTrue(result['report']['accepted_numerically'], result['report'])
        s = result['arclength_px']; out = result['target']; src = result['source']
        centers = np.arange(12.01, s[-1]-12., .073)
        def turns(points):
            rows = [np.column_stack([np.interp(centers+d,s,points[:,j]) for j in range(2)])
                    for d in [-12.,0.,12.]]
            a,b,c = rows
            return (c[:,0]-b[:,0])*(b[:,1]-a[:,1])-(b[:,0]-a[:,0])*(c[:,1]-b[:,1])
        before,after=turns(src),turns(out)
        known=abs(before)>1e-4
        self.assertTrue(np.any(before[known]>0.) and np.any(before[known]<0.))
        # This is an independent dense sampling check of the public targets;
        # interpolation at .25 px introduces a small approximation tolerance.
        self.assertGreaterEqual(float((np.sign(before[known])*after[known]).min()), -2e-3)

    def test_conflicting_preserved_motion_is_rejected_without_attenuation(self):
        y = np.arange(0., 121., 2.)
        source=np.column_stack([.01*y*y,y])
        preferred=-.02*y*y
        result=solve_boundary_curve(source,preferred,FRAME,
            preserve_through_down=40.,neck_start_down=60.,neck_end_down=112.)
        self.assertFalse(result['report']['accepted_numerically'])
        self.assertIsNone(result['target'])
        self.assertIn('preserved preferred',result['report']['rejection'])

    def test_rotated_frame_endpoint_roundoff_and_height_preservation(self):
        a=np.array([np.cos(.1),np.sin(.1)]);d=np.array([-a[1],a[0]])
        y=np.arange(0.,121.,4.); source=400*a+y[:,None]*d+500*d
        frame={'across':a.tolist(),'down':d.tolist()}
        result=solve_boundary_curve(source,np.full(len(y),2.),frame,
            preserve_through_down=530.,neck_start_down=560.,
            neck_end_down=float((source@d)[-1])+1e-11)
        self.assertTrue(result['report']['accepted_numerically'],result['report'])
        self.assertLess(abs((result['target']-result['source'])@d).max(),1e-10)

    def test_integral_mode_preserves_nonzero_upper_area_and_anchor(self):
        y=np.arange(0.,121.,2.)
        source=np.column_stack([np.zeros_like(y),y])
        t=np.clip(y/50.,0.,1.);u=np.clip((y-70.)/30.,0.,1.)
        preferred=6*t*t*(3-2*t)*(1-u*u*(3-2*u))
        result=solve_boundary_curve(source,preferred,FRAME,
            preserve_through_down=40.,neck_start_down=70.,neck_end_down=112.,
            upper_preservation='integral_and_anchor',spacing_px=.1)
        report=result['report']
        self.assertTrue(report['accepted_numerically'],report)
        self.assertGreater(report['upper_preferred_integral_px2'],50.)
        self.assertGreater(report['upper_preferred_anchor_px'],5.)
        self.assertAlmostEqual(report['upper_actual_integral_px2'],report['upper_preferred_integral_px2'],places=8)
        self.assertAlmostEqual(report['upper_actual_anchor_px'],report['upper_preferred_anchor_px'],places=8)

    def test_motion_fairing_keeps_signed_gradient_budget_and_upper_motion(self):
        y=np.arange(0.,121.,2.);source=np.column_stack([.002*y*y,y])
        t=np.clip((y-60.)/12.,0.,1.);preferred=6*(1-t*t*(3-2*t))
        result=solve_boundary_curve(source,preferred,FRAME,preserve_through_down=30.,
            neck_start_down=60.,neck_end_down=112.,controller='gradient_bounded_motion_fairing')
        report=result['report']
        self.assertTrue(report['accepted_numerically'],report)
        self.assertFalse(report['source_turn_signs_are_hard_constraints'])
        lo,hi=report['preferred_signed_gradient_interval'];a,b=report['actual_signed_gradient_interval']
        self.assertGreaterEqual(a,lo-1e-8);self.assertLessEqual(b,hi+1e-8)
        upper=result['source'][:,1]<=30.
        self.assertLess(abs((result['target']-result['source'])[upper,0]-6.).max(),1e-10)


if __name__ == '__main__':
    unittest.main()

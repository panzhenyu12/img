"""Closed-form correspondence fixtures; no solve, photograph or field."""
import copy
import hashlib
import json
import unittest

import numpy as np

from jaw_collocation import sample_jaw_targets, independent_jaw_probes
from jaw_curve_correspondence import apply_arclength_samples, arclength_contour_targets


def fixture():
    geometry={'contour':[[-6.,0.],[-6.,2.],[-3.,6.],[0.,6.],[3.,6.],
                         [6.,2.],[6.,0.],[6.,-2.],[-6.,-2.]],
        'frame':{'origin':[0.,-1.],'across':[1.,0.],'down':[0.,1.],'midline_u':0.},
        'arcs':[{'id':'left-jaw','start':{'edge':0,'t':0.},'end':{'edge':3,'t':0.},'direction':1},
                {'id':'right-jaw','start':{'edge':6,'t':0.},'end':{'edge':3,'t':0.},'direction':-1}],
        'selected_rows':[1,2,4,5]}
    profile={'schema':'jaw-arclength-piecewise-linear-profile-v1','controller':'synthetic-nonuniform-curve',
        'amount':100.,'source_geometry':copy.deepcopy(geometry),
        'source_geometry_sha256':hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'across_axis':[1.,0.],'profiles':[], 'solver_status':'AlmostSolved',
        'accepted_target_constraints':False,'diagnostic_only':True,'image_approval':False}
    for name,direction,start,sign in [('left-jaw',1,0.,1.),('right-jaw',-1,20.,-1.)]:
        profile['profiles'].append({'arc_id':name,'direction':direction,
            'original_contour_start_arclength_px':start,'arc_length_px':10.,
            'original_source_vertex_arclength_px':[0.,2.,7.,10.],
            'arclength_px':[0.,1.,2.,7.,8.5,10.],
            'across_displacement_px':(sign*np.array([0.,1.,3.,1.,.5,0.])).tolist()})
    return geometry,profile


def left_expected():
    # Explicit original target vertices, independent of either adapter.
    vertices=np.array([[-6.,0.],[-5.,1.],[-3.,2.],[-2.,6.],[-1.,6.],[0.,6.]])
    lengths=np.array([np.sqrt(2.),np.sqrt(5.),np.sqrt(17.),1.,1.])
    return vertices,np.r_[0.,lengths.cumsum()]


def sample_at(geometry,profile,positions):
    source=[];controls=[]
    for name,sign in [('left-jaw',1.),('right-jaw',-1.)]:
        for s in positions:
            point=[sign*np.interp(s,[0.,2.,7.,10.],[-6.,-6.,-3.,0.]),
                   np.interp(s,[0.,2.,7.,10.],[0.,2.,6.,6.])]
            source.append(point);controls.append({'arc_id':name,'arc_arclength_px':float(s),
                'arc_t':float(s/10.),'original_index':None,'role':'independent-test-point'})
    source.append(geometry['contour'][7]);controls.append({'arc_id':None,'original_index':7,'role':'outside'})
    return {'source':np.array(source),'target':np.array(source),'weights':np.linspace(.2,1.,len(source)),
            'controls':controls,'report':{'amount':profile['amount'],'sentinel':'unchanged metadata'}}


class JawCurveCorrespondenceTests(unittest.TestCase):
    def test_nonuniform_target_geometry_and_all_corners_are_covered_monotonically(self):
        geometry,profile=fixture();vertices,cumulative=left_expected()
        # Source positions which must visit every target corner and midpoint.
        tau=np.sort(np.r_[cumulative,(cumulative[:-1]+cumulative[1:])/2.])
        s=10.*tau/cumulative[-1];samples=sample_at(geometry,profile,s)
        before=copy.deepcopy(samples);g=copy.deepcopy(geometry);p=copy.deepcopy(profile)
        result=apply_arclength_samples(samples,profile,geometry)
        expected=np.column_stack([np.interp(tau,cumulative,vertices[:,i])for i in range(2)])
        np.testing.assert_allclose(result['target'][:len(s)],expected,atol=2e-14,rtol=0.)
        np.testing.assert_allclose(result['target'][len(s):2*len(s)],expected*[-1.,1.],atol=2e-14,rtol=0.)
        self.assertTrue(np.all(np.diff(tau)>0.))
        for first in [0,len(s)]:
            np.testing.assert_array_equal(result['target'][first],result['source'][first])
            np.testing.assert_array_equal(result['target'][first+len(s)-1],result['source'][first+len(s)-1])
        np.testing.assert_array_equal(result['target'][-1],samples['source'][-1])
        for key in ['source','weights']:np.testing.assert_array_equal(result[key],before[key])
        for key in ['source','target','weights']:np.testing.assert_array_equal(samples[key],before[key])
        self.assertEqual(result['controls'],before['controls'])
        self.assertEqual(geometry,g);self.assertEqual(profile,p)
        self.assertTrue(result['report']['new_correspondence'])
        self.assertTrue(result['report']['target_curve_set_preserved'])
        self.assertFalse(result['report']['original_pointwise_request_preserved'])
        self.assertFalse(result['report']['aesthetic_approval'])
        self.assertEqual(result['report']['input_profile_solver_status'],'AlmostSolved')
        self.assertFalse(result['report']['input_profile_target_optimization_accepted'])
        for curve,mirror in zip(result['report']['curve_correspondence'],[1.,-1.]):
            np.testing.assert_allclose(curve['target_polyline_vertices'],vertices*[mirror,1.],atol=1e-14,rtol=0.)
            np.testing.assert_allclose(curve['target_corner_source_arclength_px'],10.*cumulative/cumulative[-1],atol=1e-14,rtol=0.)
            self.assertTrue(np.all(np.diff(curve['target_corner_source_arclength_px'])>0.))

    def test_negative_baked_amount_is_not_multiplied_twice(self):
        geometry,profile=fixture();profile['amount']=-100.
        for row in profile['profiles']:
            row['across_displacement_px']=(-np.array(row['across_displacement_px'])).tolist()
        # The signed expansion geometry is again explicit and independent.
        vertices=np.array([[-6.,0.],[-7.,1.],[-9.,2.],[-4.,6.],[-2.,6.],[0.,6.]])
        cumulative=np.r_[0.,np.linalg.norm(np.diff(vertices,axis=0),axis=1).cumsum()]
        positions=np.linspace(0.,10.,27);samples=sample_at(geometry,profile,positions)
        result=apply_arclength_samples(samples,profile,geometry)
        tau=positions*cumulative[-1]/10.
        expected=np.column_stack([np.interp(tau,cumulative,vertices[:,i])for i in range(2)])
        np.testing.assert_allclose(result['target'][:len(positions)],expected,atol=2e-14,rtol=0.)
        np.testing.assert_allclose(result['target'][len(positions):2*len(positions)],expected*[-1.,1.],atol=2e-14,rtol=0.)
        self.assertEqual(result['report']['amount'],-100.)

    def test_material_can_move_down_and_contour_api_matches_same_closed_form(self):
        geometry,profile=fixture();vertices,cumulative=left_expected()
        samples=sample_at(geometry,profile,np.array([0.,2.,7.,10.]))
        result=apply_arclength_samples(samples,profile,geometry)
        tau=.2*cumulative[-1]
        t=(tau-cumulative[1])/(cumulative[2]-cumulative[1])
        expected=vertices[1]+t*(vertices[2]-vertices[1])
        np.testing.assert_allclose(result['target'][1],expected,atol=1e-14,rtol=0.)
        self.assertGreater(abs(result['target'][1,1]-samples['source'][1,1]),.5)
        contour=arclength_contour_targets(geometry,profile)
        np.testing.assert_allclose(contour[[0,1,2,3]],result['target'][:4],atol=1e-14,rtol=0.)
        np.testing.assert_allclose(contour[[6,5,4,3]],result['target'][4:8],atol=1e-14,rtol=0.)
        np.testing.assert_array_equal(contour[[0,3,6,7,8]],np.array(geometry['contour'])[[0,3,6,7,8]])

    def test_zero_identity_is_exact_despite_irrational_arc_length_roundoff(self):
        from test_jaw_target_profile import fixture as irrational_fixture
        geometry,profile=irrational_fixture();profile['amount']=0.
        for row in profile['profiles']:row['across_displacement_px']=[0.]*len(row['arclength_px'])
        for samples in [sample_jaw_targets(geometry,0.,spacing=.37),
                        independent_jaw_probes(geometry,0.,spacing=.17)]:
            result=apply_arclength_samples(samples,profile,geometry)
            np.testing.assert_array_equal(result['target'],samples['source'])
        np.testing.assert_array_equal(arclength_contour_targets(geometry,profile),geometry['contour'])

    def test_collapsed_target_segment_is_rejected_in_both_apis(self):
        geometry,profile=fixture()
        # The last original source segment is horizontal. These displacements
        # collapse all of it to the original chin (0,6), so tau is not strict.
        profile['profiles'][0]['across_displacement_px']=[0.,1.,3.,3.,1.5,0.]
        with self.assertRaisesRegex(ValueError,'positive.*target|target.*positive|zero.length'):
            apply_arclength_samples(sample_jaw_targets(geometry,100.),profile,geometry)
        with self.assertRaisesRegex(ValueError,'positive.*target|target.*positive|zero.length'):
            arclength_contour_targets(geometry,profile)

    def test_source_identity_and_original_sample_protocol_remain_required(self):
        geometry,profile=fixture();samples=sample_jaw_targets(geometry,100.)
        changed=copy.deepcopy(samples);changed['source'][5,0]+=.1
        with self.assertRaises(ValueError):apply_arclength_samples(changed,profile,geometry)
        changed=copy.deepcopy(geometry);changed['contour'][0][0]-=.1
        with self.assertRaises(ValueError):arclength_contour_targets(changed,profile)


if __name__=='__main__':unittest.main()

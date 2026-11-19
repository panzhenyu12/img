"""Independent known-truth source-contour rejection tests, no model/build.

The three fixtures separate visible material evidence from part identity,
front/protection permissions, and arbitrary content hidden by opaque sleeves.
Passing these tests is not a natural-shape or real-photo quality acceptance.
"""
import copy
import unittest
import numpy as np

from visible_rear_geometry import build_visible_rear_geometry, bounded_contour_interpolator

WIDTH, HEIGHT = 128, 160


def raster(values):
    h, w = values.shape
    return {'shape':'raster','width':w,'height':h,
            'data':np.rint(values*255).astype(np.uint8).ravel().tolist()}


def fixture(coarse=False):
    # True visible torso boundaries at x32 and x96. A separate arm beyond
    # a clear scene gap is deliberately present in the same optical layer.
    rear=np.zeros((HEIGHT,WIDTH),float)
    rear[24:136,32:96]=1
    rear[35:125,108:120]=1
    subject=rear.copy()
    part=np.zeros_like(rear);part[30:130,40:88]=1
    if coarse:part=part[::2,::2]
    params={'bodyPart':'waist','region':{'x':0,'y':0,'width':1,'height':1},
        'mask':{'shape':'rect','region':{'x':.1,'y':.2,'width':.8,'height':.6}},
        'subjectMask':raster(subject),'partMask':raster(part),
        'protectMask':raster(np.zeros_like(rear)),
        'landmarks':[{'x':.5,'y':30/HEIGHT},{'x':.5,'y':80/HEIGHT},
                     {'x':.5,'y':130/HEIGHT}],
        'adjustments':{'waist':30},'opaqueTestMetadata':{'doNotChange':[7,11]}}
    return params,np.zeros_like(rear),rear


def part_array(params):
    m=params['partMask']
    return np.asarray(m['data'],np.uint8).reshape(m['height'],m['width'])


def section_near(report,y):
    return min(report['sections'],key=lambda s:abs(s['center'][1]-y))


class VisibleRearGeometryTests(unittest.TestCase):
    def checked_call(self,params,front,rear,*,continuation='none'):
        before=copy.deepcopy(params);old_front=front.copy();old_rear=rear.copy()
        # V1 evidence-filter fixtures request their historical mode explicitly;
        # continuation fixtures below exercise the accepted PCHIP candidate.
        candidate,report=build_visible_rear_geometry(params,WIDTH,HEIGHT,front,rear,
                                                    continuation=continuation)
        self.assertEqual(params,before,'source params mutated in place')
        np.testing.assert_array_equal(front,old_front)
        np.testing.assert_array_equal(rear,old_rear)
        restored=copy.deepcopy(candidate)
        restored['partMask']['data']=before['partMask']['data']
        self.assertEqual(restored,before,'candidate changed more than geometric raster values')
        self.assertEqual(report['partMaskResolution'],[before['partMask']['width'],before['partMask']['height']])
        return candidate,report

    def test_visible_torso_edge_is_recovered_without_swallowing_separate_arm(self):
        for coarse in [False,True]:
            with self.subTest(coarse=coarse):
                params,front,rear=fixture(coarse)
                candidate,report=self.checked_call(params,front,rear)
                mid=section_near(report,80)
                self.assertEqual(mid['left']['status'],'measured')
                self.assertEqual(mid['right']['status'],'measured')
                self.assertAlmostEqual(mid['left']['boundaryPoint'][0],32,delta=.75)
                self.assertAlmostEqual(mid['right']['boundaryPoint'][0],96,delta=.75)
                result=part_array(candidate);scale=2 if coarse else 1
                self.assertGreater(result[80//scale,34//scale],240)
                self.assertGreater(result[80//scale,94//scale],240)
                self.assertEqual(int(result[:,108//scale:120//scale].max()),0,
                    'separate arm was admitted as torso geometry')
                self.assertGreater(report['addedGeometryPixels'],0)

    def test_torso_optical_layer_cannot_be_relabelled_as_an_arm(self):
        params,front,rear=fixture()
        params['bodyPart']='left-upper-arm'
        params['adjustments']={'arms':30}
        before=copy.deepcopy(params)
        with self.assertRaisesRegex(ValueError,'waist'):
            build_visible_rear_geometry(params,WIDTH,HEIGHT,front,rear)
        self.assertEqual(params,before)

    def test_front_sleeve_and_protection_make_boundary_unknown_without_bridging(self):
        for opacity in [1.0,.25]:
            with self.subTest(frontOpacity=opacity):
                params,front,rear=fixture()
                # A touching same-material sleeve removes the clear gap:
                # rear and subject now continue from torso all the way to114.
                # Visibility/protection, not the optical silhouette or color,
                # must prevent declaring its outside edge a waist boundary.
                rear[64:89,96:114]=1
                rear[64:89,114:120]=0
                params['subjectMask']=raster(rear)
                front[64:89,90:114]=opacity
                protection=np.zeros_like(front);protection[72:81,28:38]=1
                params['protectMask']=raster(protection)
                # Both an explicit lock and an unobserved side must retain
                # floating original data exactly; whole-raster byte encoding
                # would silently change these despite no admitted evidence.
                fractional_indices=[76*WIDTH+34,76*WIDTH+88]
                for at in fractional_indices:params['partMask']['data'][at]=128.4
                candidate,report=self.checked_call(params,front,rear)
                for at in fractional_indices:
                    self.assertEqual(candidate['partMask']['data'][at],128.4)
                middle=section_near(report,76)
                self.assertEqual(middle['right']['status'],'unknown')
                self.assertEqual(middle['right']['reason'],'front-occluded')
                self.assertIsNone(middle['right']['boundaryPoint'])
                self.assertEqual(middle['left']['status'],'unknown')
                self.assertEqual(middle['left']['reason'],'protected')
                self.assertIsNone(middle['left']['boundaryPoint'])
                unilateral=section_near(report,68)
                self.assertEqual(unilateral['right']['status'],'unknown')
                self.assertEqual(unilateral['right']['reason'],'front-occluded')
                self.assertEqual(unilateral['left']['status'],'measured')
                self.assertAlmostEqual(unilateral['left']['boundaryPoint'][0],32,delta=.75)
                for y in [60,96]:
                    clear=section_near(report,y)
                    self.assertEqual(clear['left']['status'],'measured')
                    self.assertEqual(clear['right']['status'],'measured')
                old,new=part_array(params),part_array(candidate)
                self.assertGreater(new[68,34],240,
                    'refusing the touching sleeve must not suppress opposite visible correction')
                blocked=(front>0)|(protection>0)
                np.testing.assert_array_equal(new[blocked],old[blocked])
                # Not just the blocked texels: do not interpolate a newly
                # asserted contour through this entire unknown cross-section.
                np.testing.assert_array_equal(new[76],old[76])

    def test_hidden_rear_hallucination_cannot_change_visible_geometry(self):
        params,front,rear=fixture()
        front[56:104,78:120]=1
        foreground=np.array([196.,138.,80.]);material=np.array([70.,128.,160.])
        scene=np.array([211.,216.,220.])
        outputs=[];observed_sources=[]
        for edge in [80,96,112]:
            variant=rear.copy()
            variant[56:104,78:120]=0
            variant[56:104,78:edge]=1
            # Source RGB is identical despite completely different hidden
            # model content. The source geometry must share that invariance.
            lower=variant[:,:,None]*material+(1-variant[:,:,None])*scene
            observed=front[:,:,None]*foreground+(1-front[:,:,None])*lower
            observed_sources.append(observed)
            candidate,report=self.checked_call(params,front,variant)
            middle=section_near(report,80)
            self.assertEqual(middle['right']['status'],'unknown')
            self.assertEqual(middle['right']['reason'],'front-occluded')
            outputs.append((candidate,report['sections']))
        for source in observed_sources[1:]:
            np.testing.assert_array_equal(source,observed_sources[0])
        for output in outputs[1:]:
            self.assertEqual(output,outputs[0],
                'unobservable rear width changed source-visible contour/geometry')

    def test_computational_continuation_retains_unknown_evidence_and_hidden_invariance(self):
        params,front,rear=fixture()
        front[56:104,78:120]=1
        protection=np.zeros_like(front);protection[72:81,28:38]=1
        params['protectMask']=raster(protection)
        params['partMask']['data'][76*WIDTH+34]=128.4
        outputs=[]
        for edge in [80,112]:
            variant=rear.copy();variant[56:104,78:120]=0;variant[56:104,78:edge]=1
            candidate,report=self.checked_call(params,front,variant,continuation='pchip')
            midpoint=section_near(report,80)
            self.assertEqual(midpoint['right']['status'],'unknown')
            self.assertEqual(midpoint['right']['reason'],'front-occluded')
            self.assertIsNone(midpoint['right']['boundaryPoint'])
            support=report['computationalSupport']['right']
            self.assertEqual(support['status'],'bounded-pchip')
            self.assertFalse(support['hiddenRearRead'])
            self.assertFalse(support['unknownEvidencePromoted'])
            self.assertFalse(support['extrapolation'])
            self.assertTrue(any(k['kind']=='visible-measurement' for k in support['knots']))
            self.assertGreater(part_array(candidate)[80,94],240,
                'v2 must actually continue computational support through the occlusion')
            original=np.asarray(params['partMask']['data'],float).reshape(HEIGHT,WIDTH)
            result=np.asarray(candidate['partMask']['data'],float).reshape(HEIGHT,WIDTH)
            np.testing.assert_array_equal(result[protection>0],original[protection>0])
            self.assertEqual(candidate['partMask']['data'][76*WIDTH+34],128.4)
            outputs.append((candidate,report['sections'],report['computationalSupport']))
        self.assertEqual(outputs[0],outputs[1],
            'hidden generated content changed visible evidence or computational prior')

    def test_actual_nonuniform_contour_helper_has_no_overshoot_or_extrapolation(self):
        t=[0.,.4,.41,1.];r=[20.,20.,40.,40.]
        curve=bounded_contour_interpolator(t,r)
        np.testing.assert_allclose(curve(t),r,rtol=0,atol=1e-12)
        for i in range(len(t)-1):
            values=curve(np.linspace(t[i],t[i+1],201))
            self.assertGreaterEqual(float(values.min()),min(r[i:i+2])-1e-12)
            self.assertLessEqual(float(values.max()),max(r[i:i+2])+1e-12)
        self.assertTrue(np.isnan(curve([-.001,1.001])).all())
        derivative=curve.derivative()
        for at in t[1:-1]:
            self.assertAlmostEqual(float(derivative(at-1e-9)),float(derivative(at+1e-9)),delta=.002)
        # These bounds apply to the actual computational curve helper. They
        # do not establish C1 after mask union, clipping, or native resampling.

    def test_missing_endpoint_support_refuses_only_that_side_without_partial_splice(self):
        # A visible interior cannot authorize a partial PCHIP domain if an
        # unknown endpoint has no bounded original-part anchor. Otherwise its
        # end is pasted onto the old mask, recreating the rejected V1 seam.
        for side in ['left','right']:
            for endpoint_y,missing_t in [(47,0.),(112,1.)]:
                with self.subTest(side=side,missingEndpoint=missing_t):
                    params,front,rear=fixture()
                    part=np.asarray(params['partMask']['data'],float).reshape(HEIGHT,WIDTH)
                    rows=slice(endpoint_y-1,endpoint_y+3)
                    rejected=slice(0,64) if side=='left' else slice(64,WIDTH)
                    part[rows,rejected]=255
                    # Block visible truth before its true edge while the
                    # original geometric prior continues beyond the domain.
                    front[rows,slice(0,38) if side=='left' else slice(90,WIDTH)]=1
                    fractional_x=34 if side=='left' else 94
                    part[80,fractional_x]=128.4
                    params['partMask']['data']=part.ravel().tolist()
                    candidate,report=self.checked_call(params,front,rear,continuation='pchip')
                    endpoint=report['sections'][0 if missing_t==0 else -1]
                    self.assertEqual(endpoint[side]['status'],'unknown')
                    self.assertEqual(endpoint[side]['reason'],'front-occluded')
                    support=report['computationalSupport'][side]
                    self.assertEqual(support['status'],'insufficient-endpoint-support')
                    self.assertEqual(support['missingEndpoints'],[missing_t])
                    self.assertGreater(support['visibleKnots'],2,
                        'fixture must have interior evidence that tempts a partial splice')
                    result=np.asarray(candidate['partMask']['data'],float).reshape(HEIGHT,WIDTH)
                    np.testing.assert_array_equal(result[:,rejected],part[:,rejected])
                    self.assertEqual(result[80,fractional_x],128.4)
                    other='right' if side=='left' else 'left'
                    self.assertEqual(report['computationalSupport'][other]['status'],'bounded-pchip')
                    self.assertGreater(result[80,94 if side=='left' else 34],240,
                        'one missing endpoint must not suppress the opposite visible correction')

        # Losing the endpoint seed itself removes anchors on both sides.
        # Plenty of valid interior rows still must not create partial domains.
        for endpoint_y,missing_t in [(47,0.),(112,1.)]:
            with self.subTest(missingSeedEndpoint=missing_t):
                params,front,rear=fixture()
                part=np.asarray(params['partMask']['data'],float).reshape(HEIGHT,WIDTH)
                part[endpoint_y-1:endpoint_y+3,62:66]=0
                params['partMask']['data']=part.ravel().tolist()
                candidate,report=self.checked_call(params,front,rear,continuation='pchip')
                endpoint=report['sections'][0 if missing_t==0 else -1]
                for side in ['left','right']:
                    self.assertEqual(endpoint[side]['status'],'unknown')
                    self.assertEqual(endpoint[side]['reason'],'no-anatomical-seed')
                    support=report['computationalSupport'][side]
                    self.assertEqual(support['status'],'insufficient-endpoint-support')
                    self.assertEqual(support['missingEndpoints'],[missing_t])
                    self.assertGreater(support['visibleKnots'],2)
                self.assertEqual(candidate,params,
                    'missing endpoint seeds must not be replaced by interior-only curve splices')

    def test_rasterized_continuation_removes_the_v1_known_unknown_width_step(self):
        # Smooth analytic material edge and a deliberately narrower coarse
        # part raster. Sleeve visibility interrupts evidence, not the physical
        # torso. Examine actual adapter partMask output, not only scipy PCHIP.
        params,front,_=fixture(coarse=True)
        yy,xx=np.mgrid[:HEIGHT,:WIDTH]
        radius=30+4*((yy+.5-80)/64)**2
        rear=np.clip(.5+radius-abs(xx+.5-64),0,1)
        rear[(yy<24)|(yy>=136)]=0
        params['subjectMask']=raster(rear)
        front[64:89,88:114]=1
        v1,r1=self.checked_call(params,front,rear)
        v2,r2=self.checked_call(params,front,rear,continuation='pchip')
        self.assertEqual(r1['sections'],r2['sections'],
            'computational continuation relabelled raw visible evidence')
        self.assertEqual(section_near(r2,76)['right']['status'],'unknown')
        self.assertIsNone(section_near(r2,76)['right']['boundaryPoint'])

        def right_edges(candidate):
            m=part_array(candidate).astype(float);scale=WIDTH/m.shape[1]
            rows=list(range(28,52));edges=[]
            for y in rows:
                row=m[y];start=m.shape[1]//2
                end=next(i for i in range(start+1,len(row)) if row[i]<127.5)
                fraction=(row[end-1]-127.5)/(row[end-1]-row[end])
                edges.append((end-.5+fraction)*scale)
            return (np.array(rows)+.5)*(HEIGHT/m.shape[0]),np.asarray(edges)

        ys,e1=right_edges(v1);_,e2=right_edges(v2)
        self.assertGreater(float(abs(np.diff(e1)).max()),4,
            'fixture must expose the former hard splice before judging v2')
        self.assertLess(float(abs(np.diff(e2)).max()),1,
            'actual raster geometry retained a known/unknown width jump')
        visible=(ys<63)|(ys>90)
        truth=64+30+4*((ys-80)/64)**2
        self.assertLess(float(abs(e2[visible]-truth[visible]).max()),.75,
            'smoothness cannot be obtained by discarding the visible waist target')
        middle=np.argmin(abs(ys-80))
        self.assertGreater(e2[middle]-e1[middle],5,
            'candidate must provide real support through the unknown interval')


if __name__=='__main__':unittest.main(verbosity=2)

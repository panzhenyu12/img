"""New-photo jaw geometry tests without a detector or photographs."""
import unittest
import numpy as np
from portable_jaw import jaw_geometry, pixels, fixed_support


def fixture():
    def oval(n, rx, ry):
        t = np.arange(n)*2*np.pi/n-np.pi/2
        return np.column_stack([100+rx*np.cos(t), 100+ry*np.sin(t)])
    def box(x, y): return np.array([[x-4,y-2],[x+4,y-2],[x+4,y+2],[x-4,y+2]])
    return oval(133, 50, 70), oval(36, 45, 65), [box(80,80), box(120,80)], box(100,130)


class PortableJawTests(unittest.TestCase):
    def test_visible_contour_is_not_replaced_by_the_inner_anatomical_oval(self):
        p, oval, eyes, lips = fixture(); before = p.copy()
        g = jaw_geometry(p, oval, eyes, lips)
        np.testing.assert_array_equal(g['contour'], before)
        self.assertEqual(len(g['contour']), 133)
        self.assertGreater(g['chin_projection_distance_px'], 4.9)
        self.assertEqual(len(g['arcs']), 2)
        self.assertTrue(np.all(p[g['selected_rows'], 1] > 130.))
        self.assertEqual(g['visibility_status'], 'unknown')
        self.assertFalse(g['automatic_edit_authorized'])

    def test_source_index_conversion_runs_once_at_pixel_centers(self):
        np.testing.assert_array_equal(pixels([{'x':.5,'y':.25}], (100,200)), [[49.5,49.5]])

    def test_source_rotation_keeps_the_same_arc_roles(self):
        args = fixture(); old = jaw_geometry(*args)
        angle=.2; rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        move=lambda p: p@rotation.T+[20,30]
        new=jaw_geometry(move(args[0]),move(args[1]),[move(p) for p in args[2]],move(args[3]))
        self.assertEqual(old['selected_rows'],new['selected_rows'])
        for a,b in zip(old['arcs'],new['arcs']):
            self.assertEqual(a['direction'],b['direction'])
            for key in ['start','end']:
                self.assertEqual(a[key]['edge'],b[key]['edge'])
                self.assertAlmostEqual(a[key]['t'],b[key]['t'],places=11)

    def test_upper_face_and_features_are_protected_before_any_field_solve(self):
        args=fixture();g=jaw_geometry(*args)
        normalized=lambda p:[{'x':(x+.5)/201,'y':(y+.5)/201} for x,y in p]
        zero={'shape':'raster','width':1,'height':1,'data':[0]}
        params={'semanticMasks':{'nose':{'points':normalized(args[3]-[0,18])},
            'lips':{'points':normalized(args[3])},'ears':zero,'accessories':zero}}
        fixed,crop,roles,_=fixed_support((201,201),params,g)
        self.assertTrue(fixed[80,80]);self.assertTrue(fixed[130,100])
        self.assertTrue(fixed[10,10]);self.assertFalse(fixed[150,140])
        self.assertEqual((crop[2]-1)%2,0);self.assertEqual((crop[3]-1)%2,0)
        p=np.asarray(g['contour'])
        self.assertTrue(np.all(p >= crop[:2]));self.assertTrue(np.all(p <= np.array(crop[:2])+crop[2:]-1))


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from layer_transport import transport_layer,project_background,prepare_fixed_plate,merge_visible_material_coverages,sample


class LayerTransportTest(unittest.TestCase):
    def test_underestimated_cloth_coverage_does_not_leave_cloth_color_in_revealed_background(self):
        yy,xx=np.mgrid[:5,:20]
        xy=np.stack([xx,yy],-1).astype(float)
        # Known solid foreground on a known dark plate. A semantic model
        # underestimates coverage; its error must not recolor hidden background.
        background=np.broadcast_to([20.,35.,40.],(5,20,3)).copy()
        source=background.copy();source[:,7:10]=[219.,234.,240.]
        estimated=np.zeros((5,20));estimated[:,7:10]=.74
        coverage,plate=prepare_fixed_plate(source,estimated,background)
        identity,_=transport_layer(source,coverage,plate,xy)
        np.testing.assert_allclose(identity,source,atol=1e-12)
        xy[...,0]-=4
        result,destination=transport_layer(source,coverage,plate,xy)
        np.testing.assert_allclose(result[:,7:10],background[:,7:10],atol=1e-12)
        np.testing.assert_array_equal(destination[:,7:10],np.zeros((5,3)))
        np.testing.assert_allclose(result[:,11:14],source[:,7:10],atol=1e-12)
        premultiplied=source-(1-coverage[...,None])*plate
        self.assertGreaterEqual(float(premultiplied.min()),-1e-12)
        self.assertLessEqual(float((premultiplied-coverage[...,None]*255).max()),1e-12)

    def test_adjacent_visible_materials_do_not_reveal_background_at_internal_seam(self):
        yy,xx=np.mgrid[:5,:25]
        xy=np.stack([xx,yy],-1).astype(float)
        background=np.stack([10.+xx*8,80.+yy*3,190.-xx*4],-1)
        left=np.zeros((5,25));right=np.zeros((5,25))
        left[:,5:12]=1.;left[:,12]=.35
        right[:,12]=.65;right[:,13:20]=1.
        # A pixel is 35% white cloth and 65% dark cloth, with no visible
        # background. These are visible subpixel materials, not overlapping
        # amodal masks or two independent estimates for the same object.
        colors=left[...,None]*[230.,210.,190.]+right[...,None]*[20.,30.,40.]
        true_coverage=np.zeros((5,25));true_coverage[:,5:20]=1.
        source=colors+(1-true_coverage[...,None])*background
        xy[...,0]-=2
        merged=merge_visible_material_coverages([left,right])
        actual,destination=transport_layer(source,merged,background,xy)
        expected=sample(colors,xy)+(1-sample(true_coverage,xy)[...,None])*background
        np.testing.assert_array_equal(destination[:,14],np.ones(5))
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def scene(self):
        yy,xx=np.mgrid[:11,:24]
        xy=np.stack([xx,yy],-1).astype(float)
        background=np.stack([20.+xx*3,35.+yy*2,100.+xx],-1)
        alpha=np.zeros((11,24));alpha[2:9,7:16]=1.;alpha[2:9,6]=.2;alpha[2:9,16]=.6
        foreground=np.broadcast_to([220.,160.,40.],background.shape)
        image=alpha[...,None]*foreground+(1-alpha[...,None])*background
        return image,alpha,foreground,background,xy

    def test_zero_motion_preserves_observed_image_even_with_an_inaccurate_plate(self):
        image,alpha,_,background,xy=self.scene()
        inaccurate=255-background
        result,destination=transport_layer(image,alpha,inaccurate,xy)
        np.testing.assert_allclose(result,image,atol=1e-12)
        np.testing.assert_array_equal(destination,alpha)

    def test_moving_mixed_edge_removes_its_old_background_without_double_alpha(self):
        image,alpha,foreground,background,xy=self.scene()
        xy[...,0]-=3
        result,destination=transport_layer(image,alpha,background,xy)
        shifted=np.zeros_like(alpha);shifted[:,3:]=alpha[:,:-3]
        expected=shifted[...,None]*foreground+(1-shifted[...,None])*background
        np.testing.assert_array_equal(destination,shifted)
        np.testing.assert_allclose(result,expected,atol=1e-12)

    def test_opaque_material_is_the_original_sample_not_a_background_blend(self):
        image,alpha,_,background,xy=self.scene()
        xy[...,0]-=3
        result,destination=transport_layer(image,alpha,background,xy)
        opaque=destination==1
        yy,xx=np.where(opaque)
        np.testing.assert_array_equal(result[opaque],image[yy,xx-3])

    def test_nonfinite_or_misaligned_evidence_is_rejected(self):
        image,alpha,_,background,xy=self.scene()
        for invalid in [alpha[:,:-1],np.full_like(alpha,np.nan),np.full_like(alpha,1.1)]:
            with self.assertRaises(ValueError):
                transport_layer(image,invalid,background,xy)

    def test_plate_projection_keeps_known_layers_and_bounds_moving_residual(self):
        image,alpha,_,background,xy=self.scene()
        np.testing.assert_allclose(project_background(image,alpha,background),background,atol=1e-12)
        projected=project_background(image,alpha,255-background)
        premultiplied=image-(1-alpha[...,None])*projected
        self.assertGreaterEqual(float(premultiplied.min()),-1e-12)
        self.assertLessEqual(float((premultiplied-alpha[...,None]*255).max()),1e-12)
        xy[...,0]-=2.3
        moved,_=transport_layer(image,alpha,projected,xy)
        self.assertGreaterEqual(float(moved.min()),-1e-12)
        self.assertLessEqual(float(moved.max()),255+1e-12)
        # Known visible background is evidence, never model-colored paint.
        np.testing.assert_array_equal(projected[alpha==0],image[alpha==0])


if __name__=='__main__':unittest.main()

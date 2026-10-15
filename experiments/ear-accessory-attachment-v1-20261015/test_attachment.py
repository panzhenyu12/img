"""Known synthetic hanging joint: observe a disconnected attachment before fix."""
import unittest
import numpy as np
from attachment import render_attachments,recompose_motion_delta,attachment_transforms
class AttachmentContract(unittest.TestCase):
    def test_jewelry_stays_attached_to_translated_lobe(self):
        h,w=64,64;ear=np.zeros((h,w));jewel=np.zeros((h,w));ear[12:30,20:32]=1;jewel[30:46,24:28]=1
        # Observable alpha shapes share an edge initially. The independent
        # required result is both source shapes shifted by exactly (+7,+2).
        rgb=np.ones((h,w,3));_,_,ea,ja=render_attachments(rgb,ear,rgb,jewel,(7,2),link_jewelry=True)
        expected_ear=np.zeros((h,w));expected_jewel=np.zeros((h,w));expected_ear[14:32,27:39]=1;expected_jewel[32:48,31:35]=1
        self.assertTrue(np.array_equal(ea,expected_ear),'ear translation mismatch')
        self.assertTrue(np.array_equal(ja,expected_jewel),'jewelry stayed at old coordinate and detached from lobe')
        ys,xs=np.where(ja>.99);self.assertEqual((int(xs.min()),int(ys.min())),(31,32))
        self.assertTrue((ea[ys.min()-1,xs.min():xs.max()+1]>.99).all(),'visible hook does not meet ear')
    def test_identity_preserves_source_when_layer_model_has_color_residual(self):
        # An observed bright edge is poorly reconstructed by the source matte.
        # With no motion, reconstruction uncertainty must not alter the image.
        source=np.array([[[.7,.4,.2],[.9,.8,.7]]])
        model=np.array([[[.6,.3,.1],[.1,.1,.1]]])
        got=recompose_motion_delta(source,model,model)
        np.testing.assert_array_equal(got,source)
    def test_upper_ear_root_stays_attached_while_hook_and_jewelry_follow(self):
        # Source root and lobe are 20 px apart. A 4 px sideways driver cannot
        # move the upper root; the ear must preserve its length and the rigid
        # hanging component must meet the achieved lobe, not an ideal target.
        p=np.array([20.,10.]);h=np.array([20.,30.]);ear,jewelry=attachment_transforms(p,h,(4.,0.))
        apply=lambda m,x:m[:,:2]@x+m[:,2]
        np.testing.assert_allclose(apply(ear,p),p,atol=1e-10,err_msg='ear root tore away from stationary head')
        np.testing.assert_allclose(apply(ear,h),apply(jewelry,h),atol=1e-10,err_msg='jewelry detached from achieved ear lobe')
        self.assertAlmostEqual(float(np.linalg.norm(apply(ear,h)-p)),20.,places=10)
        self.assertGreater(float(apply(ear,h)[0]-h[0]),3.2,'driver was suppressed below 80%')
        a=np.array([18.,38.]);b=np.array([24.,44.])
        self.assertAlmostEqual(float(np.linalg.norm(apply(jewelry,a)-apply(jewelry,b))),float(np.linalg.norm(a-b)),places=10)
if __name__=='__main__':unittest.main()

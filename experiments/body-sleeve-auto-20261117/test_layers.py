import unittest
import numpy as np
from layers import prepare_layers, render_layers


class IndependentSleeveTests(unittest.TestCase):
    def fixture(self):
        h,w=5,16
        y,x=np.mgrid[:h,:w]
        scene=np.broadcast_to([30.,60.,90.],(h,w,3)).copy()
        rear=scene.copy();rear[:,3:13]=np.stack([120+x[:,3:13]*5,100+x[:,3:13],60+x[:,3:13]*2],-1)
        sleeve=np.broadcast_to([210.,30.,40.],(h,w,3)).copy()
        a=np.zeros((h,w));a[:,6:9]=1;a[:,5]=.25;a[:,9]=.75
        b=np.zeros((h,w));b[:,3:13]=1
        source=a[...,None]*sleeve+(1-a[...,None])*rear
        return source,a,b,sleeve,rear,scene,np.stack([x,y],-1).astype(float)

    def test_zero_keeps_the_original_mixed_edge(self):
        source,a,b,fg,lower,scene,xy=self.fixture()
        pair=prepare_layers(source,a,b,fg,lower,scene)
        out,_=render_layers(pair,xy)
        np.testing.assert_allclose(out,source,atol=1e-12)

    def test_fixed_sleeve_does_not_appear_twice_after_rear_moves(self):
        source,a,b,fg,lower,scene,xy=self.fixture()
        pair=prepare_layers(source,a,b,fg,lower,scene)
        xy[...,0]=np.maximum(xy[...,0]-2,0)
        out,_=render_layers(pair,xy)
        # At x=11 the warped source used to contain an opaque red sleeve.
        # The independently moving shirt must reveal its own source stripe.
        np.testing.assert_allclose(out[:,11],lower[:,9],atol=1e-12)
        np.testing.assert_allclose(out[a==1],source[a==1],atol=1e-12)
        self.assertGreater(np.abs(out-source).max(),10)

    def test_known_source_underlay_cannot_be_repainted(self):
        source,a,b,fg,lower,scene,xy=self.fixture()
        lower[:,0]+=1
        with self.assertRaisesRegex(ValueError,"known source"):
            prepare_layers(source,a,b,fg,lower,scene)


if __name__=='__main__':unittest.main()

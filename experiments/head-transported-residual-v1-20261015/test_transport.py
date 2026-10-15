"""Known-layer counterexamples for the isolated color-residual hypothesis."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/source-skin-color-consistency-v1-20261015'))
from compositor import sample_f32


class TransportCounterexample(unittest.TestCase):
    def test_foreground_reconstruction_error_must_not_remain_at_old_location(self):
        # One material sample translates two pixels. Its observed composite is
        # known independently: .5*200 + .5*20 = 110. F estimate is wrong by 30.
        alpha=np.zeros((1,6),dtype='float32');alpha[0,1]=.5
        source=np.full((1,6,3),20,dtype='float32');source[0,1]=110
        estimated_f=np.full_like(source,170);background=np.full_like(source,20)
        residual=source-alpha[...,None]*estimated_f-(1-alpha[...,None])*background
        xy=np.array([[[-2,0],[-1,0],[0,0],[1,0],[2,0],[3,0]]],dtype='float32')
        sampled,a=sample_f32(estimated_f,alpha,xy)
        stationary=sampled+(1-a[...,None])*background+residual
        transported_r,_=sample_f32(residual,np.ones_like(alpha),xy)
        transported=sampled+(1-a[...,None])*background+transported_r
        self.assertEqual(float(stationary[0,1,0]),35) # old ghost
        self.assertEqual(float(stationary[0,3,0]),95) # new material too dark
        self.assertEqual(float(transported[0,1,0]),20)
        self.assertEqual(float(transported[0,3,0]),110)

    def test_exact_source_fit_does_not_prove_a_physical_foreground(self):
        # A wrong low alpha cannot explain a bright material on dark B.
        # The formula reconstructs C exactly but exceeds valid alpha*255.
        alpha=np.float32(.1); observed=np.float32(220); background=np.float32(20)
        premult=observed-(1-alpha)*background
        self.assertAlmostEqual(float(premult+(1-alpha)*background),220)
        self.assertGreater(float(premult),float(alpha*255))


if __name__=='__main__':unittest.main()

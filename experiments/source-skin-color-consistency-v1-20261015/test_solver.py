"""Known textured skin/scene counterexample; alpha is frozen in every test."""
import os,unittest
import numpy as np
from solver import solve

class KnownSoftEdge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        h,w=64,96;yy,xx=np.indices((h,w));cls.a=np.clip((52-xx+2)/4,0,1);cls.f=np.zeros((h,w,3))+[.72,.48,.35]+.03*np.sin(yy[...,None]/3)*[1,.8,.5];cls.b=np.zeros_like(cls.f)+[.04,.03,.02]+.01*np.cos(yy[...,None]/5);cls.c=cls.a[...,None]*cls.f+(1-cls.a[...,None])*cls.b;cls.prior=cls.f.copy();cls.band=(cls.a>=.2)&(cls.a<.995);cls.prior[cls.band]=np.clip(cls.prior[cls.band]+[.06,-.04,.03],0,1);cls.core=cls.a>=.995;cls.allow=np.ones_like(cls.a,bool)
    def result(self,**kwargs):return solve(self.c,self.a,self.prior,self.b,self.core,self.allow,enabled=os.environ.get('COLOR_BASELINE')!='1',**kwargs)
    def test_known_texture_old_edge_has_no_residual_line(self):
        r=self.result();old_exposed=self.c-self.a[...,None]*r['foreground']+self.a[...,None]*self.b
        self.assertLess(np.max(abs(old_exposed[self.band]-self.b[self.band]))*255,1e-8)
    def test_source_texture_is_preserved_in_corrected_foreground(self):
        r=self.result();self.assertLess(np.max(abs(r['foreground'][self.band]-self.f[self.band]))*255,1e-8)
    def test_small_alpha_and_unconfirmed_material_are_unchanged(self):
        r=self.result();self.assertTrue(np.array_equal(r['foreground'][self.a<.1],self.prior[self.a<.1]))
        q=solve(self.c,self.a,self.prior,self.b,self.core,np.zeros_like(self.allow));self.assertTrue(np.array_equal(q['foreground'],self.prior))
    def test_no_observed_skin_donors_abstains(self):
        q=solve(self.c,self.a,self.prior,self.b,np.zeros_like(self.core),self.allow);self.assertFalse(q['changed'].any());self.assertFalse(q['domain'].any())
    def test_wrong_background_cannot_force_out_of_local_skin_box(self):
        bad=np.ones_like(self.b)*.99;q=solve(self.c,self.a,self.prior,bad,self.core,self.allow);d=q['changed'];self.assertTrue(q['clipped'].any());self.assertTrue((q['foreground'][d]>=q['lower'][d]-1/255-1e-12).all());self.assertTrue((q['foreground'][d]<=q['upper'][d]+1/255+1e-12).all());self.assertTrue(np.array_equal(q['foreground'][~d],self.prior[~d]))
    def test_color_box_must_not_worsen_an_already_better_source_fit(self):
        c=np.zeros((5,5,3))+[.7,.5,.4];a=np.ones((5,5));f=c.copy();b=np.zeros_like(c)+.05;core=np.ones((5,5),bool)
        a[2,2]=.5;core[2,2]=False;f[2,2]=[.55,.42,.32];c[2,2]=a[2,2]*f[2,2]+(1-a[2,2])*b[2,2]
        q=solve(c,a,f,b,core,np.ones((5,5),bool));self.assertTrue(np.array_equal(q['foreground'][2,2],f[2,2]))
if __name__=='__main__':unittest.main(verbosity=2)

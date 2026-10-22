"""Rejected-but-invertible previews must never hide the failed contract."""
import unittest
import numpy as np

from evaluate_joint_jaw_field import inspect_for_render


class DiagnosticRenderTests(unittest.TestCase):
    def field(self, amplitude):
        y,x=np.mgrid[:9,:9]
        field=np.zeros((9,9,2),dtype=np.float32)
        field[:,:,0]=amplitude*np.minimum(x,8-x)*np.minimum(y,8-y)/4.
        return field

    def test_explicit_preview_preserves_rejection_on_an_invertible_map(self):
        field=self.field(.6)
        strict,allowed=inspect_for_render(field)
        self.assertFalse(allowed)
        self.assertFalse(strict['passed_strict'])
        self.assertGreater(strict['symmetric_eigenvalue_min'],0.)
        diagnostic,allowed=inspect_for_render(field,diagnostic=True)
        self.assertTrue(allowed)
        self.assertEqual(diagnostic,strict)

    def test_diagnostic_flag_does_not_allow_folds_or_a_moving_boundary(self):
        _,allowed=inspect_for_render(self.field(2.),diagnostic=True)
        self.assertFalse(allowed)
        translated=np.ones((9,9,2),dtype=np.float32)
        _,allowed=inspect_for_render(translated,diagnostic=True)
        self.assertFalse(allowed)

    def test_identity_remains_a_normal_render(self):
        inspected,allowed=inspect_for_render(self.field(0.))
        self.assertTrue(allowed)
        self.assertTrue(inspected['passed_strict'])


if __name__=='__main__':unittest.main()

import tempfile
from pathlib import Path
import unittest
import numpy as np

from audit_interface_chord_field import sha,validate_frozen_profiles,normalized_local_chord_change


class HistoricalProfileTests(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.profile=Path(folder.name).resolve()/'profile.json'
        self.profile.write_text('{"amount":100,"shape":"original"}')
        self.historical=sha(self.profile)
        self.protocol={'frozen_target_profile':str(self.profile),
            'inputs_sha256':{str(self.profile):self.historical}}

    def test_unchanged_historical_profile_is_accepted(self):
        validate_frozen_profiles({'baseline':self.protocol,'candidate':self.protocol},self.historical)

    def test_same_path_and_current_artifact_hash_cannot_replace_history(self):
        self.profile.write_text('{"amount":100,"shape":"changed"}')
        # This is exactly the old weakness: both current path and artifact
        # digest agree while the baseline protocol still binds older content.
        with self.assertRaisesRegex(ValueError,'historically frozen'):
            validate_frozen_profiles({'baseline':self.protocol},sha(self.profile))

    def test_missing_historical_digest_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'historically frozen'):
            validate_frozen_profiles({'baseline':{'frozen_target_profile':str(self.profile)}},self.historical)


class LocalChordMetricTests(unittest.TestCase):
    def test_local_reversal_is_detected_despite_identical_global_ranges(self):
        source=np.zeros((3,2,2));source[:,:,1]=[[0.,3.],[1.,4.],[2.,5.]]
        reference=source.copy();reference[:,:,0]=[[0.,0.],[1.,-1.],[0.,0.]]
        candidate=source.copy();candidate[:,:,0]=-reference[:,:,0]
        # Both global numerator ranges are [-2,+2]. The two locations have
        # nevertheless reversed their local bending directions completely.
        actual=normalized_local_chord_change(source,reference,candidate,np.eye(2))
        np.testing.assert_allclose(actual,[4.,-4.],rtol=0.,atol=1e-12)

    def test_unequal_source_heights_define_the_normalization(self):
        source=np.array([[[0.,0.]],[[0.,2.]],[[0.,5.]]])
        candidate=source.copy();candidate[1,0,0]=.012
        # N changes by -.012*(2+3)=-.06; frozen h0*h1=6, so delta=-.01.
        actual=normalized_local_chord_change(source,source,candidate,np.eye(2))
        np.testing.assert_allclose(actual,[-.01],rtol=0.,atol=1e-12)


if __name__=='__main__':unittest.main()

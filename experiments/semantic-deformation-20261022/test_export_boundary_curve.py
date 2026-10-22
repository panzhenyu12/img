"""Input-identity regressions; fixtures contain no photographs or trained data."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from export_boundary_curve import run, sha


def save(path,value):
    path.write_text(json.dumps(value))


class ExportBoundaryIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.case=self.root/'baseline'/'plus100';self.case.mkdir(parents=True)
        self.source=self.root/'source.png';self.source.write_bytes(b'identity fixture, no photograph')
        self.geometry={'frame':{'across':[1.,0.],'down':[0.,1.]}}
        self.profile=self.root/'profile.json'
        save(self.profile,{'amount':100.,'source_geometry':self.geometry})
        self.protocol=self.case.parent/'protocol.json'
        save(self.protocol,{'crop':[0,0,3,3],'source_geometry':self.geometry,
            'frozen_target_profile':str(self.profile),'inputs_sha256':{
                str(self.profile):sha(self.profile),str(self.source):sha(self.source)}})
        save(self.case/'decision.json',{'amount':100.,'final_geometry':{'passed_strict':True}})
        np.save(self.case/'inverse-field.npy',np.ones((3,3,2)))
        self.audit=self.root/'audit.json'
        save(self.audit,{'inputs_sha256':{str(p):sha(p) for p in [self.protocol,
            self.case/'decision.json',self.case/'inverse-field.npy']}})
        self.neck=self.root/'neck.json'
        save(self.neck,{'schema':'reviewed-source-neck-boundary-candidate-v1',
            'source_image_sha256':sha(self.source),'automatic_edit_authorized':False,
            'segments':[{'source_points':[[0.,565.],[0.,610.]]}],
            'input_hashes':{str(self.source):sha(self.source)}})
        self.interface=self.root/'source-interface.json'
        save(self.interface,{'schema':'reviewed-visible-source-interface-v1',
            'source_image':str(self.source),'source_image_sha256':sha(self.source),
            'source_size':[3,3],'mode':'model_source_visual_review_only',
            'visibility_status':'unknown','automatic_edit_authorized':False,
            'segments':[{'id':'left','source_points':[[0.,515.],[0.,565.],[0.,610.]]}],
            'prior_review_consistency':{'old_neck_source_path':str(self.neck),
                'old_neck_source_sha256':sha(self.neck)},
            'inputs_sha256':{str(self.source):sha(self.source)}})
        save(self.root/'visual-source-review.json',{'mode':'model_source_visual_review_only',
            'automatic_edit_authorized':False,'visibility_status':'unknown',
            'source_image_sha256':sha(self.source),'hashes':{str(self.interface):sha(self.interface)}})

    def export(self):
        return run(self.interface,self.case,self.profile,self.root/'candidate',preferred_audit=self.audit)

    def test_replaced_zero_field_cannot_impersonate_frozen_positive_request(self):
        np.save(self.case/'inverse-field.npy',np.zeros((3,3,2)))
        with self.assertRaisesRegex(ValueError,'independently frozen audit'):
            self.export()
        self.assertFalse((self.root/'candidate').exists())

    def test_same_profile_path_is_not_a_proof_of_historical_content(self):
        profile=json.loads(self.profile.read_text());profile['source_geometry']['changed']=True
        save(self.profile,profile)
        with self.assertRaisesRegex(ValueError,'same frozen jaw profile'):
            self.export()
        self.assertFalse((self.root/'candidate').exists())

    def test_changed_upper_interface_is_not_covered_by_old_low_neck_review(self):
        interface=json.loads(self.interface.read_text());interface['segments'][0]['source_points'][0][0]=1.
        save(self.interface,interface)
        with self.assertRaisesRegex(ValueError,'independently viewed and frozen'):
            self.export()
        self.assertFalse((self.root/'candidate').exists())


if __name__=='__main__':unittest.main()

"""Bound source subtype preparation, replay verification and tamper refusal."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io

import numpy as np
from PIL import Image

from apply_consistent_source_roles import apply_roles
from attachment_subtypes import prepare_partition, verify_partition, sha
from attachment_source_plan import source_barriers
from consistent_detail_control import raw_role_proposal, ROLE_CLASSES
from source_masks import mask


class BoundSubtypeEvidence(unittest.TestCase):
    def fixture(self,root):
        w,h=64,40;source=root/'source.png'
        Image.new('RGBA',(w,h),(128,100,82,255)).save(source)
        probabilities=np.zeros((19,h,w),np.float32);probabilities[1]=1
        for index,box in [(6,(8,8,42,20)),(9,(50,27,56,33))]:
            x,y,r,b=box;probabilities[1,y:b,x:r]=.1;probabilities[index,y:b,x:r]=.9
        selected={'id':'crop-1','crop':[0,0,w,h],'probabilities':probabilities}
        evidence=raw_role_proposal(selected,(w,h),(w,h),'observed-support')
        raster=lambda a:{'shape':'raster','width':w,'height':h,'data':a.ravel().tolist()}
        masks={key:raster(np.zeros((h,w),np.uint8)) for key in ROLE_CLASSES}
        for name,x in [('leftEye',12),('rightEye',29)]:
            a=np.zeros((h,w),np.uint8);a[12:16,x:x+8]=255;masks[name]=raster(a)
        original={'source':{'width':w,'height':h},'operations':[{'params':{'semanticMasks':masks}}]}
        original_path=root/'detected.json';original_path.write_text(json.dumps(original))
        cache=root/'cache';cache.mkdir();posterior=cache/'crop-1-probabilities.npz'
        np.savez_compressed(posterior,probabilities=probabilities)
        provenance={'sourceSHA256':sha(source),'sourceGraph':str(original_path),
            'sourceGraphSHA256':sha(original_path),'sourceGraphOrigin':'automatic-detection',
            'manualEvidence':False,'selection':{'id':'crop-1','crop':[0,0,w,h]},
            'posteriorSHA256':hashlib.sha256(probabilities.tobytes()).hexdigest(),
            'cache':{'modelSHA256':'1d5c459e3d4691eb6120994fd36a8d360fe3732997d1bbf245278145cdd11d20',
                'passes':[{'id':'crop-1','crop':[0,0,w,h],
                'probabilitiesFile':str(posterior),'fileSHA256':sha(posterior)}]}}
        evidence['provenance']=provenance
        (cache/'raw-source-role-sidecar.json').write_text(json.dumps(evidence))
        (cache/'report.json').write_text(json.dumps(provenance))
        graph,_=apply_roles(original,evidence);graph_path=root/'roles.json';graph_path.write_text(json.dumps(graph))
        m={k:mask(v,w,h) for k,v in graph['operations'][0]['params']['semanticMasks'].items()}
        barriers=source_barriers(m)
        return source,graph_path,cache,m['accessories'],barriers,posterior

    def test_frozen_partition_replays_same_source_and_rejects_changed_bindings_or_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source,graph,cache,accessory,barriers,posterior=self.fixture(root)
            support,items,record=prepare_partition(source,graph,cache,accessory,barriers)
            self.assertEqual([r['kind'] for r in record['components']],['eyewear','earring'])
            self.assertFalse(support[8:20,8:42].any());self.assertTrue(support[27:33,50:56].all())
            frozen=root/'partition.json';frozen.write_text(json.dumps(record))
            candidate=json.loads(graph.read_text());candidate['operations'][0]['params']['__sourceMovingAttachmentSupport']={
                'shape':'raster','width':64,'height':40,'data':(support.astype(np.uint8)*255).ravel().tolist()}
            output=root/'candidate.json';output.write_text(json.dumps(candidate))
            self.assertEqual(len(verify_partition(frozen,source,output,support,accessory,barriers)),1)
            with self.assertRaisesRegex(ValueError,'differs'):
                verify_partition(frozen,source,output,accessory>.02,accessory,barriers)
            changed=copy.deepcopy(candidate);changed['operations'][0]['params']['semanticMasks']['accessories']['data'][0]=255
            output.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'outside'):
                verify_partition(frozen,source,output,support,accessory,barriers)
            output.write_text(json.dumps(candidate))
            Image.new('RGBA',(64,40),(129,100,82,255)).save(source)
            with self.assertRaisesRegex(ValueError,'binding'):
                verify_partition(frozen,source,output,support,accessory,barriers)

    def test_altered_posterior_is_rejected_before_a_permission_is_issued(self):
        with tempfile.TemporaryDirectory() as directory:
            source,graph,cache,accessory,barriers,posterior=self.fixture(Path(directory))
            posterior.write_bytes(posterior.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'posterior file changed'):
                prepare_partition(source,graph,cache,accessory,barriers)

    def test_real_prepare_and_rebuild_admission_keep_missing_protect_mask_identity(self):
        from prepare_attachment_source_plan import prepare
        from attachment_subtypes import attachment_release_from_source
        for explicit_zero in [False,True]:
            with self.subTest(explicit_zero=explicit_zero),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                source,graph,cache,accessory,barriers,_=self.fixture(root)
                if explicit_zero:
                    original=json.loads(graph.read_text())
                    original['operations'][0]['params']['protectMask']={
                        'shape':'raster','width':64,'height':40,'data':[0]*(64*40)}
                    graph.write_text(json.dumps(original))
                output=root/'prepared'
                with contextlib.redirect_stdout(io.StringIO()):prepare(source,graph,output,cache)
                candidate=json.loads((output/'candidate-graph.json').read_text())['operations'][0]['params']
                masks={k:mask(v,64,40) for k,v in candidate['semanticMasks'].items()}
                actual=attachment_release_from_source(candidate,masks,source,output/'candidate-graph.json',output/'partition.json')
                expected=mask(json.loads((output/'support.json').read_text()),64,40)>=.5
                np.testing.assert_array_equal(actual,expected)
                self.assertFalse(actual[8:20,8:42].any())
                self.assertTrue(actual[27:33,50:56].all())
                record=json.loads((output/'partition.json').read_text())
                self.assertEqual('user' in record['barrierSHA256'],explicit_zero)


if __name__=='__main__':unittest.main()

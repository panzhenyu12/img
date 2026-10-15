"""Verify the actual ONNX feed rather than merely the caller's kwargs."""
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('original_probe',ROOT/'experiments/zim-20261014/run.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
np,torch=probe.np,probe.torch
from adapter import CombinedPromptDecoder

class RecordingSession:
    def run(self, names, feed):
        self.feed=feed
        return np.zeros((1,4,2,2),dtype='float32'),np.zeros((1,4),dtype='float32')

class OriginalDecoder:
    def __init__(self):
        self.ort_session=RecordingSession();self.num_mask_tokens=4;self.called=False
    def __call__(self,*args):
        self.called=True
        return 'unchanged-original-result'

class AdapterContract(unittest.TestCase):
    def run_combined(self,attention):
        original=OriginalDecoder()
        decoder=CombinedPromptDecoder(original,attention=attention,bbox_attention=lambda b:torch.full((1,8,8),.75))
        points=(torch.tensor([[[12.,34.],[56.,78.]]]),torch.tensor([[1.,0.]]))
        box=torch.tensor([2.,3.,90.,91.]);features=torch.zeros((1,4,8,8))
        decoder([features]*3,features,points,box,torch.full((1,8,8),.25))
        return original.ort_session.feed
    def test_negative_point_and_box_both_reach_decoder(self):
        feed=self.run_combined('point')
        np.testing.assert_array_equal(feed['point_labels'],[[1,0,2,3]])
        np.testing.assert_array_equal(feed['point_coords'],[[[12,34],[56,78],[2,3],[90,91]]])
        self.assertEqual(feed['point_coords'].dtype,np.float32)
    def test_attention_is_an_explicit_independent_factor(self):
        point=self.run_combined('point');box=self.run_combined('box')
        for key in ['point_coords','point_labels','image_embeddings','feat_D0','feat_D1','feat_D2']:
            np.testing.assert_array_equal(point[key],box[key])
        self.assertTrue(np.all(point['attn_mask']==.25))
        self.assertTrue(np.all(box['attn_mask']==.75))
    def test_single_prompt_modes_delegate_to_unmodified_vendor(self):
        for points,boxes in [(None,torch.tensor([1,2,3,4])),((torch.zeros((1,1,2)),torch.ones((1,1))),None)]:
            original=OriginalDecoder();decoder=CombinedPromptDecoder(original)
            self.assertEqual(decoder(None,None,points,boxes,None),'unchanged-original-result')
            self.assertTrue(original.called)

if __name__=='__main__':unittest.main()

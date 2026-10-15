"""Automatic background prompts must not label known facial features as scene."""
import unittest
import run
import numpy as np


class SourcePromptContract(unittest.TestCase):
    def test_open_mouth_is_not_a_negative_background_prompt(self):
        # An independently labelled face with a large open mouth. Skin
        # intentionally excludes teeth; the subject still owns the teeth.
        w=h=64
        masks={key:np.zeros((h,w),float) for key in ['skin','neck','hair','ears','accessories','teeth','lips','leftEye','rightEye','leftBrow','rightBrow','face']}
        masks['skin'][5:57,10:54]=1
        masks['skin'][25:39,18:46]=0
        masks['teeth'][25:39,18:46]=1
        masks['face'][5:57,10:54]=1
        params={'landmarks':[{'x':10/w,'y':5/h},{'x':54/w,'y':45/h}]}
        _,positive,negative,_=run.prompts(params,masks,w,h)
        self.assertGreater(len(positive),0)
        self.assertGreater(len(negative),0)
        for x,y in negative:
            self.assertEqual(masks['teeth'][y,x],0,f'Known teeth incorrectly prompted as scene at {(x,y)}')


if __name__=='__main__':unittest.main(verbosity=2)

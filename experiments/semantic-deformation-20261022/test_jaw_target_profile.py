"""Synthetic baked-strength profiles; no photograph or field solve."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jaw_collocation import sample_jaw_targets, independent_jaw_probes
from jaw_target_profile import load_profile, apply_profile_samples, profile_contour_targets


def fixture():
    root = np.sqrt(2.)
    geometry = {'contour': [[-4., 0.], [-2., 2.], [0., 4.], [2., 2.], [4., 0.], [4., -4.], [-4., -4.]],
                'frame': {'origin': [0., -2.], 'across': [1., 0.], 'midline_u': 0.},
                'arcs': [{'id': 'left-jaw', 'start': {'edge': 0, 't': .5}, 'end': {'edge': 2, 't': 0.}, 'direction': 1},
                         {'id': 'right-jaw', 'start': {'edge': 3, 't': .5}, 'end': {'edge': 2, 't': 0.}, 'direction': -1}],
                'selected_rows': [1, 3]}
    digest = hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    profile = {'schema': 'jaw-arclength-piecewise-linear-profile-v1', 'controller': 'synthetic-new-shape',
               'amount': 60., 'source_geometry': copy.deepcopy(geometry), 'source_geometry_sha256': digest,
               'across_axis': [1., 0.], 'profiles': []}
    for name, direction, start, sign in [('left-jaw', 1, root, 1.), ('right-jaw', -1, 7*root, -1.)]:
        profile['profiles'].append({'arc_id': name, 'direction': direction,
            'original_contour_start_arclength_px': start, 'arc_length_px': 3*root,
            'arclength_px': [0., root, 2*root, 3*root],
            'original_source_vertex_arclength_px': [0., root, 3*root],
            'across_displacement_px': [0., sign*.5, sign*.25, 0.]})
    return geometry, profile


class JawTargetProfileTests(unittest.TestCase):
    def load(self, profile, geometry, amount=60., *, corrupt_source=False):
        with tempfile.TemporaryDirectory() as tmp:
            profile = copy.deepcopy(profile)
            source = Path(tmp)/'source.png'; source.write_bytes(b'synthetic image identity')
            evidence = Path(tmp)/'source-evidence.json'; evidence.write_text('{"synthetic":true}')
            historical_code = Path(tmp)/'historical.py'; historical_code.write_text('# old code')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            protocol = Path(tmp)/'protocol.json'
            protocol.write_text(json.dumps({'source_geometry': profile['source_geometry'],
                'inputs_sha256': {str(p): digest(p) for p in [source, evidence, historical_code]}}))
            historical_code.write_text('# later code does not change frozen source identity')
            profile.update(source_protocol=str(protocol), source_protocol_sha256=digest(protocol))
            if corrupt_source:
                source.write_bytes(b'different source identity')
            path = Path(tmp)/'profile.json'; path.write_text(json.dumps(profile))
            return load_profile(path, geometry, amount)

    def test_baked_profile_is_not_multiplied_again_and_original_data_stay_unchanged(self):
        geometry, raw = fixture(); profile = self.load(raw, geometry)
        old_geometry, old_profile = copy.deepcopy(geometry), copy.deepcopy(profile)
        samples = sample_jaw_targets(geometry, 60., spacing=np.sqrt(2.))
        before = copy.deepcopy(samples)
        output = apply_profile_samples(samples, profile, geometry)
        for row, control in enumerate(output['controls']):
            if control['arc_id'] is None or control['arc_t'] in [0., 1.]:
                np.testing.assert_array_equal(output['target'][row], output['source'][row])
            elif abs(control['arc_arclength_px']-np.sqrt(2.)) < 1e-12:
                sign = 1. if control['arc_id'] == 'left-jaw' else -1.
                np.testing.assert_allclose(output['target'][row]-output['source'][row], [sign*.5, 0.], atol=1e-14, rtol=0)
        for key in ['source', 'target', 'weights']:
            np.testing.assert_array_equal(samples[key], before[key])
        for key in ['source', 'weights']:
            np.testing.assert_array_equal(output[key], before[key])
        self.assertEqual(output['controls'], before['controls'])
        self.assertEqual(geometry, old_geometry); self.assertEqual(profile, old_profile)
        self.assertEqual(output['report']['controller'], 'synthetic-new-shape')
        self.assertTrue(output['report']['amount_already_baked'])

    def test_fractional_source_edges_and_off_grid_probes_share_the_new_intent(self):
        geometry, raw = fixture(); profile = self.load(raw, geometry)
        contour = profile_contour_targets(geometry, profile)
        expected = np.asarray(geometry['contour']).copy(); expected[1, 0] += .5; expected[3, 0] -= .5
        np.testing.assert_allclose(contour, expected, atol=1e-14, rtol=0)
        samples = independent_jaw_probes(geometry, 60., spacing=.5)
        output = apply_profile_samples(samples, profile, geometry)
        for row, control in enumerate(output['controls']):
            s = control['arc_arclength_px']; r = np.sqrt(2.)
            if s <= r: value = .5*s/r
            elif s <= 2*r: value = .5-.25*(s-r)/r
            else: value = .25-.25*(s-2*r)/r
            if control['arc_id'] == 'right-jaw': value *= -1
            np.testing.assert_allclose(output['target'][row]-samples['source'][row], [value, 0.], atol=1e-14, rtol=0)

    def test_wrong_sign_or_source_geometry_cannot_reuse_a_profile(self):
        geometry, profile = fixture()
        with self.assertRaisesRegex(ValueError, 'amount'):
            self.load(profile, geometry, -60.)
        changed = copy.deepcopy(geometry); changed['contour'][0][0] += .01
        with self.assertRaisesRegex(ValueError, 'geometry'):
            self.load(profile, changed)
        with self.assertRaisesRegex(ValueError, 'amount'):
            apply_profile_samples(sample_jaw_targets(geometry, -60.), profile, geometry)
        with self.assertRaisesRegex(ValueError, 'source image'):
            self.load(profile, geometry, corrupt_source=True)

    def test_rejects_missing_source_knots_nonfinite_displacements_and_nonzero_endpoints(self):
        geometry, profile = fixture()
        variants = []
        missing = copy.deepcopy(profile)
        missing['profiles'][0]['arclength_px'].pop(1); missing['profiles'][0]['across_displacement_px'].pop(1)
        variants.append(missing)
        nonfinite = copy.deepcopy(profile); nonfinite['profiles'][1]['across_displacement_px'][1] = float('nan'); variants.append(nonfinite)
        endpoint = copy.deepcopy(profile); endpoint['profiles'][0]['across_displacement_px'][0] = .01; variants.append(endpoint)
        duplicate = copy.deepcopy(profile); duplicate['profiles'][1]['arc_id'] = 'left-jaw'; variants.append(duplicate)
        reversed_knots = copy.deepcopy(profile); reversed_knots['profiles'][0]['arclength_px'][1:3] = reversed(reversed_knots['profiles'][0]['arclength_px'][1:3]); variants.append(reversed_knots)
        for value in variants:
            with self.assertRaises(ValueError): self.load(value, geometry)


if __name__ == '__main__': unittest.main()

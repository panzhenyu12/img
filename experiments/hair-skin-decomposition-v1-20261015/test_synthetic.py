"""Independent known-color/alpha reconstruction and changed-backing checks."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from decompose import decompose, recompose, linearize, encode

HERE = Path(__file__).resolve().parent


def save(path, rgb):
    Image.fromarray(np.floor(np.clip(rgb, 0, 1) * 255 + .5).astype('uint8')).save(path)


def run_case(textured=False, incorrect_alpha=False, space='linear'):
    h, w = 128, 192
    y, x = np.indices((h, w), dtype='float64')
    edge = 72 + 7 * np.sin(y / 27)
    t = np.clip((x - edge + 4) / 8, 0, 1)
    alpha_true = 1 - t * t * (3 - 2 * t)
    hair = np.zeros((h, w, 3)) + [.14, .095, .065]
    skin = np.zeros((h, w, 3)) + [.85, .65, .50]
    if textured:
        hair += (.035 * np.sin(y / 7) + .012 * np.sin(x / 9))[..., None]
        skin += (.035 * np.sin(y / 19) + .0005 * (x - 90))[..., None]
    under = np.zeros((h, w, 3)) + [.08, .065, .05]
    # Real skin movement reveals a gap behind stationary fine hair.
    under[x > edge + 18] = skin[x > edge + 18]
    if space == 'linear':
        source = encode(alpha_true[..., None] * linearize(hair) + (1 - alpha_true[..., None]) * linearize(skin))
        expected = encode(alpha_true[..., None] * linearize(hair) + (1 - alpha_true[..., None]) * linearize(under))
    else:
        source = alpha_true[..., None] * hair + (1 - alpha_true[..., None]) * skin
        expected = alpha_true[..., None] * hair + (1 - alpha_true[..., None]) * under
    observed_alpha = alpha_true.copy()
    if incorrect_alpha:
        mixed = (alpha_true > .001) & (alpha_true < .995)
        observed_alpha[mixed] = np.clip(alpha_true[mixed] - .10, .002, .994)
    # Exact constant-color fixture anchors must be physically opaque/pure. A
    # .999 semantic cutoff still contains mixed RGB and is not exact truth.
    hair_core, skin_core = alpha_true == 1, alpha_true == 0
    result = decompose(source, observed_alpha, hair_core, skin_core, np.ones((h, w), bool), 15,
                       space=space, refine_alpha=incorrect_alpha)
    candidate = source.copy()
    rendered = recompose(result, under)
    candidate[result['trusted']] = rendered[result['trusted']]
    # The pure skin belongs to the moving layer; outside the edge band its new
    # position/backing is exact in this synthetic fixture.
    candidate[skin_core] = expected[skin_core]
    band = (alpha_true > .001) & (alpha_true < .995)
    error = np.abs(candidate[band] - expected[band]) * 255
    raw_error = np.abs(source[band] - expected[band]) * 255
    stats = {**result['stats'], 'textured': textured, 'incorrectInputAlpha': incorrect_alpha,
             'newBackingMeanAbsByte': float(error.mean()), 'newBackingMaxAbsByte': float(error.max()),
             'frozenMixedRgbMeanAbsByte': float(raw_error.mean()),
             'knownAlphaMeanAbsError': float(np.abs(result['alpha'][band] - alpha_true[band]).mean()),
             'fixedOpaqueHairMaxByte': float(np.abs(candidate[hair_core] - source[hair_core]).max() * 255)}
    assert stats['fixedOpaqueHairMaxByte'] == 0
    assert error.mean() < raw_error.mean() * .15, stats
    if not textured and not incorrect_alpha:
        assert stats['newBackingMeanAbsByte'] < 1e-6, stats
    if textured:
        assert stats['newBackingMeanAbsByte'] < 2, stats
    return source, expected, candidate, result, stats


def main():
    out = HERE / 'synthetic'
    out.mkdir(exist_ok=False)
    report = []
    for textured, wrong, space in [(False, False, 'linear'), (True, False, 'linear'),
                                  (True, True, 'linear'), (False, False, 'srgb'), (True, True, 'srgb')]:
        name = f"{'texture' if textured else 'constant'}-{'biased' if wrong else 'exact'}-{space}"
        source, truth, candidate, result, stats = run_case(textured, wrong, space)
        for label, image in [('source', source), ('truth', truth), ('candidate', candidate)]:
            save(out / f'{name}-{label}.png', image)
        sheet = Image.new('RGB', (source.shape[1] * 3, source.shape[0] + 24), '#20242a')
        draw = ImageDraw.Draw(sheet)
        for i, label in enumerate(['source', 'truth', 'candidate']):
            sheet.paste(Image.open(out / f'{name}-{label}.png'), (source.shape[1] * i, 24))
            draw.text((source.shape[1] * i + 3, 5), label, fill='white')
        sheet.save(out / f'{name}-comparison-1x.png')
        stats['name'] = name
        report.append(stats)
        print(json.dumps(stats), flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

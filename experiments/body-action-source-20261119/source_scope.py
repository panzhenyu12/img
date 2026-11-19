"""Freeze original-image review scopes; these are not edit controls."""
from pathlib import Path
import argparse,hashlib,json
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
SCOPE=json.loads('{"version": "body-visible-source-scope/1", "sourceSHA256": "70bfac21aa227629e24ae91ebc6b65e47177b84e0f000d5a14415cc467d8ff94", "dimensions": [1280, 1013], "frozenBeforeDetectionAndEdits": true, "manualMotionInput": false, "target": "visible suited torso material and its garment side; no claim of hidden anatomical waist width", "reviewCrop": [95, 360, 905, 935], "crops": {"right-jacket-side": [575, 525, 700, 895], "left-sleeve-laptop": [100, 555, 450, 925], "right-sleeve": [580, 355, 890, 560]}, "sourceOcclusions": ["laptop and left hand in front of lower torso", "left sleeve beside and partly over laptop edge", "open right jacket panel hangs in front of shirt; actual waist side hidden"], "sourceVisibleArcs": [{"id": "right-visible-jacket-side", "material": "open jacket torso/flap, not anatomical waist", "points": [[608, 540], [623, 591], [642, 654], [654, 701], [662, 752], [664, 787], [661, 822], [653, 852], [638, 877]]}, {"id": "left-upper-jacket-side", "material": "short visible jacket contour above laptop; lower continuation hidden", "points": [[248, 575], [243, 607], [234, 642], [224, 671]]}], "fixedMaterialProbeBoxes": {"laptop-core": [225, 730, 335, 820], "left-hand-core": [170, 840, 235, 890], "left-sleeve-core": [145, 610, 180, 650], "right-sleeve-core": [710, 455, 755, 485], "right-hand-core": [868, 273, 898, 310]}, "sceneProbeBoxes": {"bench-line": [30, 696, 90, 707]}, "successRequires": "visible torso garment responds both directions without new splits/steps; held laptop/hands/sleeves remain independent; no invented hidden naked waist measurement", "limitations": ["right loose jacket flap attachment response requires visual judgment, not a known true waist contour", "RGB probes corroborate preservation but do not establish optical alpha or complete depth order"]}')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);a=ap.parse_args()
 source=ROOT/'test/images/action-suit.jpg'
 if hashlib.sha256(source.read_bytes()).hexdigest()!=SCOPE['sourceSHA256']:raise ValueError('review scope belongs to a different original source')
 a.output.mkdir(exist_ok=False);data={**SCOPE,'source':str(source)}
 (a.output/'source-scope.json').write_text(json.dumps(data,indent=2))
 image=Image.open(source).convert('RGB');draw=ImageDraw.Draw(image)
 for arc in data['sourceVisibleArcs']:draw.line([tuple(p) for p in arc['points']],fill=(0,255,255),width=2)
 for name,box in data['fixedMaterialProbeBoxes'].items():
  draw.rectangle(box,outline=(255,0,255),width=2);draw.text((box[0],box[1]-12),name,fill=(255,0,255))
 for box in data['sceneProbeBoxes'].values():draw.rectangle(box,outline=(255,255,0),width=2)
 image.save(a.output/'source-scope-overlay.png')
if __name__=='__main__':main()

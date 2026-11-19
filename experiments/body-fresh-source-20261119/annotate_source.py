"""Freeze declared manual source-material roles after the automatic failure.

These polygons are an explicitly human-equivalent source annotation, not a
recovered model posterior. The source-only overlays must be independently
reviewed before source-layer inference or motion replay. The original graph is
only hashed: its geometry, parameters and protection are not edited here.
"""
from pathlib import Path
import hashlib
import json
import argparse
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
EXPECTED_SOURCE_SHA256='f53c0889dc4e7e5ebe39192ad99a48375b6bbd26c2626750a1fb5e1f99a92ca2'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('evidence',type=Path)
    ap.add_argument('--output-name',default='manual-role-control')
    args=ap.parse_args(); base=args.evidence.resolve()
    source=base/'source.png'; graph=base/'source-graph.json'
    if sha(source)!=EXPECTED_SOURCE_SHA256:
        raise ValueError('these reviewed manual polygons belong only to the frozen fullbody-side-fashion source')
    if Path(args.output_name).name != args.output_name: raise ValueError('output-name must be a folder name')
    out=base/args.output_name; out.mkdir(exist_ok=False)
    roles=[
      {'id':'sweater-front-arm','instance':'sweater-1','role':'stationary-front','observed':'visible pink bent sleeve; shoulder seam is a manual motion-region split within the same garment',
       'polygon':[[534,214],[555,217],[574,228],[593,244],[613,271],[636,296],[650,292],[665,294],[671,300],[675,315],[671,337],[665,357],[656,377],[644,394],[635,400],[617,399],[604,395],[590,382],[575,363],[560,343],[546,324],[529,306],[514,291],[507,278],[504,260],[510,241],[520,226]]},
      {'id':'raised-hand','instance':'person-1','role':'stationary-front','observed':'visible raised hand; hair overlap has uncertain identity',
       'polygon':[[637,297],[636,279],[633,268],[637,258],[643,249],[651,255],[657,254],[665,262],[671,269],[675,276],[676,284],[672,297],[669,305],[654,305]]},
      {'id':'lower-hand','instance':'person-1','role':'stationary-front','observed':'visible hand above bag; lower fingers overlap bag handle',
       'polygon':[[646,412],[656,415],[665,423],[670,435],[680,446],[683,462],[679,475],[671,483],[660,485],[653,479],[646,477],[638,473],[635,463],[637,452],[638,440],[636,430],[641,418]]},
      {'id':'lower-wrist-connector','instance':'sweater-1','role':'stationary-front',
       'observed':'source-review correction before inference: narrow wrist/cuff connector above black wrist line; uncertain 4-pixel boundary is passed to matting, not declared alpha truth',
       'polygon':[[638,400],[646,398],[652,401],[659,407],[666,413],[669,422],[663,425],[655,416],[645,413],[637,419]]},
      {'id':'bag-main','instance':'bag-1','role':'stationary-front','observed':'visible bag body and tassels; fine boundary is optical uncertainty',
       'polygon':[[609,489],[626,492],[646,493],[676,495],[698,503],[713,520],[717,536],[720,544],[722,564],[716,566],[722,580],[711,588],[676,592],[643,591],[612,591],[601,584],[597,568],[596,543],[591,541],[593,518],[599,501]]},
      {'id':'bag-handle-left','instance':'bag-1','role':'stationary-front','observed':'visible left handle branch, open handle interior excluded',
       'polygon':[[627,467],[649,466],[652,472],[638,476],[637,491],[627,493],[624,484],[624,474]]},
      {'id':'bag-handle-right','instance':'bag-1','role':'stationary-front','observed':'visible right handle branch below fingers, open interior excluded',
       'polygon':[[670,478],[678,478],[684,483],[685,496],[675,495],[674,486]]},
      {'id':'torso-and-skirt','instance':'person-rear-1','role':'moving-rear','instances':['sweater-1','denim-skirt-1'],
       'observed':'visible torso and denim; complete support behind sleeve/hands/bag is explicitly an estimate, not an observed waist contour',
       'polygon':[[512,241],[535,221],[565,228],[597,257],[625,293],[642,349],[651,395],[658,422],[664,454],[668,482],[672,526],[678,564],[691,606],[714,645],[744,696],[778,751],[760,760],[731,771],[713,734],[691,687],[670,642],[669,691],[666,739],[657,797],[634,808],[600,813],[565,806],[539,808],[506,804],[477,800],[481,776],[483,722],[486,662],[490,599],[495,534],[503,480],[514,442],[532,411],[538,405],[520,400],[515,383],[510,352],[506,318],[505,283]]},
      {'id':'coat-left','instance':'coat-1','role':'stationary-scene','observed':'visible left coat fragment; inner boundary occludes or is occluded locally as described in graph',
       'polygon':[[502,279],[492,300],[479,328],[471,353],[463,381],[457,413],[457,447],[451,486],[447,521],[443,556],[438,591],[433,624],[432,645],[438,656],[440,683],[480,685],[481,657],[486,643],[487,603],[491,556],[495,518],[503,477],[516,440],[532,412],[520,406],[510,384],[506,352],[504,315]]},
      {'id':'coat-right','instance':'coat-1','role':'stationary-scene','observed':'visible right coat fragment; same physical coat as left fragment',
       'polygon':[[674,326],[681,344],[685,360],[697,379],[704,401],[710,419],[717,444],[721,467],[724,493],[727,520],[724,543],[718,552],[713,526],[698,505],[684,497],[682,480],[683,461],[679,446],[670,434],[665,418],[656,407],[650,395],[658,374],[667,349]]}
    ]
    arcs=[
      {'id':'left-visible-denim-side','status':'observed-material-boundary','points':[[533,413],[523,428],[515,446],[507,461]],'outside':'coat or dark occlusion, not skin'},
      {'id':'left-visible-denim-lower-side','status':'observed-material-boundary','points':[[507,461],[500,486],[496,514]],'outside':'coat'},
      {'id':'right-waist','status':'occluded-not-measurable','points':[[650,401],[658,425],[669,456],[675,487]],'note':'schematic hidden path only; not a target/measurement arc; sleeve, hand and coat obscure boundary'},
      {'id':'visible-waist-hem','status':'internal-garment-boundary-not-side','points':[[540,405],[562,406],[591,414],[622,422],[637,429]],'note':'torso/denim interface; do not report as lateral waist contour'}
    ]
    contract={'version':'manual-body-material-control/1','source':str(source),'sourceSHA256':sha(source),'originalGraphSHA256':sha(graph),
      'annotationTiming':'after-observing-automatic-failure-before-new-layer-preparation-and-replay','manualIntervention':True,
      'automaticGeneralizationClaim':False,'sourceOnly':True,'posteriorsAvailable':False,'modelPosteriorsRecovered':False,
      'geometryEdited':False,'geometricMaskIsOpticalAlpha':False,'hiddenContentIsEstimate':True,
      'roles':roles,'sourceMeasurementArcs':arcs,
      'localOcclusionOrder':[
       {'region':'image-right sleeve and hand T-junctions','order':'sleeve/hands > coat-1','evidence':'observed-source'},
       {'region':'sleeve over lower torso; bag over denim','order':'sleeve/bag > torso/denim','evidence':'observed-source'},
       {'region':'left denim/coat dark junction around x500..518 y440..485','order':'torso/denim > coat-1','evidence':'explicit-causal-control-hypothesis','limitation':'local occlusion order is not certain from this source'},
       {'region':'complete hidden rear and scene','order':'front > torso/denim-rear > coat-1 > wall','evidence':'layered-control-hypothesis','limitation':'not a globally observed depth ordering'}],
      'limitations':['Same sweater instance contains two motion regions; the split does not prove a naturally hinged motion model.',
       'Coat instance identity does not create a geometric hard-protection region.',
       'Hidden rear support, RGB and contours are estimates, to be assessed only through final RGB.',
       'Original region, anatomical axis, target part mask, protection and signed strengths remain unchanged.']}
    (out/'source-materials.json').write_text(json.dumps(contract,indent=2))
    front={'version':'explicit-source-contour/1','source':str(source),'sourceSHA256':sha(source),'role':'stationary-front',
      'polygons':[r['polygon'] for r in roles if r['role']=='stationary-front'],'uncertaintyPixels':4,'provenance':'manual-source-image-contour'}
    (out/'front-request.json').write_text(json.dumps(front,indent=2))
    # Store rear identity separately from optical alpha; no optical estimate exists yet.
    (out/'rear-identity.json').write_text(json.dumps({'sourceSHA256':sha(source),'role':'moving-rear','manual':True,'uncertaintyPixels':4,'polygons':[r['polygon'] for r in roles if r['role']=='moving-rear'],'hiddenBoundaryIsEstimate':True},indent=2))
    image=Image.open(source).convert('RGB')
    colors={'stationary-front':(255,30,190),'moving-rear':(30,230,85),'stationary-scene':(255,185,25)}
    overlay=image.copy(); d=ImageDraw.Draw(overlay)
    for r in roles:
      v=[tuple(p) for p in r['polygon']];d.line(v+[v[0]],fill=colors[r['role']],width=2)
    for a in arcs:
      pts=[tuple(p) for p in a['points']]
      d.line(pts,fill=(0,245,255) if a['status'].startswith('observed') else (220,220,220),width=3)
    overlay.save(out/'source-role-outlines.png')
    overlay.crop((440,210,750,600)).resize((930,1170)).save(out/'source-role-close.png')
    for role in colors:
      layer=Image.new('RGBA',image.size);draw=ImageDraw.Draw(layer)
      for r in roles:
        if r['role']==role:draw.polygon([tuple(p) for p in r['polygon']],fill=colors[role]+(65,),outline=colors[role]+(255,))
      Image.alpha_composite(image.convert('RGBA'),layer).save(out/f'{role}-identity.png')
    freeze={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out/'annotation-freeze.json').write_text(json.dumps(freeze,indent=2))
    print(out)
if __name__=='__main__':main()

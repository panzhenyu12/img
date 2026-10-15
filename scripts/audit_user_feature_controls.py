"""Audit a user's fixed fresh detection against every available feature control.

Template matching measures visible eye/nose/lip/brow landmarks independently
of the renderer's deformation formula. Ambiguous/flat matches are retained as
inconclusive. Colour metrics use the actual feature polygons, excluding the
inner mouth for lipstick. Nonzero pixels alone never mean effect acceptance.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from check_skin_detail_quality import ROOT, comparison, sha
from review_face_refinement import polygon_interior, coverage


GEOMETRY = {
    "eyeSize": ["leftEye", "rightEye"], "eyeWidth": ["leftEye", "rightEye"],
    "eyeHeight": ["leftEye", "rightEye"], "eyeSpacing": ["leftEye", "rightEye"],
    "eyeTilt": ["leftEye", "rightEye"], "eyeCorner": ["leftEye", "rightEye"],
    "eyelid": ["leftEye", "rightEye"], "eyeInnerCorner": ["leftEye", "rightEye"],
    "eyeOuterCorner": ["leftEye", "rightEye"],
    "browLength": ["leftBrow", "rightBrow"], "browWidth": ["leftBrow", "rightBrow"],
    "browThickness": ["leftBrow", "rightBrow"], "browPeak": ["leftBrow", "rightBrow"],
    "browTail": ["leftBrow", "rightBrow"], "browSpacing": ["leftBrow", "rightBrow"],
    "noseBridge": ["noseBridge"], "noseWings": ["noseWings"], "noseLength": ["nose"],
    "noseTip": ["noseTip"], "noseStraight": ["noseRoot", "noseTip"],
    "nostril": ["leftNostril", "rightNostril"], "noseRoot": ["noseRoot"], "noseBase": ["noseBase"],
    "mouthWidth": ["lips"], "mouthHeight": ["lips"], "upperLip": ["lips"],
    "lowerLip": ["lips"], "philtrum": ["noseBase", "lips"],
    "lipPeak": ["lips"], "lipCorner": ["lips"],
    "symmetry": ["leftEye", "rightEye", "leftBrow", "rightBrow", "lips"],
}
COLOR = {"browColor": ["leftBrow", "rightBrow"], "lipColor": ["lips"],
         "lipBrightness": ["lips"], "lipSaturation": ["lips"], "teethWhiten": ["teeth"],
         "darkCircle": ["leftUnderEye", "rightUnderEye"]}
POSITIVE_ONLY = {"symmetry", "teethWhiten", "darkCircle"}
WIDTH_CONTROLS = {"eyeSize", "eyeWidth", "browLength", "browWidth", "noseBridge",
                  "noseWings", "mouthWidth", "noseRoot", "nostril"}
HEIGHT_CONTROLS = {"eyeSize", "eyeHeight", "browThickness", "noseLength", "mouthHeight"}


def feature_points(mask, size):
    points = np.array([[p["x"]*size[0], p["y"]*size[1]] for p in mask["points"]])
    # Independent detected extrema, not the renderer's local warp centers.
    choices = [np.argmin(points[:, 0]), np.argmax(points[:, 0]),
               np.argmin(points[:, 1]), np.argmax(points[:, 1])]
    return {name: points[index] for name, index in zip(["left", "right", "top", "bottom"], choices)}


def match_point(original, edited, point, patch_radius=7, search=14):
    x, y = np.rint(point).astype(int)
    r = patch_radius
    template = original[y-r:y+r+1, x-r:x+r+1]
    haystack = edited[y-r-search:y+r+search+1, x-r-search:x+r+search+1]
    if template.shape != (2*r+1, 2*r+1) or haystack.shape != (2*(r+search)+1, 2*(r+search)+1):
        return {"valid": False, "reason": "patch outside image"}
    windows = np.lib.stride_tricks.sliding_window_view(haystack, template.shape)
    centered = template-template.mean()
    std = float(template.std())
    if std < 2.0:
        return {"valid": False, "reason": "low source contrast", "source_std": std}
    local = windows-windows.mean(axis=(-1, -2), keepdims=True)
    denominator = np.sqrt((local*local).sum(axis=(-1, -2)) * (centered*centered).sum())
    scores = (local*centered).sum(axis=(-1, -2))/np.maximum(denominator, 1e-9)
    iy, ix = np.unravel_index(scores.argmax(), scores.shape)
    score = float(scores[iy, ix])
    if score < .88 or ix in (0, search*2) or iy in (0, search*2):
        return {"valid": False, "reason": "unreliable NCC", "score": score,
                "integer_displacement": [int(ix-search), int(iy-search)], "source_std": std}
    subpixel = []
    for low, mid, high in [(scores[iy, ix-1], scores[iy, ix], scores[iy, ix+1]),
                           (scores[iy-1, ix], scores[iy, ix], scores[iy+1, ix])]:
        denominator = low-2*mid+high
        subpixel.append(float(np.clip(.5*(low-high)/denominator, -.5, .5)) if abs(denominator)>1e-9 else 0.0)
    delta = np.array([ix-search+subpixel[0], iy-search+subpixel[1]])
    return {"valid": True, "score": score, "source_std": std,
            "source": point.tolist(), "output": (point+delta).tolist(), "displacement": delta.tolist()}


def geometry_metrics(original, edited, masks, names, size):
    result = {}
    for name in names:
        anchors = feature_points(masks[name], size)
        tracks = {label: match_point(original, edited, point) for label, point in anchors.items()}
        row = {"tracks": tracks}
        for axis, labels in [("width", ["left", "right"]), ("height", ["top", "bottom"])]:
            a, b = [tracks[label] for label in labels]
            if a["valid"] and b["valid"]:
                dimension = 0 if axis == "width" else 1
                before = abs(anchors[labels[1]][dimension]-anchors[labels[0]][dimension])
                after = abs(b["output"][dimension]-a["output"][dimension])
                row[axis] = {"source_px": float(before), "output_px": float(after),
                             "delta_px": float(after-before), "change_percent": float((after-before)/max(before, 1)*100)}
        valid = [v for v in tracks.values() if v["valid"]]
        row["valid_tracks"] = len(valid)
        row["maximum_tracked_motion_px"] = max((float(np.linalg.norm(v["displacement"])) for v in valid), default=0)
        if valid:
            row["mean_displacement"] = np.mean([v["displacement"] for v in valid], axis=0).tolist()
        result[name] = row
    return result


def color_metrics(source, edited, selected):
    a, b = source[:, :, :3].astype(float)[selected], edited[:, :, :3].astype(float)[selected]
    if not len(a):
        return {"pixels": 0}
    coefficients = np.array([.2126, .7152, .0722])
    saturation = lambda rgb: (rgb.max(axis=1)-rgb.min(axis=1))/np.maximum(rgb.max(axis=1), 1)
    return {"pixels": int(len(a)), "source_mean_rgb": a.mean(axis=0).tolist(), "output_mean_rgb": b.mean(axis=0).tolist(),
            "mean_luma_delta": float(((b-a)@coefficients).mean()),
            "mean_red_green_delta": float(((b[:, 0]-b[:, 1])-(a[:, 0]-a[:, 1])).mean()),
            "mean_saturation_delta": float((saturation(b)-saturation(a)).mean()),
            "mean_absolute_rgb_delta": float(np.abs(b-a).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=ROOT/'test/results/user-face-contour-20261014/1024')
    ap.add_argument("--binary", type=Path, default=ROOT/'test/results/user-face-contour-20261014/baseline-image-render')
    ap.add_argument("--label", default='user-feature-audit-20261014')
    args = ap.parse_args()
    source_path, base_graph_path = args.base/'canonical-source.png', args.base/'face-graph.json'
    source = Image.open(source_path).convert('RGBA'); original = np.asarray(source)
    luma = original[:, :, :3]@np.array([.2126, .7152, .0722])
    base_graph = json.loads(base_graph_path.read_text())
    params = {k: v for k,v in base_graph['operations'][0]['params'].items()
              if k in {'region', 'mask', 'protectMask', 'semanticMasks', 'landmarks', 'opacity', 'blendMode'}}
    masks = params['semanticMasks']
    polygons = {name: polygon_interior(mask, source.size) for name, mask in masks.items() if mask.get('shape')=='polygon'}
    frozen = np.zeros(original.shape[:2], dtype=bool)
    for name in ['hair', 'ears', 'accessories']:
        frozen |= coverage(masks[name], source.size)>=.5
    yy, xx = np.indices(original.shape[:2]); region = params['region']
    inside = ((xx+.5)/source.width>=region['x']) & ((xx+.5)/source.width<=region['x']+region['width']) & ((yy+.5)/source.height>=region['y']) & ((yy+.5)/source.height<=region['y']+region['height'])
    binary = args.binary.resolve(); binary_hash=sha(binary); source_hash=sha(source_path)
    out = ROOT/'test/results'/args.label; doc = ROOT/'doc/qa'/args.label
    out.mkdir(parents=True, exist_ok=True); doc.mkdir(parents=True, exist_ok=True)
    rows=[]
    for key,names in {**GEOMETRY, **COLOR}.items():
        for amount in ([100] if key in POSITIVE_ONLY else [-100, 100]):
            graph = {**base_graph, 'operations': [{'id': 'feature-audit', 'type': 'feature.adjust', 'enabled': True,
                    'params': {**params, 'adjustments': {key: amount}}}]}
            graph_path=out/f'{key}-{amount}.json'; graph_path.write_text(json.dumps(graph))
            output_path=out/f'{key}-{amount}.png'
            assert sha(binary)==binary_hash and sha(source_path)==source_hash
            start=time.perf_counter()
            result=subprocess.run([str(binary),'--input',str(source_path),'--graph',str(graph_path),'--output',str(output_path)],capture_output=True,text=True)
            row={'key':key,'amount':amount,'graph_sha256':sha(graph_path),'native_total_ms':(time.perf_counter()-start)*1000}
            assert sha(binary)==binary_hash and sha(source_path)==source_hash
            if result.returncode:
                row['error']=result.stderr; rows.append(row); continue
            edited_image=Image.open(output_path).convert('RGBA'); edited=np.asarray(edited_image)
            changed=np.any(edited!=original, axis=2)
            row.update({'output_sha256':sha(output_path),'changed_pixels':int(changed.sum()),
                'outside_region_changes':int(changed[~inside].sum()),'frozen_changes':int(changed[frozen].sum()),
                'alpha_changes':int((edited[:,:,3]!=original[:,:,3]).sum())})
            target = np.logical_or.reduce([polygons[name] for name in names])
            if key in {'lipColor','lipBrightness','lipSaturation'}:
                target &= ~polygons['teeth']
            row['target_color']=color_metrics(original, edited, target)
            if key in GEOMETRY:
                output_luma=edited[:,:,:3]@np.array([.2126,.7152,.0722])
                row['geometry']=geometry_metrics(luma,output_luma,masks,names,source.size)
                dimensions=(["width"] if key in WIDTH_CONTROLS else [])+(["height"] if key in HEIGHT_CONTROLS else [])
                signs=[]
                for dimension in dimensions:
                    deltas=[item[dimension]['delta_px'] for item in row['geometry'].values() if dimension in item]
                    signs.append({'dimension':dimension,'median_delta_px':float(np.median(deltas)) if deltas else None,
                        'direction_response':bool(deltas and np.median(deltas)*amount>0),
                        'material_response_over_1px':bool(deltas and abs(np.median(deltas))>1.0)})
                row['dimension_checks']=signs
            points=[p for name in names for p in masks[name]['points']]
            crop=(max(0,int(min(p['x'] for p in points)*source.width)-22),max(0,int(min(p['y'] for p in points)*source.height)-22),
                  min(source.width,int(max(p['x'] for p in points)*source.width)+23),min(source.height,int(max(p['y'] for p in points)*source.height)+23))
            comparison([(source,'Original'),(edited_image,f'{key} {amount}')],crop,doc/f'{key}-{amount}-comparison.png')
            row['comparison']=str((doc/f'{key}-{amount}-comparison.png').relative_to(ROOT))
            rows.append(row)
            print(json.dumps({k:row[k] for k in ['key','amount','changed_pixels','target_color']},ensure_ascii=False),flush=True)
    report={'input_sha256':source_hash,'original_photo':str(ROOT/'test/images/user-face-contour-20261014.jpg'),
        'base_graph_sha256':sha(base_graph_path),'binary_sha256':binary_hash,'independent_originals':1,
        'scope':'Fresh source detection is fixed; no output re-detection. NCC is diagnostic, flat/ambiguous matches are inconclusive. One user photo cannot establish 8/10.',
        'source_hashes':{p:sha(ROOT/p) for p in ['apps/web/src/core/registry.ts','apps/web/src/components/ToolPanel.tsx','apps/web/src/App.tsx','core/image-core/src/geometry.rs','core/image-core/src/color.rs']},
        'rows':rows}
    (doc/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(rows),'errors':[r['key'] for r in rows if r.get('error')],
        'geometric_dimensions_without_material_response':[{'key':r['key'],'amount':r['amount'],'checks':r['dimension_checks']} for r in rows if r.get('dimension_checks') and any(not c['direction_response'] or not c['material_response_over_1px'] for c in r['dimension_checks'])]},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

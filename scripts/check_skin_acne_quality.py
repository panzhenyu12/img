"""Real acne photo challenge; fixed manual skin selections, frozen native cores.

Lesion annotations are fixed before candidate changes. Chroma/contrast metrics
describe marked defects; they do not award aesthetic quality automatically.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps

from check_skin_detail_quality import ROOT, comparison, sha

SAMPLES = {
    "acne-teen": {
        "region": {"x":.06,"y":.43,"width":.36,"height":.28},
        "points": [[.095,.43],[.35,.445],[.42,.54],[.34,.625],[.26,.68],[.12,.70],[.065,.59]],
        "lesions": [{"id":"cheek-red-1","x":206,"y":540,"radius":5,"ring":[14,22]},
                    {"id":"cheek-red-2","x":232,"y":578,"radius":5,"ring":[14,22]}],
        "detail_patch": [252,478,284,512],
        "crop": [80,460,305,650],
    },
    "acne-oily": {
        "region": {"x":.20,"y":.09,"width":.67,"height":.48},
        "points": [[.32,.09],[.75,.09],[.87,.21],[.85,.47],[.65,.56],[.33,.52],[.20,.34]],
        "lesions": [{"id":"forehead-red","x":790,"y":180,"radius":7,"ring":[20,30]},
                    {"id":"forehead-whitehead","x":370,"y":112,"radius":4,"ring":[12,19]}],
        "detail_patch": [490,230,545,285],
        "crop": [300,65,870,275],
    },
    "acne-cystic": {
        "region": {"x":.26,"y":.50,"width":.44,"height":.27},
        "points": [[.32,.50],[.67,.52],[.70,.65],[.60,.76],[.35,.74],[.26,.62]],
        "lesions": [{"id":"cheek-red","x":278,"y":611,"radius":6,"ring":[15,24]},
                    {"id":"outside-selection-control","x":312,"y":810,"radius":6,"ring":[15,24],
                     "purpose":"negative control outside the manual selection; must stay unchanged"}],
        "detail_patch": [235,685,275,725],
        "crop": [140,550,355,790],
    },
}
CASES = {"natural":{"smooth":60,"texture":60},
         "strong":{"smooth":100,"texture":0},
         "texture":{"smooth":80,"texture":100}}


def lesion_metrics(pixels, lesion):
    yy,xx = np.indices(pixels.shape[:2])
    distance = np.hypot(xx-lesion["x"], yy-lesion["y"])
    center = distance <= lesion["radius"]
    ring = (distance>=lesion["ring"][0]) & (distance<=lesion["ring"][1])
    rgb = pixels[:,:,:3].astype(float)
    light = rgb@np.array([.2126,.7152,.0722])
    # Relative chroma is less sensitive to the source's strong lateral light.
    red = (rgb[:,:,0]-rgb[:,:,1]) / (light+20) * 100
    return {"red_excess":float(np.median(red[center])-np.median(red[ring])),
            "light_contrast":float(np.median(light[ring])-np.median(light[center]))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--candidate",type=Path,default=ROOT/".run/skin-detail-v2-image-render")
    parser.add_argument("--label",default="v2")
    parser.add_argument("--reference",type=Path,help="Optional immutable previous candidate for same-parameter comparison")
    parser.add_argument("--reference-label",default="v2")
    parser.add_argument("--samples",nargs="+",default=list(SAMPLES))
    args=parser.parse_args()
    out=ROOT/f"test/results/skin-acne/{args.label}";out.mkdir(exist_ok=True,parents=True)
    builds=[("ea5b404",ROOT/".run/quality-baseline/image-render")]
    if args.reference:
        builds.append((args.reference_label,args.reference))
    builds.append((args.label,args.candidate))
    assert len({name for name,_ in builds})==len(builds), "build labels must be unique"
    hashes={name:sha(binary) for name,binary in builds}
    manifest={row["id"]:row for row in json.loads((ROOT/"test/images/manifest.json").read_text())}
    rows=[]
    for sid in args.samples:
        config=SAMPLES[sid];source=ROOT/f"test/results/skin-acne/{sid}-source.png"
        entry=manifest[sid];original_path=ROOT/"test/images"/entry["path"]
        assert sha(original_path)==entry["sha256"],f"original changed: {original_path}"
        # Recreate the exact test input from the versioned original, so a clean
        # checkout does not depend on ignored QA caches.
        # Normalize EXIF orientation before resizing. The fixed annotations are
        # in display coordinates, not the raw JPEG storage orientation.
        prepared=ImageOps.exif_transpose(Image.open(original_path)).convert("RGBA")
        longest=max(prepared.size);ratio=1024/longest
        prepared=prepared.resize(tuple(round(v*ratio) for v in prepared.size),Image.Resampling.LANCZOS)
        source.parent.mkdir(parents=True,exist_ok=True);prepared.save(source)
        image=Image.open(source).convert("RGBA");original=np.asarray(image)
        region=config["region"]
        xx,yy=np.meshgrid((np.arange(image.width)+.5)/image.width,(np.arange(image.height)+.5)/image.height)
        outside=(xx<region["x"])|(xx>region["x"]+region["width"])|(yy<region["y"])|(yy>region["y"]+region["height"])
        mask={"shape":"polygon","points":[{"x":x,"y":y}for x,y in config["points"]],"feather":3}
        for case,values in CASES.items():
            graph={"version":"1.0","sourceAssetId":sid,"operations":[{"id":"real-acne","type":"skin.retouch","enabled":True,"params":{"region":region,"semanticMasks":{"skin":mask},**values}}]}
            graph_path=out/f"{sid}-{case}.json";graph_path.write_text(json.dumps(graph))
            panels=[(image,"Original")]
            row={"sample":sid,"case":case,"parameters":values,"selection":"fixed manual polygon; no detector invoked",
                 "original_path":str(original_path.relative_to(ROOT)),"original_sha256":sha(original_path),
                 "source_sha256":sha(source),"graph_sha256":sha(graph_path),"lesions":{l["id"]:{"annotation":l,"original":lesion_metrics(original,l)}for l in config["lesions"]}}
            for name,binary in builds:
                assert sha(binary)==hashes[name]
                output=out/f"{sid}-{case}-{name}.png";begin=time.perf_counter()
                subprocess.run([str(binary),"--input",str(source),"--graph",str(graph_path),"--output",str(output)],check=True,capture_output=True)
                row[name+"_native_total_ms"]=(time.perf_counter()-begin)*1000
                assert sha(binary)==hashes[name]
                edited_image=Image.open(output).convert("RGBA");edited=np.asarray(edited_image);changed=np.any(original!=edited,axis=2)
                row[name+"_outside_changes"]=int(changed[outside].sum());row[name+"_alpha_changes"]=int((original[:,:,3]!=edited[:,:,3]).sum())
                for lesion in config["lesions"]:row["lesions"][lesion["id"]][name]=lesion_metrics(edited,lesion)
                panels.append((edited_image,name))
            comparison(panels,config["crop"],out/f"{sid}-{case}-100pct.png")
            row["evidence"]=str((out/f"{sid}-{case}-100pct.png").relative_to(ROOT));rows.append(row)
    assert all(sha(binary)==hashes[name] for name,binary in builds)
    report={"binary_sha256":hashes,"selection_protocol":SAMPLES,"rows":rows,"note":"No automatic aesthetic score; annotations include known failures."}
    (ROOT/f"doc/qa/skin-acne-{args.label}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(rows,indent=2))
    assert all(r[name+"_outside_changes"]==0 and r[name+"_alpha_changes"]==0 for r in rows for name,_ in builds)


if __name__=="__main__":
    main()

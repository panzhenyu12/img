"""Known pore/blemish image; identical parameters for old/new native cores."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from check_skin_detail_quality import ROOT, comparison, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--v1-binary", type=Path)
    args = parser.parse_args()
    builds = [("baseline", ROOT / ".run/quality-baseline/image-render")]
    if args.v1_binary:
        builds.append(("v1", args.v1_binary))
    builds.append(("current", args.binary))
    build_hashes = {str(binary): sha(binary) for _, binary in builds}
    out = ROOT / "test/results/skin-detail"
    out.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((128, 192), dtype=float)
    base = 156 + xx * .08
    pores = np.zeros_like(base)
    for cy in [32, 56, 80, 104]:
        for cx in [32, 56, 80]:
            pores += 10 * np.exp(-((xx-cx)**2 + (yy-cy)**2) / 2)
    blemish = 44 * np.exp(-((xx-136)**2 + (yy-64)**2) / 14)
    image = Image.fromarray(np.clip(np.floor(np.stack([
        base-pores-blemish*.55, base-17-pores-blemish,
        base-29-pores-blemish, np.full_like(base, 173)], axis=2) + .5), 0, 255).astype(np.uint8))
    source = out / "synthetic-pore-blemish-source.png"
    image.save(source)
    def light(img, x, y):
        p = np.asarray(img, dtype=float)[y, x]
        return float(.2126*p[0] + .7152*p[1] + .0722*p[2])
    rows = []
    for texture in [0, 100]:
        graph = {"version":"1.0", "sourceAssetId":"synthetic", "operations":[{
            "id":"skin-detail-regression", "type":"skin.retouch", "enabled":True,
            "params":{"smooth":100, "texture":texture,
                      "region":{"x":0,"y":0,"width":1,"height":1},
                      "mask":{"shape":"rect","region":{"x":0,"y":0,"width":1,"height":1}}}}]}
        graph_path = out / f"synthetic-texture{texture}.json"
        graph_path.write_text(json.dumps(graph))
        panels = [(image, "Original")]
        row = {"smooth":100, "texture":texture, "source_sha256":sha(source),
               "original_pore_contrast":light(image,32,27)-light(image,32,32),
               "original_blemish_contrast":light(image,136,51)-light(image,136,64)}
        for label, binary in [("baseline", ROOT / ".run/quality-baseline/image-render"), ("current", args.binary)]:
            assert sha(binary) == build_hashes[str(binary)]
            output = out / f"synthetic-texture{texture}-{label}.png"
            subprocess.run([str(binary), "--input",str(source), "--graph",str(graph_path),
                            "--output",str(output)], check=True, capture_output=True)
            assert sha(binary) == build_hashes[str(binary)]
            edited = Image.open(output).convert("RGBA")
            row[label+"_pore_contrast"] = light(edited,32,27)-light(edited,32,32)
            row[label+"_blemish_contrast"] = light(edited,136,51)-light(edited,136,64)
            row[label+"_blemish_brightened"] = light(edited,136,64)-light(image,136,64)
            panels.append((edited, "ea5b404" if label=="baseline" else "Multiscale"))
        comparison(panels, (0,0,192,128), out / f"synthetic-texture{texture}-comparison.png")
        rows.append(row)
    (ROOT / "doc/qa/skin-detail-regression.json").write_text(json.dumps(rows,indent=2)+"\n")
    red = np.exp(-((xx-96)**2 + (yy-64)**2) / 14)
    red_image = Image.fromarray(np.floor(np.stack([180+6*red,150-9*red,130-5*red,
                               np.full_like(red,255)],axis=2)+.5).astype(np.uint8))
    red_source = out / "synthetic-red-patch-source.png"
    red_image.save(red_source)
    panels = [(red_image, "Original")]
    red_row = {"smooth":100,"texture":100,"original_red_excess":15}
    for label, binary in builds:
        assert sha(binary) == build_hashes[str(binary)]
        output = out / f"synthetic-red-patch-{label}.png"
        subprocess.run([str(binary),"--input",str(red_source),"--graph",str(graph_path),
                        "--output",str(output)],check=True,capture_output=True)
        assert sha(binary) == build_hashes[str(binary)]
        edited = Image.open(output).convert("RGBA")
        pixel = np.asarray(edited,dtype=int)[64,96]
        red_row[label+"_red_excess"] = int(pixel[0]-pixel[1]-30)
        panels.append((edited,label))
    comparison(panels,(72,40,120,88),out / "synthetic-red-patch-100pct.png")
    (ROOT / "doc/qa/skin-detail-red-patch-regression.json").write_text(json.dumps(red_row,indent=2)+"\n")
    assert red_row["current_red_excess"] <= 9
    assert all(sha(Path(path)) == expected for path,expected in build_hashes.items())
    print(json.dumps(rows,indent=2))
    print(json.dumps(red_row,indent=2))
    assert rows[0]["current_pore_contrast"] >= rows[0]["original_pore_contrast"]*.55
    assert rows[0]["current_blemish_contrast"] <= rows[0]["original_blemish_contrast"]*.60
    assert rows[1]["current_blemish_brightened"] >= 15


if __name__ == "__main__":
    main()

"""Audit original-source conventions and target differences; never solve/warp pixels.

Uses only frozen local photo evidence and the explicit Rust JSON interface.
Run with services/background-worker/.venv/bin/python from the repository root.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEGACY = ROOT / "experiments/face-user-source-roles-20261021"
sys.path.insert(0, str(LEGACY))
from whole_face_width import derive_whole_face_width
from intent import derive
from prepare_bounded_face import raw_controls, validate_trace_source
from refine_face_width_intent import smoothstep

APPLICABILITY = ROOT / "test/results/face-source-relative-width-20261020/whole-face-width-source-applicability-v2"
USER_BASE = ROOT / "test/results/face-source-relative-width-20261020/user-+100-anatomical-upper-A1-step1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def checked(path, digest):
    assert sha(path) == digest, f"frozen source changed: {path}"


def load_source(name, manifest):
    if name == "user":
        frozen_path = Path(manifest["user_frozen_intent"])
        checked(frozen_path, manifest["user_frozen_intent_sha256"])
        frozen = read(frozen_path)
        provenance_path = USER_BASE / "provenance.json"
        provenance = read(provenance_path)
        frame_path = Path(provenance["source_relative_width"]["report"])
        frame = read(frame_path)["frame"]
        source = np.asarray(frozen["source"], float)
        anatomy = provenance["anatomical_upper_transition"]
        oval, eyes = anatomy["source_oval"], anatomy["source_eyes"]
        np.testing.assert_array_equal(source, anatomy["original_128_source"])
        frozen_outputs = {100.: np.asarray(frozen["requested"], float)}
        source_record = frozen_path
    else:
        source_record = APPLICABILITY / f"{name}-source-only.json"
        record = read(source_record)
        for path, digest in record["inputs_sha256"].items():
            checked(path, digest)
        provenance_path = Path(record["base"]) / "provenance.json"
        provenance = read(provenance_path)
        frame_path = Path(provenance["source_relative_width"]["report"])
        frame = record["frame"]
        assert frame == read(frame_path)["frame"]
        source = np.asarray(record["source_128"], float)
        oval, eyes = record["source_oval"], record["source_eyes"]
        frozen_outputs = {float(c["amount"]): np.asarray(c["new_requested_128"], float)
                          for c in record["cases"]}
    source_path = Path(provenance["feature_transport"]["source"])
    checked(source_path, provenance["feature_transport"]["source_sha256"])
    image = Image.open(source_path).convert("RGBA")
    trace = Path(provenance["source_trace"])
    source_proof = validate_trace_source(trace, np.asarray(image))
    native = provenance["legacy_control_trace"]
    checked(Path(native["path"]) / native["file"], native["sha256"])
    native_source, _ = raw_controls(Path(native["path"]))
    np.testing.assert_array_equal(source, native_source)
    native_proof = validate_trace_source(Path(native["path"]), np.asarray(image))
    flow = read(trace / "flow.json")
    for key in ("across", "down", "bounds"):
        assert frame[key] == flow[key]
    np.testing.assert_array_equal(frame["origin"], np.asarray(flow["origin"]) - .5)
    for amount, frozen in frozen_outputs.items():
        old = derive_whole_face_width(source, frame, oval, eyes, amount)
        np.testing.assert_array_equal(old["requested"], frozen)
    return dict(name=name, frame=frame, source=source, oval=oval, eyes=eyes,
                frozen_outputs=frozen_outputs, image=image,
                source_path=source_path, source_record=source_record,
                frame_path=frame_path, provenance_path=provenance_path,
                source_proof=source_proof, native_proof=native_proof)


def invoke(binary, request):
    process = subprocess.run([str(binary)], input=json.dumps(request), text=True,
                             capture_output=True, check=False)
    if process.returncode:
        raise RuntimeError(process.stderr)
    return json.loads(process.stdout)


def request_for(case, points, amount, k):
    return {"frame": {**case["frame"], "midline_u": 0.},
            "calibration": {"max_radius_fraction": float(k), "chin_fade_start": .88},
            "amount": amount,
            "contour": [{"id": f"source-{i}", "source": p.tolist()}
                        for i, p in enumerate(points)],
            "parts": [], "relations": []}


def chin_probe(case, binary, k):
    frame = case["frame"]
    basis = np.asarray([frame["across"], frame["down"]], float)
    left, top, right, bottom = frame["bounds"]
    # Invert the actual stored basis without renormalizing or changing frame.
    heights = [.88-1e-6, .88, .88+1e-6, 1.-1e-6, 1., 1.+1e-6]
    radius = .5 * (right-left)
    local = np.array([[radius, top+h*(bottom-top)] for h in heights])
    points = np.linalg.solve(basis, local.T).T + frame["origin"]
    output = invoke(binary, request_for(case, points, 100., k))
    target = np.array([p["target"] for p in output["targets"]])
    displacements = target-points
    return {"probe_local_radius": radius, "normalized_heights": heights,
            "displacements": displacements.tolist(),
            "span_across_fade_start_px": float(np.linalg.norm(displacements[2]-displacements[0])),
            "span_across_fade_end_px": float(np.linalg.norm(displacements[5]-displacements[3])),
            "end_and_below_identity_exact": bool(np.array_equal(target[4:], points[4:])),
            "scope": "Source target envelope has C1 endpoints analytically; probes confirm no finite jump at h=.88/1. No rasterized connection/neck guarantee."}


def overlay(case, cases, out):
    source = case["source"]
    image = case["image"].convert("RGB")
    lo = np.maximum(np.floor(source.min(0)-30).astype(int), 0)
    hi = np.minimum(np.ceil(source.max(0)+30).astype(int), image.size)
    box = tuple(np.r_[lo, hi]); w, h = hi-lo
    panel = Image.new("RGB", (int(w*3), int(h+56)), "white")
    pd = ImageDraw.Draw(panel)
    for col, amount in enumerate([-60., 0., 60.]):
        entry = next(c for c in cases if c["amount"] == amount)
        tile = image.crop(box); draw = ImageDraw.Draw(tile)
        for points, color in [(source, "yellow"),
                              (np.asarray(entry["legacy_target"]), "cyan"),
                              (np.asarray(entry["rust_target"]), "magenta")]:
            draw.line([tuple(p-lo) for p in np.vstack([points, points[0]])], fill=color, width=1)
        for index in entry["changed_indices_gt_1e_9_px"]:
            point = np.asarray(entry["rust_target"])[index]-lo
            draw.ellipse([point[0]-2,point[1]-2,point[0]+2,point[1]+2], outline="red")
        panel.paste(tile, (int(col*w), 56))
        pd.text((int(col*w+4), 4), f"{case['name']} {amount:+.0f} TARGETS ONLY", fill="black")
        pd.text((int(col*w+4), 20), "yellow=source cyan=old magenta=Rust", fill="black")
        pd.text((int(col*w+4), 36), "red ring=changed >1e-9px; image unwarped", fill="black")
    panel.save(out/f"{case['name']}-target-overlay.png")


def audit(case, binary, out):
    frame, source = case["frame"], case["source"]
    origin = np.asarray(frame["origin"])
    basis = np.stack([frame["across"], frame["down"]], 1)
    local = (source-origin)@basis
    left, top, right, bottom = frame["bounds"]
    radius, height = local[:,0], (local[:,1]-top)/(bottom-top)
    lateral = smoothstep(.08, .28, abs((radius-(left+right)*.5)/((right-left)*.5)))
    full = derive(source, frame, 100.)
    k = float(full["k"])
    np.testing.assert_array_equal(full["signed_radius"], radius)
    cases=[]; signed={}
    for amount in [-60., 0., 60., 100.]:
        old = derive_whole_face_width(source, frame, case["oval"], case["eyes"], amount)
        request = request_for(case, source, amount, k)
        output = invoke(binary, request)
        target = np.array([p["target"] for p in output["targets"]]); signed[amount]=target
        delta = target-old["requested"]
        norms = np.linalg.norm(delta, axis=1)
        envelope = 1-smoothstep(.88, 1., height)
        predicted = -(k*amount/100*radius*envelope*(1-lateral))[:,None]*basis[:,0]
        prediction_residual = float(np.max(np.linalg.norm(delta-predicted, axis=1)))
        assert prediction_residual < 1e-10
        upper = (height<.34)&(abs(radius)>1e-6)
        moved = np.linalg.norm(target-source, axis=1)
        if amount:
            assert np.all(moved[upper]>0), "nonzero-radius upper contour was unintentionally frozen"
        else:
            np.testing.assert_array_equal(target, source)
        record={"amount": amount, "legacy_reference":
            "frozen_array_exactly_reproduced" if amount in case["frozen_outputs"] else "same_frozen_module_recomputed_from_original_source",
            "k_at_full_amount":k,
            "same_request":False,
            "max_difference_px":float(norms.max()),
            "changed_indices_exact_float":np.flatnonzero(np.any(delta!=0,axis=1)).tolist(),
            "changed_indices_gt_1e_9_px":np.flatnonzero(norms>1e-9).tolist(),
            "removing_lateral_gate_prediction_residual_max_px":prediction_residual,
            "upper_nonzero_radius_count":int(np.count_nonzero(upper)),
            "upper_unexpected_zero_motion_indices":np.flatnonzero(upper&(moved==0)).tolist() if amount else [],
            "max_displacement_px":float(moved.max()),
            "legacy_target":old["requested"].tolist(),"rust_target":target.tolist(),
            "rows":[{"index":i,"source":source[i].tolist(),"local_radius":float(radius[i]),
                     "local_height":float(height[i]),"legacy_lateral":float(lateral[i]),
                     "delta_from_legacy":delta[i].tolist(),"difference_px":float(norms[i])}
                    for i in range(len(source))]}
        save(out/f"{case['name']}-{amount:+.0f}-request.json",request)
        save(out/f"{case['name']}-{amount:+.0f}-output.json",output)
        cases.append(record)
    reversal=float(abs(signed[-60.]+signed[60.]-2*source).max())
    assert reversal<1e-12
    result={"name":case["name"],"source_image":str(case["source_path"]),
            "source_record":str(case["source_record"]),
            "source_note":"user image has an AI-creation watermark; not claimed to be an ungenerated camera photograph" if case["name"]=="user" else "existing independent portrait source",
            "input_sha256":{str(p):sha(p) for p in [case["source_path"],case["source_record"],case["frame_path"],case["provenance_path"]]},
            "source128_matches_own_native_trace_exact":True,
            "native_and_photo_coordinate_conversion":case["source_proof"],
            "native_intent_source_proof":case["native_proof"],
            "frame_raw":frame,"frame_renormalized":False,"midline_u":0.,
            "midline_reason":"Original derive uses signed_radius=dot(source-origin,across); bounds midpoint only belongs to the removed lateral envelope, not the radius origin.",
            "bounds_midpoint_u_not_used_as_midline":float((left+right)*.5),
            "frame_norm_squared_error":float(np.dot(basis[:,0],basis[:,0])-1),
            "calibration_k_full":k,"calibration_reference":full["reference"],
            "source_contour":source.tolist(),"source_oval":case["oval"],"source_eyes":case["eyes"],
            "zero_identity_exact":True,"signed_reversal_max_px":reversal,
            "chin_envelope_probe":chin_probe(case,binary,k),"cases":cases,
            "status":"target_only_new_controller_semantics_no_solver_no_image_quality_rating"}
    save(out/f"{case['name']}-audit.json",result)
    overlay(case,cases,out)
    return {"name":case["name"],"calibration_k_full":k,"source_image":str(case["source_path"]),
            "frame_norm_squared_error":result["frame_norm_squared_error"],
            "bounds_midpoint_u_not_used_as_midline":result["bounds_midpoint_u_not_used_as_midline"],
            "signed_reversal_max_px":reversal,"chin_envelope_probe":result["chin_envelope_probe"],
            "cases":[{key:c[key] for key in ["amount","legacy_reference","max_difference_px","changed_indices_gt_1e_9_px","upper_nonzero_radius_count","upper_unexpected_zero_motion_indices"]} for c in cases]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary",type=Path,default=HERE/"target/debug/face-width-targets")
    ap.add_argument("--output",type=Path,default=ROOT/"test/results/semantic-deformation-20261022/controller-source-audit")
    args=ap.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    manifest=read(APPLICABILITY/"report.json")
    for path,digest in manifest["code_sha256"].items(): checked(path,digest)
    results=[audit(load_source(name,manifest),args.binary.resolve(),out) for name in ["user","woman","hair"]]
    report={"no_inference_no_solve_no_warp":True,"beauty_accepted":False,
            "different_targets_are_not_same_request":True,
            "code_sha256":{str(p):sha(p) for p in [Path(__file__),HERE/"src/lib.rs",args.binary.resolve(),LEGACY/"whole_face_width.py"]},
            "applicability_manifest_sha256":sha(APPLICABILITY/"report.json"),
            "cases":results}
    save(out/"report.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()

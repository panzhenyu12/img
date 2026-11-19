"""One frozen, explicitly manual multi-point SAM2 capability diagnostic."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = ROOT / "experiments/sam2matting-tiny-probe-20261015"
WEIGHTS = ROOT / ".models/sam2matting-20261015/SAM2Matting-SAM2.1Tiny.pt"
CONFIG = "configs/sam2matting-sam2.1tiny.yaml"
SOURCE = ROOT / "test/results/body-material-candidates-20261119/side-fashion-source/source.png"
EVALUATION = Path("/tmp/body-independent-20261119/side-fashion/frozen-source-evaluation.json")
POINTS = {"sleeve": [[588, 315], [637, 370]], "torso": [[535, 335], [559, 393]]}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    evaluation = json.loads(EVALUATION.read_text())
    for points in POINTS.values():
        for x, y in points:
            assert not any(x0 <= x < x1 and y0 <= y < y1 for x0,y0,x1,y1 in evaluation["rectangles_xyxy"].values())
    files = {"source": SOURCE, "checkpoint": WEIGHTS, "script": Path(__file__),
             "predictor": RUNTIME / "vendor/sam2/sam2_image_predictor.py",
             "config": RUNTIME / "vendor/sam2" / CONFIG}
    protocol = {"status": "frozen_before_inference", "manualControl": True,
                "claim": "Capability of interactive manual correction, never automatic segmentation success",
                "sourceObservation": "Pink front sleeve and visible pink torso belong to one sweater but are different motion parts. Manual points lie in visible RGB interiors; precise shoulder seam remains uncertain.",
                "files": {key: {"path":str(path),"sha256":sha(path)} for key,path in files.items()},
                "pointsXY": POINTS, "queries": [
                    {"id":"sleeve","positive":POINTS["sleeve"],"negative":POINTS["torso"]},
                    {"id":"torso","positive":POINTS["torso"],"negative":POINTS["sleeve"]}],
                "evaluationExclusion": {"path":str(EVALUATION),"sha256":sha(EVALUATION),
                    "use":"Coordinates read only to exclude all prompt points from existing evaluation rectangles; no evaluation mask or pixel used for inference or candidate selection"},
                "rules": {"imageEncodings":1,"queries":2,"positivePerQuery":2,"negativePerQuery":2,
                          "allMultimasks":True,"maskThreshold":0,"alphaHeadCalls":0,"additionalQueries":0,
                          "postfilter":None,"candidateScoreSelection":False},
                "acceptance": "Independent material cores and whole visible RGB boundary must separate sleeve/torso without swallowing either part. Correct prompt-point labels alone do not pass. No threshold tuning, extra points, matting, deformation or background reconstruction.",
                "stopRule":"If neither query has a useful full visible material candidate, stop this head/input rule. No adaptive requery."}
    (out / "protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    source=Image.open(SOURCE).convert("RGB")
    draw=ImageDraw.Draw(source)
    for role,points in POINTS.items():
        color="cyan" if role=="sleeve" else "yellow"
        for n,(x,y) in enumerate(points):
            draw.ellipse((x-4,y-4,x+4,y+4),fill=color,outline="black",width=1)
            draw.text((x+6,y-8),f"{role}{n+1}",fill=color,stroke_width=1,stroke_fill="black")
    source.save(out / "source-points.png")
    source.crop((490,210,685,430)).resize((585,660)).save(out / "source-points-detail.png")
    print(json.dumps({"protocol":str(out/"protocol.json"),"sha256":sha(out/"protocol.json"),"manualControl":True}),flush=True)


def run(out):
    assert not (out / "run.json").exists()
    protocol=json.loads((out / "protocol.json").read_text())
    for entry in protocol["files"].values():
        assert sha(entry["path"])==entry["sha256"],entry["path"]
    sys.path[:0]=[str(RUNTIME/"deps"),str(RUNTIME/"vendor")]
    import torch
    from sam2.build_sam import build_sam2matting
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    report={"status":"running","manualControl":True,"protocolSHA256":sha(out/"protocol.json"),
            "torch":torch.__version__,"device":"cpu","imageEncodings":0,"alphaHeadCalls":0,"candidates":[]}
    (out/"run.json").write_text(json.dumps(report,indent=2)+"\n")
    model=build_sam2matting(CONFIG,str(WEIGHTS),device="cpu")
    forward=model.forward_image
    def encode(*args,**kwargs):
        report["imageEncodings"]+=1
        assert report["imageEncodings"]==1
        return forward(*args,**kwargs)
    def forbidden(*args,**kwargs):
        report["alphaHeadCalls"]+=1
        raise AssertionError("No alpha head in this diagnostic")
    model.forward_image=encode
    model._forward_alpha_heads=forbidden
    predictor=SAM2ImagePredictor(model,mask_threshold=0,max_hole_area=0,max_sprinkle_area=0)
    source=Image.open(SOURCE).convert("RGB")
    rgb=np.array(source)
    masks_all,logits_all,scores_all=[],[],[]
    with torch.inference_mode():
        start=time.perf_counter();predictor.set_image(source);report["encodeSeconds"]=time.perf_counter()-start
        for q in protocol["queries"]:
            start=time.perf_counter()
            masks,scores,logits=predictor.predict(point_coords=np.asarray(q["positive"]+q["negative"],dtype=np.float32),
                point_labels=np.array([1,1,0,0],dtype=np.int32),multimask_output=True,return_logits=False)
            seconds=time.perf_counter()-start
            assert masks.shape==(3,source.height,source.width)
            assert np.isfinite(logits).all() and np.isfinite(scores).all()
            for index in range(3):
                name=f'{q["id"]}-m{index}';mask=masks[index].astype(bool)
                Image.fromarray(mask.astype(np.uint8)*255).save(out/f"{name}-mask.png")
                preview=np.where(mask[...,None],rgb,np.round(rgb*.2)).astype(np.uint8)
                Image.fromarray(preview).save(out/f"{name}-preview.png")
                Image.fromarray(preview).crop((480,205,700,435)).resize((660,690)).save(out/f"{name}-detail.png")
                masks_all.append(np.packbits(mask.reshape(-1)));logits_all.append(logits[index]);scores_all.append(scores[index])
                report["candidates"].append({"id":name,"query":q["id"],"maskIndex":index,
                    "predictedIOUNotTruth":float(scores[index]),"pixels":int(mask.sum()),"queryDecodeSeconds":seconds,
                    "maskSHA256":sha(out/f"{name}-mask.png")})
            print(q["id"],seconds,flush=True)
    np.savez_compressed(out/"candidates.npz",masksPacked=np.stack(masks_all),lowResolutionLogits=np.stack(logits_all),
                        predictedIOU=np.asarray(scores_all),imageShape=[source.height,source.width])
    assert report["imageEncodings"]==1 and report["alphaHeadCalls"]==0
    report.update(status="completed_independent_review_pending",candidateBankSHA256=sha(out/"candidates.npz"))
    (out/"run.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="candidates"}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","run"])
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    (prepare if args.stage=="prepare" else run)(args.output)

"""Accessory color counterfactual and exact all-attachment control replay.

This tests output dependence on classified accessory texels, not merely whether
the visible accessory moved. The source graph remains frozen intentionally.
"""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replay import ROOT, raster_coverage, sha, snapshot, source_freeze, OUTPUT_STAGE, full_alpha


def main():
    source_freeze()
    stage = HERE/'outputs'/OUTPUT_STAGE
    root_report = json.loads((stage/'report.json').read_text())
    binary = Path(root_report['candidateImmutableBinary'])
    if sha(binary) != root_report['candidateBinarySha256']:
        raise RuntimeError('candidate binary changed')
    case = root_report['input']
    source_path = ROOT/case['canonical']
    before = np.asarray(Image.open(source_path).convert('RGBA'))
    h,w=before.shape[:2]
    graph=json.loads((ROOT/case['sourceGraph']).read_text())
    semantic=graph['operations'][0]['params']['semanticMasks']
    hard=raster_coverage(semantic['accessories'],w,h)>=.5
    # All >1/255 confidence is excluded from the dependence measurement so
    # edge reencoding/counterfactual output does not count as an outside leak.
    extent=raster_coverage(semantic['accessories'],w,h)>1/255
    target=stage/'occlusion-contract'
    target.mkdir(parents=True,exist_ok=True)
    results={'candidateBinarySha256':sha(binary),'accessoryHardPixels':int(hard.sum()),
             'accessoryNonzeroConfidencePixels':int(extent.sum()),
             'scope':'two source RGB accessory perturbations; same immutable graph, no redetection',
             'cases':[]}
    for amount in [30,100]:
        graph_path=stage/f'faceWidth{amount}.json'
        original=np.asarray(Image.open(stage/f'faceWidth{amount}-v21.png').convert('RGBA'))
        old=np.asarray(Image.open(stage/f'faceWidth{amount}-v19.png').convert('RGBA'))
        env=os.environ.copy();env.pop('FACE_V21_DEBUG_DIR',None)
        env['FACE_V21_USE_ALL_ATTACHMENTS']='1'
        env['FACE_V21_ALLOW_ACCESSORY_DONORS']='1'
        old_output=target/f'faceWidth{amount}-all-attachment-control.png'
        subprocess.run([str(binary),'--input',str(source_path),'--graph',str(graph_path),'--output',str(old_output)],env=env,check=True,capture_output=True)
        baseline_matches=np.array_equal(old,np.asarray(Image.open(old_output).convert('RGBA')))
        if not baseline_matches:
            raise RuntimeError('all-attachment control is not exact production v19')
        env.pop('FACE_V21_ALLOW_ACCESSORY_DONORS', None)
        safe_old_output=target/f'faceWidth{amount}-safe-all-attachment-control.png'
        env['FACE_V21_DEBUG_DIR']=str(target/f'safe-all-attachment-debug{amount}')
        subprocess.run([str(binary),'--input',str(source_path),'--graph',str(graph_path),'--output',str(safe_old_output)],env=env,check=True,capture_output=True)
        safe_old=np.asarray(Image.open(safe_old_output).convert('RGBA'))
        env.pop('FACE_V21_DEBUG_DIR',None)
        source_alpha_same=np.array_equal(np.asarray(Image.open(stage/f'debug{amount}/source-physical-alpha.png')),
                                         np.asarray(Image.open(target/f'safe-all-attachment-debug{amount}/source-physical-alpha.png')))
        if not source_alpha_same:
            raise RuntimeError('same donor guard does not yield same source physical alpha')
        for label,color in [('magenta',[255,0,255]),('green',[0,255,0])]:
            altered=before.copy();altered[hard,:3]=color
            image_path=target/f'accessory-{label}-source.png'
            Image.fromarray(altered).save(image_path)
            output=target/f'faceWidth{amount}-accessory-{label}.png'
            env.pop('FACE_V21_USE_ALL_ATTACHMENTS',None)
            subprocess.run([str(binary),'--input',str(image_path),'--graph',str(graph_path),'--output',str(output)],env=env,check=True,capture_output=True)
            after=np.asarray(Image.open(output).convert('RGBA'))
            changed=np.any(original!=after,axis=2)
            maximum=int(np.abs(original.astype(np.int16)-after.astype(np.int16))[~extent].max(initial=0))
            row={'amount':amount,'color':label,'v19AllAttachmentModeMatchesAllRGBA':baseline_matches,
                 'guardedAllAttachmentSourcePhysicalAlphaMatchesCandidate':source_alpha_same,
                 'guardedAllAttachmentChangedPixelsFromProductionV19':int(np.count_nonzero(np.any(safe_old!=old,axis=2))),
                 'guardedAllAttachmentOutputSha256':sha(safe_old_output),
                 'classifiedAccessoryOutputPreservedRGBA':bool(np.array_equal(after[hard],altered[hard])),
                 'outsideAccessoryConfidenceChangedPixels':int(changed[~extent].sum()),
                 'outsideAccessoryMaximumChannelDelta':maximum,
                 'inputSha256':sha(image_path),'graphSha256':sha(graph_path),'outputSha256':sha(output),
                 'wholePipelineAccessoryDonorIndependence':not bool(changed[~extent].any())}
            results['cases'].append(row)
            print(json.dumps(row),flush=True)
            (target/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    source_freeze()


if __name__=='__main__':
    main()

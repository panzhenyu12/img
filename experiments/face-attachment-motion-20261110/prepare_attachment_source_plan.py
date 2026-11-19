"""Plan source accessory motion before native face/ear connectivity is built."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from attachment_source_plan import plan, source_barriers
from source_masks import mask
from source_masks import sha


def prepare(source,graph_path,output,subtype_cache=None):
    graph=json.loads(graph_path.read_text())
    with Image.open(source) as image:w,h=image.size
    if graph.get('source')!={'width':w,'height':h}:
        raise ValueError('actual source dimensions required')
    params=graph['operations'][0]['params']
    if '__sourceMovingAttachmentSupport' in params:
        raise ValueError('source already has an attachment plan')
    masks={k:mask(v,w,h) for k,v in params['semanticMasks'].items()}
    zero=np.zeros((h,w),np.float32)
    protection=mask(params['protectMask'],w,h) if 'protectMask' in params else None
    garment=mask(params['__sourceGarment'],w,h) if '__sourceGarment' in params else None
    barriers=source_barriers(masks,protection,garment)
    partition_record=None
    if subtype_cache is None:
        support,selected=plan(masks.get('accessories',zero),barriers)
    else:
        from attachment_subtypes import prepare_partition
        support,selected,partition_record=prepare_partition(source,graph_path,subtype_cache,
                                                           masks.get('accessories',zero),barriers)
    raster={'shape':'raster','width':w,'height':h,'data':np.uint8(support.ravel()*255).tolist()}
    candidate=copy.deepcopy(graph)
    candidate['operations'][0]['params']['__sourceMovingAttachmentSupport']=raster
    output.mkdir(parents=True,exist_ok=False)
    (output/'support.json').write_text(json.dumps(raster,separators=(',',':')))
    (output/'candidate-graph.json').write_text(json.dumps(candidate,separators=(',',':')))
    if partition_record is not None:
        (output/'partition.json').write_text(json.dumps(partition_record,indent=2)+'\n')
    report={'sourceSHA256':sha(source),'originalGraphSHA256':sha(graph_path),
        'candidateGraphSHA256':sha(output/'candidate-graph.json'),
        'supportSHA256':sha(output/'support.json'),'source':str(source),'graph':str(graph_path),
        'sourceComponents':[{'label':r['label'],'corePixels':int(r['core'].sum()),
                            'supportPixels':int(r['support'].sum())} for r in selected],
        'policy':'existing source core>=.5 with at least30pixels; connected soft>.02; true protected material rejects',
        'motionPreparedBeforeSourceConnectivity':True,'physicalAlphaChanged':False,
        'semanticMasksChanged':False,'sourceRGBChanged':False,'manualEvidence':False,
        'scriptSHA256':sha(Path(__file__))}
    if partition_record is not None:
        report.update(policy=partition_record['policy'],partition=str((output/'partition.json').resolve()),
            partitionSHA256=sha(output/'partition.json'),components=partition_record['components'],
            sourceSubtypeEvidence=partition_record['evidence'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','graph','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--subtype-cache',type=Path,help='Exact-source selected class6/class9 posterior cache')
    a=p.parse_args();prepare(a.source.resolve(),a.graph.resolve(),a.out.resolve(),a.subtype_cache)

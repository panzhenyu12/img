"""Same-source class6/class9 evidence and a frozen attachment partition.

This is a preparation sidecar, not a new public graph field or material alpha.
The ordinary native API receives the existing binary movement-permission mask.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter

from attachment_source_plan import partition, plan, verify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(values, dtype):
    return hashlib.sha256(np.ascontiguousarray(values,dtype=dtype).tobytes()).hexdigest()


def load_source_subtypes(source, graph_path, cache):
    # Projection and mask sampling are source-only helpers.
    from source_masks import mask
    from consistent_detail_control import _project_probability, _coverage
    from apply_consistent_source_roles import apply_roles
    source,graph_path,cache=map(lambda p:Path(p).resolve(),(source,graph_path,cache))
    report_path=cache/'report.json';sidecar_path=cache/'raw-source-role-sidecar.json'
    report=json.loads(report_path.read_text());sidecar=json.loads(sidecar_path.read_text())
    provenance=sidecar['provenance'];selection=provenance['selection']
    if (report.get('sourceSHA256')!=sha(source) or provenance.get('sourceSHA256')!=sha(source)
            or report.get('selection')!=selection or provenance.get('manualEvidence') is not False
            or provenance.get('sourceGraphOrigin')!='automatic-detection'):
        raise ValueError('subtype evidence is not bound to this automatic unchanged source')
    original_graph=Path(provenance['sourceGraph'])
    if (sha(original_graph)!=provenance['sourceGraphSHA256']
            or report['sourceGraphSHA256']!=sha(original_graph)):
        raise ValueError('subtype source detection graph changed')
    graph=json.loads(graph_path.read_text())
    with Image.open(source) as image:w,h=image.size
    if graph.get('source')!={'width':w,'height':h}:
        raise ValueError('subtype graph source dimensions differ')
    original=json.loads(original_graph.read_text());original['source']={'width':w,'height':h}
    applied,_=apply_roles(original,sidecar)
    accessory=graph['operations'][0]['params']['semanticMasks']['accessories']
    if accessory!=applied['operations'][0]['params']['semanticMasks']['accessories']:
        raise ValueError('subtype and current accessory masks do not share the same source revision')
    cache_record=provenance['cache'];reused_reports=[]
    while 'passes' not in cache_record and cache_record.get('reused') is True:
        inherited=Path(cache_record['report'])
        if len(reused_reports)>=4 or sha(inherited)!=cache_record['reportSHA256']:
            raise ValueError('subtype inherited source cache binding changed')
        inherited_report=json.loads(inherited.read_text())
        if inherited_report.get('sourceSHA256')!=sha(source):
            raise ValueError('subtype inherited posterior belongs to a different source')
        reused_reports.append({'path':str(inherited),'sha256':sha(inherited)})
        cache_record=inherited_report['cache']
    model_sha='1d5c459e3d4691eb6120994fd36a8d360fe3732997d1bbf245278145cdd11d20'
    if cache_record.get('modelSHA256')!=model_sha:
        raise ValueError('subtype class IDs require the frozen source parsing model')
    rows=[row for row in cache_record.get('passes',[]) if row['id']==selection['id']]
    if len(rows)!=1 or rows[0]['crop']!=selection['crop']:
        raise ValueError('subtype selected source crop differs')
    row=rows[0];posterior_path=Path(row['probabilitiesFile'])
    if sha(posterior_path)!=row['fileSHA256']:
        raise ValueError('subtype source posterior file changed')
    with np.load(posterior_path,allow_pickle=False) as saved:probabilities=saved['probabilities']
    if (probabilities.ndim!=3 or probabilities.shape[0]!=19
            or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1))
            or not np.allclose(probabilities.sum(0),1,rtol=0,atol=1e-5)
            or hashlib.sha256(np.ascontiguousarray(probabilities).tobytes()).hexdigest()!=provenance['posteriorSHA256']):
        raise ValueError('subtype source requires the frozen complete normalized posterior')
    grid=(accessory['width'],accessory['height'])
    valid=np.asarray(sidecar['sourceValidityMask']['data']).reshape(grid[1],grid[0])==255
    selected=minimum_filter(valid,size=3,mode='nearest')
    raster=lambda a:{'shape':'raster','width':grid[0],'height':grid[1],'data':a.ravel().tolist()}
    subtypes={'observed':mask(raster(selected.astype(np.uint8)*255),w,h)==1}
    for name,index in [('eyewear',6),('earring',9)]:
        projected,covered=_project_probability(probabilities[index].astype(np.float64),selection['crop'],(w,h),grid)
        if np.any(valid&~covered):
            raise ValueError('subtype observed source support exceeds the selected crop')
        subtypes[name]=mask(raster(_coverage(projected)),w,h)
    evidence={'sourceSHA256':sha(source),'cache':str(cache),'reportSHA256':sha(report_path),
        'sidecarSHA256':sha(sidecar_path),'posterior':str(posterior_path),
        'posteriorFileSHA256':sha(posterior_path),'posteriorSHA256':provenance['posteriorSHA256'],
        'sourceGraphSHA256':sha(original_graph),'selection':selection,'classIds':{'eyewear':6,'earring':9},
        'modelSHA256':model_sha,'inheritedReports':reused_reports,
        'mapping':'existing per-class coverage and PreparedMask sampling; same observed-cell domain',
        'opticalAlphaClaim':False,'manualEvidence':False}
    return subtypes,evidence


def prepare_partition(source, graph_path, cache, accessory, barriers):
    source,graph_path,cache=map(lambda p:Path(p).resolve(),(source,graph_path,cache))
    subtypes,evidence=load_source_subtypes(source,graph_path,cache)
    groups=partition(accessory,barriers,subtypes)
    support,moving=plan(accessory,barriers,subtypes=subtypes)
    rows=[]
    for item in groups:
        rows.append({key:item[key] for key in ('label','coreLabels','kind','motion','observedSupport',
                                              'subtypeEvidence','protectedConflicts')})
        rows[-1].update(corePixels=int(item['core'].sum()),supportPixels=int(item['support'].sum()),
                       supportSHA256=array_sha(item['support'],'u1'))
    record={'version':1,'source':str(source),'sourceSHA256':sha(source),
        'originalGraph':str(graph_path),'originalGraphSHA256':sha(graph_path),
        'subtypeCache':str(cache),'evidence':evidence,'components':rows,
        'accessorySHA256':array_sha(accessory,'<f4'),
        'barrierSHA256':{name:array_sha(values,'u1') for name,values in sorted(barriers.items())},
        'permissionSHA256':array_sha(support,'u1'),
        'policy':'whole connected source instance; eyewear fixed; unambiguous earring may move; mixed or unobserved fixed-unresolved; moving conflicts reject',
        'implementationSHA256':{'partition':sha(Path(__file__).with_name('attachment_source_plan.py')),
                                'binding':sha(__file__),
                                'sampling':sha(Path(__file__).with_name('source_masks.py')),
                                'projection':sha(Path(__file__).with_name('consistent_detail_control.py')),
                                'roleApplication':sha(Path(__file__).with_name('apply_consistent_source_roles.py'))},
        'semanticMasksChanged':False,'physicalAlphaChanged':False,'automaticInstanceTruthClaim':False}
    return support,moving,record


def verify_partition(path, source, graph_path, support, accessory, barriers):
    path=Path(path).resolve();record=json.loads(path.read_text())
    if sha(source)!=record['sourceSHA256'] or sha(record['originalGraph'])!=record['originalGraphSHA256']:
        raise ValueError('frozen attachment partition source/graph binding changed')
    graph=json.loads(Path(graph_path).read_text());without=copy.deepcopy(graph)
    permission=without['operations'][0]['params'].pop('__sourceMovingAttachmentSupport',None)
    if permission is None or without!=json.loads(Path(record['originalGraph']).read_text()):
        raise ValueError('partition candidate graph differs outside its explicit permission')
    subtypes,_=load_source_subtypes(source,record['originalGraph'],record['subtypeCache'])
    selected=verify(support,accessory,barriers,subtypes=subtypes)
    _,_,expected=prepare_partition(source,record['originalGraph'],record['subtypeCache'],accessory,barriers)
    if expected!=record:
        raise ValueError('frozen attachment partition or source evidence changed')
    return selected


def attachment_release_from_source(graph, masks, source_path, graph_path, attachment_partition=None):
    """Replay the preparation permission before rebuilding any optical layer.

    Absence of protectMask stays absent in the frozen barrier identity. An
    explicitly supplied zero mask is still a distinct, source-bound input.
    """
    from source_masks import mask
    from attachment_source_plan import verify as verify_attachment_plan, source_barriers
    ih,iw=masks['accessories'].shape
    protect=mask(graph['protectMask'],iw,ih) if 'protectMask' in graph else np.zeros((ih,iw))
    release=(masks['accessories']>=.5)&(masks['hair']<.5)&(protect==0)
    if '__sourceMovingAttachmentSupport' in graph:
        release=mask(graph['__sourceMovingAttachmentSupport'],iw,ih)>=.5
        garment=mask(graph['__sourceGarment'],iw,ih) if '__sourceGarment' in graph else None
        barriers=source_barriers(masks,protect if 'protectMask' in graph else None,garment)
        if attachment_partition is None:
            verify_attachment_plan(release,masks['accessories'],barriers)
        else:
            verify_partition(attachment_partition,source_path,graph_path,release,masks['accessories'],barriers)
    elif attachment_partition is not None:
        raise ValueError('typed attachment partition requires an explicit permission, including zero')
    return release

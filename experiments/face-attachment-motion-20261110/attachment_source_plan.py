"""Source attachment topology and its planned motion support.

This reuses the existing >=.5 core / >.02 connected soft support and minimum
30-core-pixel rule. Optional same-source class6/class9 evidence distinguishes
stationary eyewear from candidate ear attachments. It never measures optical
opacity, changes the accessory mask, or cuts an instance around protection.
"""
import numpy as np
from scipy import ndimage as ndi


def components(accessory):
    accessory=np.asarray(accessory)
    if accessory.ndim!=2 or not np.isfinite(accessory).all() or np.any((accessory<0)|(accessory>1)):
        raise ValueError('finite source accessory evidence in [0,1] required')
    labels,count=ndi.label(accessory>=.5)
    result=[]
    for label in range(1,count+1):
        core=labels==label
        if np.count_nonzero(core)<30:
            continue
        support=ndi.binary_propagation(core,mask=accessory>.02)
        result.append({'label':label,'core':core,'support':support})
    return result


def partition(accessory, barriers, subtypes):
    """Conservative whole-component roles, not a precise instance classifier.

    Multiple hard cores joined by existing soft evidence are one instance.
    A disconnected weak branch without a >=30px core gets no permission; an
    actually disconnected frame piece cannot be inferred from this topology.
    """
    accessory=np.asarray(accessory)
    selected=components(accessory)
    checked={}
    for name,raw in barriers.items():
        blocked=np.asarray(raw,dtype=bool)
        if blocked.shape!=accessory.shape:
            raise ValueError('source barrier dimensions differ: '+name)
        checked[name]=blocked
    for name in ('eyewear','earring','observed'):
        if name not in subtypes:
            raise ValueError('both source subtypes and observed domain required')
        layer=np.asarray(subtypes[name])
        if layer.shape!=accessory.shape or not np.isfinite(layer).all():
            raise ValueError('source subtype dimensions or values differ: '+name)
        if name!='observed' and np.any((layer<0)|(layer>1)):
            raise ValueError('source subtype coverage outside [0,1]')
    if np.asarray(subtypes['observed']).dtype!=np.bool_:
        raise ValueError('source subtype observed domain must be boolean')
    groups=[]
    for item in selected:
        # Propagation uses the same soft connected domain; intersecting
        # supports are identical and must not receive different dispositions.
        previous=next((row for row in groups if np.any(row['support']&item['core'])),None)
        if previous is not None:
            previous['core']|=item['core']
            previous['coreLabels'].append(item['label'])
        else:
            groups.append({**item,'core':item['core'].copy(),'coreLabels':[item['label']]})
    for row in groups:
        support=row['support']
        kinds=[name for name in ('eyewear','earring')
               if np.any(support&(np.asarray(subtypes[name])>=.5))]
        observed=bool(np.asarray(subtypes['observed'])[support].all())
        conflicts=[name for name,blocked in checked.items() if np.any(support&blocked)]
        if not observed or len(kinds)!=1:
            kind='unresolved';motion='fixed'
        elif kinds==['eyewear']:
            kind='eyewear';motion='fixed'
        else:
            kind='earring';motion='moving'
            if conflicts:
                raise ValueError('source attachment conflicts with protected '+conflicts[0])
        row.update(kind=kind,motion=motion,observedSupport=observed,
                   subtypeEvidence=kinds,protectedConflicts=conflicts)
    return groups


def plan(accessory, barriers, subtypes=None):
    selected=components(accessory) if subtypes is None else [
        row for row in partition(accessory,barriers,subtypes) if row['motion']=='moving']
    support=np.zeros(np.shape(accessory),bool)
    for item in selected:support|=item['support']
    for name,raw in barriers.items():
        blocked=np.asarray(raw,dtype=bool)
        if blocked.shape!=support.shape:
            raise ValueError('source barrier dimensions differ: '+name)
        if np.any(support&blocked):
            raise ValueError('source attachment conflicts with protected '+name)
    return support,selected


def verify(support,accessory,barriers,subtypes=None):
    expected,selected=plan(accessory,barriers,subtypes=subtypes)
    if not np.array_equal(np.asarray(support,dtype=bool),expected):
        raise ValueError('attachment plan differs from the unchanged source component support')
    return selected


def source_barriers(masks,protection=None,garment=None):
    result={}
    for name in ('hair','leftEye','rightEye','leftBrow','rightBrow','nose','lips','teeth'):
        if name in masks:result[name]=np.asarray(masks[name])>=.5
    if protection is not None:result['user']=np.asarray(protection)>0
    if garment is not None:result['garment']=np.asarray(garment)>0
    return result

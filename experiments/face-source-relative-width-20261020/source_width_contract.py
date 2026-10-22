"""Prepare a separately named target contract from source evidence only."""
import argparse,copy,hashlib,json,sys,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from intent import derive,neck_request,attachment_request,intersections

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'experiments/face-cubic-monotone-20261020'))
from prepare_face_neck_support import verified_frame
from prepare_bounded_face import swept_attachment_nodes,validate_trace_source,touched_nodes,raw_controls
from audit_cubic_support import expand


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')


def build(base,provenance,amount=100.,baseline_amount=None,baseline_kind='clean-face-width'):
    c=copy.deepcopy(base);flow=verified_frame(Path(provenance['source_trace']),provenance)
    frame={k:flow[k] for k in ['origin','across','down','bounds']};frame['origin']=(np.array(frame['origin'])-.5).tolist()
    source=np.array([np.array(t['position'])+t['delta'] for t in base['targets'][:128]])
    old=np.array([t['position'] for t in base['targets'][:128]])
    r=derive(source,frame,amount)
    original_request=r if baseline_amount is None else derive(source,frame,baseline_amount)
    if baseline_kind=='clean-face-width':
        if np.max(np.linalg.norm(original_request['legacy']-old,axis=1))>1e-8:raise ValueError('expected original clean faceWidth contract at the explicitly supplied baseline amount')
    elif baseline_kind=='native-trace':
        trace=provenance['control_trace'];path=Path(trace['path']);assert sha(path/trace['file'])==trace['sha256']
        old_source,old_requested=raw_controls(path)
        assert np.array_equal(source,old_source) and np.array_equal(old,old_requested),'original native target trace does not match baseline'
    else:raise ValueError('unknown baseline contract')
    # A different requested level is recomputed from source coordinates. The
    # legacy comparison is also evaluated at this level, never a resized photo.
    old=r['legacy']
    for item,p,q in zip(c['targets'][:128],source,r['requested']):item.update(position=q.tolist(),delta=(p-q).tolist())
    neck=provenance.get('source_neck_support');start=neck['base_target_count'] if neck else 128+provenance['unknown_motion_prior']['pixels'];count=neck['added_target_count'] if neck else 0
    assert start==128+provenance['unknown_motion_prior']['pixels'] and start+count==len(base['targets'])
    ns=np.array(neck['source']) if neck else np.empty((0,2));nq=neck_request(ns,neck['profile'],frame,r['k']) if neck else ns.copy()
    assert count==len(ns) and (not count or np.allclose(np.array([np.array(t['position'])+t['delta'] for t in base['targets'][start:]]),ns,atol=1e-12,rtol=0))
    for item,p,q in zip(c['targets'][start:],ns,nq):item.update(position=q.tolist(),delta=(p-q).tolist())
    assert c['targets'][128:start]==base['targets'][128:start]
    g=c['grid'];w,h=g['width'],g['height'];origin=np.array(g['origin']);spacing=np.array(g['spacing'])
    border={i for i in range(w*h) if i%w in (0,w-1) or i//w in (0,h-1)}
    hard=set(expand(sorted(set(c['fixed_nodes'])|border),w,h));attachments=[]
    for i,e in enumerate(provenance['rigid_evidence']):
        assert i<2 and base['rigid_groups'][i]['target_weight']==100.
        delta,interpolation=attachment_request(e,source,r['requested']);points=np.array(e['source_material']+e['source_contact'])
        needed=swept_attachment_nodes(points,delta,origin,spacing,w,h,True);old_nodes=set(base['rigid_groups'][i]['nodes']);nodes=old_nodes|needed
        c['rigid_groups'][i]['nodes']=sorted(nodes);c['rigid_groups'][i]['delta']=delta.tolist()
        attachments.append({'group':i,'source_contact_count':len(e['source_contact']),'original_delta':base['rigid_groups'][i]['delta'],
                            'new_delta':delta.tolist(),'source_request_interpolation':interpolation,'added_raw_nodes':sorted(nodes-old_nodes),
                            'original_nodes_retained':True})
    for i,group in enumerate(c['rigid_groups']):
        expanded=set(expand(group['nodes'],w,h));collision=expanded&hard
        compatible_fixed_zero=group.get('target_weight') is None and group['delta']==[0.,0.]
        if collision and not compatible_fixed_zero:raise ValueError(f'new group {i} conflicts with original hard/border protection: {len(collision)}')
        for j,other in enumerate(c['rigid_groups'][:i]):
            if expanded&set(expand(other['nodes'],w,h)):raise ValueError(f'new groups {i}/{j} overlap')
    touched_nodes(np.array([t['position'] for t in c['targets']]),origin,spacing,w,h)
    assert {k:v for k,v in c.items() if k not in ['targets','rigid_groups']}=={k:v for k,v in base.items() if k not in ['targets','rigid_groups']}
    assert [t['weight'] for t in c['targets']]==[t['weight'] for t in base['targets']]
    assert c['rigid_groups'][len(provenance['rigid_evidence']):]==base['rigid_groups'][len(provenance['rigid_evidence']):]
    basis=np.stack([frame['across'],frame['down']],1);uv=(source-frame['origin'])@basis
    rows=[]
    for y in [.1,.17,.25,.34,.48,.6,.72,.78,.88,.9,.95]:
        height=frame['bounds'][1]+y*(frame['bounds'][3]-frame['bounds'][1]);cuts={name:intersections((points-frame['origin'])@basis,height) for name,points in [('source',source),('legacy',old),('candidate',r['requested'])]}
        rows.append({'normalized_height':y,**{name:cut.tolist() for name,cut in cuts.items()},
                     'legacy_inward_px':[float(cuts['legacy'][0]-cuts['source'][0]),float(cuts['source'][1]-cuts['legacy'][1])],
                     'new_inward_px':[float(cuts['candidate'][0]-cuts['source'][0]),float(cuts['source'][1]-cuts['candidate'][1])]})
    report={'name':'source-relative-width-same-envelope','amount':float(amount),'baseline_amount':float(amount if baseline_amount is None else baseline_amount),'source_parameter_recomputed':True,'frame':frame,'k':r['k'],'reference':r['reference'],
            'contract_change':'signed source-radius proportional width, keeping original faceWidth height/lateral envelope and eye-line origin; neck taper and attachment priors rederived from same source',
            'legacy_contract_preserved_separately':True,'pointwise_requests_unchanged':False,'no_parameter_search':True,
            'unchanged_unknown_targets':start-128,'contour_count':128,'neck_target_count':count,'height_rows':rows,'attachments':attachments,
            'neck':{'source':ns.tolist(),'original_requested':neck['requested'],'new_requested':nq.tolist(),'profile':neck['profile'],'source_topology_preserved':True} if neck else None,
            'original_target_weights_material_terms_true_protection_unchanged':True,
            'limitations':['One exact source photo with manual ear evidence; no generalization or beauty acceptance.','Source contour is semantic geometry, not independently verified facial anatomy.','Reference .48 is an existing profile design point, not a claimed anatomical optimum.']}
    report['baseline_contract_kind']=baseline_kind
    if not provenance.get('ear_material_strain'):
        report['limitations'][0]='One source photo; only its own source evidence is used, with no transferred manual ear annotation or ear-strain prior. No generalization or beauty acceptance.'
    return c,report


def expanded_controls(expected,provenance,audit,evidence):
    """Reconstruct only measured whole-material support additions.

    Old single-right-eye audits remain readable. New audits may cover both
    eyes, with their indices and source pixels taken from this photograph.
    No requested motion, weights, protection, or missing support is dropped.
    """
    old=read(evidence['controls']);ignore={'material_strain','material_orientation_groups','material_reference'}
    assert {k:v for k,v in old.items() if k not in ignore}=={k:v for k,v in expected.items() if k not in ignore},'support expansion does not continue the registered contract'
    proof=read(evidence['provenance']);assert audit['alpha']==1.
    assert proof['feature_transport']['source_sha256']==provenance['feature_transport']['source_sha256']
    proposal=read(evidence['proposal']);g=proposal['grid'];assert all(g[k]==expected['grid'][k] for k in ['width','height'])
    assert all(np.allclose(g[k],expected['grid'][k],rtol=0,atol=1e-12) for k in ['origin','spacing'])
    result=copy.deepcopy(expected);seen=set();w,h=g['width'],g['height']
    for record in audit.get('groups',[audit]):
        gi=record['group'];index=gi-proof['feature_transport']['original_groups_count']
        assert gi not in seen and 0<=index<len(proof['feature_transport']['groups']);seen.add(gi)
        assert gi==provenance['feature_transport']['original_groups_count']+index
        group=proof['feature_transport']['groups'][index];registered=provenance['feature_transport']['groups'][index]
        assert group['source_material']==registered['source_material'] and record['features']==group['features']==registered['features']
        assert record['features'] in [['leftEye','leftBrow'],['rightEye','rightBrow']]
        delta=np.array(proposal['rigid_translations'][gi]['actual']);assert delta.tolist()==record['trial_inverse_translation']
        old_nodes=set(expected['rigid_groups'][gi]['nodes'])
        needed=swept_attachment_nodes(np.array(group['source_material']),delta,np.array(g['origin']),np.array(g['spacing']),w,h,True)
        new_nodes=old_nodes|needed;new_coeff=set(expand(sorted(new_nodes),w,h));old_coeff=set(expand(sorted(old_nodes),w,h))
        assert sorted(needed-old_nodes)==record['new_raw_node_ids'] and sorted(new_coeff-old_coeff)==record['new_B3_coefficient_ids']
        result['rigid_groups'][gi]['nodes']=sorted(new_nodes)
    border={y*w+x for y in range(h) for x in range(w) if x in (0,w-1) or y in (0,h-1)}
    fixed=set(expand(sorted(set(expected['fixed_nodes'])|border),w,h))
    for gi in seen:
        coeff=set(expand(result['rigid_groups'][gi]['nodes'],w,h));assert not coeff&fixed
        assert all(not coeff&set(expand(other['nodes'],w,h)) for i,other in enumerate(result['rigid_groups']) if i!=gi)
    return result


def validated_controls(provenance):
    spec=provenance['source_relative_width'];base=Path(spec['baseline']);contract=Path(spec['contract'])
    if sha(base/'controls.json')!=spec['baseline_controls_sha256'] or sha(base/'provenance.json')!=spec['baseline_provenance_sha256'] or sha(contract)!=spec['contract_sha256']:raise ValueError('source-relative contract provenance changed')
    expected,_=build(read(base/'controls.json'),read(base/'provenance.json'),spec.get('amount',100.),spec.get('baseline_amount'),spec.get('baseline_kind','clean-face-width'))
    if expected!=read(contract):raise ValueError('source-relative contract does not reconstruct from original source evidence')
    for record in spec.get('support_expansions',[]):
        path=Path(record['audit']);assert sha(path)==record['sha256'],'support audit changed'
        audit=read(path);evidence={name:Path(item['path']) for name,item in audit['evidence'].items()}
        assert all(sha(evidence[name])==item['sha256'] for name,item in audit['evidence'].items()),'support sweep evidence changed'
        expected=expanded_controls(expected,provenance,audit,evidence)
    return expected


def register_support_expansion(provenance,path):
    path=Path(path).resolve();provenance['source_relative_width'].setdefault('support_expansions',[]).append({'audit':str(path),'sha256':sha(path)})
    return validated_controls(provenance)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--amount',type=float,default=100.);ap.add_argument('--baseline-amount',type=float);ap.add_argument('--baseline-kind',choices=['clean-face-width','native-trace'],default='clean-face-width');a=ap.parse_args();base=a.base.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    c0=read(base/'controls.json');p=read(base/'provenance.json');source=Path(p['feature_transport'].get('source') or p['ear_material_strain']['source'])
    assert sha(source)==p['control_trace']['source_evidence']['source_sha256']
    p['feature_transport'].update(source=str(source.resolve()),source_sha256=sha(source))
    image=Image.open(source).convert('RGBA');validate_trace_source(Path(p['source_trace']),np.asarray(image));c,report=build(c0,p,a.amount,a.baseline_amount,a.baseline_kind)
    save(out/'original-controls.json',c0);save(out/'contract.json',c);save(out/'controls.json',c);save(out/'target-comparison.json',report)
    p['source_relative_width']={'baseline':str(base),'baseline_controls_sha256':sha(base/'controls.json'),'baseline_provenance_sha256':sha(base/'provenance.json'),
                               'contract':str(out/'contract.json'),'contract_sha256':sha(out/'contract.json'),'report':str(out/'target-comparison.json'),'name':report['name'],'amount':a.amount}
    if a.baseline_amount is not None:p['source_relative_width']['baseline_amount']=a.baseline_amount
    if a.baseline_kind!='clean-face-width':p['source_relative_width']['baseline_kind']=a.baseline_kind
    p['legacy_control_trace']=copy.deepcopy(p['control_trace']);raw=out/'source-intent';raw.mkdir()
    pairs=[{'source':(np.array(t['position'])+t['delta']+.5).tolist(),'requested':(np.array(t['position'])+.5).tolist()} for t in c['targets'][:128]]
    save(raw/'controls.json',pairs)
    for name in ['layers.json','crop-source.png']:shutil.copyfile(Path(p['source_trace'])/name,raw/name)
    evidence=validate_trace_source(raw,np.asarray(image));p['control_trace'].update({'path':str(raw),'file':'controls.json','sha256':sha(raw/'controls.json'),'meaning':'Explicit source-relative-width-same-envelope target contract; legacy target trace separately retained.'});p['control_trace']['source_evidence']['intent']=evidence
    if report['neck']:
        p['source_neck_support']['requested']=report['neck']['new_requested'];p['source_neck_support']['target_semantics']=report['name'];p['source_neck_support']['legacy_policy']=p['source_neck_support']['policy'];p['source_neck_support']['policy']='Same source neck arc/profile with signed source radius; explicit target semantic change.'
    for e,record in zip(p['rigid_evidence'],report['attachments']):e.update(delta=record['new_delta'],request_interpolation=record['source_request_interpolation'],node_count=len(c['rigid_groups'][record['group']]['nodes']))
    comparison=str(base) if (base/'photo.png').exists() else p['feature_transport'].get('comparison',str(base))
    p['targets']=report['name'];p['feature_transport'].update(comparison=comparison,comparison_labels=['Source','Constant-width request','Source-relative request'])
    if a.baseline_kind=='native-trace':p['feature_transport']['comparison_labels']=['Source','Original native edit (different contract)','Source-relative width '+str(a.amount)]
    save(out/'provenance.json',p);assert validated_controls(p)==c
    overlay=image.convert('RGB');draw=ImageDraw.Draw(overlay)
    for name,color in [('source','yellow'),('legacy','red'),('requested','cyan')]:
        if name=='source':rows=[np.array(t['position'])+t['delta'] for t in c0['targets'][:128]]
        elif name=='legacy':rows=derive(np.array([np.array(t['position'])+t['delta'] for t in c0['targets'][:128]]),report['frame'],a.amount)['legacy'].tolist()
        else:rows=[t['position'] for t in c['targets'][:128]]
        draw.line([tuple(v) for v in rows+[rows[0]]],fill=color,width=1)
    overlay.save(out/'source-target-overlay.png');print(json.dumps({k:report[k] for k in ['name','k','reference','height_rows','attachments']},indent=2))


if __name__=='__main__':main()

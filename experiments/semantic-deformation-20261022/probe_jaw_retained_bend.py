"""Fixed-rho source-bend target proposal on the existing plateau preference.

Only the known source-convex, height-monotone window lower bound changes:
N_target >= 0.5*N_source. This is a finite-chord numerator ratio, not a
curvature ratio. Defaults, source geometry and unknown regions are untouched.
"""
import argparse
import copy
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
from unittest.mock import patch

import numpy as np
import scipy

import jaw_curve_qp as core
from export_jaw_curve_profile import verify_interfaces_and_area
from jaw_target_profile import load_profile
from probe_jaw_plateau import OriginalProblem, PlateauProblem, motion_statistics, save, sha


RHO = .5


class RetainedProblem(PlateauProblem):
    def numerator(self, arc, centers, values=None):
        target, denominator = OriginalProblem.numerator(self,arc,centers,values)
        if values is not None:
            source,_ = OriginalProblem.numerator(self,arc,centers)
            target = target-RHO*source
        return target,denominator

    def turn_rows(self, arc, centers):
        rows,source = OriginalProblem.turn_rows(self,arc,centers)
        return rows,(1.-RHO)*source


def _margin_report(rows):
    rows=copy.deepcopy(rows)
    for row in rows:
        for old,new in [('continuous_min_numerator','continuous_min_retained_bend_margin_px2'),
                        ('continuous_violating_subintervals','continuous_margin_violating_subintervals'),
                        ('continuous_min_signed_slope_at_numerator_minima','normalized_margin_at_margin_minima'),
                        ('violating_center_arclength_px','margin_violating_center_arclength_px')]:
            if old in row:row[new]=row.pop(old)
    return rows


def solve_retained_bend_profile(geometry, amount, *, max_radius_fraction=.13,
        node_spacing_px=2., smoothing_length_px=4., window_radius_px=16., bilateral_weight=1.):
    parameters=dict(max_radius_fraction=max_radius_fraction,node_spacing_px=node_spacing_px,
        smoothing_length_px=smoothing_length_px,window_radius_px=window_radius_px,bilateral_weight=bilateral_weight)
    if max_radius_fraction != .13 or bilateral_weight != 1.:
        raise ValueError('fixed proposal retains radius .13 and bilateral weight 1')
    with patch.object(core,'_Problem',RetainedProblem):
        result=core.solve_jaw_curve_profile(geometry,amount,**parameters)
    report=result['report']
    result['controller']='probe-plateau-retain-half-source-bend-v1'
    result['retained_bend_ratio']=RHO
    result['parameters']={'amount':float(amount),**parameters}
    report['constraints']+='; N_target >= .5*N_source on the SAME known source-convex height-monotone windows'
    report['retained_bend_definition']={
        'rho':RHO,'margin':'N_target - rho*N_source; finite-chord cross-multiplied signed slope numerator in px^2',
        'continuous_check':'Minimum of the entire piecewise quadratic margin, including its own interior vertex; not subtraction at the target-only minimum',
        'unchanged':'plateau preferred nodes, objective, total continuous original beta-cubed area, endpoint/sign bounds, source geometry and known/unknown domains',
        'limits':'Not a curvature-ratio, anatomical, aesthetic or rendered-field guarantee. Original absolute margin tolerance remains 1e-8 px^2.'}
    report['refinement_quantity']='retained-bend margin; original target geometry checked separately after this solve'
    report['refinement']=_margin_report(report['refinement'])
    if 'precision_fallback' in report:
        report['precision_fallback']['refinement']=_margin_report(report['precision_fallback']['refinement'])
    if 'bilateral_coupling' in report:
        report['bilateral_coupling']['reference_kind']='piecewise-linear scaled C2 plateau control preference; not a lambda-zero solution'
    if 'continuous_check' in report:
        report['retained_bend_continuous_check']=_margin_report(report.pop('continuous_check'))
    if report['accepted_numerically']:
        # Outside the patched problem: these are the original target numerator
        # checks, not retained-bend margins relabeled as geometry.
        checked=core.evaluate_profile_constraints(geometry,result['profiles'],window_radius_px=window_radius_px)
        report['continuous_check']=checked
        if any(row['continuous_violating_subintervals'] for row in checked):
            report['accepted_numerically']=False
            report['rejection_reason']='Original target geometry check failed after retained-bend solve'
            result['profiles']=[]
    else:
        report['continuous_check']=[]
    return result


def run(baselines, output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('refusing to overwrite frozen retained-bend probe')
    output.mkdir(parents=True);folder=output/'executed-code';folder.mkdir()
    records=[]
    for name in ['probe_jaw_retained_bend.py','probe_jaw_plateau.py','jaw_curve_qp.py',
                 'jaw_bilateral.py','jaw_collocation.py','jaw_target_profile.py','export_jaw_curve_profile.py']:
        source=Path(__file__).resolve().with_name(name);target=folder/name
        digest=sha(source);shutil.copy2(source,target)
        if sha(target)!=digest:raise ValueError('source changed during snapshot')
        records.append({'source':str(source),'snapshot':str(target),'sha256':digest})
    save(folder/'manifest.json',records)
    report={'experiment':'fixed rho=.5 retained-source finite-chord numerator; one new constraint only',
        'baseline':'six frozen plateau profiles; no baseline solve or changed preference',
        'source_bend_ratio_is_not_curvature_ratio':True,'no_parameter_scan':True,
        'no_photo_render':True,'aesthetic_approval':False,'cases':[],
        'executed_code_manifest':str(folder/'manifest.json'),'executed_code_manifest_sha256':sha(folder/'manifest.json'),
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'clarabel':version('clarabel')}}
    bound_inputs={}
    for path in baselines:
        path=Path(path).resolve();baseline=json.loads(path.read_text())
        if baseline['controller']!='probe-equal-original-beta-area-c2-plateau-bilateral1-v1':
            raise ValueError('baseline must be the frozen plateau proposal')
        geometry=baseline['source_geometry'];parameters=baseline['parameters'].copy()
        load_profile(path,geometry,parameters['amount'])
        subject=path.parent.name.split('-jaw-')[0]
        sign=('plus' if parameters['amount']>=0 else 'minus')+f"{abs(parameters['amount']):g}"
        case=output/(subject+'-'+sign);case.mkdir()
        frozen_baseline=case/'baseline-profile.json';shutil.copy2(path,frozen_baseline)
        protocol=Path(baseline['source_protocol']);frozen_protocol=case/'source-protocol.json';shutil.copy2(protocol,frozen_protocol)
        inputs={str(path):sha(path),str(frozen_baseline):sha(frozen_baseline),
            str(protocol):sha(protocol),str(frozen_protocol):sha(frozen_protocol),**baseline['source_evidence_sha256']}
        bound_inputs.update(inputs)
        candidate=solve_retained_bend_profile(geometry,**parameters)
        if candidate['source_geometry']!=geometry or candidate['required_signed_area_delta_px2']!=baseline['required_signed_area_delta_px2']:
            raise ValueError('original source or area request changed')
        preferred=PlateauProblem(geometry,parameters['node_spacing_px'],parameters['window_radius_px'])
        if preferred.scale!=baseline['preference_global_scale']:
            raise ValueError('plateau preference scale changed')
        candidate.update(source_protocol=str(frozen_protocol),source_protocol_sha256=sha(frozen_protocol),
            source_evidence_sha256=baseline['source_evidence_sha256'],inputs_sha256=inputs,
            baseline_profile=str(frozen_baseline),baseline_profile_sha256=sha(frozen_baseline),
            preference_global_scale=preferred.scale,
            executed_code_manifest=str(folder/'manifest.json'),executed_code_manifest_sha256=sha(folder/'manifest.json'),
            runtime=report['runtime'])
        for prefix,name in [('qp','jaw_curve_qp.py'),('export','probe_jaw_retained_bend.py'),
                            ('bilateral','jaw_bilateral.py'),('plateau','probe_jaw_plateau.py')]:
            candidate[prefix+'_source_path']=str(folder/name);candidate[prefix+'_source_sha256']=sha(folder/name)
        metrics={'case':case.name,'baseline_original_path':str(path),'baseline_profile_sha256':sha(path),
            'inputs_sha256':inputs,'parameters':parameters,'rho':RHO,
            'same_original_beta_area_and_plateau_preference':True,
            'accepted_numerically':candidate['report']['accepted_numerically'],
            'feasibility_message':candidate['report']['feasibility_message'],
            'retained_bend_continuous_check':candidate['report'].get('retained_bend_continuous_check',[]),
            'original_target_geometry_check':candidate['report']['continuous_check'],'sides':[]}
        if candidate['report']['accepted_numerically']:
            candidate['verification']=verify_interfaces_and_area(geometry,candidate)
            problem=OriginalProblem(geometry,parameters['node_spacing_px'],parameters['window_radius_px'])
            for arc in problem.arcs:
                old=next(p for p in baseline['profiles'] if p['arc_id']==arc['id'])
                new=next(p for p in candidate['profiles'] if p['arc_id']==arc['id'])
                metrics['sides'].append({'arc_id':arc['id'],
                    'frozen_plateau':motion_statistics(problem,arc,old),
                    'retained_bend':motion_statistics(problem,arc,new)})
        save(case/'qp-report.json',candidate)
        metrics['qp_report_path']=str(case/'qp-report.json');metrics['qp_report_sha256']=sha(case/'qp-report.json')
        if candidate['report']['accepted_numerically']:
            candidate['qp_report_path']=str(case/'qp-report.json');candidate['qp_report_sha256']=sha(case/'qp-report.json')
            save(case/'candidate-profile.json',candidate)
            load_profile(case/'candidate-profile.json',geometry,parameters['amount'])
            metrics['candidate_profile_path']=str(case/'candidate-profile.json')
            metrics['candidate_profile_sha256']=sha(case/'candidate-profile.json')
        save(case/'comparison.json',metrics);report['cases'].append(metrics)
        print(json.dumps({'case':case.name,'accepted_numerically':metrics['accepted_numerically'],
            'min_margin_px2':[r['continuous_min_retained_bend_margin_px2'] for r in metrics['retained_bend_continuous_check']]}),flush=True)
    if any(sha(p)!=digest for p,digest in bound_inputs.items()) or any(sha(row['source'])!=row['sha256'] for row in records):
        raise ValueError('bound source evidence or executed code changed during probe')
    save(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.baseline,args.output)

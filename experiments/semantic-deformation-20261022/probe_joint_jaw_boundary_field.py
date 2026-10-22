"""Freeze user ±100 joint source-field diagnostics; never render or export profiles."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

import numpy as np
from PIL import Image

from export_jaw_curve_profile import check_protocol
from jaw_target_profile import load_profile
from joint_jaw_boundary_field import solve_joint_field, interface_displacement_bending_energy, jaw_start_displacement_energy, check_jaw_prefix_intent, qp_fingerprint
from portable_field import build_problem
from portable_jaw import fixed_support, jaw_geometry, pixels
from run_portable_jaw import snapshot_local_code, validate_review_boundaries


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_first_round_reuse(folder,problem,geometry,amount,current_inputs):
    """Bind one frozen failed first round, then let the solver re-separate it."""
    from scipy.sparse import load_npz
    folder=Path(folder).resolve();path=folder/'report.json';report=read(path)
    if (len(report['refinement'])!=1 or report['refinement'][0]['solver_status']!='AlmostSolved'
        or report['refinement'][0]['same_subproblem_strict_polish']
        or report['source_continuous_accepted'] or report['source_geometry']!=geometry
        or report['amount']!=amount or report['grid']!=problem['grid']
        or any(report['inputs_sha256'].get(p)!=digest for p,digest in current_inputs.items())):
        raise ValueError('reuse requires one unchanged signed-source first-round AlmostSolved report and matching input hashes')
    state_path=folder/'diagnostic-solver-state.npz';metadata_path=folder/'diagnostic-qp'/'metadata.json'
    source_state_path=folder/'diagnostic-source-state.npz'
    if (sha(state_path)!=report['diagnostic_solver_state_sha256']
        or sha(metadata_path)!=report['diagnostic_qp_metadata_sha256']
        or sha(source_state_path)!=report['diagnostic_source_state_sha256']):
        raise ValueError('frozen first-round solver state or QP metadata changed')
    source_state=np.load(source_state_path)
    if not np.array_equal(source_state['mapping'],problem['mapping']):
        raise ValueError('reused source variable mapping differs')
    metadata=read(metadata_path);bindings={str(path):sha(path),str(state_path):sha(state_path),str(metadata_path):sha(metadata_path),
                                         str(source_state_path):sha(source_state_path)}
    for name,digest in metadata['files_sha256'].items():
        file=metadata_path.parent/name
        if sha(file)!=digest:raise ValueError('frozen first-round complete QP file changed')
        bindings[str(file)]=digest
    p=load_npz(metadata_path.parent/'P.npz');a=load_npz(metadata_path.parent/'A.npz')
    vectors=np.load(metadata_path.parent/'vectors.npz')
    fingerprint=qp_fingerprint(p,vectors['q'],a,vectors['b'],metadata['cones'])
    if (fingerprint!=metadata['sha256'] or fingerprint!=report['refinement'][0]['subproblem_sha256']
        or metadata['scalar_variables']!=problem['hessian'].shape[0]):
        raise ValueError('complete frozen first-round QP fingerprint or scalar dimension changed')
    state={key:value.copy() for key,value in np.load(state_path).items()}
    if not np.array_equal(source_state['scalar_solution'],state['primal_vector'][:metadata['scalar_variables']]):
        raise ValueError('reused source solution differs from the full frozen primal state')
    return {'source_report':str(path),'subproblem_sha256':fingerprint,'state':state,
            'row':report['refinement'][0],'prior_elapsed_seconds':report['total_source_solve_seconds']},bindings


def polygon_oracle(result,z):
    full=result['full_curve'];old=result['jaw_problem'];across=full.across
    area=lambda p:float(.5*np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))
    samples=[]
    for density in [4,8,16]:
        lo,hi=full.breaks[:-1],full.breaks[1:]
        s=(lo[:,None]+(hi-lo)[:,None]*np.linspace(0.,1.,density,endpoint=False)).ravel()
        source=full.source(s);motion=full.op(s)@z;target=source+motion[:,None]*across
        outside=np.ones(len(s),bool)
        for arc in old.arcs:
            local=((s-arc['start'])*arc['direction'])%full.length
            outside &= (local>arc['length']+1e-9)&(local<full.length-1e-9)
        samples.append({'subdivisions_per_original_edge_grid_piece':density,
            'sample_count':len(s),'closed_polygon_area_delta_px2':area(target)-area(source),
            'nonjaw_sampled_max_motion_px':float(abs(motion[outside]).max(initial=0.))})
    extrapolated=[(4*b['closed_polygon_area_delta_px2']-a['closed_polygon_area_delta_px2'])/3
                  for a,b in zip(samples[:-1],samples[1:])]
    return {'method':'independent dense source walk and target polygon shoelace; quadratic target pieces give O(h^2) area error',
            'measurements':samples,'richardson_area_delta_px2':extrapolated,
            'richardson_difference_px2':abs(extrapolated[1]-extrapolated[0]),
            'nonjaw_area_cancellation_does_not_prove_no_motion':True}


def run(references,interface_path,soft_protocol_path,output,*,interface_bend_band=None,allow_strict_polish=True,
        regularize_interface_displacement=False,comparison_source_root=None,regularize_jaw_start=False,
        constrain_jaw_start_intent=False,reuse_first_round_source_root=None):
    output=Path(output).resolve();interface_path=Path(interface_path).resolve()
    soft_protocol_path=Path(soft_protocol_path).resolve()
    if output.exists():raise FileExistsError('refusing to overwrite frozen joint diagnostic')
    interface=read(interface_path);visual_path=interface_path.parent/'visual-source-review.json';visual=read(visual_path)
    if (interface.get('schema')!='reviewed-visible-source-interface-v1'
        or interface.get('anatomical_role_status','').split(';')[0]!='unknown'
        or interface.get('automatic_edit_authorized') is not False
        or visual['hashes'].get(str(interface_path))!=sha(interface_path)):
        raise ValueError('unchanged source-only reviewed full interface required')
    neck_path=Path(interface['prior_review_consistency']['old_neck_source_path']);neck=read(neck_path)
    if sha(neck_path)!=interface['prior_review_consistency']['old_neck_source_sha256']:
        raise ValueError('old reviewed neck source changed')
    soft_protocol=read(soft_protocol_path);soft=soft_protocol['soft_source_boundary']
    if (soft['density_per_source_px']!=.05 or soft['fade_length_px']!=16. or soft['spacing_px']!=2.
        or soft_protocol['inputs_sha256'].get(str(neck_path))!=sha(neck_path)):
        raise ValueError('same prior reviewed soft0 quadrature and evidence required')
    inputs={**interface['inputs_sha256'],**visual['hashes'],**neck['input_hashes'],
            str(interface_path):sha(interface_path),str(visual_path):sha(visual_path),
            str(neck_path):sha(neck_path),str(soft_protocol_path):sha(soft_protocol_path)}
    if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('frozen source evidence changed')
    if len(interface['segments'])!=len(neck['segments']):raise ValueError('one reviewed low segment per full interface required')
    for a,b in zip(interface['segments'],neck['segments']):
        p=np.asarray(a['source_points']);q=np.asarray(b['source_points'])
        if not np.array_equal(q[:,1],np.arange(565.,611.,5.)) or not np.allclose(
            np.interp(q[:,1],p[:,1],p[:,0]),q[:,0],atol=1e-10,rtol=0.):
            raise ValueError('known neck source must coincide with reviewed y565–610 subset')
    output.mkdir(parents=True);manifest,code_hashes=snapshot_local_code(output)
    aggregate={'experiment':'joint-jaw-full-visible-interface-source-diagnostic-v1',
        'new_target_distribution':True,'executable':False,'no_render':True,'no_parameter_scan':True,
        'inputs_sha256':inputs,'executed_code_manifest':str(manifest),'cases':[],
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'clarabel':version('clarabel')},
        'interface_source_relative_bend_band':interface_bend_band,
        'interface_shape_constraint_change':interface_bend_band is not None,
        'interface_displacement_bending_enabled':bool(regularize_interface_displacement),
        'jaw_start_displacement_regularization_enabled':bool(regularize_jaw_start),
        'jaw_start_intent_constraints_enabled':bool(constrain_jaw_start_intent),
        'reuse_first_round_source_root':str(Path(reuse_first_round_source_root).resolve()) if reuse_first_round_source_root else None,
        'comparison_source_root':str(Path(comparison_source_root).resolve()) if comparison_source_root is not None else None,
        'same_subproblem_strict_polish_enabled':bool(allow_strict_polish)}
    for reference in references:
        reference=Path(reference).resolve();protocol_path=reference/'protocol.json'
        protocol,evidence=check_protocol(protocol_path);profile_path=Path(protocol['frozen_target_profile'])
        profile=read(profile_path);geometry=profile['source_geometry'];amount=profile['amount']
        if abs(amount)!=100 or protocol['amounts']!=[amount]:raise ValueError('one frozen user signed100 reference required')
        if profile['controller']!='probe-plateau-retain-half-source-bend-v1':raise ValueError('frozen retained baseline required')
        load_profile(profile_path,geometry,amount)
        if protocol['source_geometry']!=geometry or protocol['field_reduction_stride']!=1:
            raise ValueError('same geometry and full stride1 reference required')
        source_path=Path(interface['source_image']);source_folder=source_path.parent
        record_path=source_folder/'cached-source-evidence.json';proof_path=source_folder/'report.json';review_path=source_folder/'visual-source-review.json'
        proof,review,record=read(proof_path),read(review_path),read(record_path)
        if evidence.get(str(source_path))!=interface['source_image_sha256'] or neck['source_image_sha256']!=sha(source_path):
            raise ValueError('reference must use the same reviewed source image')
        operations=[c for c in record['checks'] if c.get('key')=='faceWidth'][0]['graph']['operations']
        if len(operations)!=1 or operations[0]['type']!='face.adjust':raise ValueError('one source face operation required')
        params=operations[0]['params'];size=tuple(interface['source_size'])
        rebuilt=jaw_geometry(pixels(params['semanticMasks']['faceContour']['points'],size),pixels(params['landmarks'],size),
            [pixels(params['semanticMasks'][k]['points'],size) for k in ['leftEye','rightEye']],pixels(params['semanticMasks']['lips']['points'],size))
        if rebuilt!=geometry:raise ValueError('reviewed cached source geometry cannot be reproduced exactly')
        validate_review_boundaries(rebuilt,review)
        fixed,crop,_,radius=fixed_support(size,params,geometry)
        frozen_fixed=np.asarray(Image.open(reference/'fixed-source.png'))>0
        if crop!=protocol['crop'] or not np.array_equal(fixed,frozen_fixed) or radius!=protocol['support_radius_px']:
            raise ValueError('same reconstructed crop/fixed mask/support required')
        y,x=np.where(fixed)
        dummy=np.array([[crop[0]+4.,crop[1]+4.]])
        problem=build_problem((size[1],size[0]),dummy,dummy,np.column_stack([x,y]),crop=crop,
            spacing=2.,target_weights=0.,stride=1,bending_weight=protocol['bending_weight'],membrane_weight=protocol['membrane_weight'])
        case=output/('user-plus100' if amount>0 else 'user-minus100');case.mkdir()
        bound={**inputs,**evidence,str(protocol_path):sha(protocol_path),str(profile_path):sha(profile_path),
               str(reference/'fixed-source.png'):sha(reference/'fixed-source.png')}
        reuse=None
        if reuse_first_round_source_root is not None:
            reuse,reuse_bindings=load_first_round_reuse(Path(reuse_first_round_source_root)/case.name,
                problem,geometry,amount,bound)
            bound.update(reuse_bindings)
        result=solve_joint_field(problem,geometry,amount,interface['segments'],neck['segments'],
            smoothing_length_px=profile['parameters']['smoothing_length_px'],
            jaw_radius_px=profile['parameters']['window_radius_px'],
            neck_density=soft['density_per_source_px'],neck_fade_px=soft['fade_length_px'],
            interface_bend_band=interface_bend_band,allow_strict_polish=allow_strict_polish,
            regularize_interface_displacement=regularize_interface_displacement,
            regularize_jaw_start=regularize_jaw_start,constrain_jaw_start_intent=constrain_jaw_start_intent,
            reuse_first_round=reuse)
        report=result['report'];report.update(inputs_sha256=bound,grid=problem['grid'],
            fixed_source_pixels=int(fixed.sum()),free_scalar_variables=problem['hessian'].shape[0],
            source_geometry=geometry,reference_profile=str(profile_path),reference_profile_sha256=sha(profile_path),
            original_fixed_mask_exactly_reconstructed=True,original_crop_reconstructed=True,
            executed_code_manifest=str(manifest),executed_code_manifest_sha256=sha(manifest))
        z=result['diagnostic_solution']
        if z is not None and comparison_source_root is not None:
            baseline_folder=Path(comparison_source_root).resolve()/case.name
            baseline_report_path=baseline_folder/'report.json';baseline=read(baseline_report_path)
            baseline_state_path=baseline_folder/'diagnostic-source-state.npz'
            if (baseline['grid']!=problem['grid'] or baseline['source_geometry']!=geometry
                or baseline['amount']!=amount
                or baseline['inputs_sha256'].get(str(interface_path))!=sha(interface_path)
                or baseline['diagnostic_source_state_sha256']!=sha(baseline_state_path)):
                raise ValueError('comparison must bind the same source, geometry, amount and frozen scalar state')
            baseline_state=np.load(baseline_state_path)
            if not np.array_equal(baseline_state['mapping'],problem['mapping']):
                raise ValueError('comparison must use the unchanged source variable mapping')
            baseline_z=baseline_state['scalar_solution']
            if baseline_z.shape!=z.shape or not np.isfinite(baseline_z).all():
                raise ValueError('comparison requires finite complete source variables')
            metric=interface_displacement_bending_energy(result['interface_displacement_regularizers'],baseline_z)
            bound.update({str(baseline_report_path):sha(baseline_report_path),str(baseline_state_path):sha(baseline_state_path)})
            report['interface_displacement_bending']['comparison']={
                'source_report':str(baseline_report_path),'source_report_sha256':sha(baseline_report_path),
                'same_operator_and_weights':True,**metric}
            report['jaw_start_displacement_regularization']['comparison']={
                'source_report':str(baseline_report_path),'source_report_sha256':sha(baseline_report_path),
                'same_operator_and_weights':True,
                **jaw_start_displacement_energy(result['jaw_start_regularizers'],baseline_z)}
            report['jaw_start_intent_constraints']['comparison']={
                'source_report':str(baseline_report_path),'source_report_sha256':sha(baseline_report_path),
                'same_operator_and_caps':True,**check_jaw_prefix_intent(result['jaw_start_intent_blocks'],baseline_z)}
        if result.get('diagnostic_solver_state') is not None:
            np.savez_compressed(case/'diagnostic-solver-state.npz',**result['diagnostic_solver_state'])
            report['diagnostic_solver_state_sha256']=sha(case/'diagnostic-solver-state.npz')
        if result.get('diagnostic_qp') is not None:
            from scipy.sparse import save_npz
            qp=result['diagnostic_qp'];folder=case/'diagnostic-qp';folder.mkdir()
            save_npz(folder/'P.npz',qp['P']);save_npz(folder/'A.npz',qp['A'])
            np.savez_compressed(folder/'vectors.npz',q=qp['q'],b=qp['b'])
            metadata={key:value for key,value in qp.items() if key not in ['P','q','A','b']}
            metadata['files_sha256']={name:sha(folder/name) for name in ['P.npz','A.npz','vectors.npz']}
            metadata['scope']='Exact final executed source conic QP; diagnostic only, no rendering or executable profile authorization'
            save(folder/'metadata.json',metadata)
            report['diagnostic_qp_metadata_sha256']=sha(folder/'metadata.json')
        if z is not None:
            report['closed_polygon_oracle']=polygon_oracle(result,z)
            np.savez_compressed(case/'diagnostic-source-state.npz',scalar_solution=z,mapping=problem['mapping'])
            report['diagnostic_source_state_sha256']=sha(case/'diagnostic-source-state.npz')
            curve_rows=[]
            for index,curve in enumerate(result['curves']):
                positions=np.unique(np.r_[curve.breaks,np.linspace(0.,curve.length,int(np.ceil(curve.length*4))+1)])
                source=curve.source(positions);motion=curve.op(positions)@z
                np.savez_compressed(case/f'diagnostic-curve-{index}.npz',source=source,
                    target=source+motion[:,None]*curve.across,source_arclength_px=positions)
                curve_rows.append({'index':index,'maximum_across_motion_px':float(abs(motion).max()),
                                   'sha256':sha(case/f'diagnostic-curve-{index}.npz')})
            report['diagnostic_curves']=curve_rows
        if any(sha(p)!=h for p,h in bound.items()):raise ValueError('bound source inputs changed during diagnostic')
        save(case/'report.json',report)
        aggregate['cases'].append({'case':case.name,'report':str(case/'report.json'),'sha256':sha(case/'report.json'),
            'source_continuous_accepted':report['source_continuous_accepted'],'rejection':report.get('rejection')})
        print(json.dumps(aggregate['cases'][-1]),flush=True)
    if any(sha(p)!=h for p,h in code_hashes.items()):raise ValueError('executed code changed during diagnostic')
    save(output/'report.json',aggregate)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,action='append',required=True)
    parser.add_argument('--interface-bend-band',type=float,nargs=2,default=None,metavar=('LOWER','UPPER'))
    parser.add_argument('--no-strict-polish',action='store_true')
    parser.add_argument('--regularize-interface-displacement',action='store_true')
    parser.add_argument('--regularize-jaw-start',action='store_true')
    parser.add_argument('--constrain-jaw-start-intent',action='store_true')
    parser.add_argument('--comparison-source-root',type=Path)
    parser.add_argument('--reuse-first-round-source-root',type=Path)
    for key in ['interface','soft-protocol','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();run(args.reference,args.interface,args.soft_protocol,args.output,
        interface_bend_band=args.interface_bend_band,allow_strict_polish=not args.no_strict_polish,
        regularize_interface_displacement=args.regularize_interface_displacement,
        comparison_source_root=args.comparison_source_root,regularize_jaw_start=args.regularize_jaw_start,
        constrain_jaw_start_intent=args.constrain_jaw_start_intent,
        reuse_first_round_source_root=args.reuse_first_round_source_root)

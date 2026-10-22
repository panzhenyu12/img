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
from joint_jaw_boundary_field import solve_joint_field
from portable_field import build_problem
from portable_jaw import fixed_support, jaw_geometry, pixels
from run_portable_jaw import snapshot_local_code, validate_review_boundaries


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


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


def run(references,interface_path,soft_protocol_path,output):
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
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'clarabel':version('clarabel')}}
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
        result=solve_joint_field(problem,geometry,amount,interface['segments'],neck['segments'],
            smoothing_length_px=profile['parameters']['smoothing_length_px'],
            jaw_radius_px=profile['parameters']['window_radius_px'],
            neck_density=soft['density_per_source_px'],neck_fade_px=soft['fade_length_px'])
        report=result['report'];report.update(inputs_sha256=bound,grid=problem['grid'],
            fixed_source_pixels=int(fixed.sum()),free_scalar_variables=problem['hessian'].shape[0],
            source_geometry=geometry,reference_profile=str(profile_path),reference_profile_sha256=sha(profile_path),
            original_fixed_mask_exactly_reconstructed=True,original_crop_reconstructed=True,
            executed_code_manifest=str(manifest),executed_code_manifest_sha256=sha(manifest))
        z=result['diagnostic_solution']
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
    for key in ['interface','soft-protocol','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();run(args.reference,args.interface,args.soft_protocol,args.output)

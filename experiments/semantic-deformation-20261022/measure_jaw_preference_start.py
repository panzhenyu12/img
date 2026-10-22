"""Measure existing jaw intent start energies; no optimization or rendering."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

from probe_jaw_plateau import OriginalProblem, PlateauProblem


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preference_polynomial(problem,arc,lo,hi,amount,fraction,kind):
    """Actual radius-times-envelope preference in local q=(s-lo)/(hi-lo)."""
    if kind not in ['original_beta_cubed','equal_area_plateau']:
        raise ValueError('known existing preference required')
    if not (0.<=lo<hi<=arc['length']):
        raise ValueError('preference piece must fit original arc')
    # Caller splits at original radius knots and every envelope branch point.
    for knot in arc['knots']:
        if lo<knot<hi:raise ValueError('radius knot inside polynomial piece')
    if kind=='equal_area_plateau' and any(lo<k<hi for k in [.3*arc['length'],.7*arc['length']]):
        raise ValueError('plateau branch inside polynomial piece')
    radius=(problem.source(arc,np.array([lo,hi]))-problem.origin)@problem.across-problem.midline
    radius_poly=Polynomial([radius[0],radius[1]-radius[0]])
    s=Polynomial([lo,hi-lo]);t=s/arc['length']
    if kind=='original_beta_cubed':
        envelope=64.*(t*(1.-t))**3;scale=1.
    else:
        middle=(lo+hi)/2.
        if middle<.3*arc['length']:ramp=t/.3
        elif middle>.7*arc['length']:ramp=(1.-t)/.3
        else:ramp=None
        envelope=Polynomial([1.]) if ramp is None else 10.*ramp**3-15.*ramp**4+6.*ramp**5
        scale=problem.scale
    return -amount/100.*fraction*scale*radius_poly*envelope


def measure_preference_start(problem,arc,amount,fraction,length_px,kind):
    if not np.isfinite(length_px) or not (0.<length_px<=arc['length']):
        raise ValueError('original preference window must fit known source jaw')
    branches=np.array([.3*arc['length'],.7*arc['length']]) if kind=='equal_area_plateau' else np.array([])
    cuts=np.unique(np.r_[0.,arc['knots'][(arc['knots']>0.)&(arc['knots']<length_px)],
                         branches[(branches>0.)&(branches<length_px)],length_px])
    g,gauss_weights=np.polynomial.legendre.leggauss(9);q=(g+1.)/2.
    energy=0.;antiderivative_energy=0.;unweighted_energy=0.;unweighted_analytic=0.;pieces=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        motion=preference_polynomial(problem,arc,lo,hi,amount,fraction,kind)
        derivative=motion.deriv()/(hi-lo);s=lo+(hi-lo)*q;t=s/length_px
        taper=(1.-t)**3*(1.+3.*t+6.*t**2)
        value=length_px**2*(hi-lo)/2.*float(gauss_weights@(taper*derivative(q)**2))
        local_t=Polynomial([lo/length_px,(hi-lo)/length_px])
        weight=(1.-local_t)**3*(1.+3.*local_t+6.*local_t**2)
        integrand=length_px**2*(hi-lo)*weight*derivative**2
        primitive=integrand.integ();analytic=float(primitive(1.)-primitive(0.))
        unweighted=length_px**2*(hi-lo)/2.*float(gauss_weights@derivative(q)**2)
        primitive0=(length_px**2*(hi-lo)*derivative**2).integ()
        unweighted_energy+=unweighted;unweighted_analytic+=float(primitive0(1.)-primitive0(0.))
        energy+=value;antiderivative_energy+=analytic
        pieces.append({'source_interval_px':[float(lo),float(hi)],'motion_degree':motion.degree(),
                       'integrand_degree':integrand.degree(),'energy_px3':value,'unweighted_energy_px3':unweighted})
    samples=[]
    for position in [0.,.5,1.,2.,6.,12.]:
        if position>length_px:continue
        i=min(int(np.searchsorted(cuts,position,side='right')-1),len(cuts)-2)
        lo,hi=cuts[i:i+2];motion=preference_polynomial(problem,arc,lo,hi,amount,fraction,kind)
        local=(position-lo)/(hi-lo)
        samples.append({'source_arclength_px':position,'motion_px':float(motion(local)),
                        'derivative':float(motion.deriv()(local)/(hi-lo)),
                        'derivative_side':'right original piece except final window endpoint uses left'})
    return {'id':arc['id'],'arc_length_px':float(arc['length']),'window_end_px':float(length_px),
            'energy_px3':energy,'soc_radius_sqrt_energy':float(np.sqrt(energy)),
            'polynomial_antiderivative_energy_px3':antiderivative_energy,
            'quadrature_antiderivative_difference_px3':energy-antiderivative_energy,
            'unweighted_energy_px3':unweighted_energy,
            'unweighted_soc_radius_sqrt_energy':float(np.sqrt(unweighted_energy)),
            'unweighted_polynomial_antiderivative_energy_px3':unweighted_analytic,
            'unweighted_quadrature_antiderivative_difference_px3':unweighted_energy-unweighted_analytic,
            'unweighted_cauchy_motion_bounds':[{'source_arclength_px':float(s),
                'maximum_abs_motion_from_start_px':float(np.sqrt(s*unweighted_energy)/length_px)}
                for s in [2.,6.,12.] if s<=length_px],
            'samples':samples,'pieces':pieces}


def existing_field_unweighted_energy(path,original,length):
    from scipy.sparse import csc_matrix
    from joint_jaw_boundary_field import GridCurve,jaw_start_displacement_regularizer
    report=json.loads(path.read_text());state_path=path.parent/'diagnostic-source-state.npz'
    if sha(state_path)!=report['diagnostic_source_state_sha256']:
        raise ValueError('frozen comparison source field changed')
    state=np.load(state_path);z=state['scalar_solution'];mapping=state['mapping']
    problem={'grid':report['grid'],'mapping':mapping,'hessian':csc_matrix((len(z),len(z)))}
    rows=[]
    for arc in original.arcs:
        curve=GridCurve(original.source(arc,arc['knots']),problem,original.across,original.down,source_knots=arc['knots'])
        quadrature=jaw_start_displacement_regularizer(curve,length)
        t=quadrature['positions']/length;taper=(1.-t)**3*(1.+3.*t+6.*t**2)
        # All Gauss sites are interior, hence taper > 0; this removes only
        # the weighting, retaining the original exact ds quadrature.
        weights=quadrature['weights']/taper;derivative=quadrature['operator']@z
        energy=float(weights@derivative**2)
        rows.append({'id':arc['id'],'unweighted_energy_px3':energy})
    return {'method':'four-Gauss exact degree-two derivative-square integral on the original field source/grid pieces',
            'curves':rows,'total_unweighted_energy_px3':sum(row['unweighted_energy_px3'] for row in rows),
            'source_state':str(state_path),'source_state_sha256':sha(state_path)}


def run(source_report_path,output):
    source_report_path=Path(source_report_path).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('refusing to overwrite preference energy evidence')
    source=json.loads(source_report_path.read_text());geometry=source['source_geometry'];amount=source['amount']
    profile_path=Path(source['reference_profile']);profile=json.loads(profile_path.read_text())
    if sha(profile_path)!=source['reference_profile_sha256'] or profile['source_geometry']!=geometry:
        raise ValueError('original intent geometry/profile binding changed')
    parameters=profile['parameters'];spacing=parameters['node_spacing_px'];radius=parameters['window_radius_px']
    length=parameters['smoothing_length_px'];fraction=parameters['max_radius_fraction']
    original=OriginalProblem(geometry,spacing,radius);plateau=PlateauProblem(geometry,spacing,radius)
    report={'schema':'original-jaw-preference-start-energy-v1','source_report':str(source_report_path),
            'inputs_sha256':{str(source_report_path):sha(source_report_path),str(profile_path):sha(profile_path)},
            'amount':amount,'max_radius_fraction':fraction,'window_length_px':length,
            'formula':'E=length^2 * integral_0^length [1-10(s/length)^3+15(s/length)^4-6(s/length)^5] * beta_prime(s)^2 ds',
            'unweighted_formula':'E0=length^2 * integral_0^length beta_prime(s)^2 ds',
            'unweighted_cauchy_bound':'abs(u(s)-u(0)) <= sqrt(s*E0)/length; a separate budget for each original jaw',
            'method':'analytic local polynomial derivative on original radius/envelope pieces; nine-point Gauss exactly integrates degree<=17 in real arithmetic; separate polynomial antiderivative cross-check',
            'no_solve':True,'no_changed_intent':True,'feasibility_not_assumed':True,
            'plateau_equal_area_scale':plateau.scale,
            'original_requested_area_px2':original.beta_area(amount,fraction),
            'plateau_preferred_area_px2':plateau.scale*plateau.plateau_area(amount,fraction),
            'profiles':{}}
    for name,problem in [('original_beta_cubed',original),('equal_area_plateau',plateau)]:
        rows=[measure_preference_start(problem,arc,amount,fraction,length,name) for arc in problem.arcs]
        report['profiles'][name]={'curves':rows,'total_energy_px3':sum(r['energy_px3'] for r in rows),
                                 'total_unweighted_energy_px3':sum(r['unweighted_energy_px3'] for r in rows)}
    for version in ['v10','v11']:
        path=source_report_path.parent.parent.parent/('joint-jaw-visible-interface-source-probe-'+version)/source_report_path.parent.name/'report.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old['source_geometry']!=geometry or old['amount']!=amount:
                raise ValueError('comparison field must share original geometry and strength')
            metric=old.get('jaw_start_displacement_regularization')
            # v10 predates its metric: v11 freezes v10 under the same operator.
            if version=='v10':metric=source.get('jaw_start_displacement_regularization',{}).get('comparison')
            if metric:
                metric=dict(metric);metric['unweighted']=existing_field_unweighted_energy(path,original,length)
                report.setdefault('existing_field_start_energy',{})[version]=metric
                report['inputs_sha256'][str(path)]=sha(path)
                report['inputs_sha256'][metric['unweighted']['source_state']]=metric['unweighted']['source_state_sha256']
    if any(sha(p)!=value for p,value in report['inputs_sha256'].items()):raise ValueError('bound evidence changed')
    output.mkdir(parents=True);(output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(output/'report.json'),'profiles':{
        name:{'weighted_energy_px3':value['total_energy_px3'],
              'unweighted_energy_px3':value['total_unweighted_energy_px3']}
        for name,value in report['profiles'].items()}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source_report,args.output)

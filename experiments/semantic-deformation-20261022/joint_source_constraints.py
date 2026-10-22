"""Complete original source gates for corrections of a frozen joint field.

The frozen QP retains its objective and constraints. This oracle additionally
checks the complete original continuous domains; newly found violating source
corners and curve points supply valid convex cuts. A finite cut list never
replaces continuous verification. No source geometry or budget is fitted here.
"""
import clarabel
import numpy as np
from scipy.sparse import csr_matrix, diags, eye, hstack, vstack

from joint_jaw_boundary_field import (GridCurve, OriginalProblem, PlateauProblem,
    _refine_regions, all_corner_inverse_check, check_jaw_prefix_intent,
    corner_gradient_rows, interface_chord_conditions, jaw_prefix_intent_blocks,
    source_sign_regions)


class SourceConstraintAudit:
    def __init__(self,reference,interface,mapping):
        mapping=np.asarray(mapping)
        grid=reference['grid']
        if (mapping.shape!=(grid['width']*grid['height'],) or mapping.dtype.kind not in 'iu'
                or np.any(mapping < -1)):
            raise ValueError('complete original source mapping required')
        active=mapping[mapping>=0]
        if not len(active) or not np.array_equal(np.sort(active),np.arange(len(active))):
            raise ValueError('original source variables must be independent and contiguous')
        self.size=len(active)
        self.problem={'grid':grid,'mapping':mapping.copy(),'hessian':eye(self.size,format='csc')}
        geometry=reference['source_geometry'];frame=geometry['frame']
        self.across=np.asarray(frame['across'],float);self.down=np.asarray(frame['down'],float)
        axes=np.array([self.across,self.down])
        if not np.allclose(axes@axes.T,np.eye(2),atol=1e-10,rtol=0.):
            raise ValueError('original orthonormal source frame required')
        self.amount=float(reference['amount'])
        if not np.isfinite(self.amount) or abs(self.amount)>100.:
            raise ValueError('finite original signed amount required')
        step=reference['interface_displacement_bending']['requested_node_spacing_px']
        length=reference['jaw_start_displacement_regularization']['length_px']
        radius=reference['jaw_radius_px']
        self.old=OriginalProblem(geometry,step,radius)
        preference=PlateauProblem(geometry,step,radius)
        self.jaws=[GridCurve(self.old.source(a,a['knots']),self.problem,self.across,self.down,
                            source_knots=a['knots']) for a in self.old.arcs]
        self.full=GridCurve(np.vstack([geometry['contour'],geometry['contour'][0]]),
                            self.problem,self.across,self.down)
        self.domains=[source_sign_regions(c,self.old.origin,self.old.midline,self.amount) for c in self.jaws]
        self.zero_rows=[c.op(np.unique(np.r_[0.,c.length,zeros])) for c,(_,zeros) in zip(self.jaws,self.domains)]
        self.specifications=[]
        for arc,curve in zip(self.old.arcs,self.jaws):
            regions=_refine_regions(self.old.convex_regions(arc),curve.chord_partition(radius),arc['side'])
            self.specifications.append((arc['id'],curve,regions,radius,.5))
        for segment in interface['segments']:
            curve=GridCurve(segment['source_points'],self.problem,self.across,self.down)
            self.specifications.extend(interface_chord_conditions(segment['id'],curve,
                reference['interface_radius_px'],reference['interface_source_relative_bend_band']))
        self.wanted=self.old.beta_area(self.amount,.13)
        if self.wanted!=reference['requested_jaw_and_full_contour_area_px2']:
            raise ValueError('original requested area changed')
        intent=reference['jaw_start_intent_constraints']
        self.prefix=jaw_prefix_intent_blocks(preference,self.jaws,self.amount,.13,length,step) if intent['enabled'] else []
        # Compare every original budget record, including its side, window,
        # outer scale and unsolved preference. Do not silently rebuild a
        # changed cap and subsequently report that the frozen cap was checked.
        if [b['report'] for b in self.prefix]!=intent['budgets']:
            raise ValueError('frozen original prefix budget differs from its unsolved preference')
        self.jaw_area=sum((a['direction']*c.area_row() for a,c in zip(self.old.arcs,self.jaws)),
                          csr_matrix((1,self.size)))
        self.full_area=self.full.area_row()
        self.seen_corners=set();self.seen_planes=set()

    def _vector(self,z):
        z=np.asarray(z,float)
        if z.shape!=(self.size,) or not np.isfinite(z).all():
            raise ValueError('finite original source scalar coefficients required')
        return z

    def audit(self,z):
        z=self._vector(z)
        curves=[{'id':name,**curve.continuous_chord_check(z,regions,radius,rho)[0]}
                for name,curve,regions,radius,rho in self.specifications]
        signs=[{'id':arc['id'],**curve.continuous_sign_check(z,None,regions)[0]}
               for arc,curve,(regions,_) in zip(self.old.arcs,self.jaws,self.domains)]
        zero_max=max(float(abs(row@z).max(initial=0.)) for row in self.zero_rows)
        jaw=float((self.jaw_area@z)[0]);full=float((self.full_area@z)[0])
        area_ok=max(abs(jaw-self.wanted),abs(full-self.wanted))<=max(1e-8,abs(self.wanted)*1e-10)
        geometry,_=all_corner_inverse_check(self.problem,z,self.across)
        prefix=check_jaw_prefix_intent(self.prefix,z)
        return {'curves':curves,'sign':signs,'zero_max_px':zero_max,'jaw_area_px2':jaw,
                'full_contour_area_px2':full,'requested_area_px2':self.wanted,
                'jaw_area_residual_px2':jaw-self.wanted,'full_area_residual_px2':full-self.wanted,
                'area_accepted':bool(area_ok),'all_source_corner_geometry':geometry,'jaw_start_intent':prefix,
                'accepted':bool(area_ok and zero_max<=1e-10 and geometry['accepted'] and prefix['accepted']
                                and not any(row['violations'] for row in curves+signs)),
                'scope':'all original source corners, continuous jaw/interface band/sign pieces, all original zero sites, both areas and independent prefix budgets; final map excluded'}

    def additional_cuts(self,z,total_variables):
        """Separate original conditions; auxiliary columns are always zero.

        Prefix cones and hard equalities already exist in the frozen base QP.
        Their failure remains an audit rejection, never a relaxed replacement.
        """
        z=self._vector(z)
        if total_variables<self.size:raise ValueError('complete frozen variable dimension required')
        rows=[];rhs=[];cones=[];new_planes=0
        def add_planes(matrix,lower):
            nonlocal new_planes
            for row,value in zip(matrix.tocsr(),np.asarray(lower)):
                scale=max(float(np.linalg.norm(row.data)),1.)
                row=row/scale;value=float(value)/scale
                # Full floating-point rows are deduplicated exactly. Rounding
                # the key can erase distinct near-endpoint valid cuts.
                key=(tuple(row.indices),tuple(row.data),value)
                if key in self.seen_planes:continue
                self.seen_planes.add(key);new_planes+=1
                rows.append(-row);rhs.append(np.array([-value]));cones.append(clarabel.NonnegativeConeT(1))
        for _,curve,regions,radius,rho in self.specifications:
            _,positions,signs=curve.continuous_chord_check(z,regions,radius,rho)
            if len(positions):
                ns,operator,_=curve.chord(positions,radius)
                add_planes(diags(signs)@operator,-signs*(1.-rho)*ns)
        for curve,(regions,_) in zip(self.jaws,self.domains):
            _,positions=curve.continuous_sign_check(z,None,regions)
            if len(positions):
                signs=-np.sign(self.amount*((curve.source(positions)-self.old.origin)@self.across-self.old.midline))
                add_planes(diags(signs)@curve.op(positions),np.zeros(len(positions)))
        _,bad=all_corner_inverse_check(self.problem,z,self.across)
        fresh=[tuple(map(int,index)) for index in bad if tuple(map(int,index)) not in self.seen_corners]
        if fresh:
            p_rows,b_rows=corner_gradient_rows(self.problem,fresh,self.across,self.down)
            zero=csr_matrix((1,self.size))
            for index,p,b in zip(fresh,p_rows,b_rows):
                rows.extend([vstack([zero,-1.04*p,-np.sqrt(.52/.48)*b]),
                             vstack([-3.6*p,zero,-np.sqrt(1.8/.8)*b]),
                             vstack([-.48*p-b,-.48*p+b])])
                rhs.extend([np.array([1.,.04,0.]),np.array([2.6,1.,0.]),np.array([.48,.48])])
                cones.extend([clarabel.SecondOrderConeT(3),clarabel.SecondOrderConeT(3),clarabel.NonnegativeConeT(2)])
                self.seen_corners.add(index)
        matrix=vstack(rows,format='csr') if rows else csr_matrix((0,self.size))
        return {'A':hstack([matrix,csr_matrix((matrix.shape[0],total_variables-self.size))],format='csc'),
                'b':np.concatenate(rhs) if rhs else np.empty(0),'cones':cones,
                'new_constraints':new_planes+len(fresh),'new_geometry_corners':len(fresh),
                'new_curve_planes':new_planes}

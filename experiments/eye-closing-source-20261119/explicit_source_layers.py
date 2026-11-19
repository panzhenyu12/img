"""Estimate source layers after a human confirms moving material identity.

This permission does NOT apply to automatic ridge/texture classification.
Color-model uncertainty is reported separately from absent source information.
The source foreground may mix a fine lash and its attached moving eyelid edge.
"""
import copy

import numpy as np

from robust_source_surface import infer_moving_source_layer
from source_estimator import triangle_weights


def infer_explicit_moving_source_layer(source,moving_support,ocular_donors,
        ocular_domain,noise,*,source_skin_prior,radius=6.0):
    """Infer a bounded source layer from explicit categorical material labels.

    `source_skin_prior` is a source-only appearance proposal, not a true layer
    or opacity. Neither this function nor its callees receives edited output,
    requested motion, true alpha, or a known hidden ocular background.

    `estimated` means that the underlay has source support and the layers are
    finite, physically bounded and reconstruct source RGB. It is not a claim
    that optical alpha was recovered. `modelUncertain` records an inadequate
    approximate palette fit without reversing the explicit material label.
    `unknown` remains blocking where actual source support or validity is absent.
    """
    baseline=infer_moving_source_layer(source,moving_support,ocular_donors,
                                      ocular_domain,noise,radius=radius)
    source=np.asarray(source,float);skin=np.asarray(source_skin_prior,float)
    if skin.shape!=source.shape or not np.isfinite(skin).all() or np.any((skin<0)|(skin>255)):
        raise ValueError('finite bounded source-sized skin appearance proposal is required')
    moving=np.asarray(moving_support);domain=np.asarray(ocular_domain)
    background=baseline['ocular'];lash=baseline['pigment']
    available=np.zeros(moving.shape,bool)
    for record in baseline['report']['sourceSurfaceFitRecords']:
        x,y=record['pixel'];available[y,x]=True

    # The foreground simplex is source skin + source lash. A fixed pure-RGB
    # fit chooses ocular/skin/lash weights; no target alpha or output tuning.
    weights,_=triangle_weights(source,background,skin,lash,np.zeros(moving.shape),0.)
    exposure=weights[...,0]
    front=weights[...,1,None]*skin+weights[...,2,None]*lash
    foreground_prior=np.divide(front,(1-exposure)[...,None],out=lash.copy(),
                               where=(1-exposure)[...,None]>1e-12)
    cap=np.ones(moving.shape)
    for channel in range(3):
        b=background[...,channel];c=source[...,channel]
        cap=np.minimum(cap,np.divide(c,b,out=np.ones_like(b),where=b>0))
        cap=np.minimum(cap,np.divide(255-c,255-b,out=np.ones_like(b),where=b<255))
    unconstrained_exposure=exposure.copy()
    exposure=np.minimum(exposure,cap)
    residual=source-exposure[...,None]*background
    reconstruction=residual+exposure[...,None]*background
    finite=np.isfinite(background).all(axis=-1)&np.isfinite(residual).all(axis=-1)&np.isfinite(exposure)
    bounded=np.all((background>=0)&(background<=255),axis=-1)&(exposure>=0)&(exposure<=1)
    bounded&=np.all(residual>=-1e-10,axis=-1)&np.all(residual<=255*(1-exposure[...,None])+1e-10,axis=-1)
    reconstructs=np.max(abs(reconstruction-source),axis=-1)<=1e-10
    estimated=moving&domain&available&finite&bounded&reconstructs
    unknown=moving&~estimated
    model=exposure[...,None]*background+(1-exposure[...,None])*foreground_prior
    error=np.sqrt(np.mean((source-model)**2,axis=-1))
    uncertain=estimated&(error>baseline['predictionScale'])
    report=copy.deepcopy(baseline['report'])
    report.update(method='explicit-source-ocular-skin-lash-robust-layers-v2',
        explicitMaterialIdentityRequired=True,automaticMaterialPermission=False,
        estimatedPixels=int(estimated.sum()),unknownPixels=int(unknown.sum()),
        modelUncertainPixels=int(uncertain.sum()),
        modelConflictPixels=int(uncertain.sum()),
        modelUncertaintyIsMaterialUnknown=False,
        finitePhysicalFailurePixels=int((moving&available&~(finite&bounded&reconstructs)).sum()),
        sourcePalette='ocular robust surface / source skin proposal / explicit source lash pigment',
        feasibilityAdjustedPixels=int((moving&available&(unconstrained_exposure>exposure+1e-12)).sum()),
        sourceReconstructionMaxError=float(np.max(abs(reconstruction-source))),
        sourceForegroundResidualMinimum=float(residual[estimated].min()) if estimated.any() else None,
        sourceForegroundResidualUpperViolation=float(np.max(residual[estimated]-255*(1-exposure[estimated,None]))) if estimated.any() else None,
        outputClipping=False,
        restriction='requires explicit source moving-material identity; fine lashes and their attached moving eyelid mixtures only; color-model error is diagnostic, not an identity veto')
    return dict(ocular=background,exposure=exposure,residual=residual,
        estimated=estimated,unknown=unknown,modelConsistent=estimated&~uncertain,
        modelUncertain=uncertain,fitError=error,predictionScale=baseline['predictionScale'],
        pigment=lash,foregroundPrior=foreground_prior,materialWeights=weights,report=report)

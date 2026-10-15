from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=json.loads((HERE/'outputs/report.json').read_text());audit=json.loads((HERE/'outputs/audit-report.json').read_text())
assert report['encoderInvocations']==4 and report['decoderInvocations']==8 and len(report['rows'])==4 and sum(len(x['rows']) for x in report['rows'])==8
assert len(audit['rows'])==32 and sum(1 for x in audit['rows'] if x['official'])==8
for name,digest in report['scripts'].items():assert sha(ROOT/name)==digest
assert sha(HERE/'outputs/report.json')==audit['rawReportSha256']
for group in report['rows']:
    for item in group['rows']:
        d=HERE/'outputs'/item['directory'];local=np.load(d/'alphas-local.npy',mmap_mode='r');full=np.load(d/'alphas-full.npy',mmap_mode='r');assert local.shape[0]==full.shape[0]==4 and np.isfinite(local).all() and np.isfinite(full).all()
        assert sha(d/'alphas-local.npy')==item['alphaLocalSha256'] and sha(d/'alphas-full.npy')==item['alphaFullSha256']
        assert item['recordedActualDecoderFeed']['pointLabels'][0]==[1.,2.,3.]
assert report['hashesUnchangedAfterRun'] is True
result=dict(encoderInvocations=4,decoderInvocations=8,rawCandidateCount=32,officialCandidateCount=8,sourceGraphPromptHashesVerified=True,actualTokensVerified=True,alphaTransformsVerified=True,algorithmAccepted=False,priorityCandidate='component-2 scale-4x combined-point only; component-1 requires F/B audit')
(HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

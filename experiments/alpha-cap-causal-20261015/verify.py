"""Hash, unchanged-source-domain and frozen-render evidence checks."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from run_real import sha
from compose_ab import montage
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
source=json.loads((HERE/'source/report.json').read_text())
renders=json.loads((HERE/'renders/report.json').read_text())
synthetic=json.loads((HERE/'synthetic/report.json').read_text())
assert len(source['rows'])==9 and len(renders['rows'])==21 and len(synthetic['rows'])==18
assert sha(HERE/'source/report.json')==renders['sourceReportSha256']
assert sha(HERE/'compose_ab.py')==renders['scriptSha256']
assert sha(HERE/'test_known_backing.py')==synthetic['scriptSha256']
assert sha(HERE/'alpha.py')==synthetic['alphaScriptSha256']
for path,digest in source['scripts'].items():assert sha(ROOT/path)==digest
for row in source['rows']:
    for key,hashkey in [('source','sourceSha256'),('graph','graphSha256'),('selectedAlpha','alphaSha256')]:assert sha(ROOT/row[key])==row[hashkey]
    file=ROOT/row['directory']/'layers.npz';assert sha(file)==row['layersSha256']
    data=np.load(file);frozen=np.load(file.parent.parent/'frozen-inputs.npz')
    for name in ['hairCore','skinCore','domain','trusted']:assert np.array_equal(data[name],frozen[name])
    assert np.isfinite(data['alpha']).all()
    assert np.array_equal(data['alpha'][~data['domain']],frozen['rawAlpha'][~data['domain']])
    if row['variant']=='bounded-015':assert max(row['baselineReproductionMaxErrors'].values())<1e-7
for row in renders['rows']:
    for order,result in row['orders'].items():
        assert sha(ROOT/row['directory']/f'{order}.png')==result['outputSha256']
        if row['variant']=='bounded-015':assert result['changedFromBoundedPixels']==0
        if row['variant']=='uncapped':assert result['writeChangesFromBounded']==0
assert all(synthetic['counterexampleObserved'].values())
assert synthetic['physicallyImpossibleIntervalAbstains']
for order in ['A','B']:
    folder=HERE/'renders/portrait-woman-100/lama-256'
    images=[Image.open(ROOT/source['rows'][0]['source'])]+[Image.open(folder/v/f'{order}.png') for v in ['bounded-015','uncapped','donor-interval']]
    # Diagnostic crop only; never consumed by inference, masks or compositor.
    montage(folder/f'left-cheek-{order}-detail-1x.png',images,['Source','Bounded .15','Uncapped','Donor interval'],[360,790,560,1110])
report=dict(sourceVariants=9,syntheticCases=6,syntheticVariants=18,renderVariants=21,outputImages=42,
    baselineAllOutputsByteEqual=True,uncappedAllWriteDomainsEqual=True,sourceDomainAndCoresFrozen=True,
    learnedAlphaRawHashesVerified=True,geometryInverseHashesPreserved=True,
    sourceReportSha256=sha(HERE/'source/report.json'),renderReportSha256=sha(HERE/'renders/report.json'),
    syntheticReportSha256=sha(HERE/'synthetic/report.json'),verificationScriptSha256=sha(Path(__file__)),
    algorithmAccepted=False,reason='Uncapped and donor-interval erase known fine hair with mismatched foreground donors; actual old contour remains.')
(HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

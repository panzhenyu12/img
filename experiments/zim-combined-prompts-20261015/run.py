"""Automatic source-only mixed-prompt ablation, retaining every raw candidate."""
from pathlib import Path
import argparse
import importlib.util
import json
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('component_probe', ROOT / 'experiments/hair-accessory-matting-zim-v2-20261015/run.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
zim, np, Image = probe.zim, probe.np, probe.Image
from adapter import CombinedPromptDecoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', nargs='+', choices=list(probe.SOURCES), default=list(probe.SOURCES))
    parser.add_argument('--out', type=Path, default=HERE / 'outputs')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    Encoder, Decoder, Zim, Predictor, vendor = zim.load_vendor()
    zim.torch.set_num_threads(3)
    weights = ROOT / '.models/zim-20261014'
    decoder = Decoder(str(weights / 'decoder.onnx'), num_threads=3)
    model = Zim(Encoder(str(weights / 'encoder.onnx'), num_threads=3), decoder)
    predictor = Predictor(model)
    adapters = {mode: CombinedPromptDecoder(decoder, attention=mode, bbox_attention=model.bbox_attn_mask) for mode in ['point','box']}
    report = {'scope':'Explicit mixed-prompt integration experiment, no production or source-graph modifications.',
              'vendorCommit':vendor['commit'], 'weights':{p.name:zim.sha(p) for p in weights.glob('*.onnx')},
              'scripts':{p.name:zim.sha(p) for p in [Path(__file__), HERE/'adapter.py']},
              'onnxInputs':[{'name':i.name,'shape':i.shape,'type':i.type} for i in decoder.ort_session.get_inputs()], 'images':[]}
    for name in args.names:
        source, graph = [ROOT/p for p in probe.SOURCES[name]]
        rgb = np.asarray(Image.open(source).convert('RGB')); h,w = rgb.shape[:2]
        params = json.loads(graph.read_text())['operations'][0]['params']
        masks = {k:zim.coverage(v,w,h) for k,v in params['semanticMasks'].items()}
        chin = max(p['y']*h for p in params['landmarks'])
        prompts,_ = probe.prompts_from_components(masks['hair'],masks['skin'],'hair',chin)
        folder = args.out/name; folder.mkdir()
        crop = probe.face_crop(params,w,h)
        start=time.perf_counter(); predictor.set_image(rgb)
        row = {'name':name,'source':str(source.relative_to(ROOT)),'sourceSha256':zim.sha(source),
               'graph':str(graph.relative_to(ROOT)),'graphSha256':zim.sha(graph),'faceCrop':crop,
               'encodeMs':(time.perf_counter()-start)*1000,'rows':[]}
        for prompt in prompts:
            for mode in ['upstream-mixed','box-only','points-only','combined-point','combined-box']:
                # A duplicate box-only output under the negative protocol is
                # retained; it establishes the controls without hidden caching.
                case=folder/(prompt['protocol']+'-'+str(prompt['component'])+'-'+mode);case.mkdir()
                coords=[prompt['positive']]+prompt['negative'];labels=[1]+[0]*len(prompt['negative'])
                kw={'multimask_output':True}
                if mode!='points-only':kw['box']=np.array(prompt['box'],dtype=np.float32)
                if mode!='box-only':
                    kw['point_coords']=np.array(coords,dtype=np.float32)
                    kw['point_labels']=np.array(labels,dtype=np.float32)
                model.decoder=adapters[mode.removeprefix('combined-')] if mode.startswith('combined-') else decoder
                start=time.perf_counter();alphas,scores,_=predictor.predict(**kw);elapsed=(time.perf_counter()-start)*1000
                assert alphas.shape==(4,h,w) and np.isfinite(alphas).all()
                np.save(case/'alphas.npy',alphas)
                summaries=[]
                skin=(masks['skin']>=.9)&(masks['hair']<.05)&(masks['accessories']<.05)
                hair=masks['hair']>=.9
                for index,a in enumerate(alphas):
                    probe.png(case/f'alpha-{index}.png',a)
                    summaries.append({'index':index,'score':float(scores[index]),'positiveAlpha':float(a[prompt['positive'][1],prompt['positive'][0]]),
                                      'negativeAlpha':[float(a[y,x]) for x,y in prompt['negative']],
                                      'parserCoreRetainedFraction':float((a[hair]>=.5).mean()),
                                      'skinCoreRetainedFraction':float((a[skin]>=.5).mean()),
                                      'skinCoreMeanAlpha':float(a[skin].mean()),'hairCoreMeanAlpha':float(a[hair].mean())})
                probe.make_panel(case/'face-alpha-1x.png',rgb,masks,alphas,scores,prompt,crop)
                item={'mode':mode,'prompt':prompt,'directory':str(case.relative_to(args.out)), 'decodeMs':elapsed,
                      'officialScoreArgmax':int(np.argmax(scores)),'alphas':summaries,'alphaSha256':zim.sha(case/'alphas.npy')}
                if mode.startswith('combined-'):item['actualDecoderPrompt']=model.decoder.last_prompt
                row['rows'].append(item)
                print(json.dumps({'name':name,'protocol':prompt['protocol'],'mode':mode,'best':summaries[item['officialScoreArgmax']]}),flush=True)
        report['images'].append(row)
        (args.out/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()

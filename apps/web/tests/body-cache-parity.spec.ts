import { expect, test } from '@playwright/test';
import fs from 'node:fs/promises';

test('身体前景缓存跨滑杆、图片、部位和保护变化与无缓存native逐字节一致', async ({page,request})=>{
  await page.goto('/');
  const fixture=await page.evaluate(()=>{
    const width=128,height=128;
    const ellipse=(cx:number,cy:number,rx:number,ry:number)=>Array.from({length:width*height},(_,i)=>{
      const r=Math.hypot((i%width+.5-cx)/rx,(Math.floor(i/width)+.5-cy)/ry);
      return Math.round(255*Math.max(0,Math.min(1,(1.04-r)/.08)));
    });
    const left=ellipse(42,64,17,45),right=ellipse(85,64,17,45),subject=left.map((v,i)=>Math.max(v,right[i]));
    const sources=[0,1].map(version=>{
      const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
      const ctx=canvas.getContext('2d')!,data=ctx.createImageData(width,height);
      for(let i=0;i<subject.length;i++){
        const x=i%width,y=Math.floor(i/width),a=subject[i]/255,bg=[210+(x%8<4?15:0),210,200];
        const fg=left[i]>0?[90+version*60+y%13,80+version*35,65]:[45,100+y%19,160];
        for(let c=0;c<3;c++)data.data[i*4+c]=Math.round(bg[c]*(1-a)+fg[c]*a);
        data.data[i*4+3]=255;
      }
      ctx.putImageData(data,0,0);return canvas.toDataURL('image/png').split(',')[1];
    });
    const raster=(data:number[])=>({shape:'raster',width,height,data});
    const mask={shape:'rect',region:{x:.1,y:.05,width:.8,height:.9}};
    const cases=[
      {source:0,part:0,strength:20,protect:false},
      {source:0,part:0,strength:60,protect:false},
      {source:0,part:1,strength:60,protect:false},
      {source:0,part:0,strength:60,protect:false},
      {source:1,part:0,strength:60,protect:false},
      {source:0,part:0,strength:60,protect:true},
      {source:0,part:0,strength:60,protect:false},
    ].map(c=>({...c,operation:{id:'body',type:'body.adjust',enabled:true,params:{
      region:mask.region,mask,bodyPart:c.part===0?'left-upper-arm':'right-upper-arm',
      subjectMask:raster(subject),partMask:raster(c.part===0?left:right),
      landmarks:[{x:(c.part===0?42:85)/width,y:20/height},{x:(c.part===0?42:85)/width,y:108/height}],
      adjustments:{arms:c.strength},
      ...(c.protect?{protectMask:{shape:'rect',region:{x:.2,y:.45,width:.2,height:.15}}}:{}),
    },controlPoints:[]}}));
    return {sources,cases};
  });
  const sessions=[]; const assets=[];
  for (let sourceIndex=0; sourceIndex<fixture.sources.length; sourceIndex++) {
    const created=await request.post('http://127.0.0.1:8080/api/v1/sessions');expect(created.ok()).toBeTruthy();
    const {id}=await created.json();
    const uploaded=await request.post(`http://127.0.0.1:8080/api/v1/sessions/${id}/assets`,{multipart:{file:{name:'body.png',mimeType:'image/png',buffer:Buffer.from(fixture.sources[sourceIndex],'base64')}}});
    expect(uploaded.ok(),await uploaded.text()).toBeTruthy(); sessions.push(id); assets.push(await uploaded.json());
  }
  const evidence=[];
  try {
    for(const [index,c] of fixture.cases.entries()){
      const graph={version:'1.0',sourceAssetId:assets[c.source].id,operations:[c.operation]};
      const native=await request.post(`http://127.0.0.1:8080/api/v1/sessions/${sessions[c.source]}/render`,{data:{editGraph:graph,format:'png'}});
      expect(native.ok(),await native.text()).toBeTruthy();
      const result=await page.evaluate(async({source,native,graph})=>{
        const {decodeImage,renderImage,getRenderSchedulerState}=await import('/src/core/engine.ts');
        const decode=(b64:string)=>decodeImage(new File([Uint8Array.from(atob(b64),c=>c.charCodeAt(0))],'body.png',{type:'image/png'}));
        const input=await decode(source),expected=await decode(native),start=performance.now();
        const actual=await renderImage(input,graph,{kind:'export'});
        return {ms:performance.now()-start,mismatches:actual.pixels.reduce((n,v,i)=>n+Number(v!==expected.pixels[i]),0),changed:actual.pixels.reduce((n,v,i)=>n+Number(v!==input.pixels[i]),0),workerGenerations:getRenderSchedulerState().workerGenerations};
      },{source:fixture.sources[c.source],native:(await native.body()).toString('base64'),graph});
      expect(result.mismatches,`case ${index}`).toBe(0);expect(result.changed).toBeGreaterThan(0);
      expect(result.workerGenerations).toBe(1);evidence.push({index,source:c.source,part:c.part,strength:c.strength,protect:c.protect,...result});
    }
    const overlap=await page.evaluate(async({source,operation})=>{
      const {decodeImage,renderImage,getRenderSchedulerState}=await import('/src/core/engine.ts');
      const input=await decodeImage(new File([Uint8Array.from(atob(source),c=>c.charCodeAt(0))],'body.png',{type:'image/png'}));
      const graph=(strength:number)=>({version:'1.0',sourceAssetId:'slider',operations:[{...operation,params:{...operation.params,adjustments:{arms:strength}}}]});
      // React effect cleanup aborts before the replacement debounce dispatch.
      const controller=new AbortController();
      const first=renderImage(input,graph(20),{signal:controller.signal});
      controller.abort();
      await new Promise(resolve=>setTimeout(resolve,75));
      const latest=renderImage(input,graph(60));
      const results=await Promise.allSettled([first,latest]);
      const actual=results[1].status==='fulfilled'?results[1].value:undefined;
      const reference=await renderImage(input,graph(60),{kind:'export'});
      return {first:results[0].status,error:results[0].status==='rejected'?results[0].reason.name:null,last:results[1].status,
        mismatches:actual?actual.pixels.reduce((n,v,i)=>n+Number(v!==reference.pixels[i]),0):-1,
        generations:getRenderSchedulerState().workerGenerations};
    },{source:fixture.sources[0],operation:fixture.cases[0].operation});
    expect(overlap).toEqual({first:'rejected',error:'AbortError',last:'fulfilled',mismatches:0,generations:1});
    await fs.writeFile(test.info().outputPath('body-cache-parity.json'),JSON.stringify(evidence,null,2));
  } finally { await Promise.all(sessions.map(id=>request.delete(`http://127.0.0.1:8080/api/v1/sessions/${id}`))); }
});

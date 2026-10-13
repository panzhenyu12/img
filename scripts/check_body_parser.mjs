/** Real ORT-Web inference + portable input tensor for OpenCV comparison. */
import {chromium} from '../apps/web/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
const root=path.resolve(import.meta.dirname,'..'),id=process.env.BODY_PARSER_ID??'fullbody-side-fashion';
const source=await fs.readFile(path.join(root,'test/results/body-contour-v2',id,'source.png'));
const browser=await chromium.launch({headless:true});
try {
 const page=await browser.newPage();await page.goto(process.env.RETOUCH_WEB_URL??'http://127.0.0.1:5173/');
 const result=JSON.parse(await page.evaluate(async encoded=>{
  const image=new Image();image.src='data:image/png;base64,'+encoded;await image.decode();const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;canvas.getContext('2d').drawImage(image,0,0);
  const {bodyInput,parseBodyClasses,bodyParserRuntime}=await import('/src/core/bodyParsing.ts');
  const input=bodyInput(canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data,canvas.width,canvas.height);
  const bytes=new Uint8Array(input.buffer);let binary='';for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
  const started=performance.now(),classes=await parseBodyClasses(canvas);const elapsedMs=performance.now()-started;
  return JSON.stringify({input:btoa(binary),classes,elapsedMs,width:canvas.width,height:canvas.height,runtime:bodyParserRuntime(),provider:'wasm',modelPath:'/models/bodyparser.onnx'});
 },source.toString('base64')));
 const dir=path.join(root,'test/results/body-parser-browser',id);await fs.mkdir(dir,{recursive:true});await fs.writeFile(path.join(dir,'input.f32'),Buffer.from(result.input,'base64'));delete result.input;
 await fs.writeFile(path.join(dir,'classes.json'),JSON.stringify(result.classes));delete result.classes;
 await fs.writeFile(path.join(root,'doc/qa/body-parser-browser-runtime.json'),JSON.stringify({id,...result},null,2));console.log(JSON.stringify({id,...result}));
}finally {await browser.close();}

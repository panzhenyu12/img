import {expect,test} from '@playwright/test';
import fs from 'node:fs/promises';

test('像素级保护蒙版可显示、手工擦除，并随工程和撤销保留',async({page})=>{
  await page.goto('/');
  const encoded=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=c.height=64;const g=c.getContext('2d')!;g.fillStyle='rgb(100,100,100)';g.fillRect(0,0,64,64);return c.toDataURL().split(',')[1];});
  await page.locator('input[accept="image/jpeg,image/png,image/webp"]').setInputFiles({name:'raster.png',mimeType:'image/png',buffer:Buffer.from(encoded,'base64')});
  const data=Array.from({length:64*64},(_,i)=>i%64<32?255:0);
  const graph={version:'1.0',operations:[{id:'raster-protect',type:'color.basic',enabled:true,label:'像素保护测试',params:{brightness:40,protectMask:{shape:'raster',width:64,height:64,data}}}]};
  await page.locator('input[accept=".json"]').setInputFiles({name:'raster.edit.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(graph))});
  await expect(page.locator('.statusbar')).toContainText('操作工程已载入');
  await expect(page.locator('.canvas-overlay image')).toHaveCount(1);
  const read=async(x:number)=>page.locator('canvas[aria-label="图片预览"]').evaluate((node,x)=>(node as HTMLCanvasElement).getContext('2d')!.getImageData(x,32,1,1).data[0],x);
  expect(await read(16)).toBe(100);expect(await read(48)).toBeGreaterThan(100);
  await page.getByRole('button',{name:'蒙版与冻结',exact:true}).click();
  await page.getByRole('button',{name:'擦除保护',exact:true}).click();
  const box=(await page.locator('.canvas-overlay').boundingBox())!;
  await page.mouse.move(box.x+box.width*.25,box.y+box.height*.5);await page.mouse.down();await page.mouse.up();
  await expect.poll(()=>read(16)).toBeGreaterThan(100);
  const[download]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:'保存工程',exact:true}).click()]);
  const saved=JSON.parse(await fs.readFile((await download.path())!,'utf8')).operations[0].params.protectMask;
  expect(saved.shape).toBe('raster');expect(saved.data).toEqual(data);expect(saved.strokes.some((s:{erase:boolean})=>s.erase)).toBe(true);
  await page.getByRole('button',{name:'撤销',exact:true}).click();await expect.poll(()=>read(16)).toBe(100);
  await page.getByRole('button',{name:'重做',exact:true}).click();await expect.poll(()=>read(16)).toBeGreaterThan(100);
});

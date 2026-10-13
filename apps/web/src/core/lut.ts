export interface CubeLUT {
  title: string;
  size: number;
  data: number[][];
  domainMin: number[];
  domainMax: number[];
}

/** Parse the open .cube 3D LUT format, with red as the fastest varying axis. */
export function parseCube(text: string): CubeLUT {
  let title = '自定义 LUT';
  let size = 0;
  const data: number[][] = [];
  let domainMin = [0, 0, 0];
  let domainMax = [1, 1, 1];
  for (const original of text.split(/\r?\n/)) {
    const line = original.replace(/#.*$/, '').trim();
    if (!line) continue;
    const [key, ...rest] = line.split(/\s+/);
    if (key === 'TITLE') { title = rest.join(' ').replace(/^"|"$/g, ''); continue; }
    if (key === 'LUT_1D_SIZE') throw new Error('请导入 3D .cube LUT');
    if (key === 'LUT_3D_SIZE') {
      size = Number(rest[0]);
      if (!Number.isInteger(size) || size < 2 || size > 65) throw new Error('LUT 尺寸需要在 2～65 之间');
      continue;
    }
    if (key === 'DOMAIN_MIN' || key === 'DOMAIN_MAX') {
      const values = rest.map(Number);
      if (values.length !== 3 || !values.every(Number.isFinite)) throw new Error('LUT DOMAIN 格式错误');
      if (key === 'DOMAIN_MIN') domainMin = values;
      else domainMax = values;
      continue;
    }
    const values = [key, ...rest].map(Number);
    if (values.length !== 3 || !values.every(Number.isFinite)) throw new Error('LUT 数据行需要三个有限数值');
    data.push(values);
  }
  if (!size || data.length !== size ** 3) throw new Error(`LUT 需要 ${size ** 3} 个颜色点，实际 ${data.length}`);
  if (domainMin.some((v, i) => v >= domainMax[i])) throw new Error('LUT DOMAIN_MAX 必须大于 DOMAIN_MIN');
  return { title, size, data, domainMin, domainMax };
}

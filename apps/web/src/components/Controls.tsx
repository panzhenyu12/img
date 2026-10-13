import type { Parameter } from '../core/registry';
export function Slider({ label, value, min = -100, max = 100, step = 1, defaultValue, onChange }: { label: string; value: number; min?: number; max?: number; step?: number; defaultValue?: number; onChange: (value: number) => void }) {
  const reset = defaultValue === undefined ? undefined : () => onChange(defaultValue);
  const title = reset ? `双击恢复默认值 ${defaultValue}` : undefined;
  return <label className="slider-field"><span onDoubleClick={reset} title={title}>{label}</span><input aria-label={label} title={title} type="range" min={min} max={max} step={step} value={value} onDoubleClick={reset} onChange={(event) => onChange(Number(event.target.value))} /><input className="slider-number" aria-label={`${label}数值`} title={title} type="number" min={min} max={max} step={step} value={value} onDoubleClick={reset} onChange={(event) => onChange(Math.max(min, Math.min(max, Number(event.target.value))))} /></label>;
}
export function ParameterGroups({ parameters, values, onChange }: { parameters: Parameter[]; values: Record<string, unknown>; onChange: (key: string, value: number) => void }) {
  const groups = [...new Set(parameters.map((item) => item.group))];
  return <>{groups.map((group) => <details className="parameter-group" key={group} open><summary>{group}</summary>{parameters.filter((item) => item.group === group).map((item) => <Slider key={item.key} label={item.label} min={item.min} max={item.max} step={item.key === 'gamma' ? .1 : 1} defaultValue={item.default} value={Number(values[item.key] ?? item.default)} onChange={(value) => onChange(item.key, value)} />)}</details>)}</>;
}
export function NumberField({ label, value, min = 0, max, step = 1, onChange }: { label: string; value: number; min?: number; max?: number; step?: number; onChange: (value: number) => void }) {
  return <label className="number-field"><span>{label}</span><input aria-label={label} type="number" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}

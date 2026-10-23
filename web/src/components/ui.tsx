// Small shared pieces: LaTeX, sliders, segmented controls, formula cards with their sources.

import katex from 'katex'
import 'katex/dist/katex.min.css'
import { useEffect, useMemo, useState } from 'react'
import type { BibEntry, Formula, ScriptView } from '../api'
import { pythonScript, useFormulas, useReferences } from '../api'
import { fmt } from '../lib/colormap'

export function Tex({ src, block = false }: { src: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(src, { displayMode: block, throwOnError: false }), [src, block])
  return <span className={block ? 'tex tex-block' : 'tex'} dangerouslySetInnerHTML={{ __html: html }} />
}

export function Slider({
  label,
  tex,
  value,
  min,
  max,
  step,
  onChange,
  format = (v) => fmt(v),
  testId,
}: {
  label: string
  tex?: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  format?: (v: number) => string
  testId?: string
}) {
  return (
    <label className="slider">
      <span className="slider-head">
        <span>{tex ? <Tex src={tex} /> : null} {label}</span>
        <span className="mono">{format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        data-testid={testId}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  )
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
  label: string
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={o.value === value}
          className={o.value === value ? 'is-active' : ''}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

function sourceLink(entry: BibEntry | undefined): string | undefined {
  if (entry === undefined) return undefined
  if (entry.doi) return `https://doi.org/${entry.doi}`
  if (entry.eprint) return `https://arxiv.org/abs/${entry.eprint}`
  return entry.url
}

export function shortCite(entry: BibEntry | undefined, key: string): string {
  if (entry === undefined) return key
  const first = (entry.author ?? key).split(' and ')[0]?.split(',')[0]?.replace(/[{}\\'"]/g, '') ?? key
  const etal = (entry.author ?? '').includes(' and ') ? ' et al.' : ''
  return `${first}${etal} ${entry.year ?? ''}`.trim()
}

export function SourceList({ sources }: { sources: Formula['sources'] }) {
  const refs = useReferences().data
  return (
    <ul className="sources">
      {sources.map((s) => {
        const entry = refs?.[s.key]
        const href = sourceLink(entry)
        return (
          <li key={s.key + s.locator}>
            <span className={`kind kind-${s.kind}`}>{s.kind.replace('_', ' ')}</span>{' '}
            {href ? (
              <a href={href} target="_blank" rel="noreferrer">
                {shortCite(entry, s.key)}
              </a>
            ) : (
              shortCite(entry, s.key)
            )}
            , {s.locator}
            {s.note ? <span className="muted">. {s.note}</span> : null}
          </li>
        )
      })}
    </ul>
  )
}

/** A formula with its sources, and optionally the theory and simulation numbers side by side. */
export function FormulaCard({
  formulaKey,
  theory,
  sim,
  unit = '',
  simLabel,
}: {
  formulaKey: string
  theory?: number | null
  sim?: number | null
  unit?: string
  simLabel?: string
}) {
  const f = useFormulas().data?.find((x) => x.key === formulaKey)
  if (f === undefined) return null
  const rel = theory != null && sim != null && theory !== 0 ? Math.abs(sim - theory) / Math.abs(theory) : null
  return (
    <section className="formula-card" data-formula={f.key}>
      <h4>{f.title}</h4>
      <Tex src={f.latex} block />
      {theory !== undefined || sim !== undefined ? (
        <div className="vs">
          {theory !== undefined ? (
            <div>
              <span className="tag tag-theory">theory</span> <span className="mono">{fmt(theory)}{unit}</span>
            </div>
          ) : null}
          {sim !== undefined ? (
            <div>
              <span className="tag tag-sim">simulation</span> <span className="mono">{fmt(sim)}{unit}</span>
              {simLabel ? <span className="muted small"> {simLabel}</span> : null}
            </div>
          ) : null}
          {rel !== null ? <div className="muted small">differ by {fmt(rel * 100, 2)}%</div> : null}
        </div>
      ) : null}
      <p className="small muted">Valid when: {f.valid_when}</p>
      <SourceList sources={f.sources} />
    </section>
  )
}

export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) return null
  return (
    <div className="error-banner" role="alert">
      The engine rejected the input or is unreachable: {error instanceof Error ? error.message : String(error)}
    </div>
  )
}

/** "Copy as Python": fetch the script for this view at its current inputs, copy it, and show it. */
export function CopyPython({ view, inputs }: { view: ScriptView; inputs: unknown }) {
  const [code, setCode] = useState<string | null>(null)
  const [state, setState] = useState<'idle' | 'copied' | 'error'>('idle')
  const copy = async () => {
    try {
      const { python } = await pythonScript(view, inputs)
      setCode(python)
      await navigator.clipboard?.writeText(python)
      setState('copied')
    } catch {
      setState('error')
    }
    window.setTimeout(() => setState('idle'), 1500)
  }
  return (
    <div className="copy-python">
      <h3>Take it with you</h3>
      <button className="btn" onClick={() => void copy()} data-testid="copy-python">
        {state === 'copied' ? 'Copied' : state === 'error' ? 'Could not copy' : 'Copy as Python'}
      </button>
      {code ? (
        <pre className="code" data-testid="python-code">
          {code}
        </pre>
      ) : null}
    </div>
  )
}

/** What is running and for how long, so a long first solve never looks frozen. */
export function Solving({ what = 'solving', boot = false }: { what?: string; boot?: boolean }) {
  const [start] = useState(() => performance.now())
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    const id = window.setInterval(() => setSeconds((performance.now() - start) / 1000), 100)
    return () => window.clearInterval(id)
  }, [start])
  const line = (
    <span>
      <span className="spinner" aria-hidden /> {what} <span className="mono">{seconds.toFixed(1)} s</span>
    </span>
  )
  if (!boot) return <div className="solving muted" role="status" data-testid="solving">{line}</div>
  return (
    <div className="boot" role="status" data-testid="solving">
      <div className="solving-boot">
        {line}
        {seconds > 3 ? (
          <p className="small muted">The first run of a simulation compiles it, which can take 10 to 30 s. Later runs reuse it.</p>
        ) : null}
      </div>
    </div>
  )
}

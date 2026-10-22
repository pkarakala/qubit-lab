// Playground: type your own Hamiltonian and losses for one oscillator mode, solve them with
// Dynamiqs, and take the script home. No theory layer here; the presets are the lessons.

import { useEffect, useState } from 'react'
import { type Initial, type PlaygroundSpec, usePlayground } from '../api'
import { CONFIG, Plot, SERIES, layout } from '../components/Plot'
import { ErrorBanner, Segmented, Slider } from '../components/ui'
import { fmt } from '../lib/colormap'
import { useExplorer } from '../store'
import { WignerScene } from '../three/WignerSurface'

interface Preset {
  id: string
  label: string
  blurb: string
  spec: PlaygroundSpec
}

const base: PlaygroundSpec = {
  hamiltonian: '0',
  jumps: [],
  params: {},
  dim: 30,
  initial: 'vacuum',
  alpha_sq: 4,
  t_final: 3,
  n_frames: 41,
  grid: 61,
  extent: 4,
}

export const PLAYGROUND_PRESETS: Preset[] = [
  {
    id: 'two-photon',
    label: 'Two-photon cat',
    blurb: 'The same model as the Phase space view, written out by hand. Edit it and see what breaks.',
    spec: {
      ...base,
      jumps: [
        { op: 'a**2 - alpha2', rate: '1' },
        { op: 'a', rate: 'kappa_1' },
      ],
      params: { alpha2: 4, kappa_1: 0.05 },
    },
  },
  {
    id: 'kerr',
    label: 'Kerr cat',
    blurb: 'A Hamiltonian with |±α⟩ as degenerate states, α² = eps2 / K. No dissipation, so vacuum sloshes.',
    spec: { ...base, hamiltonian: '-K*ad**2*a**2 + eps2*(ad**2 + a**2)', params: { K: 1, eps2: 4 } },
  },
  {
    id: 'revival',
    label: 'Kerr revival',
    blurb: 'A coherent state under (K/2) a†²a² smears out, then comes back at K t = 2π, when every phase is a whole turn.',
    spec: { ...base, hamiltonian: 'K/2*ad**2*a**2', params: { K: 1 }, initial: 'coherent_plus', t_final: 2 * Math.PI, n_frames: 61 },
  },
  {
    id: 'driven',
    label: 'Driven, damped',
    blurb: 'A drive against loss settles in a coherent state with ⟨n⟩ = (2 eps / kappa)² = 4.',
    spec: { ...base, hamiltonian: 'eps*(a + ad)', jumps: [{ op: 'a', rate: 'kappa' }], params: { eps: 1, kappa: 1 }, t_final: 8 },
  },
  {
    id: 'squeeze',
    label: 'Squeezing',
    blurb: 'A two-photon drive i(r/2)(a†² − a²) squeezes vacuum into an ellipse.',
    spec: { ...base, hamiltonian: '1j*r/2*(ad**2 - a**2)', params: { r: 1 }, t_final: 1.2, dim: 50 },
  },
  {
    id: 'lossy-cat',
    label: 'Unprotected cat',
    blurb: 'An even cat in a lossy cavity. The fringes go first, then the blobs fall to vacuum.',
    spec: { ...base, jumps: [{ op: 'a', rate: 'kappa' }], params: { kappa: 0.5 }, initial: 'even_cat', t_final: 4 },
  },
]

const INITIALS: { value: Initial; label: string }[] = [
  { value: 'vacuum', label: 'Vacuum' },
  { value: 'coherent_plus', label: '|α⟩' },
  { value: 'even_cat', label: 'Even cat' },
  { value: 'odd_cat', label: 'Odd cat' },
]

export function Playground() {
  const theme = useExplorer((s) => s.theme)
  const c = SERIES[theme]
  const [draft, setDraft] = useState<PlaygroundSpec>(PLAYGROUND_PRESETS[0]?.spec ?? base)
  const [submitted, setSubmitted] = useState<PlaygroundSpec>(draft)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [copied, setCopied] = useState(false)
  const q = usePlayground(submitted)
  const data = q.data
  const n = data?.sim.times.length ?? 1
  const f = Math.min(frame, n - 1)
  const t = data?.sim.times ?? []
  const edge = Math.max(...(data?.sim.edge_population ?? [0]))
  const dirty = JSON.stringify(draft) !== JSON.stringify(submitted)
  const set = (p: Partial<PlaygroundSpec>) => setDraft((d) => ({ ...d, ...p }))
  const run = (spec: PlaygroundSpec) => {
    setDraft(spec)
    setSubmitted(spec)
    setFrame(0)
    setPlaying(true)
  }

  useEffect(() => {
    if (!playing || !data) return
    const id = window.setInterval(() => {
      setFrame((k) => {
        if (k >= n - 1) {
          setPlaying(false)
          return k
        }
        return k + 1
      })
    }, 90)
    return () => window.clearInterval(id)
  }, [playing, data, n])

  const params = Object.entries(draft.params)
  const setParam = (i: number, name: string, value: number) =>
    set({ params: Object.fromEntries(params.map(([k, v], j) => (j === i ? [name, value] : [k, v]))) })

  return (
    <div className="view">
      <aside className="pane pane-left">
        <h3>Start from a model</h3>
        <div className="presets">
          {PLAYGROUND_PRESETS.map((p) => (
            <button
              key={p.id}
              className={`preset ${JSON.stringify(p.spec) === JSON.stringify(submitted) ? 'is-active' : ''}`}
              onClick={() => run(p.spec)}
              data-testid={`playground-preset-${p.id}`}
            >
              <strong>{p.label}</strong>
              <span className="muted small">{p.blurb}</span>
            </button>
          ))}
        </div>
        <h3>Initial state</h3>
        <Segmented label="Initial state" value={draft.initial} options={INITIALS} onChange={(initial) => set({ initial })} />
        <Slider label="initial size" tex="|\alpha|^2" value={draft.alpha_sq} min={0} max={9} step={0.5} onChange={(alpha_sq) => set({ alpha_sq })} />
        <Slider label="duration" tex="t" value={draft.t_final} min={0.2} max={20} step={0.1} onChange={(t_final) => set({ t_final })} />
        <Slider label="Fock states kept" tex="N" value={draft.dim} min={10} max={60} step={1} onChange={(dim) => set({ dim })} format={(v) => String(v)} />
        <Slider label="Wigner window" tex="x_{max}" value={draft.extent} min={2} max={7} step={0.5} onChange={(extent) => set({ extent })} />
      </aside>

      <main className="pane pane-center">
        <ErrorBanner error={q.error} />
        <div className="viewport">
          {data ? <WignerScene xvec={data.sim.xvec} w={data.sim.wigner[f] ?? []} ghost={null} theme={theme} /> : <div className="boot">Solving your model…</div>}
          <div className="hud">
            <div>
              <span className="muted">time</span> <span className="mono">t = {fmt(t[f] ?? 0)}</span>
            </div>
            <div>
              <span className="muted">⟨n⟩</span> <span className="mono">{fmt(data?.sim.photon_number[f])}</span>
            </div>
            <div>
              <span className="muted">parity</span> <span className="mono">{fmt(data?.sim.parity[f])}</span>
            </div>
            <div>
              <span className="muted">purity</span> <span className="mono">{fmt(data?.sim.purity[f])}</span>
            </div>
            {q.isFetching ? <div className="muted">solving…</div> : null}
          </div>
        </div>
        <div className="transport">
          <button className="btn" onClick={() => (f >= n - 1 ? (setFrame(0), setPlaying(true)) : setPlaying(!playing))}>
            {playing ? 'Pause' : 'Play'}
          </button>
          <input
            type="range"
            min={0}
            max={n - 1}
            value={f}
            onChange={(e) => {
              setPlaying(false)
              setFrame(Number(e.target.value))
            }}
            aria-label="time"
          />
        </div>
        <div className="dock">
          {data ? (
            <Plot
              data={[
                { x: t, y: data.sim.photon_number, name: '⟨n⟩', line: { color: c.a, width: 2 } },
                { x: t, y: data.sim.parity, name: 'parity', line: { color: c.b, width: 2 }, yaxis: 'y2' },
                { x: t, y: data.sim.purity, name: 'purity', line: { color: c.c, width: 2 }, yaxis: 'y2' },
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 't (in the units of your rates)' }, range: [0, t[t.length - 1] ?? 1] },
                yaxis: { title: { text: 'photons' }, rangemode: 'tozero' },
                yaxis2: { overlaying: 'y', side: 'right', range: [-1.05, 1.05], showgrid: false },
                shapes: [{ type: 'line', x0: t[f], x1: t[f], yref: 'paper', y0: 0, y1: 1, line: { color: c.muted, width: 1 } }],
                margin: { l: 56, r: 40, t: 10, b: 40 },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : null}
        </div>
      </main>

      <aside className="pane pane-right">
        <h3>Your model</h3>
        <label className="field">
          <span className="small">Hamiltonian H</span>
          <textarea rows={2} value={draft.hamiltonian} onChange={(e) => set({ hamiltonian: e.target.value })} data-testid="playground-h" spellCheck={false} />
        </label>
        <span className="small">Losses: rate × D[operator]</span>
        {draft.jumps.map((j, i) => (
          <div className="row" key={i}>
            <input value={j.rate} onChange={(e) => set({ jumps: draft.jumps.map((x, k) => (k === i ? { ...x, rate: e.target.value } : x)) })} style={{ width: '35%' }} aria-label="rate" spellCheck={false} />
            <input value={j.op} onChange={(e) => set({ jumps: draft.jumps.map((x, k) => (k === i ? { ...x, op: e.target.value } : x)) })} aria-label="jump operator" spellCheck={false} />
            <button className="btn" onClick={() => set({ jumps: draft.jumps.filter((_, k) => k !== i) })} aria-label="remove loss">
              ×
            </button>
          </div>
        ))}
        <button className="btn" onClick={() => set({ jumps: [...draft.jumps, { op: 'a', rate: '0.1' }] })} disabled={draft.jumps.length >= 6}>
          Add loss
        </button>
        <h3>Parameters</h3>
        {params.map(([k, v], i) => (
          <div className="row" key={i}>
            <input value={k} onChange={(e) => setParam(i, e.target.value, v)} style={{ width: '45%' }} aria-label="parameter name" spellCheck={false} />
            <input type="number" value={v} step="any" onChange={(e) => setParam(i, k, Number(e.target.value))} aria-label="parameter value" />
            <button className="btn" onClick={() => set({ params: Object.fromEntries(params.filter((_, j) => j !== i)) })} aria-label="remove parameter">
              ×
            </button>
          </div>
        ))}
        <button className="btn" onClick={() => set({ params: { ...draft.params, [`p${params.length + 1}`]: 1 } })} disabled={params.length >= 12}>
          Add parameter
        </button>
        <div className="row" style={{ marginTop: 10 }}>
          <button className="btn" onClick={() => run(draft)} disabled={!dirty} data-testid="playground-run">
            {dirty ? 'Run' : 'Up to date'}
          </button>
        </div>
        <p className="small muted">
          Write with a, ad (a†), n and I, your parameters, numbers (1j for i), + − * /, ** with a whole number, and
          sqrt(). a*ad is a matrix product. A number added to an operator means number × I.
        </p>
        {data && edge > 1e-3 ? (
          <p className="small warn" data-testid="truncation-warning">
            Up to {fmt(edge * 100, 2)}% of the state sits in the top two Fock states. Keep more states or the result is
            wrong.
          </p>
        ) : null}
        <h3>Take it with you</h3>
        <button
          className="btn"
          disabled={!data}
          onClick={() => {
            if (!data) return
            void navigator.clipboard?.writeText(data.python).then(() => setCopied(true))
            window.setTimeout(() => setCopied(false), 1500)
          }}
          data-testid="copy-python"
        >
          {copied ? 'Copied' : 'Copy as Python'}
        </button>
        {data ? <pre className="code" data-testid="python-code">{data.python}</pre> : null}
      </aside>
    </div>
  )
}

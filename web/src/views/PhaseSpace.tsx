// Phase space: one cat qubit's Wigner function evolving under the Lindblad equation (Dynamiqs),
// with the closed-form ideal cat as an optional ghost, and the observables over time below.

import { useEffect } from 'react'
import { type Initial, type Scheme, useEvolution } from '../api'
import { CopyPython, ErrorBanner, FormulaCard, Segmented, Slider } from '../components/ui'
import { CONFIG, Plot, SERIES, layout } from '../components/Plot'
import { fmt } from '../lib/colormap'
import { PRESETS, useExplorer } from '../store'
import { WignerScene } from '../three/WignerSurface'

const INITIALS: { value: Initial; label: string }[] = [
  { value: 'vacuum', label: 'Vacuum' },
  { value: 'coherent_plus', label: '|+α⟩' },
  { value: 'even_cat', label: 'Even cat' },
  { value: 'odd_cat', label: 'Odd cat' },
]

const SCHEMES: { value: Scheme; label: string }[] = [
  { value: 'dissipative', label: 'Two-photon' },
  { value: 'kerr', label: 'Kerr' },
  { value: 'none', label: 'None' },
]

function usePlayback(nFrames: number) {
  const playing = useExplorer((s) => s.playing)
  useEffect(() => {
    if (!playing) return
    const id = window.setInterval(() => {
      const s = useExplorer.getState()
      if (s.frame >= nFrames - 1) s.setPlaying(false)
      else s.setFrame(s.frame + 1)
    }, 90)
    return () => window.clearInterval(id)
  }, [playing, nFrames])
}

export function PhaseSpace() {
  const theme = useExplorer((s) => s.theme)
  const params = useExplorer((s) => s.params)
  const setParams = useExplorer((s) => s.setParams)
  const frame = useExplorer((s) => s.frame)
  const setFrame = useExplorer((s) => s.setFrame)
  const playing = useExplorer((s) => s.playing)
  const setPlaying = useExplorer((s) => s.setPlaying)
  const showTheory = useExplorer((s) => s.showTheory)
  const toggleTheory = useExplorer((s) => s.toggleTheory)
  const q = useEvolution(params)
  const data = q.data
  const nFrames = data?.sim.times.length ?? 1
  usePlayback(nFrames)
  const f = Math.min(frame, nFrames - 1)
  const parityNow = data?.sim.parity[f] ?? 1
  const ghost = showTheory && data ? (parityNow >= 0 ? data.theory.even_cat_wigner : data.theory.odd_cat_wigner) : null
  const c = SERIES[theme]
  const t = data?.sim.times ?? []
  const preset = PRESETS.find((p) => JSON.stringify(p.params) === JSON.stringify(params))

  return (
    <div className="view">
      <aside className="pane pane-left">
        <h3>Start from a lesson</h3>
        <div className="presets">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              className={`preset ${preset?.id === p.id ? 'is-active' : ''}`}
              onClick={() => useExplorer.setState({ params: p.params, frame: 0, playing: true })}
            >
              <strong>{p.label}</strong>
              <span className="muted small">{p.blurb}</span>
            </button>
          ))}
        </div>
        <h3>Stabilization</h3>
        <Segmented label="Stabilization" value={params.scheme} options={SCHEMES} onChange={(scheme) => setParams({ scheme })} />
        <h3>Initial state</h3>
        <Segmented label="Initial state" value={params.initial} options={INITIALS} onChange={(initial) => setParams({ initial })} />
        <h3>Knobs</h3>
        <Slider label="cat size" tex="|\alpha|^2" value={params.alpha_sq} min={0.5} max={8} step={0.5} onChange={(alpha_sq) => setParams({ alpha_sq })} testId="alpha-sq" />
        <Slider label="photon loss" tex="\kappa_1/\kappa_2" value={params.kappa_1} min={0} max={0.5} step={0.01} onChange={(kappa_1) => setParams({ kappa_1 })} />
        <Slider label="dephasing" tex="\kappa_\phi/\kappa_2" value={params.kappa_phi} min={0} max={0.5} step={0.01} onChange={(kappa_phi) => setParams({ kappa_phi })} />
        <Slider label="thermal photons" tex="n_{th}" value={params.n_th} min={0} max={0.2} step={0.01} onChange={(n_th) => setParams({ n_th })} />
        <Slider label="duration" tex="\kappa_2 t" value={params.t_final} min={0.5} max={20} step={0.5} onChange={(t_final) => setParams({ t_final })} />
        <p className="small muted">
          Rates are in units of the stabilization rate (κ₂, or K for the Kerr cat). Each change runs a Dynamiqs solve
          with {data?.sim.n_fock ?? '…'} Fock states; results are cached.
        </p>
        <CopyPython view='phase' inputs={params} />
      </aside>

      <main className="pane pane-center">
        <ErrorBanner error={q.error} />
        <div className="viewport">
          {data ? (
            <WignerScene xvec={data.sim.xvec} w={data.sim.wigner[f] ?? []} ghost={ghost} theme={theme} />
          ) : (
            <div className="boot">Solving the master equation…</div>
          )}
          <div className="hud">
            <div>
              <span className="muted">time</span> <span className="mono">κ₂t = {fmt(t[f] ?? 0)}</span>
            </div>
            <div>
              <span className="muted">⟨n⟩</span> <span className="mono">{fmt(data?.sim.photon_number[f])}</span>
            </div>
            <div>
              <span className="muted">parity ⟨X⟩</span> <span className="mono">{fmt(parityNow)}</span>
            </div>
            <div>
              <span className="muted">side ⟨Z⟩</span> <span className="mono">{fmt(data?.sim.sign_x[f])}</span>
            </div>
            {q.isFetching ? <div className="muted">solving…</div> : null}
          </div>
          <div className="legend-wigner">
            <span className="swatch swatch-neg" /> W &lt; 0 (quantum) <span className="swatch swatch-pos" /> W &gt; 0
          </div>
        </div>
        <div className="transport">
          <button className="btn" onClick={() => (f >= nFrames - 1 ? (setFrame(0), setPlaying(true)) : setPlaying(!playing))} data-testid="play">
            {playing ? 'Pause' : 'Play'}
          </button>
          <input
            type="range"
            min={0}
            max={nFrames - 1}
            value={f}
            onChange={(e) => {
              setPlaying(false)
              setFrame(Number(e.target.value))
            }}
            aria-label="time"
            data-testid="scrubber"
          />
          <label className="check">
            <input type="checkbox" checked={showTheory} onChange={toggleTheory} data-testid="ghost" /> Show ideal cat (theory)
          </label>
        </div>
        <div className="dock">
          {data ? (
            <Plot
              data={[
                { x: t, y: data.sim.photon_number, name: '⟨n⟩', line: { color: c.a, width: 2 } },
                { x: t, y: data.sim.parity, name: 'parity ⟨X⟩', line: { color: c.b, width: 2 }, yaxis: 'y2' },
                { x: t, y: data.sim.sign_x, name: 'side ⟨Z⟩', line: { color: c.c, width: 2 }, yaxis: 'y2' },
                ...(data.theory.parity_curve
                  ? [{ x: t, y: data.theory.parity_curve, name: 'parity, theory', line: { color: c.theory, width: 2, dash: 'dash' }, yaxis: 'y2' }]
                  : []),
                ...(data.theory.stabilized_photon_number !== undefined
                  ? [
                      {
                        x: [t[0], t[t.length - 1]],
                        y: [data.theory.stabilized_photon_number, data.theory.stabilized_photon_number],
                        name: '|α∞|², theory',
                        line: { color: c.theory, width: 2, dash: 'dash' },
                        mode: 'lines',
                      },
                    ]
                  : []),
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 'κ₂ t' } },
                yaxis: { title: { text: 'photons' }, rangemode: 'tozero' },
                yaxis2: { overlaying: 'y', side: 'right', range: [-1.05, 1.05], tickvals: [-1, -0.5, 0, 0.5, 1], showgrid: false, title: { text: '⟨X⟩, ⟨Z⟩' } },
                shapes: [{ type: 'line', x0: t[f], x1: t[f], yref: 'paper', y0: 0, y1: 1, line: { color: c.muted, width: 1 } }],
                margin: { l: 56, r: 56, t: 10, b: 40 },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : null}
        </div>
      </main>

      <aside className="pane pane-right">
        <h3>What you are looking at</h3>
        <p className="small">
          The surface is the Wigner function W(x, p) of the resonator, solved with Dynamiqs. Two red peaks are the
          coherent states |±α⟩. Blue dips between them are negative W: the fringes that make it a cat, not a coin
          flip. Their sign at the origin is the parity.
        </p>
        <h3>Theory for these settings</h3>
        <FormulaCard formulaKey="cat.wigner_even_cat" />
        {params.scheme === 'dissipative' && params.kappa_1 > 0 ? (
          <>
            <FormulaCard
              formulaKey="cat.stabilized_photon_number"
              theory={data?.theory.stabilized_photon_number}
              sim={data?.sim.photon_number[nFrames - 1]}
              simLabel="⟨n⟩ at the end of the run"
            />
            <FormulaCard formulaKey="cat.parity_decay_rate" theory={data?.theory.parity_decay_rate} />
          </>
        ) : (
          <p className="small muted">Add photon loss with two-photon stabilization to compare the loss formulas.</p>
        )}
        <FormulaCard formulaKey="cat.coherent_overlap" />
      </aside>
    </div>
  )
}

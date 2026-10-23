// Two-qubit gates: a transmon CZ (park |11> on the |20> resonance for a full cycle) and a
// bias-preserving CNOT between two dissipative cats. Simulation from Dynamiqs; dashed lines and
// markers are the papers' formulas.

import { useEffect, useState } from 'react'
import { type GatesParams, useGates } from '../api'
import { CONFIG, Plot, SERIES, layout } from '../components/Plot'
import { CopyPython, ErrorBanner, FormulaCard, Segmented, Slider, Solving } from '../components/ui'
import { fmt } from '../lib/colormap'
import { useShared } from '../lib/share'
import { useExplorer } from '../store'

const DEFAULTS: GatesParams = { g_mhz: 10, alpha_mhz: -300, n: 4, log_kappa_ratio: -3 }

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms)
    return () => window.clearTimeout(id)
  }, [value, ms])
  return v
}

export function Gates() {
  const theme = useExplorer((s) => s.theme)
  const c = SERIES[theme]
  const [draft, setDraft] = useShared<GatesParams>('gates', DEFAULTS)
  const params = useDebounced(draft, 500)
  const q = useGates(params)
  const d = q.data
  const set = (p: Partial<GatesParams>) => setDraft((x) => ({ ...x, ...p }))
  const tcz = d?.cz.theory.cz_time_ns
  const atTcz = d ? d.cz.sim.phase[Math.round((d.cz.sim.phase.length - 1) / 2)] : undefined // t_CZ is the midpoint of the sweep

  return (
    <div className="view">
      <aside className="pane pane-left">
        <h3>Transmon CZ</h3>
        <Slider label="coupling" tex="g/2\pi" value={draft.g_mhz} min={1} max={60} step={1} onChange={(g_mhz) => set({ g_mhz })} format={(v) => `${fmt(v)} MHz`} />
        <Slider label="anharmonicity" tex="\alpha/2\pi" value={draft.alpha_mhz} min={-400} max={-100} step={10} onChange={(alpha_mhz) => set({ alpha_mhz })} format={(v) => `${fmt(v)} MHz`} />
        <p className="small muted">
          Qubit 1 jumps to where |11⟩ and |20⟩ are degenerate, waits, and jumps back. A full cycle through |20⟩ leaves a
          phase of −1 on |11⟩ only.
        </p>
        <h3>Cat CNOT</h3>
        <Segmented<'2' | '4'>
          label="Cat size"
          value={String(draft.n) as '2' | '4'}
          options={[
            { value: '2', label: '|α|² = 2' },
            { value: '4', label: '|α|² = 4' },
          ]}
          onChange={(v) => set({ n: Number(v) as 2 | 4 })}
        />
        <Slider label="photon loss" tex="\kappa_1/\kappa_2" value={draft.log_kappa_ratio} min={-4} max={-2} step={0.25} onChange={(log_kappa_ratio) => set({ log_kappa_ratio })} format={(v) => (10 ** v).toExponential(1)} />
        <p className="small muted">
          The target's two-photon pump turns half a circle only when the control is in |−α⟩, so the gate never stops
          protecting against bit-flips (Guillaud and Mirrahimi 2019).
        </p>
        <CopyPython view="gates" inputs={params} />
      </aside>

      <main className="pane pane-center">
        <ErrorBanner error={q.error} />
        <div className="chart-main">
          {d ? (
            <Plot
              data={[
                { x: d.cz.sim.times_ns, y: d.cz.sim.phase.map((p) => p / Math.PI), name: '|conditional phase| / π, simulation', line: { color: c.a, width: 2 } },
                { x: d.cz.sim.times_ns, y: d.cz.sim.leakage, name: 'leakage out of |11⟩', line: { color: c.b, width: 2 }, yaxis: 'y2' },
                { x: [tcz, tcz], y: [0, 2.2], name: 't_CZ = π/(√2 g), theory', mode: 'lines', line: { color: c.theory, width: 2, dash: 'dash' } },
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 'hold time at the 11-20 resonance (ns)' } },
                yaxis: { title: { text: '|conditional phase| / π' }, range: [0, 2.2] },
                yaxis2: { overlaying: 'y', side: 'right', range: [0, 1.1], dtick: 0.25, showgrid: false, title: { text: 'leakage' } },
                margin: { l: 56, r: 56, t: 10, b: 40 },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : (
            <Solving what="Evolving the transmon pair and three cat CNOTs" boot />
          )}
          {q.isFetching && d ? <div className="hud-corner"><Solving /></div> : null}
        </div>
        <div className="dock">
          {d ? (
            <Plot
              data={[
                { x: d.cnot.theory.T, y: d.cnot.theory.control, name: 'control flips, theory', line: { color: c.a, width: 2, dash: 'dash' } },
                { x: d.cnot.theory.T, y: d.cnot.theory.target, name: 'target flips, theory', line: { color: c.b, width: 2, dash: 'dash' } },
                { x: d.cnot.sim.T, y: d.cnot.sim.control, name: 'control, simulation', mode: 'markers', marker: { color: c.a, size: 9 } },
                { x: d.cnot.sim.T, y: d.cnot.sim.target, name: 'target, simulation', mode: 'markers', marker: { color: c.b, size: 9 } },
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 'cat CNOT gate time κ₂T' }, type: 'log' },
                yaxis: { title: { text: 'parity-flip chance' }, type: 'log', exponentformat: 'power' },
                shapes: [{ type: 'line', x0: d.cnot.theory.t_opt, x1: d.cnot.theory.t_opt, yref: 'paper', y0: 0, y1: 1, line: { color: c.muted, width: 1, dash: 'dot' } }],
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : null}
        </div>
      </main>

      <aside className="pane pane-right">
        <h3>What the numbers say</h3>
        <p className="small" data-testid="gates-summary">
          CZ: hold {fmt(tcz)} ns; the simulated conditional phase there is {fmt(atTcz !== undefined ? atTcz / Math.PI : undefined)}π.
          Cat CNOT: the fewest phase-flips at κ₂T = {fmt(d?.cnot.theory.t_opt)}, where the simulated control flips with
          chance {fmt(d?.cnot.sim.control[1])}. Bit-flips stay exponentially small throughout.
        </p>
        {d ? (
          <div style={{ height: 220 }}>
            <Plot
              data={[
                { x: d.cz.sim.zz_delta_mhz, y: d.cz.sim.zz_mhz.map(Math.abs), name: 'static ZZ, simulation', mode: 'markers', marker: { color: c.a, size: 4 } },
                { x: d.cz.sim.zz_delta_mhz, y: d.cz.theory.zz_mhz.map(Math.abs), name: 'theory', line: { color: c.theory, width: 2, dash: 'dash' } },
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 'qubit detuning Δ/2π (MHz)' } },
                yaxis: { title: { text: '|ζ|/2π (MHz)' }, type: 'log', exponentformat: 'power' },
                margin: { l: 48, r: 8, t: 8, b: 36 },
                showlegend: false,
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        ) : null}
        <FormulaCard formulaKey="gates.cz_time" />
        <FormulaCard formulaKey="gates.static_zz" />
        <FormulaCard formulaKey="gates.cat_cnot_z1" />
        <FormulaCard formulaKey="gates.cat_cnot_z2" />
      </aside>
    </div>
  )
}

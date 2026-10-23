// Dispersive readout: the qubit state shifts a resonator's frequency by +-chi, so a drive on the
// resonator ends up in a different spot of the IQ plane for |g> and |e>. chi comes from the
// dressed spectrum (scqubits + NumPy), the fields from Dynamiqs, the curves from the papers.

import { useEffect, useState } from 'react'
import { type ReadoutParams, useReadout } from '../api'
import { CONFIG, Plot, SERIES, layout } from '../components/Plot'
import { CopyPython, ErrorBanner, FormulaCard, Slider, Solving } from '../components/ui'
import { fmt } from '../lib/colormap'
import { useShared } from '../lib/share'
import { useExplorer } from '../store'

const DEFAULTS: ReadoutParams = { ratio: 50, E_C: 0.3, f_r: 7.0, g_n: 0.05, kappa_mhz: 1.0, n_bar: 1.0, delta_r_mhz: 0 }

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms)
    return () => window.clearTimeout(id)
  }, [value, ms])
  return v
}

export function Readout() {
  const theme = useExplorer((s) => s.theme)
  const c = SERIES[theme]
  const [draft, setDraft] = useShared<ReadoutParams>('readout', DEFAULTS)
  const params = useDebounced(draft, 400)
  const q = useReadout(params)
  const d = q.data
  const set = (p: Partial<ReadoutParams>) => setDraft((x) => ({ ...x, ...p }))
  const err = (theory?: number) => (d && theory ? `${fmt((Math.abs(theory - d.sim.chi_mhz) / Math.abs(d.sim.chi_mhz)) * 100, 2)}%` : '')

  return (
    <div className="view">
      <aside className="pane pane-left">
        <h3>Transmon</h3>
        <Slider label="E_J/E_C" value={draft.ratio} min={20} max={200} step={1} onChange={(ratio) => set({ ratio })} format={(v) => String(v)} />
        <Slider label="charging energy" tex="E_C" value={draft.E_C} min={0.1} max={0.5} step={0.01} onChange={(E_C) => set({ E_C })} format={(v) => `${fmt(v)} GHz`} />
        <h3>Resonator</h3>
        <Slider label="frequency" tex="f_r" value={draft.f_r} min={4} max={9} step={0.05} onChange={(f_r) => set({ f_r })} format={(v) => `${fmt(v)} GHz`} />
        <Slider label="coupling to n(a + a†)" tex="g_n" value={draft.g_n} min={0.005} max={0.15} step={0.005} onChange={(g_n) => set({ g_n })} format={(v) => `${fmt(v * 1e3)} MHz`} />
        <Slider label="linewidth" tex="\kappa/2\pi" value={draft.kappa_mhz} min={0.1} max={10} step={0.1} onChange={(kappa_mhz) => set({ kappa_mhz })} format={(v) => `${fmt(v)} MHz`} />
        <h3>Drive</h3>
        <Slider label="photons on resonance" tex="\bar n" value={draft.n_bar} min={0.05} max={10} step={0.05} onChange={(n_bar) => set({ n_bar })} />
        <Slider label="drive detuning" tex="\delta_r/2\pi" value={draft.delta_r_mhz} min={-5} max={5} step={0.05} onChange={(delta_r_mhz) => set({ delta_r_mhz })} format={(v) => `${fmt(v)} MHz`} />
        <p className="small muted">
          χ comes from diagonalizing the transmon (scqubits) coupled to the resonator, with no dispersive
          approximation. The fields come from Dynamiqs, solving the resonator for each qubit state.
        </p>
        <CopyPython view="readout" inputs={params} />
      </aside>

      <main className="pane pane-center">
        <ErrorBanner error={q.error} />
        <div className="chart-main">
          {d ? (
            <Plot
              data={[
                { x: d.sim.field_g[0], y: d.sim.field_g[1], name: 'qubit in |g⟩', mode: 'lines', line: { color: c.b, width: 2 } },
                { x: d.sim.field_e[0], y: d.sim.field_e[1], name: 'qubit in |e⟩', mode: 'lines', line: { color: c.a, width: 2 } },
                {
                  x: [d.sim.field_g[0].at(-1), d.sim.field_e[0].at(-1)],
                  y: [d.sim.field_g[1].at(-1), d.sim.field_e[1].at(-1)],
                  name: 'steady state',
                  mode: 'markers',
                  marker: { color: c.ink, size: 9 },
                },
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 'I = Re⟨a⟩' }, zeroline: true },
                yaxis: { title: { text: 'Q = Im⟨a⟩' }, scaleanchor: 'x', zeroline: true },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : (
            <Solving what="Diagonalizing the transmon and resonator, then solving both pointer states" boot />
          )}
          {q.isFetching && d ? <div className="hud-corner"><Solving /></div> : null}
        </div>
        <div className="dock">
          {d ? (
            <Plot
              data={[
                { x: d.theory.probe_mhz, y: d.theory.response_g, name: '|g⟩, theory', line: { color: c.b, width: 2, dash: 'dash' } },
                { x: d.theory.probe_mhz, y: d.theory.response_e, name: '|e⟩, theory', line: { color: c.a, width: 2, dash: 'dash' } },
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 'drive detuning δ_r/2π (MHz)' } },
                yaxis: { title: { text: 'photons in steady state' }, rangemode: 'tozero' },
                shapes: [{ type: 'line', x0: draft.delta_r_mhz, x1: draft.delta_r_mhz, yref: 'paper', y0: 0, y1: 1, line: { color: c.muted, width: 1 } }],
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : null}
        </div>
      </main>

      <aside className="pane pane-right">
        <h3>Dispersive shift χ</h3>
        <table className="facts" data-testid="chi-table">
          <tbody>
            <tr>
              <td>Dressed spectrum (simulation)</td>
              <td className="mono">{fmt(d?.sim.chi_mhz)} MHz</td>
              <td />
            </tr>
            <tr>
              <td>Koch Eqs. 27 and 28</td>
              <td className="mono">{fmt(d?.theory.chi_mhz.transmon)} MHz</td>
              <td className="small muted">{err(d?.theory.chi_mhz.transmon)}</td>
            </tr>
            <tr>
              <td>Same, with g₁₂ = √2 g₀₁</td>
              <td className="mono">{fmt(d?.theory.chi_mhz.harmonic)} MHz</td>
              <td className="small muted">{err(d?.theory.chi_mhz.harmonic)}</td>
            </tr>
            <tr>
              <td>Two-level g²/Δ</td>
              <td className="mono">{fmt(d?.theory.chi_mhz.two_level)} MHz</td>
              <td className="small muted">{err(d?.theory.chi_mhz.two_level)}</td>
            </tr>
          </tbody>
        </table>
        <p className="small muted">
          Qubit at {fmt(d?.sim.f01_ghz)} GHz, anharmonicity {fmt(d?.sim.alpha_mhz)} MHz, g₀₁ = {fmt(d?.sim.g01_mhz)} MHz and
          g₁₂ = {fmt(d?.sim.g12_mhz)} MHz. The √2 shortcut is the large E_J/E_C limit; the two terms of χ nearly
          cancel, so a small error in g₁₂ shows up large.
        </p>
        <h3>The price of looking</h3>
        <p className="small" data-testid="dephasing">
          Photons in the resonator dephase the qubit at Γ_m = {fmt(d?.theory.dephasing_per_us)} per µs (Gambetta), and{' '}
          {fmt(d?.sim.dephasing_per_us)} per µs in the simulation. The small-χ formula 8χ²n̄/κ gives{' '}
          {fmt(d?.theory.dephasing_weak_per_us)}.
        </p>
        <FormulaCard formulaKey="readout.chi_transmon" theory={d?.theory.chi_mhz.transmon} sim={d?.sim.chi_mhz} unit=" MHz" />
        <FormulaCard formulaKey="readout.dephasing_rate" theory={d?.theory.dephasing_per_us} sim={d?.sim.dephasing_per_us} unit=" /µs" />
        <FormulaCard formulaKey="readout.pointer_photons" />
      </aside>
    </div>
  )
}

// Flip rates vs cat size: the bias that makes cat qubits worth building. Rates come from the
// Liouvillian spectrum (simulation), theory lines from the closed forms, experiments as facts.

import { useEffect, useState } from 'react'
import { type CatParams, useFacts, useRates, useReferences } from '../api'
import { CONFIG, Plot, SERIES, layout } from '../components/Plot'
import { CopyPython, ErrorBanner, FormulaCard, PaperSettings, shortCite, Slider, Solving } from '../components/ui'
import { fmt } from '../lib/colormap'
import { useExplorer } from '../store'
import { useShared } from '../lib/share'

const SIZES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms)
    return () => window.clearTimeout(id)
  }, [value, ms])
  return v
}

export function FlipRates() {
  const theme = useExplorer((s) => s.theme)
  const [draft, setDraft] = useShared<CatParams>('rates', { alpha_sq: 4, kappa_1: 0.1, kappa_phi: 0, n_th: 0, scheme: 'dissipative' })
  const params = useDebounced(draft, 500)
  const q = useRates(params, SIZES)
  const facts = (useFacts().data ?? []).filter((f) => f.qubit === 'cat')
  const refs = useReferences().data
  const c = SERIES[theme]
  const rows = q.data?.rows ?? []
  const x = rows.map((r) => r.alpha_sq)
  const bit = rows.map((r) => r.sim.bit_flip_rate)
  const phase = rows.map((r) => r.sim.phase_flip_rate)
  const ratios = rows.slice(1).map((r, i) => {
    const prev = rows[i]?.sim.bit_flip_rate
    const cur = r.sim.bit_flip_rate
    return prev && cur && r.alpha_sq - (rows[i]?.alpha_sq ?? 0) > 0 ? (prev / cur) ** (1 / (r.alpha_sq - (rows[i]?.alpha_sq ?? 0))) : null
  })
  const at = (n: number) => rows.find((r) => r.alpha_sq === n)
  const set = (p: Partial<CatParams>) => setDraft((d) => ({ ...d, ...p }))

  return (
    <div className="view">
      <aside className="pane pane-left">
        <PaperSettings view="rates" onPick={(r) => set(r as Partial<CatParams>)} />
        <h3>Noise</h3>
        <p className="small muted">
          Two-photon dissipation only. For a Kerr cat the slowest modes of the Liouvillian are not the flips, so this
          method would mislabel them; use the Phase space view to watch a Kerr cat instead.
        </p>
        <Slider label="photon loss" tex="\kappa_1/\kappa_2" value={draft.kappa_1} min={0} max={0.5} step={0.01} onChange={(kappa_1) => set({ kappa_1 })} />
        <Slider label="dephasing" tex="\kappa_\phi/\kappa_2" value={draft.kappa_phi} min={0} max={0.5} step={0.01} onChange={(kappa_phi) => set({ kappa_phi })} />
        <Slider label="thermal photons" tex="n_{th}" value={draft.n_th} min={0} max={0.2} step={0.01} onChange={(n_th) => set({ n_th })} />
        <p className="small muted">
          Each point is the slowest decay rate in one parity block of the Liouvillian, so no curve fitting is involved.
          Bit-flip rates under {fmt(q.data?.rate_floor)} are below double precision and are not drawn.
        </p>
        <h3>Measured on hardware</h3>
        <table className="facts">
          <tbody>
            {facts.map((f) => (
              <tr key={f.key}>
                <td>
                  {f.label}
                  {f.alpha_sq !== null ? <span className="muted"> at |α|² = {f.alpha_sq}</span> : null}
                </td>
                <td className="mono">
                  {fmt(f.value)} {f.unit}
                </td>
                <td className="small">
                  <span className={`kind kind-${f.kind}`}>{f.kind.replace('_', ' ')}</span> {shortCite(refs?.[f.cite], f.cite)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <CopyPython view='rates' inputs={{ ...params, alpha_sq_values: SIZES }} />
      </aside>

      <main className="pane pane-center">
        <ErrorBanner error={q.error} />
        <div className="chart-main">
          {q.data ? (
            <Plot
              data={[
                { x, y: bit, name: 'bit-flip Γ_X', mode: 'lines+markers', line: { color: c.a, width: 2 }, connectgaps: false },
                { x, y: phase, name: 'phase-flip Γ_Z', mode: 'lines+markers', line: { color: c.b, width: 2 } },
                {
                  x,
                  y: rows.map((r) => r.theory.phase_flip_rate),
                  name: 'Γ_Z = κ₁|α|², theory',
                  line: { color: c.theory, width: 2, dash: 'dash' },
                },
                {
                  x,
                  y: rows.map((r) => r.theory.bit_flip_rate_dephasing),
                  name: 'Γ_X from dephasing, theory',
                  line: { color: c.theory, width: 2, dash: 'dot' },
                },
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 'cat size |α|² (mean photon number)' } },
                yaxis: { title: { text: 'rate / κ₂' }, type: 'log', exponentformat: 'power' },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : (
            <Solving boot what={`Diagonalizing ${SIZES.length} Liouvillians`} />
          )}
          {q.isFetching && q.data ? (
            <div className="hud-corner">
              <Solving what={`${q.data.rows.length} of ${q.data.total} sizes`} />
            </div>
          ) : null}
        </div>
        <div className="dock">
          {q.data ? (
            <Plot
              data={[
                { x: x.slice(1), y: ratios, name: 'gain per added photon', mode: 'lines+markers', line: { color: c.a, width: 2 } },
                {
                  x: [x[1], x[x.length - 1]],
                  y: [1 / q.data.theory.bit_flip_suppression_per_photon, 1 / q.data.theory.bit_flip_suppression_per_photon],
                  name: 'e² ≈ 7.39, theory',
                  line: { color: c.theory, width: 2, dash: 'dash' },
                },
                { x: [x[1], x[x.length - 1]], y: [4.2, 4.2], name: '4.2, measured (Lescanne 2020)', line: { color: c.muted, width: 1.5, dash: 'dot' } },
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: '|α|²' } },
                yaxis: { title: { text: 'bit-flip gain per photon' }, rangemode: 'tozero' },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : null}
        </div>
      </main>

      <aside className="pane pane-right">
        <h3>The trade</h3>
        <p className="small">
          Bigger cats: bit-flips fall exponentially, phase-flips rise linearly. At |α|² = 4 with these settings the
          simulation gives Γ_X = {fmt(at(4)?.sim.bit_flip_rate)} and Γ_Z = {fmt(at(4)?.sim.phase_flip_rate)}, a bias
          of {fmt((at(4)?.sim.phase_flip_rate ?? 0) / (at(4)?.sim.bit_flip_rate ?? 1))}.
        </p>
        <FormulaCard formulaKey="cat.bit_flip_suppression" />
        <FormulaCard formulaKey="cat.phase_flip_rate" theory={at(4)?.theory.phase_flip_rate} sim={at(4)?.sim.phase_flip_rate} />
        <FormulaCard
          formulaKey="cat.bit_flip_rate_dephasing"
          theory={at(2)?.theory.bit_flip_rate_dephasing}
          sim={draft.kappa_phi > 0 && draft.kappa_1 === 0 ? at(2)?.sim.bit_flip_rate : undefined}
        />
      </aside>
    </div>
  )
}

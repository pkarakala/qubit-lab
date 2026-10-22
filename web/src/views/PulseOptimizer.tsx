// Pulse optimizer: an X gate on a 4-level transmon. The DRAG pulse (Motzoi 2009) is the starting
// point; gradient descent through the Dynamiqs propagator (GRAPE) reshapes it. The knobs only take
// effect on "Optimize", because each run is a few seconds of solves.

import { useEffect, useState } from 'react'
import { type PulseParams, type PulseScore, usePulse } from '../api'
import { CONFIG, Plot, SERIES, layout } from '../components/Plot'
import { CopyPython, ErrorBanner, FormulaCard, Segmented, Slider } from '../components/ui'
import { fmt } from '../lib/colormap'
import { useExplorer } from '../store'
import { BlochScene } from '../three/BlochSphere'
import { useShared } from '../lib/share'

type Which = 'drag' | 'optimized'

const START: PulseParams = {
  alpha_ghz: -0.2,
  duration_ns: 6,
  omega_max_ghz: 0.3,
  t1_ns: 50_000,
  t2_ns: 50_000,
  steps: 300,
}

// Plotly step trace: one value per slice, held until the next edge.
const steps = (edges: number[], v: number[]) => ({
  x: edges,
  y: [...v, v[v.length - 1] ?? 0].map((w) => w * 1e3),
})

export function PulseOptimizer() {
  const theme = useExplorer((s) => s.theme)
  const [req, setReq] = useShared<PulseParams>('pulse', START)
  const [knobs, setKnobs] = useState<PulseParams>(req)
  const [which, setWhich] = useState<Which>('optimized')
  const run = usePulse(req)
  const d = run.data?.sim
  const shown: PulseScore | undefined = d?.[which]
  const n = shown?.times.length ?? 1
  const [frame, setFrame] = useState(0)
  const c = SERIES[theme]
  const set = (p: Partial<PulseParams>) => setKnobs((k) => ({ ...k, ...p }))
  const stale = JSON.stringify(knobs) !== JSON.stringify(req)

  useEffect(() => setFrame(0), [shown])
  useEffect(() => {
    const id = window.setInterval(() => setFrame((f) => (f >= n - 1 ? f : f + 1)), 40)
    return () => window.clearInterval(id)
  }, [n, shown])

  const f = Math.min(frame, n - 1)
  const pops = shown?.populations[f] ?? []

  return (
    <div className="view">
      <aside className="pane pane-left">
        <h3>Gate</h3>
        <p className="small muted">An X gate (a π pulse) on a transmon. Short gates leak to level 2; that is what the optimizer fights.</p>
        <Slider
          label="gate length (ns)"
          value={knobs.duration_ns}
          min={2}
          max={30}
          step={1}
          onChange={(v) => set({ duration_ns: v })}
          testId="pulse-length"
        />
        <Slider
          label="largest drive allowed (MHz)"
          tex="\Omega_{max}"
          value={knobs.omega_max_ghz * 1e3}
          min={50}
          max={600}
          step={10}
          onChange={(v) => set({ omega_max_ghz: v / 1e3 })}
        />
        <Slider label="gradient steps" value={knobs.steps} min={50} max={1000} step={50} onChange={(v) => set({ steps: v })} />
        <h3>Device</h3>
        <Slider
          label="anharmonicity (MHz)"
          tex="\alpha"
          value={knobs.alpha_ghz * 1e3}
          min={-500}
          max={-100}
          step={10}
          onChange={(v) => set({ alpha_ghz: v / 1e3 })}
        />
        <Slider
          label="T1 = T2 (µs)"
          value={Math.log10(knobs.t1_ns / 1e3)}
          min={0}
          max={3}
          step={0.05}
          format={() => fmt(knobs.t1_ns / 1e3)}
          onChange={(v) => {
            const t = Math.round(10 ** v * 1e3)
            set({ t1_ns: t, t2_ns: t })
          }}
        />
        <button className="btn" onClick={() => setReq(knobs)} disabled={!stale && !run.isFetching} data-testid="pulse-run">
          {run.isFetching ? 'Optimizing…' : stale ? 'Optimize' : 'Up to date'}
        </button>
        <CopyPython view='pulse' inputs={req} />
      </aside>

      <main className="pane pane-center">
        <ErrorBanner error={run.error} />
        <div className="viewport">
          {shown ? <BlochScene bloch={shown.bloch} frame={f} theme={theme} /> : <div className="boot">Optimizing the pulse…</div>}
          <div className="hud">
            <Segmented
              label="pulse"
              value={which}
              options={[
                { value: 'drag', label: 'DRAG' },
                { value: 'optimized', label: 'Optimized' },
              ]}
              onChange={setWhich}
            />
            <div className="mono">t = {fmt(shown?.times[f])} ns</div>
            <div className="small">
              P0 {fmt(pops[0], 3)} · P1 {fmt(pops[1], 3)} · leaked {fmt((1 - (pops[0] ?? 0) - (pops[1] ?? 0)) * 100, 3)}%
            </div>
            <button className="btn" onClick={() => setFrame(0)}>
              Replay
            </button>
          </div>
        </div>
        <div className="dock">
          {d ? (
            <Plot
              data={[
                {
                  ...steps(d.edges, d.drag.omega_x),
                  name: 'Ωx, DRAG',
                  line: {
                    shape: 'hv',
                    color: c.muted,
                    width: 1.5,
                    dash: 'dash',
                  },
                },
                {
                  ...steps(d.edges, d.drag.omega_y),
                  name: 'Ωy, DRAG',
                  line: { shape: 'hv', color: c.b, width: 1.5, dash: 'dash' },
                },
                {
                  ...steps(d.edges, d.optimized.omega_x),
                  name: 'Ωx, optimized',
                  line: { shape: 'hv', color: c.a, width: 2 },
                },
                {
                  ...steps(d.edges, d.optimized.omega_y),
                  name: 'Ωy, optimized',
                  line: { shape: 'hv', color: c.c, width: 2 },
                },
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 'time (ns)' } },
                yaxis: { title: { text: 'drive (MHz)' } },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : null}
        </div>
      </main>

      <aside className="pane pane-right">
        <h3>DRAG vs optimized</h3>
        {d ? (
          <table className="checks small" data-testid="pulse-table">
            <thead>
              <tr>
                <th />
                <th>DRAG</th>
                <th>optimized</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>gate error</td>
                <td className="mono">{fmt(d.drag.error)}</td>
                <td className="mono">{fmt(d.optimized.error)}</td>
              </tr>
              <tr>
                <td>leakage</td>
                <td className="mono">{fmt(d.drag.leakage)}</td>
                <td className="mono">{fmt(d.optimized.leakage)}</td>
              </tr>
              <tr>
                <td>error with T1, T2</td>
                <td className="mono">{fmt(d.drag.error_with_noise)}</td>
                <td className="mono">{fmt(d.optimized.error_with_noise)}</td>
              </tr>
            </tbody>
          </table>
        ) : null}
        <p className="small muted">
          Gate error is 1 − F, the average over all input states (Pedersen et al. 2007, Eq. 3), so leakage counts. Both pulses are simulated
          the same way; only the shape differs.
        </p>
        {d ? (
          <div style={{ height: 180 }}>
            <Plot
              data={[
                {
                  y: d.optimized.history,
                  name: '1 − F',
                  line: { color: c.a, width: 2 },
                },
              ]}
              layout={layout(theme, {
                margin: { l: 50, r: 10, t: 10, b: 36 },
                showlegend: false,
                xaxis: { title: { text: 'gradient step' } },
                yaxis: { type: 'log', title: { text: 'gate error' } },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        ) : null}
        <FormulaCard formulaKey="qubit.drag_quadrature" />
        <p className="small muted">
          The optimizer starts from DRAG and follows the gradient of the gate error through the simulation (GRAPE, Khaneja et al. 2005). It
          removes the phase error DRAG leaves behind, until the drive limit stops it.
        </p>
      </aside>
    </div>
  )
}

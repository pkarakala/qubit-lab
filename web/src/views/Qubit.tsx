// Qubit control: one transmon, kept to 4 levels, under the pulses and noise of a real lab.
// Simulation (Dynamiqs) drives the Bloch sphere; theory (two-level closed forms) is the dashed line.

import { useEffect, useState } from 'react'
import { type Experiment, useQubit } from '../api'
import { CONFIG, Plot, SERIES, layout } from '../components/Plot'
import { CopyPython, ErrorBanner, FormulaCard, Segmented, Slider, Solving } from '../components/ui'
import { fmt } from '../lib/colormap'
import { QUBIT_PRESETS, useExplorer } from '../store'
import { BlochScene } from '../three/BlochSphere'

const EXPERIMENTS: { value: Experiment; label: string }[] = [
  { value: 'rabi', label: 'Rabi' },
  { value: 'pulse', label: 'Pulse' },
  { value: 't1', label: 'T1' },
  { value: 'ramsey', label: 'Ramsey' },
  { value: 'echo', label: 'Echo' },
]

const FORMULA: Record<Experiment, string> = {
  rabi: 'qubit.rabi_population',
  pulse: 'qubit.drag_quadrature',
  t1: 'qubit.t1_decay',
  ramsey: 'qubit.ramsey',
  echo: 'qubit.echo',
}

export function Qubit() {
  const theme = useExplorer((s) => s.theme)
  const p = useExplorer((s) => s.qubit)
  const set = useExplorer((s) => s.setQubit)
  const run = useQubit(p)
  const d = run.data
  const n = d?.sim.times.length ?? 1
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(true)
  const c = SERIES[theme]

  useEffect(() => {
    setFrame(0)
    setPlaying(true)
  }, [d])
  useEffect(() => {
    if (!playing) return
    // About 4 s per run, whatever the number of time points.
    const step = Math.max(1, Math.round(n / 100))
    const id = window.setInterval(() => {
      setFrame((f) => {
        if (f >= n - 1) {
          setPlaying(false)
          return f
        }
        return Math.min(n - 1, f + step)
      })
    }, 40)
    return () => window.clearInterval(id)
  }, [playing, n])

  const f = Math.min(frame, n - 1)
  const pops = d?.sim.populations[f] ?? []
  const leak = d?.sim.leakage[f] ?? 0
  const ex = p.experiment
  const tracePops = ex === 'rabi' || ex === 'pulse' || ex === 't1'
  const simTrace = d ? (tracePops ? d.sim.populations.map((r) => r[1] ?? 0) : d.sim.bloch.map((r) => r[0] ?? 0)) : []
  const lastTheory = d?.theory.curve?.[n - 1] ?? null
  const omegaMax = d ? Math.max(1e-12, ...d.sim.omega_x.map(Math.abs)) : 1

  return (
    <div className="view">
      <aside className="pane pane-left">
        <h3>Lessons</h3>
        <div className="presets">
          {QUBIT_PRESETS.map((q) => (
            <button key={q.id} className="preset" onClick={() => useExplorer.setState({ qubit: q.params })} data-testid={`qubit-preset-${q.id}`}>
              <strong>{q.label}</strong>
              <span className="small muted">{q.blurb}</span>
            </button>
          ))}
        </div>
        <h3>Experiment</h3>
        <Segmented label="experiment" value={ex} options={EXPERIMENTS} onChange={(v) => set({ experiment: v })} />
        {ex === 'rabi' ? (
          <Slider label="drive strength (MHz)" tex="\Omega" value={p.omega_ghz * 1e3} min={1} max={150} step={1} onChange={(v) => set({ omega_ghz: v / 1e3 })} />
        ) : null}
        {ex === 'pulse' ? (
          <>
            <Slider label="pulse length (ns)" value={p.duration_ns} min={2} max={40} step={1} onChange={(v) => set({ duration_ns: v })} />
            <Slider label="DRAG coefficient" value={p.drag} min={0} max={2} step={0.05} onChange={(v) => set({ drag: v })} testId="drag" />
          </>
        ) : null}
        {ex !== 'pulse' && ex !== 't1' ? (
          <Slider label="detuning (MHz)" tex="\Delta" value={p.delta_ghz * 1e3} min={-50} max={50} step={1} onChange={(v) => set({ delta_ghz: v / 1e3 })} />
        ) : null}
        {ex !== 'pulse' ? (
          <Slider label="duration (ns)" value={p.t_max_ns} min={20} max={1000} step={10} onChange={(v) => set({ t_max_ns: v })} />
        ) : null}
        <h3>Device</h3>
        <Slider label="anharmonicity (MHz)" tex="\alpha" value={p.alpha_ghz * 1e3} min={-500} max={-100} step={10} onChange={(v) => set({ alpha_ghz: v / 1e3 })} />
        <Slider
          label="T1 (ns)"
          tex="T_1"
          value={Math.log10(p.t1_ns)}
          min={1}
          max={5}
          step={0.05}
          format={() => fmt(p.t1_ns)}
          onChange={(v) => {
            const t1 = Math.round(10 ** v)
            set({ t1_ns: t1, t2_ns: Math.min(p.t2_ns, 2 * t1) })
          }}
        />
        <Slider
          label="T2 (ns), at most 2 T1"
          tex="T_2"
          value={Math.log10(p.t2_ns)}
          min={1}
          max={Math.log10(2 * p.t1_ns)}
          step={0.05}
          format={() => fmt(p.t2_ns)}
          onChange={(v) => set({ t2_ns: Math.min(Math.round(10 ** v), 2 * p.t1_ns) })}
        />
        {ex === 'ramsey' || ex === 'echo' ? (
          <Slider label="slow frequency noise (MHz)" tex="\sigma" value={p.sigma_ghz * 1e3} min={0} max={10} step={0.5} onChange={(v) => set({ sigma_ghz: v / 1e3 })} />
        ) : null}
        <CopyPython view='qubit' inputs={p} />
      </aside>

      <main className="pane pane-center">
        <ErrorBanner error={run.error} />
        <div className="viewport">
          {d ? <BlochScene bloch={d.sim.bloch} frame={f} theme={theme} /> : <Solving boot what="Solving the master equation" />}
          <div className="hud">
            <div className="mono">t = {fmt(d?.sim.times[f])} ns</div>
            <div className="small">
              P0 {fmt(pops[0], 3)} · P1 {fmt(pops[1], 3)} · <span title="population above level 1">leaked {fmt(leak * 100, 3)}%</span>
            </div>
            <div className="leak-bar" aria-label="leakage">
              <div style={{ width: `${Math.min(100, leak * 100 * 10)}%` }} />
            </div>
            <div className="muted small">Bar: leakage to level 2 and up, ×10.</div>
            <button className="btn" onClick={() => (f >= n - 1 ? (setFrame(0), setPlaying(true)) : setPlaying(!playing))} data-testid="qubit-play">
              {playing ? 'Pause' : 'Play'}
            </button>
          </div>
        </div>
        <div className="dock">
          {d ? (
            <Plot
              data={[
                {
                  x: d.sim.times,
                  y: simTrace,
                  name: tracePops ? 'P1 (sim)' : '⟨X⟩ (sim)',
                  line: { color: c.a, width: 2 },
                },
                ...(d.theory.curve
                  ? [{ x: d.sim.times, y: d.theory.curve, name: `${d.theory.curve_label} (theory)`, line: { color: c.theory, width: 1.5, dash: 'dash' } }]
                  : []),
                ...(ex === 'pulse'
                  ? [
                      { x: d.sim.times, y: d.sim.leakage, name: 'leakage (sim)', line: { color: c.c, width: 2 } },
                      { x: d.sim.times, y: d.sim.omega_x.map((v) => v / omegaMax), name: 'Ωx / peak', line: { color: c.muted, width: 1 } },
                      { x: d.sim.times, y: d.sim.omega_y.map((v) => v / omegaMax), name: 'Ωy / peak Ωx', line: { color: c.b, width: 1 } },
                    ]
                  : []),
                { x: [d.sim.times[f], d.sim.times[f]], y: [-1, 1], name: 'now', showlegend: false, line: { color: c.muted, width: 1 } },
              ]}
              layout={layout(theme, { xaxis: { title: { text: 'time (ns)' } }, yaxis: { range: [-1.05, 1.05] } })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : null}
        </div>
      </main>

      <aside className="pane pane-right">
        <h3>Theory vs simulation</h3>
        <FormulaCard
          formulaKey={FORMULA[ex]}
          theory={ex === 'pulse' ? undefined : lastTheory}
          sim={ex === 'pulse' ? undefined : simTrace[n - 1]}
          simLabel={ex === 'pulse' ? undefined : 'at the last time'}
        />
        {ex === 'ramsey' || ex === 'echo' ? <FormulaCard formulaKey="qubit.t2" /> : null}
        <FormulaCard formulaKey="qubit.idle_fidelity" theory={d?.theory.idle_fidelity_at_end} unit="" />
        <p className="small muted">
          The simulation keeps four transmon levels, so it shows effects the two-level formulas cannot: leakage, and the
          level-2 Stark shift that the checks found in detuned Rabi.
        </p>
      </aside>
    </div>
  )
}

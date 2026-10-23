// Error correction: the surface code that transmons need, against the repetition code that a
// biased cat qubit can get away with. Simulation: Stim + PyMatching. Theory: Fowler et al.

import { useCatCode, useQecCurves, useQecShot } from '../api'
import { CONFIG, Plot, SERIES, layout } from '../components/Plot'
import { CopyPython, ErrorBanner, FormulaCard, PaperSettings, Segmented, Slider, Solving } from '../components/ui'
import { fmt } from '../lib/colormap'
import { useExplorer } from '../store'
import { SpaceTimeScene } from '../three/SpaceTime'
import { useShared } from '../lib/share'

const DISTANCES = [3, 5, 7]

export function ErrorCorrection() {
  const theme = useExplorer((s) => s.theme)
  const c = SERIES[theme]
  const [d, setD] = useShared('qec.d', 5)
  const [p, setP] = useShared('qec.p', 0.005)
  const [seed, setSeed] = useShared('qec.seed', 1)
  const [pPhase, setPPhase] = useShared('qec.p_phase', 0.005)
  const [bias, setBias] = useShared('qec.bias', 1e4) // p_phase / p_bit
  const curves = useQecCurves()
  const shot = useQecShot(d, p, seed)
  const cat = useCatCode(pPhase, pPhase / bias)
  const s = shot.data?.sim
  const colors: Record<number, string> = { 3: c.b, 5: c.a, 7: c.c }

  const catRows = cat.data?.rows ?? []
  // Rows where the phase-flips left are below what we sampled have no total: no guessing.
  const total = catRows.map((r) => (r.sim.phase_flip_logical === null ? null : r.sim.phase_flip_logical + r.theory.bit_flip_floor))
  const resolved = total.filter((t): t is number => t !== null)
  const best = resolved.length ? catRows[total.indexOf(Math.min(...resolved))] : undefined
  const unresolved = catRows.find((r) => r.sim.phase_flip_logical === null)

  return (
    <div className="view">
      <aside className="pane pane-left">
        <PaperSettings
          view="qec"
          onPick={(s) => {
            setD(Number(s.d))
            setP(Number(s.p))
            setSeed(Number(s.seed))
          }}
        />
        <h3>One surface code run</h3>
        <Segmented
          label="distance"
          value={String(d)}
          options={DISTANCES.map((x) => ({ value: String(x), label: `d = ${x}` }))}
          onChange={(v) => setD(Number(v))}
        />
        <Slider label="error per operation (%)" tex="p" value={p * 100} min={0.1} max={2} step={0.1} onChange={(v) => setP(v / 100)} />
        <button className="btn" onClick={() => setSeed(seed + 1)} data-testid="new-shot">
          New run
        </button>
        <p className="small muted">
          Every gate, reset and measurement fails with probability p. Stim samples the circuit; PyMatching pairs up the
          detection events. Raise p and watch the pairs get long enough to fool the decoder.
        </p>
        <h3>A repetition code of cats</h3>
        <PaperSettings
          view="qec_cat"
          title="Cat codes in the literature"
          onPick={(s) => {
            setPPhase(Number(s.p_phase))
            setBias(Number(s.p_phase) / Number(s.p_bit))
          }}
        />
        <Slider label="cat phase-flip per round (%)" tex="p_Z" value={pPhase * 100} min={0.1} max={5} step={0.1} onChange={(v) => setPPhase(v / 100)} />
        <Slider
          label="noise bias p_Z / p_X"
          tex="\eta"
          value={Math.log10(bias)}
          min={0}
          max={8}
          step={0.5}
          format={() => fmt(bias)}
          onChange={(v) => setBias(10 ** v)}
          testId="bias"
        />
        <p className="small muted">
          A transmon has η ≈ 1. Cats reach very large η because bit-flips fall exponentially with photon number (see
          Flip rates). The code fixes phase-flips; bit-flips get through, and more qubits let more through.
        </p>
        <CopyPython view='qec' inputs={{ p, p_phase: pPhase, p_bit: pPhase / bias }} />
      </aside>

      <main className="pane pane-center">
        <ErrorBanner error={curves.error ?? shot.error ?? cat.error} />
        <div className="viewport">
          {s ? (
            <SpaceTimeScene qubits={s.qubits} detectors={s.detectors} fired={s.fired} matched={s.matched} theme={theme} />
          ) : (
            <Solving boot what="Sampling a surface code run" />
          )}
          <div className="hud">
            <div className="mono">
              d = {d}, p = {fmt(p * 100)}%, {s?.fired.length ?? 0} detection events
            </div>
            <div className={s?.logical_error ? 'small tag-bad' : 'small'}>
              {s ? (s.logical_error ? 'Decoder fooled: logical error' : 'Decoded correctly') : ''}
            </div>
            <div className="muted small">Floor: the chip. Up: rounds of measurement. Red: detection events. Purple: decoder pairs.</div>
          </div>
        </div>
        <div className="dock">
          {curves.data ? (
            <Plot
              data={[
                ...curves.data.sim.curves.map((cv) => ({
                  x: curves.data.p,
                  y: cv.per_round,
                  name: `${cv.code} d=${cv.d} (${cv.qubits} qubits)`,
                  mode: 'lines+markers',
                  line: { color: colors[cv.d], width: 2, dash: cv.code === 'repetition' ? 'dot' : 'solid' },
                  marker: { size: 5 },
                })),
                ...DISTANCES.map((dd) => ({
                  x: curves.data.p,
                  y: curves.data.theory.fowler_surface[String(dd)],
                  name: `Fowler Eq. 11, his noise model (theory)`,
                  line: { color: c.theory, width: 1.2, dash: 'dash' },
                  showlegend: dd === 3,
                })),
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 'physical error per operation p' }, type: 'log' },
                yaxis: { title: { text: 'logical error per round' }, type: 'log' },
                legend: { orientation: 'v', x: 1.02, y: 1, font: { size: 10 } },
                margin: { l: 56, r: 210, t: 10, b: 40 },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : (
            <Solving boot what="Running 36 memory experiments" />
          )}
        </div>
      </main>

      <aside className="pane pane-right">
        <h3>Cats: how big a code?</h3>
        {catRows.length ? (
          <div style={{ height: 270 }}>
            <Plot
              data={[
                { x: catRows.map((r) => r.qubits), y: catRows.map((r) => r.sim.phase_flip_logical), name: 'phase-flips left (sim)', line: { color: c.b, width: 2 } },
                { x: catRows.map((r) => r.qubits), y: catRows.map((r) => r.theory.bit_flip_floor), name: 'bit-flips d·p_X (theory)', line: { color: c.theory, dash: 'dash', width: 1.5 } },
                { x: catRows.map((r) => r.qubits), y: total, name: 'total', line: { color: c.a, width: 2.5 } },
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 'physical qubits' } },
                yaxis: { type: 'log', title: { text: 'logical error / round' } },
                legend: { orientation: 'h', y: -0.42, font: { size: 10 } },
                margin: { l: 50, r: 8, t: 6, b: 90 },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        ) : null}
        {best ? (
          <p className="small" data-testid="best-cat-code">
            Best sampled: d = {best.d} ({best.qubits} cats), {fmt((best.sim.phase_flip_logical ?? 0) + best.theory.bit_flip_floor)} per
            round.{' '}
            {unresolved ? `From d = ${unresolved.d} the phase-flips left are too rare to sample with ${cat.data?.shots} runs. ` : ''}
            A surface code with d = 5 needs {curves.data?.sim.curves.find((x) => x.code === 'surface' && x.d === 5)?.qubits ?? 49} qubits.
          </p>
        ) : null}
        <FormulaCard formulaKey="qec.error_exponent" />
        <FormulaCard formulaKey="qec.repetition_bit_flip_floor" />
        <FormulaCard formulaKey="qec.surface_code_logical_error" />
        <FormulaCard formulaKey="qec.surface_code_qubits" theory={(2 * d - 1) ** 2} sim={curves.data?.sim.curves.find((x) => x.code === 'surface' && x.d === d)?.qubits} simLabel="Stim's rotated layout" />
      </aside>
    </div>
  )
}

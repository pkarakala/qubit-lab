// Transmon: exact charge-basis spectrum (simulation) against Koch et al.'s closed forms (theory).

import { useBands, useSurface } from '../api'
import { CONFIG, Plot, SERIES, layout } from '../components/Plot'
import { ErrorBanner, FormulaCard, Slider } from '../components/ui'
import { fmt } from '../lib/colormap'
import { useExplorer } from '../store'
import { BandScene } from '../three/BandSurface'

export function Transmon() {
  const theme = useExplorer((s) => s.theme)
  const ratio = useExplorer((s) => s.ratio)
  const setRatio = useExplorer((s) => s.setRatio)
  const E_C = useExplorer((s) => s.E_C)
  const surface = useSurface()
  const E_J = Number((ratio * E_C).toFixed(4))
  const bands = useBands(E_J, E_C)
  const b = bands.data
  const c = SERIES[theme]
  const colors = [c.b, c.a, c.c, c.d]

  return (
    <div className="view">
      <aside className="pane pane-left">
        <h3>Circuit</h3>
        <Slider
          label="Josephson / charging"
          tex="E_J/E_C"
          value={Math.log10(ratio)}
          min={0}
          max={2}
          step={0.02}
          format={() => fmt(ratio)}
          onChange={(v) => setRatio(Math.round(10 ** v * 10) / 10)}
          testId="ratio"
        />
        <Slider
          label="charging energy (GHz)"
          tex="E_C"
          value={E_C}
          min={0.1}
          max={0.5}
          step={0.01}
          onChange={(v) => useExplorer.setState({ E_C: v })}
        />
        <p className="small muted">
          E_J = {fmt(E_J)} GHz. The spectrum comes from diagonalizing H = 4E_C(n − n_g)² − E_J cos φ in the charge
          basis with JAX, over one period of offset charge n_g.
        </p>
        <h3>At this ratio</h3>
        <table className="facts">
          <tbody>
            <tr>
              <td>Qubit frequency E₀₁</td>
              <td className="mono">{fmt(b?.sim.e01_mean)} GHz</td>
            </tr>
            <tr>
              <td>Anharmonicity</td>
              <td className="mono">{fmt(b ? b.sim.anharmonicity * 1000 : null)} MHz</td>
            </tr>
            <tr>
              <td>Charge dispersion, level 1</td>
              <td className="mono">{fmt(b ? Math.abs(b.sim.dispersion[1] ?? 0) * 1e6 : null)} kHz</td>
            </tr>
          </tbody>
        </table>
      </aside>

      <main className="pane pane-center">
        <ErrorBanner error={surface.error ?? bands.error} />
        <div className="viewport">
          {surface.data ? (
            <BandScene
              ng={surface.data.sim.n_g}
              ratios={surface.data.sim.ratios}
              energies={surface.data.sim.ripple_over_ec}
              ratio={ratio}
              theme={theme}
            />
          ) : (
            <div className="boot">Diagonalizing the transmon over E_J/E_C…</div>
          )}
          <div className="hud">
            <div className="muted">How much each level wiggles with offset charge n_g (units of E_C).<br />Blue: level 0, red: level 1, green: level 2. Flat means immune to charge noise.</div>
          </div>
        </div>
        <div className="dock">
          {b ? (
            <Plot
              data={[
                ...[0, 1, 2, 3].map((m) => ({
                  x: b.sim.n_g,
                  y: b.sim.energies.map((row) => row[m] ?? null),
                  name: `E${m} (sim)`,
                  line: { color: colors[m], width: 2 },
                })),
                ...[1, 2, 3].map((m) => ({
                  x: [-1, 1],
                  y: [b.theory.level_spacing_from_ground[m], b.theory.level_spacing_from_ground[m]].map(
                    (v) => (v ?? 0) + Math.min(...b.sim.energies.map((r) => r[0] ?? 0)),
                  ),
                  name: `E${m} Koch Eq. 2.11 (theory)`,
                  line: { color: colors[m], width: 1.2, dash: 'dash' },
                  showlegend: m === 1,
                })),
              ]}
              layout={layout(theme, {
                xaxis: { title: { text: 'offset charge n_g' } },
                yaxis: { title: { text: 'energy (GHz)' } },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : null}
        </div>
      </main>

      <aside className="pane pane-right">
        <h3>Theory vs exact spectrum</h3>
        <FormulaCard formulaKey="transmon.qubit_frequency" theory={b?.theory.e01} sim={b?.sim.e01_mean} unit=" GHz" />
        <FormulaCard formulaKey="transmon.anharmonicity" theory={b?.theory.anharmonicity} sim={b?.sim.anharmonicity} unit=" GHz" />
        <FormulaCard formulaKey="transmon.charge_dispersion" theory={b?.theory.dispersion[1]} sim={b?.sim.dispersion[1]} unit=" GHz" />
      </aside>
    </div>
  )
}

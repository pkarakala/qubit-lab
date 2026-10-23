// Architecture gallery: the leading superconducting qubit designs as 3D parts that explode
// apart, each with its role, true size and source, next to the design's energy levels.

import { useState } from 'react'
import { useArchitectures, useArchSpectrum, useReferences } from '../api'
import { CONFIG, Plot, SERIES, layout } from '../components/Plot'
import { ErrorBanner, Slider, Solving, shortCite } from '../components/ui'
import { fmt } from '../lib/colormap'
import { useShared } from '../lib/share'
import { useExplorer } from '../store'
import { ArchitectureScene } from '../three/ArchitectureScene'

const um = (v: number) => (v >= 1000 ? `${fmt(v / 1000)} mm` : v >= 1 ? `${fmt(v)} µm` : `${fmt(v * 1000)} nm`)

export function Architectures() {
  const theme = useExplorer((s) => s.theme)
  const c = SERIES[theme]
  const all = useArchitectures()
  const refs = useReferences().data
  const [id, setId] = useShared('arch.id', 'xmon')
  const [explode, setExplode] = useShared('arch.explode', 0.6)
  const [selected, setSelected] = useState<string | null>(null)
  const scene = all.data?.find((s) => s.id === id) ?? all.data?.[0]
  const spectrum = useArchSpectrum(scene?.id ?? 'xmon')
  const part = scene?.parts.find((p) => p.id === selected)
  const levels = spectrum.data?.sim.levels ?? []
  const nLevels = levels[0]?.length ?? 0
  const palette = [c.muted, c.a, c.b, c.c, c.d]

  return (
    <div className="view">
      <aside className="pane pane-left">
        <h3>Designs</h3>
        <div className="presets">
          {(all.data ?? []).map((s) => (
            <button
              key={s.id}
              className={`preset ${s.id === scene?.id ? 'is-active' : ''}`}
              onClick={() => {
                setId(s.id)
                setSelected(null)
              }}
              data-testid={`arch-${s.id}`}
            >
              <strong>{s.name}</strong>
              <span className="muted small">{s.summary}</span>
            </button>
          ))}
        </div>
        <h3>View</h3>
        <Slider label="explode" value={explode} min={0} max={1} step={0.01} onChange={setExplode} testId="explode" />
        <p className="small muted">
          Drag to rotate, scroll to zoom. Click a part to fly to it; click empty space to fly back. Parts flagged "drawn at a different size" would be invisible, or
          would hide everything else, at their true size.
        </p>
      </aside>

      <main className="pane pane-center">
        <ErrorBanner error={all.error} />
        <div className="viewport" data-testid="arch-viewport">
          {scene ? (
            <ArchitectureScene parts={scene.parts} explode={explode} selected={selected} onPick={setSelected} />
          ) : (
            <Solving what="Loading the designs" boot />
          )}
        </div>
        <div className="dock">
          {spectrum.data ? (
            <Plot
              data={Array.from({ length: nLevels - 1 }, (_, i) => ({
                x: spectrum.data!.sim.x,
                y: levels.map((row) => row[i + 1]),
                name: `level ${i + 1}`,
                line: { color: palette[(i + 1) % palette.length], width: 2 },
              }))}
              layout={layout(theme, {
                xaxis: { title: { text: spectrum.data.sim.x_label } },
                yaxis: { title: { text: `energy above level 0 (${spectrum.data.sim.unit})` } },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : spectrum.isFetching ? (
            <Solving what="Diagonalizing with scqubits" />
          ) : (
            <p className="small muted" style={{ padding: 12 }}>
              The dissipative cat's physics lives in the Phase space and Flip rates views.
            </p>
          )}
        </div>
      </main>

      <aside className="pane pane-right">
        {part ? (
          <section data-testid="arch-inspector">
            <h3>{part.name}</h3>
            <p className="small">{part.role}</p>
            <table className="facts">
              <tbody>
                <tr>
                  <td>Size</td>
                  <td className="mono">
                    {um(part.size[0])} × {um(part.size[1])} × {um(part.size[2])}
                  </td>
                </tr>
                <tr>
                  <td>Source</td>
                  <td>
                    <span className={`kind kind-${part.dims_class === 'MEASURED' ? 'peer_reviewed' : 'assumption'}`}>
                      {part.dims_class.toLowerCase()}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
            <p className="small muted">{part.dims_source}</p>
            {part.draw_size ? <p className="small warn">Drawn at a different size; the numbers above are the real ones.</p> : null}
          </section>
        ) : (
          <>
            <h3>{scene?.name}</h3>
            <p className="small">{scene?.summary}</p>
            <p className="small muted">Click a part to see what it does and where its size comes from.</p>
          </>
        )}
        <h3>Energy levels</h3>
        <p className="small muted" data-testid="arch-physics">
          {scene?.physics.source} ({String(scene?.physics.kind ?? '').replace('_', ' ')})
        </p>
        <h3>Papers</h3>
        <ul className="sources small">
          {(scene?.refs ?? []).map((r) => (
            <li key={r}>{shortCite(refs?.[r], r)}</li>
          ))}
        </ul>
      </aside>
    </div>
  )
}

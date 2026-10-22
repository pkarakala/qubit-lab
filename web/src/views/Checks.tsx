// Every closed-form formula against the simulation, with the verdict predicted beforehand.

import { useChecks, useFormulas } from '../api'
import { ErrorBanner, Tex } from '../components/ui'
import { fmt } from '../lib/colormap'

export function Checks() {
  const q = useChecks()
  const formulas = useFormulas().data ?? []
  const rows = q.data ?? []
  const surprises = rows.filter((r) => r.verdict !== r.expected).length
  return (
    <div className="view view-single">
      <main className="pane pane-scroll">
        <ErrorBanner error={q.error} />
        <h2>Theory vs simulation</h2>
        <p className="small muted lede">
          Each row evaluates one closed-form formula and the matching Dynamiqs or JAX simulation. <em>Expected</em> is
          the verdict predicted from the formula's validity range before running: a formula should agree inside it
          and diverge outside it. {q.data ? `${rows.length} checks, ${surprises} surprises.` : 'Running every check (about two minutes the first time)…'}
        </p>
        {formulas
          .filter((f) => rows.some((r) => r.formula === f.key))
          .map((f) => (
            <section key={f.key} className="check-group">
              <h4>
                {f.title} <Tex src={f.latex} />
              </h4>
              <table className="checks">
                <thead>
                  <tr>
                    <th>Regime</th>
                    <th>Theory</th>
                    <th>Simulation</th>
                    <th>Rel. error</th>
                    <th>Tolerance</th>
                    <th>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {rows
                    .filter((r) => r.formula === f.key)
                    .map((r) => (
                      <tr key={r.key} className={r.verdict === r.expected ? '' : 'is-surprise'}>
                        <td>{r.regime}</td>
                        <td className="mono">{fmt(r.theory, 6)}</td>
                        <td className="mono">{fmt(r.sim, 6)}</td>
                        <td className="mono">{fmt(r.rel_error, 2)}</td>
                        <td className="mono">{fmt(r.rel_tol)}</td>
                        <td>
                          <span className={`verdict verdict-${r.verdict.toLowerCase()}`}>{r.verdict}</span>
                          {r.verdict === r.expected ? <span className="muted small"> as predicted</span> : <strong> surprise</strong>}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </section>
          ))}
      </main>
    </div>
  )
}

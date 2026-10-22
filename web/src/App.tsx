// The explorer shell: top bar with the views, then the active view.

import { useEffect } from 'react'
import { CatVsTransmon } from './views/CatVsTransmon'
import { Checks } from './views/Checks'
import { FlipRates } from './views/FlipRates'
import { PhaseSpace } from './views/PhaseSpace'
import { Playground } from './views/Playground'
import { Qubit } from './views/Qubit'
import { Transmon } from './views/Transmon'
import { type View, useExplorer } from './store'

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: 'phase', label: 'Phase space', hint: 'One cat qubit, live' },
  { id: 'rates', label: 'Flip rates', hint: 'Bit-flips vs phase-flips' },
  { id: 'transmon', label: 'Transmon', hint: 'The other superconducting qubit' },
  { id: 'qubit', label: 'Qubit control', hint: 'Pulses and noise on a transmon' },
  { id: 'compare', label: 'Cat vs transmon', hint: 'Measured error rates and noise bias' },
  { id: 'playground', label: 'Playground', hint: 'Type your own Hamiltonian and losses' },
  { id: 'checks', label: 'Checks', hint: 'Theory vs simulation' },
]

const REPO = 'https://github.com/pkarakala/qubit-lab'

export default function App() {
  const view = useExplorer((s) => s.view)
  const setView = useExplorer((s) => s.setView)
  const theme = useExplorer((s) => s.theme)
  const toggleTheme = useExplorer((s) => s.toggleTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      const i = Number(e.key) - 1
      const v = VIEWS[i]
      if (v) setView(v.id)
      if (e.key === ' ' && useExplorer.getState().view === 'phase') {
        e.preventDefault()
        const s = useExplorer.getState()
        s.setPlaying(!s.playing)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setView])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-name">QUBIT LAB</span>
          <span className="brand-sub">cat qubits and transmons, theory vs simulation</span>
        </div>
        <nav className="tabs" role="tablist">
          {VIEWS.map((v, i) => (
            <button
              key={v.id}
              role="tab"
              aria-selected={v.id === view}
              className={`tab ${v.id === view ? 'is-active' : ''}`}
              onClick={() => setView(v.id)}
              title={`${v.hint} (${i + 1})`}
            >
              {v.label}
            </button>
          ))}
        </nav>
        <div className="topbar-spacer" />
        <a className="btn" href={`${REPO}/tree/main/docs/learn`} target="_blank" rel="noreferrer">
          Learn
        </a>
        <button className="btn" onClick={toggleTheme} aria-label="Toggle theme">
          {theme === 'light' ? 'Dark' : 'Light'}
        </button>
      </header>
      {view === 'phase' ? <PhaseSpace /> : null}
      {view === 'rates' ? <FlipRates /> : null}
      {view === 'transmon' ? <Transmon /> : null}
      {view === 'qubit' ? <Qubit /> : null}
      {view === 'compare' ? <CatVsTransmon /> : null}
      {view === 'playground' ? <Playground /> : null}
      {view === 'checks' ? <Checks /> : null}
      <footer className="footer small muted">
        Independent learning project. Every equation is cited; numbers from companies are labeled as claims. Not
        affiliated with Alice &amp; Bob or Qolab. Simulation: Dynamiqs + JAX.
      </footer>
    </div>
  )
}

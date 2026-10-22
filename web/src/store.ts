// Explorer state: which view, the knobs, and the playhead. Physics lives in the API.

import { create } from 'zustand'
import type { EvolveParams } from './api'

export type View = 'phase' | 'rates' | 'transmon' | 'checks'
export type Theme = 'light' | 'dark'

export interface Preset {
  id: string
  label: string
  blurb: string
  params: EvolveParams
}

const base: EvolveParams = {
  alpha_sq: 4,
  kappa_1: 0,
  n_th: 0,
  kappa_phi: 0,
  scheme: 'dissipative',
  initial: 'vacuum',
  t_final: 3,
  n_frames: 31,
  grid: 61,
}

// Guided starting points, one lesson each.
export const PRESETS: Preset[] = [
  { id: 'grow', label: 'Grow a cat', blurb: 'Vacuum flows into an even cat under two-photon dissipation.', params: base },
  {
    id: 'decay',
    label: 'Lose the cat',
    blurb: 'Single-photon loss flips parity and washes out the fringes. The blobs survive.',
    params: { ...base, initial: 'even_cat', kappa_1: 0.1, t_final: 6 },
  },
  {
    id: 'unprotected',
    label: 'No protection',
    blurb: 'Photon loss with no stabilization: the even cat decays to vacuum.',
    params: { ...base, initial: 'even_cat', kappa_1: 0.5, alpha_sq: 4, t_final: 6, scheme: 'none' },
  },
  {
    id: 'kerr',
    label: 'Kerr cat',
    blurb: 'A Hamiltonian instead of dissipation. Vacuum is not a ground state here, so it sloshes instead of settling.',
    params: { ...base, initial: 'vacuum', scheme: 'kerr', t_final: 3 },
  },
]

interface ExplorerState {
  view: View
  theme: Theme
  params: EvolveParams
  frame: number
  playing: boolean
  showTheory: boolean
  ratio: number // transmon E_J / E_C
  E_C: number // transmon charging energy, GHz
  setView: (v: View) => void
  toggleTheme: () => void
  setParams: (p: Partial<EvolveParams>) => void
  setFrame: (f: number) => void
  setPlaying: (p: boolean) => void
  toggleTheory: () => void
  setRatio: (r: number) => void
}

const prefersDark = typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches

export const useExplorer = create<ExplorerState>((set) => ({
  view: 'phase',
  theme: prefersDark ? 'dark' : 'light',
  params: base,
  frame: 0,
  playing: false,
  showTheory: false,
  ratio: 50,
  E_C: 0.3,
  setView: (view) => set({ view }),
  toggleTheme: () => set((s) => ({ theme: s.theme === 'light' ? 'dark' : 'light' })),
  setParams: (p) => set((s) => ({ params: { ...s.params, ...p }, frame: 0 })),
  setFrame: (frame) => set({ frame }),
  setPlaying: (playing) => set({ playing }),
  toggleTheory: () => set((s) => ({ showTheory: !s.showTheory })),
  setRatio: (ratio) => set({ ratio }),
}))

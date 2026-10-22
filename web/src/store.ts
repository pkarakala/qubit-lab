// Explorer state: which view, the knobs, and the playhead. Physics lives in the API.

import { create } from 'zustand'
import type { EvolveParams, QubitParams } from './api'

export type View = 'phase' | 'rates' | 'transmon' | 'qubit' | 'pulse' | 'qec' | 'compare' | 'playground' | 'checks'
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

const qubitBase: QubitParams = {
  experiment: 'rabi',
  alpha_ghz: -0.3,
  t1_ns: 20000,
  t2_ns: 20000,
  sigma_ghz: 0,
  delta_ghz: 0,
  omega_ghz: 0.02,
  duration_ns: 20,
  drag: 0,
  t_max_ns: 100,
  n_points: 121,
}

export interface QubitPreset {
  id: string
  label: string
  blurb: string
  params: QubitParams
}

// One lesson each, in the order a lab would run them.
export const QUBIT_PRESETS: QubitPreset[] = [
  { id: 'rabi', label: 'Rabi', blurb: 'A steady drive rotates |0> to |1> and back. Detune it and the circle tilts.', params: qubitBase },
  {
    id: 'pi',
    label: 'Fast π pulse',
    blurb: 'A 6 ns Gaussian pulse is too fast for a 200 MHz anharmonicity: some population leaks to level 2.',
    params: { ...qubitBase, experiment: 'pulse', alpha_ghz: -0.2, duration_ns: 6, n_points: 161 },
  },
  {
    id: 'drag',
    label: 'DRAG',
    blurb: 'The same pulse plus a quadrature drive (Motzoi 2009). Leakage nearly vanishes, but P1 still falls short: level 2 also shifts the qubit frequency, which first-order DRAG does not correct.',
    params: { ...qubitBase, experiment: 'pulse', alpha_ghz: -0.2, duration_ns: 6, drag: 1, n_points: 161 },
  },
  {
    id: 't1',
    label: 'T1',
    blurb: 'Start in |1> and wait. Energy leaks to the environment and the state falls to |0>.',
    params: { ...qubitBase, experiment: 't1', t1_ns: 50, t2_ns: 80, t_max_ns: 200 },
  },
  {
    id: 'ramsey',
    label: 'Ramsey',
    blurb: 'Start on the equator, detuned 20 MHz. The state precesses and shrinks; slow frequency noise makes it shrink faster.',
    params: { ...qubitBase, experiment: 'ramsey', delta_ghz: 0.02, t1_ns: 200, t2_ns: 150, sigma_ghz: 0.003, t_max_ns: 200 },
  },
  {
    id: 'echo',
    label: 'Echo',
    blurb: 'Same noise, plus a π pulse halfway. The slow noise cancels and only T2 is left.',
    params: { ...qubitBase, experiment: 'echo', t1_ns: 200, t2_ns: 150, sigma_ghz: 0.003, t_max_ns: 200, n_points: 41 },
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
  qubit: QubitParams
  setQubit: (p: Partial<QubitParams>) => void
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
  qubit: qubitBase,
  setQubit: (p) => set((s) => ({ qubit: { ...s.qubit, ...p } })),
}))

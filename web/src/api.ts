// Typed client for the qubitlab API. Every response keeps `sim` (Dynamiqs) and `theory`
// (closed-form SymPy formulas) apart, and the explorer never computes physics itself.

import { keepPreviousData, useQuery } from '@tanstack/react-query'

export type Scheme = 'dissipative' | 'kerr' | 'none'
export type Initial = 'vacuum' | 'coherent_plus' | 'coherent_minus' | 'even_cat' | 'odd_cat'

export interface CatParams {
  alpha_sq: number
  kappa_1: number
  n_th: number
  kappa_phi: number
  scheme: Scheme
}

export interface EvolveParams extends CatParams {
  initial: Initial
  t_final: number
  n_frames: number
  grid: number
}

export interface Source {
  key: string
  kind: string
  locator: string
  note: string
}

export interface Formula {
  key: string
  title: string
  latex: string
  args: string[]
  valid_when: string
  sources: Source[]
}

export interface Evolution {
  sim: {
    n_fock: number
    times: number[]
    photon_number: number[]
    parity: number[]
    sign_x: number[]
    purity: number[]
    xvec: number[]
    wigner: number[][][]
  }
  theory: {
    even_cat_wigner: number[][] | null
    odd_cat_wigner: number[][] | null
    stabilized_photon_number?: number
    parity_decay_rate?: number
    parity_curve?: number[]
  }
}

export interface RateRow {
  alpha_sq: number
  sim: { bit_flip_rate: number | null; phase_flip_rate: number | null; n_fock: number }
  theory: { phase_flip_rate: number | null; bit_flip_rate_dephasing: number | null }
}

export interface Rates {
  rows: RateRow[]
  theory: { bit_flip_suppression_per_photon: number }
  rate_floor: number
}

export interface Bands {
  sim: { n_g: number[]; energies: number[][]; dispersion: number[]; e01_mean: number; anharmonicity: number }
  theory: { level_spacing_from_ground: number[]; e01: number; anharmonicity: number; dispersion: number[] }
}

export interface Surface {
  sim: { ratios: number[]; n_g: number[]; energies_over_e01: number[][][]; ripple_over_ec: number[][][] }
}

export type Experiment = 'rabi' | 'pulse' | 't1' | 'ramsey' | 'echo'

export interface QubitParams {
  experiment: Experiment
  alpha_ghz: number
  t1_ns: number
  t2_ns: number
  sigma_ghz: number
  delta_ghz: number
  omega_ghz: number
  duration_ns: number
  drag: number
  t_max_ns: number
  n_points: number
}

export interface QubitRun {
  sim: {
    times: number[]
    bloch: number[][] // [time][x, y, z], z = P0 - P1
    populations: number[][] // [time][level]
    leakage: number[]
    omega_x: number[]
    omega_y: number[]
  }
  theory: { curve: number[] | null; curve_label: string; idle_fidelity_at_end: number }
}

export interface CheckRow {
  key: string
  formula: string
  regime: string
  theory: number
  sim: number
  rel_error: number
  rel_tol: number
  verdict: 'AGREE' | 'DIVERGE'
  expected: 'AGREE' | 'DIVERGE'
}

export interface Fact {
  key: string
  label: string
  value: number
  unit: string
  alpha_sq: number | null
  cite: string
  kind: string
  locator: string
  qubit: 'cat' | 'transmon'
}

export interface CompareDevice {
  key: string
  label: string
  qubit: 'cat' | 'transmon'
  t_bit: string // fact key
  t_phase: string // fact key
  theory: { p_bit: number; p_phase: number; bias: number; idle_error: number | null }
  sim: { p_bit: number; p_phase: number; idle_error: number } | null
}

export interface Compare {
  t_ns: number
  devices: CompareDevice[]
}

export type BibEntry = Record<string, string>

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`)
  return (await r.json()) as T
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`)
  return (await r.json()) as T
}

const forever = { staleTime: Infinity, gcTime: Infinity }

export const useFormulas = () => useQuery({ queryKey: ['formulas'], queryFn: () => get<Formula[]>('/api/formulas'), ...forever })
export const useReferences = () =>
  useQuery({ queryKey: ['references'], queryFn: () => get<Record<string, BibEntry>>('/api/references'), ...forever })
export const useFacts = () => useQuery({ queryKey: ['facts'], queryFn: () => get<Fact[]>('/api/facts'), ...forever })
export const useChecks = () => useQuery({ queryKey: ['checks'], queryFn: () => get<CheckRow[]>('/api/checks'), ...forever })

export const useEvolution = (p: EvolveParams) =>
  useQuery({
    queryKey: ['evolve', p],
    queryFn: () => post<Evolution>('/api/cat/evolve', p),
    placeholderData: keepPreviousData,
    ...forever,
  })

export const useRates = (p: CatParams, values: number[]) =>
  useQuery({
    queryKey: ['rates', p, values],
    queryFn: () => post<Rates>('/api/cat/rates', { ...p, alpha_sq_values: values }),
    placeholderData: keepPreviousData,
    ...forever,
  })

export const useBands = (E_J: number, E_C: number) =>
  useQuery({
    queryKey: ['bands', E_J, E_C],
    queryFn: () => get<Bands>(`/api/transmon/bands?E_J=${E_J}&E_C=${E_C}&n_levels=4&n_points=101`),
    placeholderData: keepPreviousData,
    ...forever,
  })

export const useSurface = () =>
  useQuery({ queryKey: ['surface'], queryFn: () => get<Surface>('/api/transmon/surface?n_levels=3'), ...forever })

export const useQubit = (p: QubitParams) =>
  useQuery({
    queryKey: ['qubit', p],
    queryFn: () => post<QubitRun>('/api/qubit/run', p),
    placeholderData: keepPreviousData,
    ...forever,
  })

export interface QecCurve {
  code: 'surface' | 'repetition'
  d: number
  qubits: number
  per_round: (number | null)[]
  failures: number[]
}

export interface QecCurves {
  p: number[]
  shots: number
  sim: { curves: QecCurve[] }
  theory: { fowler_surface: Record<string, number[]>; fowler_p_th: number; surface_qubits: Record<string, number> }
}

export interface CatCodeRow {
  d: number
  qubits: number
  sim: { phase_flip_logical: number | null; failures: number | null }
  theory: { bit_flip_floor: number }
}

export interface QecShot {
  sim: {
    qubits: [number, number, boolean][]
    detectors: [number, number, number][]
    fired: number[]
    matched: [number, number][]
    logical_error: boolean
  }
}

export const useQecCurves = () => useQuery({ queryKey: ['qec-curves'], queryFn: () => get<QecCurves>('/api/qec/curves'), ...forever })

export const useCatCode = (p_phase: number, p_bit: number) =>
  useQuery({
    queryKey: ['qec-cat', p_phase, p_bit],
    queryFn: () => post<{ rows: CatCodeRow[]; shots: number }>('/api/qec/cat', { p_phase, p_bit }),
    placeholderData: keepPreviousData,
    ...forever,
  })

export const useQecShot = (d: number, p: number, seed: number) =>
  useQuery({
    queryKey: ['qec-shot', d, p, seed],
    queryFn: () => get<QecShot>(`/api/qec/shot?d=${d}&p=${p}&seed=${seed}`),
    placeholderData: keepPreviousData,
    ...forever,
  })

export const useCompare = (t_ns: number) =>
  useQuery({
    queryKey: ['compare', t_ns],
    queryFn: () => get<Compare>(`/api/compare?t_ns=${t_ns}`),
    placeholderData: keepPreviousData,
    ...forever,
  })

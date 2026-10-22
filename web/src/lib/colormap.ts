// Diverging colormap for the Wigner function: negative (quantum) values blue, positive red,
// zero the neutral surface color. Symmetric about zero so the sign is never ambiguous.

export type RGB = [number, number, number]

export const PALETTES: Record<'light' | 'dark', { neg: RGB; zero: RGB; pos: RGB }> = {
  light: { neg: [0.16, 0.38, 0.85], zero: [0.93, 0.93, 0.91], pos: [0.86, 0.27, 0.18] },
  dark: { neg: [0.35, 0.58, 1.0], zero: [0.2, 0.2, 0.22], pos: [1.0, 0.42, 0.3] },
}

/** Map w in [-limit, limit] to a color. Values outside are clamped. */
export function diverging(w: number, limit: number, theme: 'light' | 'dark'): RGB {
  const { neg, zero, pos } = PALETTES[theme]
  const t = Math.max(-1, Math.min(1, limit > 0 ? w / limit : 0))
  const end = t < 0 ? neg : pos
  const a = Math.abs(t) ** 0.8
  return [zero[0] + (end[0] - zero[0]) * a, zero[1] + (end[1] - zero[1]) * a, zero[2] + (end[2] - zero[2]) * a]
}

/** Engineering format: 3 significant figures, exponent for very small or large values. */
export function fmt(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '–'
  if (x === 0) return '0'
  const a = Math.abs(x)
  if (a < 1e-3 || a >= 1e5) return x.toExponential(digits - 1).replace('e', '×10^').replace('+', '')
  return Number(x.toPrecision(digits)).toString()
}

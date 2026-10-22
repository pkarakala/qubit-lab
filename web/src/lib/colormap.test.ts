import { describe, expect, it } from 'vitest'
import { PALETTES, diverging, fmt } from './colormap'

describe('diverging', () => {
  it('maps zero to the neutral color', () => {
    expect(diverging(0, 1, 'light')).toEqual(PALETTES.light.zero)
  })
  it('maps the limits to the end colors and clamps beyond', () => {
    diverging(-1, 1, 'light').forEach((v, i) => expect(v).toBeCloseTo(PALETTES.light.neg[i] ?? NaN, 12))
    diverging(5, 1, 'light').forEach((v, i) => expect(v).toBeCloseTo(PALETTES.light.pos[i] ?? NaN, 12))
  })
  it('is symmetric in magnitude', () => {
    const p = diverging(0.3, 1, 'dark')
    const n = diverging(-0.3, 1, 'dark')
    const z = PALETTES.dark.zero
    const dist = (c: number[]) => Math.hypot(...c.map((v, i) => v - (z[i] ?? 0)))
    expect(dist(p) / dist(n)).toBeCloseTo(dist([...PALETTES.dark.pos]) / dist([...PALETTES.dark.neg]), 6)
  })
})

describe('fmt', () => {
  it('formats small rates with an exponent', () => {
    expect(fmt(4.93e-8)).toBe('4.93×10^-8')
    expect(fmt(0.08)).toBe('0.08')
    expect(fmt(null)).toBe('–')
  })
})

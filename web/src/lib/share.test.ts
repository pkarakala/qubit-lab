import { describe, expect, it } from 'vitest'
import { decode, encode } from './share'

describe('share links', () => {
  it('round-trips state, including non-ASCII text', () => {
    const state = { view: 'playground', playground: { hamiltonian: '-K*ad**2*a**2', note: 'κ₁ ≈ |α|²' }, qec: { d: 5 } }
    expect(decode(encode(state))).toEqual(state)
  })
  it('uses only URL-safe characters', () => {
    expect(encode({ x: '???>>>~~~' })).toMatch(/^[A-Za-z0-9_-]*$/)
  })
  it('ignores a broken or non-object hash', () => {
    expect(decode('not base64!')).toEqual({})
    expect(decode(encode({}).slice(0, 0))).toEqual({})
    expect(decode(btoa('[1,2]'))).toEqual({})
  })
})

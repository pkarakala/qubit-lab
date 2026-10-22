// Shareable links: every knob that matters lives in the URL hash, as base64url JSON keyed by
// name ('view', 'qubit', 'pulse.req', ...). Opening the link restores the same view and inputs.

import { useCallback, useState } from 'react'

export function encode(state: Record<string, unknown>): string {
  const bytes = new TextEncoder().encode(JSON.stringify(state))
  let bin = ''
  for (const b of bytes) bin += String.fromCharCode(b)
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

export function decode(hash: string): Record<string, unknown> {
  try {
    const bin = atob(hash.replace(/-/g, '+').replace(/_/g, '/'))
    const parsed: unknown = JSON.parse(new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0))))
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

const shared: Record<string, unknown> = typeof window === 'undefined' ? {} : decode(window.location.hash.slice(1))

/** The value stored under `key` in the link, or `fallback`. */
export function fromLink<T>(key: string, fallback: T): T {
  return key in shared ? (shared[key] as T) : fallback
}

/** Store `value` under `key` in the link, without adding a browser history entry. */
export function toLink(key: string, value: unknown): void {
  shared[key] = value
  window.history.replaceState(null, '', `#${encode(shared)}`)
}

/** useState whose value is also kept in the link. */
export function useShared<T>(key: string, initial: T): [T, (v: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => fromLink(key, initial))
  const set = useCallback(
    (v: T | ((prev: T) => T)) =>
      setValue((prev) => {
        const next = typeof v === 'function' ? (v as (prev: T) => T)(prev) : v
        toLink(key, next)
        return next
      }),
    [key],
  )
  return [value, set]
}

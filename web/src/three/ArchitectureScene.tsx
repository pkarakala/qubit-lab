// A qubit design as a stack of parts that explodes apart. Each part is a box at its cited size
// (or at draw_size when the true size would be invisible). Click a part to inspect it.

import { OrbitControls } from '@react-three/drei'
import { type ThreeEvent, useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { ArchPart } from '../api'
import { LabeledCanvas } from './LabeledCanvas'

const COLORS: Record<string, { color: string; opacity: number; metalness: number }> = {
  sapphire: { color: '#9cc4e4', opacity: 0.45, metalness: 0 },
  silicon: { color: '#7d8590', opacity: 0.85, metalness: 0.1 },
  aluminium: { color: '#d4d7dc', opacity: 1, metalness: 0.8 },
  aluminium_bulk: { color: '#c9ccd2', opacity: 0.35, metalness: 0.8 },
  tantalum: { color: '#8f9bb3', opacity: 1, metalness: 0.7 },
  aluminium_film: { color: '#d4d7dc', opacity: 0.3, metalness: 0.6 },
  tantalum_film: { color: '#8f9bb3', opacity: 0.3, metalness: 0.6 },
  junction: { color: '#ff7a3d', opacity: 1, metalness: 0.2 },
  copper: { color: '#c47a45', opacity: 0.9, metalness: 0.8 },
}

type V = [number, number, number]

/** Scene micrometres to world units, fitted to every part at the current explode: the largest
 * extent becomes 10, and the scene is centred. µm z is up; three.js y is up. */
function useFit(parts: ArchPart[], t: number) {
  return useMemo(() => {
    const lo = [Infinity, Infinity, Infinity]
    const hi = [-Infinity, -Infinity, -Infinity]
    for (const p of parts) {
      const d = p.draw_size ?? p.size
      for (let i = 0; i < 3; i++) {
        const c = p.position[i]! + p.explode[i]! * t
        lo[i] = Math.min(lo[i]!, c - d[i]! / 2)
        hi[i] = Math.max(hi[i]!, c + d[i]! / 2)
      }
    }
    const extent = Math.max(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!, 1)
    const center: V = [(lo[0]! + hi[0]!) / 2, (lo[1]! + hi[1]!) / 2, (lo[2]! + hi[2]!) / 2]
    return { k: 10 / extent, center }
  }, [parts, t])
}

const toWorld = (v: V, k: number, c: V = [0, 0, 0]): V => [(v[0] - c[0]) * k, (v[2] - c[2]) * k, -(v[1] - c[1]) * k]

function Box({
  part,
  k,
  c,
  t,
  selected,
  onPick,
}: {
  part: ArchPart
  k: number
  c: V
  t: number
  selected: boolean
  onPick: (id: string) => void
}) {
  const ref = useRef<THREE.Mesh>(null)
  const d = part.draw_size ?? part.size
  const minDim = 0.03 / k // never thinner than 0.03 world units, so films stay visible
  const dims: V = [Math.max(d[0], minDim) * k, Math.max(d[2], minDim) * k, Math.max(d[1], minDim) * k]
  const home = toWorld(part.position, k, c)
  const away = toWorld(part.explode, k)
  const look = COLORS[part.material] ?? COLORS.aluminium!
  useFrame(() => {
    const m = ref.current
    if (!m) return
    const goal = new THREE.Vector3(home[0] + away[0] * t, home[1] + away[1] * t, home[2] + away[2] * t)
    m.position.lerp(goal, 0.15)
  })
  return (
    <mesh
      ref={ref}
      position={home}
      onClick={(e: ThreeEvent<MouseEvent>) => {
        e.stopPropagation()
        onPick(part.id)
      }}
    >
      <boxGeometry args={dims} />
      <meshStandardMaterial
        color={look.color}
        transparent={look.opacity < 1}
        opacity={look.opacity}
        metalness={look.metalness}
        roughness={0.4}
        emissive={selected ? '#ffd166' : '#000000'}
        emissiveIntensity={selected ? 0.6 : 0}
        depthWrite={look.opacity >= 1}
      />
    </mesh>
  )
}

export function ArchitectureScene({
  parts,
  explode,
  selected,
  onPick,
}: {
  parts: ArchPart[]
  explode: number
  selected: string | null
  onPick: (id: string | null) => void
}) {
  const { k, center } = useFit(parts, explode)
  const sel = parts.find((p) => p.id === selected)
  const labels = sel
    ? [{ position: toWorld(sel.position.map((x, i) => x + sel.explode[i]! * explode) as V, k, center).map((x, i) => (i === 1 ? x + 0.6 : x)) as V, text: sel.name }]
    : []
  return (
    <LabeledCanvas labels={labels} camera={{ position: [8, 6, 10], fov: 45 }} onPointerMissed={() => onPick(null)}>
      <ambientLight intensity={0.7} />
      <directionalLight position={[8, 12, 6]} intensity={1.4} />
      <directionalLight position={[-6, 4, -8]} intensity={0.5} />
      {parts.map((p) => (
        <Box key={p.id} part={p} k={k} c={center} t={explode} selected={p.id === selected} onPick={onPick} />
      ))}
      <OrbitControls makeDefault enableDamping />
    </LabeledCanvas>
  )
}

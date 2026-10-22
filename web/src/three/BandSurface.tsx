// How each transmon level wiggles with offset charge n_g, for E_J/E_C from 1 to 100.
// Sheet m sits at height m and shows E_m(n_g) minus its average, in units of E_C. Near the
// Cooper pair box (E_J/E_C ~ 1) the sheets ripple; in the transmon regime they are flat.

import { Html, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useMemo } from 'react'
import * as THREE from 'three'
import { HtmlWarmup } from './HtmlWarmup'

const SPAN_X = 6 // n_g from -1 to 1
const SPAN_Z = 7 // log10(E_J/E_C) from 0 to 2
const HEIGHT = 0.22 // world units per E_C of ripple
const GAP = 1.9 // world units between sheets

export function sheetGeometry(ng: number[], ratios: number[], energies: number[][][], level: number) {
  const nx = ng.length
  const nz = ratios.length
  const g = new THREE.BufferGeometry()
  const pos = new Float32Array(nx * nz * 3)
  for (let k = 0; k < nz; k++) {
    for (let i = 0; i < nx; i++) {
      const o = (k * nx + i) * 3
      pos[o] = ((ng[i] ?? 0) / 1) * (SPAN_X / 2)
      pos[o + 1] = level * GAP + (energies[k]?.[i]?.[level] ?? 0) * HEIGHT
      pos[o + 2] = SPAN_Z / 2 - (Math.log10(ratios[k] ?? 1) / 2) * SPAN_Z
    }
  }
  const idx: number[] = []
  for (let k = 0; k < nz - 1; k++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = k * nx + i
      idx.push(a, a + 1, a + nx, a + 1, a + nx + 1, a + nx)
    }
  }
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  g.setIndex(idx)
  g.computeVertexNormals()
  return g
}

const LEVEL_COLORS = ['#3d7be0', '#e0533d', '#2f8a57']

/** One polyline over n_g per E_J/E_C value: a ridge plot that reads well from any angle. */
export function ridgeGeometry(ng: number[], ratios: number[], energies: number[][][], level: number) {
  const pts: number[] = []
  const at = (k: number, i: number) => [
    (ng[i] ?? 0) * (SPAN_X / 2),
    level * GAP + (energies[k]?.[i]?.[level] ?? 0) * HEIGHT,
    SPAN_Z / 2 - (Math.log10(ratios[k] ?? 1) / 2) * SPAN_Z,
  ]
  for (let k = 0; k < ratios.length; k += 2) {
    for (let i = 0; i < ng.length - 1; i++) pts.push(...at(k, i), ...at(k, i + 1))
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts), 3))
  return g
}

export function BandScene({
  ng,
  ratios,
  energies,
  ratio,
  theme,
}: {
  ng: number[]
  ratios: number[]
  energies: number[][][]
  ratio: number
  theme: 'light' | 'dark'
}) {
  const sheets = useMemo(() => [0, 1, 2].map((m) => sheetGeometry(ng, ratios, energies, m)), [ng, ratios, energies])
  const ridges = useMemo(() => [0, 1, 2].map((m) => ridgeGeometry(ng, ratios, energies, m)), [ng, ratios, energies])
  const ink = theme === 'dark' ? '#d8d6cf' : '#30302c'
  const zMark = SPAN_Z / 2 - (Math.log10(ratio) / 2) * SPAN_Z
  return (
    <Canvas camera={{ position: [2.5, 7.5, 11], fov: 34 }} dpr={[1, 2]} gl={{ preserveDrawingBuffer: true }} data-testid="band-canvas">
      <HtmlWarmup />
      <color attach="background" args={[theme === 'dark' ? '#121214' : '#f6f5f2']} />
      <ambientLight intensity={0.8} />
      <directionalLight position={[5, 9, 6]} intensity={1.4} />
      <group position={[0, -1.8, 0]}>
        {sheets.map((g, m) => (
          <mesh key={m} geometry={g}>
            <meshStandardMaterial color={LEVEL_COLORS[m]} side={THREE.DoubleSide} roughness={0.8} transparent opacity={0.22} depthWrite={false} />
          </mesh>
        ))}
        {ridges.map((g, m) => (
          <lineSegments key={m} geometry={g}>
            <lineBasicMaterial color={LEVEL_COLORS[m]} />
          </lineSegments>
        ))}
        <lineSegments position={[0, 0, zMark]}>
          <edgesGeometry args={[new THREE.PlaneGeometry(SPAN_X + 0.4, 5.2).translate(0, 1.8, 0)]} />
          <lineBasicMaterial color={ink} transparent opacity={0.6} />
        </lineSegments>
        <Html position={[0, 4.6, zMark]} center className="axis-label">
          E_J/E_C = {Math.round(ratio)}
        </Html>
        <Html position={[SPAN_X / 2 + 0.4, 0, SPAN_Z / 2]} center className="axis-label">
          n_g
        </Html>
        <Html position={[-SPAN_X / 2 - 0.9, 0, SPAN_Z / 2]} center className="axis-label">
          E_J/E_C = 1
        </Html>
        <Html position={[-SPAN_X / 2 - 0.9, 0, -SPAN_Z / 2]} center className="axis-label">
          E_J/E_C = 100
        </Html>
        {[0, 1, 2].map((m) => (
          <Html key={m} position={[SPAN_X / 2 + 0.5, m * GAP, -SPAN_Z / 2]} center className="axis-label">
            level {m}
          </Html>
        ))}
        <gridHelper args={[SPAN_X, 4, ink, ink]} position={[0, -0.01, 0]} scale={[1, 1, SPAN_Z / SPAN_X]} material-opacity={0.15} material-transparent />
      </group>
      <OrbitControls makeDefault enableDamping target={[0, 0.6, 0]} />
    </Canvas>
  )
}

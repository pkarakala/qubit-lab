// The Wigner function W(x, p) as a 3D height field. Height and color both show W; blue dips
// are negative values, the signature of a non-classical state. The optional ghost is the
// closed-form ideal cat (theory layer), drawn as a wireframe over the simulated surface.

import { OrbitControls } from '@react-three/drei'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { type Label, LabeledCanvas } from './LabeledCanvas'
import { diverging } from '../lib/colormap'

const HEIGHT = 2.4 // world units for W = 2/pi
const W_MAX = 2 / Math.PI
const SPAN = 6 // world width of the phase-space window
const COLOR_LIMIT = 1 / Math.PI // a cat blob's peak; brighter colors for typical states

function useGridGeometry(n: number) {
  return useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * n * 3), 3))
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * n * 3), 3))
    const idx: number[] = []
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < n - 1; j++) {
        const a = i * n + j
        idx.push(a, a + n, a + 1, a + 1, a + n, a + n + 1)
      }
    }
    g.setIndex(idx)
    return g
  }, [n])
}

function fill(g: THREE.BufferGeometry, xvec: number[], w: number[][], theme: 'light' | 'dark', colors: boolean) {
  const n = xvec.length
  const ext = xvec[n - 1] ?? 1
  const pos = g.getAttribute('position') as THREE.BufferAttribute
  const col = g.getAttribute('color') as THREE.BufferAttribute
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const k = i * n + j
      const v = w[i]?.[j] ?? 0
      // x along world X, p along world -Z (so +p points away from the default camera)
      pos.setXYZ(k, ((xvec[i] ?? 0) / ext) * (SPAN / 2), (v / W_MAX) * HEIGHT, (-(xvec[j] ?? 0) / ext) * (SPAN / 2))
      if (colors) {
        const [r, gg, b] = diverging(v, COLOR_LIMIT, theme)
        // vertex colors are linear; the palette is sRGB
        col.setXYZ(k, r ** 2.2, gg ** 2.2, b ** 2.2)
      }
    }
  }
  pos.needsUpdate = true
  col.needsUpdate = true
  g.computeVertexNormals()
  g.computeBoundingSphere()
}

function Surface({ xvec, w, theme }: { xvec: number[]; w: number[][]; theme: 'light' | 'dark' }) {
  const g = useGridGeometry(xvec.length)
  useEffect(() => fill(g, xvec, w, theme, true), [g, xvec, w, theme])
  return (
    <mesh geometry={g}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} roughness={0.55} metalness={0.05} />
    </mesh>
  )
}

function Ghost({ xvec, w, color }: { xvec: number[]; w: number[][]; color: string }) {
  // Every third grid line in each direction, so the simulated surface stays readable underneath.
  const geometry = useMemo(() => {
    const n = xvec.length
    const ext = xvec[n - 1] ?? 1
    const stride = 3
    const pts: number[] = []
    const at = (i: number, j: number) => [
      ((xvec[i] ?? 0) / ext) * (SPAN / 2),
      ((w[i]?.[j] ?? 0) / W_MAX) * HEIGHT + 0.01,
      (-(xvec[j] ?? 0) / ext) * (SPAN / 2),
    ]
    for (let i = 0; i < n; i += stride) {
      for (let j = 0; j < n - 1; j++) pts.push(...at(i, j), ...at(i, j + 1), ...at(j, i), ...at(j + 1, i))
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts), 3))
    return g
  }, [xvec, w])
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={color} transparent opacity={0.45} />
    </lineSegments>
  )
}

function Grid({ ext, ink }: { ext: number; ink: string }) {
  const ticks = useMemo(() => {
    const out: number[] = []
    for (let t = -Math.floor(ext); t <= Math.floor(ext); t++) out.push(t)
    return out
  }, [ext])
  return (
    <gridHelper args={[SPAN, ticks.length - 1 || 1, ink, ink]} position={[0, -0.001, 0]} material-opacity={0.18} material-transparent />
  )
}

const LABELS: Label[] = [
  { position: [SPAN / 2 + 0.35, 0, 0], text: 'x = Re β' },
  { position: [0, 0, -SPAN / 2 - 0.35], text: 'p = Im β' },
]

export function WignerScene({
  xvec,
  w,
  ghost,
  theme,
}: {
  xvec: number[]
  w: number[][]
  ghost: number[][] | null
  theme: 'light' | 'dark'
}) {
  const ink = theme === 'dark' ? '#d8d6cf' : '#30302c'
  const controls = useRef(null)
  return (
    <LabeledCanvas
      labels={LABELS}
      camera={{ position: [5.2, 4.6, 6.4], fov: 38 }}
      dpr={[1, 2]}
      gl={{ preserveDrawingBuffer: true, antialias: true }}
      data-testid="wigner-canvas"
    >
      <color attach="background" args={[theme === 'dark' ? '#121214' : '#f6f5f2']} />
      <ambientLight intensity={0.75} />
      <directionalLight position={[4, 8, 5]} intensity={1.6} />
      <directionalLight position={[-6, 3, -4]} intensity={0.5} />
      <Surface xvec={xvec} w={w} theme={theme} />
      {ghost ? <Ghost xvec={xvec} w={ghost} color={ink} /> : null}
      <Grid ext={xvec[xvec.length - 1] ?? 3} ink={ink} />
      <OrbitControls ref={controls} makeDefault enableDamping target={[0, 0.4, 0]} maxPolarAngle={Math.PI / 2.05} />
    </LabeledCanvas>
  )
}

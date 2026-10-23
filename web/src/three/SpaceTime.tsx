// One surface code run in space-time. The chip is the floor: data qubits (light) and measure
// qubits (dark). Each round of stabilizer measurements stacks one layer up. A red dot is a
// detection event; lines are the pairs the decoder (PyMatching) joined, or a detector it sent
// to the boundary. Errors leave pairs of events; decoding is guessing which errors made them.

import { Line, OrbitControls } from '@react-three/drei'
import { useMemo } from 'react'
import { type Label, LabeledCanvas } from './LabeledCanvas'

type Vec = [number, number, number]
const LAYER = 1.4 // world units per round

export function SpaceTimeScene({
  qubits,
  detectors,
  fired,
  matched,
  theme,
}: {
  qubits: [number, number, boolean][]
  detectors: [number, number, number][]
  fired: number[]
  matched: [number, number][]
  theme: 'light' | 'dark'
}) {
  const span = Math.max(...qubits.map((q) => Math.max(q[0], q[1])), 1)
  const rounds = Math.max(...detectors.map((d) => d[2]), 0)
  const at = (x: number, y: number, t: number): Vec => [x - span / 2, t * LAYER, y - span / 2]
  const ink = theme === 'dark' ? '#d8d6cf' : '#30302c'
  const hot = theme === 'dark' ? '#ff7a5c' : '#c4432f'
  const edge = theme === 'dark' ? '#b29cff' : '#6b4fc4'
  const firedSet = useMemo(() => new Set(fired), [fired])
  const lines = useMemo(
    () =>
      matched.map(([a, b]) => {
        const da = detectors[a >= 0 ? a : b]
        if (!da) return null
        const pa = at(da[0], da[1], da[2])
        if (a >= 0 && b >= 0) {
          const db = detectors[b]
          return db ? [pa, at(db[0], db[1], db[2])] : null
        }
        // Boundary: draw a short stub toward the nearest edge of the patch.
        const x = da[0] < span / 2 ? -1 : span + 1
        return [pa, at(x, da[1], da[2])]
      }),
    [matched, detectors, span],
  )
  const labels: Label[] = [
    { position: at(span + 1.5, 0, -0.6), text: 'chip' },
    { position: at(span + 1.5, 0, rounds), text: `round ${rounds}` },
  ]
  return (
    <LabeledCanvas labels={labels} camera={{ position: [span * 1.1, rounds * LAYER * 0.9 + 4, span * 1.5], fov: 40 }} dpr={[1, 2]} gl={{ preserveDrawingBuffer: true }} data-testid="spacetime-canvas">
      <color attach="background" args={[theme === 'dark' ? '#121214' : '#f6f5f2']} />
      <ambientLight intensity={0.9} />
      <directionalLight position={[5, 10, 6]} intensity={1.1} />
      {qubits.map(([x, y, data], i) => (
        <mesh key={i} position={at(x, y, -0.6)}>
          <sphereGeometry args={[data ? 0.18 : 0.12, 16, 12]} />
          <meshStandardMaterial color={data ? (theme === 'dark' ? '#e8e6df' : '#ffffff') : ink} />
        </mesh>
      ))}
      {detectors.map(([x, y, t], i) =>
        firedSet.has(i) ? (
          <mesh key={i} position={at(x, y, t)}>
            <sphereGeometry args={[0.16, 16, 12]} />
            <meshStandardMaterial color={hot} />
          </mesh>
        ) : (
          <mesh key={i} position={at(x, y, t)}>
            <sphereGeometry args={[0.04, 8, 6]} />
            <meshBasicMaterial color={ink} transparent opacity={0.25} />
          </mesh>
        ),
      )}
      {lines.map((pts, i) => (pts ? <Line key={i} points={pts} color={edge} lineWidth={2.5} /> : null))}
      <OrbitControls makeDefault enableDamping target={[0, (rounds * LAYER) / 2, 0]} />
    </LabeledCanvas>
  )
}

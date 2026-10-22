// The qubit's state on the Bloch sphere, with its path so far. |0> is the north pole.
// Bloch (x, y, z) maps to three.js (x, z, -y), since three.js points y up.
// The arrow can be shorter than 1: decoherence, or population that left for level 2.

import { Html, Line, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useMemo } from 'react'
import * as THREE from 'three'
import { HtmlWarmup } from './HtmlWarmup'

type Vec = [number, number, number]

export const toScene = (b: number[] | undefined): Vec => [b?.[0] ?? 0, b?.[2] ?? 0, -(b?.[1] ?? 0)]

function Circle({ axis, color }: { axis: 'x' | 'y' | 'z'; color: string }) {
  const pts = useMemo(() => {
    const out: Vec[] = []
    for (let i = 0; i <= 96; i++) {
      const a = (i / 96) * Math.PI * 2
      const c = Math.cos(a)
      const s = Math.sin(a)
      out.push(axis === 'y' ? [c, 0, s] : axis === 'x' ? [0, c, s] : [c, s, 0])
    }
    return out
  }, [axis])
  return <Line points={pts} color={color} lineWidth={1} transparent opacity={0.35} />
}

export function BlochScene({
  bloch,
  frame,
  theme,
}: {
  bloch: number[][]
  frame: number
  theme: 'light' | 'dark'
}) {
  const ink = theme === 'dark' ? '#d8d6cf' : '#30302c'
  const path = useMemo(() => bloch.slice(0, frame + 1).map(toScene), [bloch, frame])
  const tip = toScene(bloch[frame])
  const len = Math.hypot(...tip)
  const dir = len > 1e-6 ? new THREE.Vector3(...tip).normalize() : new THREE.Vector3(0, 1, 0)
  const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir)
  const accent = theme === 'dark' ? '#ff7a5c' : '#c4432f'
  return (
    <Canvas camera={{ position: [3.0, 1.7, 3.4], fov: 38 }} dpr={[1, 2]} gl={{ preserveDrawingBuffer: true }} data-testid="bloch-canvas">
      <HtmlWarmup />
      <color attach="background" args={[theme === 'dark' ? '#121214' : '#f6f5f2']} />
      <ambientLight intensity={0.9} />
      <directionalLight position={[3, 5, 4]} intensity={1.1} />
      <mesh>
        <sphereGeometry args={[1, 48, 32]} />
        <meshStandardMaterial color={theme === 'dark' ? '#3a3a44' : '#c9c7c0'} transparent opacity={0.16} depthWrite={false} />
      </mesh>
      <Circle axis="x" color={ink} />
      <Circle axis="y" color={ink} />
      <Circle axis="z" color={ink} />
      <Line points={[[-1.2, 0, 0], [1.2, 0, 0]]} color={ink} lineWidth={1} transparent opacity={0.5} />
      <Line points={[[0, -1.2, 0], [0, 1.2, 0]]} color={ink} lineWidth={1} transparent opacity={0.5} />
      <Line points={[[0, 0, -1.2], [0, 0, 1.2]]} color={ink} lineWidth={1} transparent opacity={0.5} />
      {path.length > 1 ? <Line points={path} color={accent} lineWidth={2} /> : null}
      {len > 1e-3 ? (
        <group quaternion={quat}>
          <mesh position={[0, len / 2, 0]}>
            <cylinderGeometry args={[0.018, 0.018, len, 12]} />
            <meshStandardMaterial color={accent} />
          </mesh>
          <mesh position={[0, len, 0]}>
            <sphereGeometry args={[0.05, 16, 12]} />
            <meshStandardMaterial color={accent} />
          </mesh>
        </group>
      ) : null}
      <Html position={[0, 1.38, 0]} center className="axis-label">|0⟩</Html>
      <Html position={[0, -1.38, 0]} center className="axis-label">|1⟩</Html>
      <Html position={[1.38, 0, 0]} center className="axis-label">x</Html>
      <Html position={[0, 0, -1.38]} center className="axis-label">y</Html>
      <OrbitControls makeDefault enableDamping enablePan={false} />
    </Canvas>
  )
}

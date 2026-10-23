// A Canvas with axis labels drawn as plain DOM over it. drei's <Html> mounts a separate React
// root per label, and React 19 unmounts those mid-render whenever a view closes ("Attempted to
// synchronously unmount a root"). Here the labels live in the page's own tree and one useFrame
// call moves them to where their 3D positions project.

import { Canvas, type CanvasProps, useFrame } from '@react-three/fiber'
import { type RefObject, useMemo, useRef } from 'react'
import * as THREE from 'three'

export interface Label {
  position: [number, number, number]
  text: string
}

function Track({ labels, els }: { labels: Label[]; els: RefObject<(HTMLDivElement | null)[]> }) {
  const v = useMemo(() => new THREE.Vector3(), [])
  useFrame(({ camera, size }) => {
    labels.forEach((label, i) => {
      const el = els.current[i]
      if (!el) return
      v.set(...label.position).project(camera)
      el.style.transform = `translate(${((v.x + 1) / 2) * size.width}px, ${((1 - v.y) / 2) * size.height}px) translate(-50%, -50%)`
      el.style.visibility = v.z < 1 ? 'visible' : 'hidden' // behind the camera
    })
  })
  return null
}

export function LabeledCanvas({ labels, children, ...props }: CanvasProps & { labels: Label[] }) {
  const els = useRef<(HTMLDivElement | null)[]>([])
  return (
    <div className="scene">
      <Canvas {...props}>
        {children}
        <Track labels={labels} els={els} />
      </Canvas>
      <div className="scene-labels" aria-hidden>
        {labels.map((label, i) => (
          <div
            key={i}
            ref={(el) => {
              els.current[i] = el
            }}
            className="axis-label"
          >
            {label.text}
          </div>
        ))}
      </div>
    </div>
  )
}

// Workaround: with React 19, drei's first <Html> in a new Canvas mounts before the canvas's
// event target is connected, gets unmounted mid-render ("Attempted to synchronously unmount a
// root") and never shows. Rendering one empty <Html> first absorbs that, so real labels show.

import { Html } from '@react-three/drei'

export function HtmlWarmup() {
  return <Html />
}

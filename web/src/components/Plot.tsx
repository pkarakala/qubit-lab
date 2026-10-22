// Plotly, styled from the theme tokens. Simulation is solid in a series color; theory is
// always a purple dashed line, so the two layers never look alike.

export const THEORY_DASH = 'dash'

import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'
import type { Theme } from '../store'

export const Plot = createPlotlyComponent(Plotly)

export const SERIES = {
  light: { a: '#c4432f', b: '#2c63c9', c: '#2f8a57', d: '#8a5cc4', theory: '#6b4fc4', ink: '#1a1a18', muted: '#85837c', grid: '#e3e1da' },
  dark: { a: '#ff7a5c', b: '#6fa0ff', c: '#5cc98b', d: '#b78cff', theory: '#b29cff', ink: '#ecebe6', muted: '#8f8d86', grid: '#2e2e31' },
}

export function layout(theme: Theme, extra: Record<string, unknown> = {}): Record<string, unknown> {
  const c = SERIES[theme]
  const axis = { gridcolor: c.grid, zerolinecolor: c.grid, linecolor: c.grid, tickcolor: c.muted }
  return {
    autosize: true,
    margin: { l: 56, r: 16, t: 10, b: 40 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: c.ink, size: 11, family: 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif' },
    legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 11 } },
    hoverlabel: { font: { size: 11 } },
    ...extra,
    xaxis: { ...axis, ...((extra.xaxis as object | undefined) ?? {}) },
    yaxis: { ...axis, ...((extra.yaxis as object | undefined) ?? {}) },
  }
}

export const CONFIG = { displayModeBar: false, responsive: true }

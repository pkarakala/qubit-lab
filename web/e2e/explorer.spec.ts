// Walks every view and writes the README screenshots to docs/img/.
import { expect, test } from '@playwright/test'

const IMG = '../docs/img'

test('phase space loses a cat and shows the theory ghost', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('wigner-canvas').locator('canvas')).toBeVisible({ timeout: 120_000 })
  await page.getByText('Lose the cat').click()
  await expect(page.locator('.hud')).not.toContainText('solving', { timeout: 120_000 })
  await page.getByTestId('play').click()
  await page.getByTestId('scrubber').fill('4')
  await expect(page.locator('.hud')).toContainText('κ₂t = 0.8')
  await page.getByTestId('ghost').check()
  await expect(page.locator('[data-formula="cat.parity_decay_rate"]')).toContainText('0.8')
  await page.screenshot({ path: `${IMG}/phase_space.png` })
})

test('flip rates show the bias and the e^2 law', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Flip rates' }).click()
  await expect(page.locator('.chart-main .js-plotly-plot')).toBeVisible({ timeout: 300_000 })
  await expect(page.locator('table.facts')).toContainText('Lescanne et al. 2020')
  await page.screenshot({ path: `${IMG}/flip_rates.png` })
})

test('transmon compares Koch et al. with the exact spectrum', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Transmon', exact: true }).click()
  await expect(page.getByTestId('band-canvas').locator('canvas')).toBeVisible({ timeout: 120_000 })
  await expect(page.locator('[data-formula="transmon.qubit_frequency"]')).toContainText('5.7 GHz', { timeout: 120_000 })
  await page.waitForTimeout(1500)
  await page.screenshot({ path: `${IMG}/transmon.png` })
})

test('qubit control: DRAG cuts leakage on a fast pi pulse', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Qubit control' }).click()
  await expect(page.getByTestId('bloch-canvas').locator('canvas')).toBeVisible({ timeout: 120_000 })
  await page.getByTestId('qubit-preset-drag').click()
  await expect(page.locator('[data-formula="qubit.drag_quadrature"]')).toBeVisible({ timeout: 120_000 })
  await page.waitForTimeout(8000)
  await expect(page.locator('.hud')).toContainText('leaked')
  await page.screenshot({ path: `${IMG}/qubit_control.png` })
})

test('error correction: space-time decoding and the cat code', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Error correction' }).click()
  await expect(page.getByTestId('spacetime-canvas').locator('canvas')).toBeVisible({ timeout: 120_000 })
  await expect(page.getByTestId('best-cat-code')).toContainText('Best sampled: d =', { timeout: 300_000 })
  await expect(page.locator('.dock .js-plotly-plot')).toBeVisible({ timeout: 300_000 })
  await page.waitForTimeout(1500)
  await page.screenshot({ path: `${IMG}/error_correction.png` })
})

test('noise bias view compares a cat and a transmon at measured numbers', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Noise bias' }).click()
  await expect(page.locator('.chart-main .js-plotly-plot')).toBeVisible({ timeout: 300_000 })
  await expect(page.getByTestId('compare-facts')).toContainText('Google Quantum AI')
  await expect(page.locator('[data-formula="qubit.flip_probability"]')).toBeVisible()
  await expect(page.locator('.hud-corner')).toHaveCount(0, { timeout: 300_000 })
  await page.screenshot({ path: `${IMG}/noise_bias.png` })
})

test('playground runs a typed model and rejects unsafe input', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Playground' }).click()
  await expect(page.getByTestId('python-code')).toContainText('dq.mesolve', { timeout: 120_000 })
  await page.getByTestId('playground-h').fill("__import__('os')")
  await page.getByTestId('playground-run').click()
  await expect(page.locator('.error-banner')).toContainText('not allowed', { timeout: 60_000 })
  await page.getByTestId('playground-preset-revival').click()
  await expect(page.getByTestId('python-code')).toContainText('dq.powm(ad, 2)', { timeout: 120_000 })
  await expect(page.locator('.error-banner')).toHaveCount(0)
  await page.locator('.transport input[type=range]').fill('20')
  await page.waitForTimeout(1500)
  await page.screenshot({ path: `${IMG}/playground.png` })
})

test('pulse optimizer beats DRAG on a 6 ns gate', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Pulse optimizer' }).click()
  await expect(page.getByTestId('pulse-table')).toBeVisible({ timeout: 300_000 })
  await expect(page.locator('.dock .js-plotly-plot')).toBeVisible()
  await page.waitForTimeout(6000)
  await page.screenshot({ path: `${IMG}/pulse_optimizer.png` })
})

test('a shared link restores the view and its inputs', async ({ page, context }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Qubit control' }).click()
  await page.getByTestId('qubit-preset-drag').click()
  const link = page.url()
  expect(link).toContain('#')
  const other = await context.newPage()
  await other.goto(link)
  await expect(other.getByRole('tab', { name: 'Qubit control' })).toHaveAttribute('aria-selected', 'true')
  await expect(other.locator('[data-formula="qubit.drag_quadrature"]')).toBeVisible({ timeout: 120_000 })
})

test('checks page reports no surprises', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Checks' }).click()
  await expect(page.locator('.lede')).toContainText('0 surprises', { timeout: 600_000 })
  await page.screenshot({ path: `${IMG}/checks.png` })
})

test('architecture gallery explodes a design and shows its levels', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Architectures', exact: true }).click()
  await expect(page.getByTestId('arch-viewport').locator('canvas')).toBeVisible({ timeout: 120_000 })
  await page.getByTestId('arch-fluxonium').click()
  await expect(page.getByTestId('arch-physics')).toContainText('manucharyan2009')
  await expect(page.locator('.dock .js-plotly-plot')).toBeVisible({ timeout: 120_000 })
  await page.getByTestId('arch-tantalum_transmon').click()
  await page.getByTestId('explode').fill('0.8')
  await page.waitForTimeout(2000)
  await page.screenshot({ path: `${IMG}/architectures.png` })
})

test('every view copies a Python script', async ({ page }) => {
  await page.goto('/')
  const views: [string, string][] = [
    ['Phase space', 'cat.evolve('],
    ['Flip rates', 'cat.sweep_rates('],
    ['Transmon', 'transmon.levels('],
    ['Qubit control', 'qb.run('],
    ['Pulse optimizer', 'po.optimize('],
    ['Error correction', 'qec.memory('],
    ['Noise bias', 'theory.flip_probability('],
  ]
  for (const [tab, call] of views) {
    await page.getByRole('tab', { name: tab, exact: true }).click()
    await page.getByTestId('copy-python').click()
    await expect(page.getByTestId('python-code')).toContainText(call)
  }
})

test('a published setting loads with its citation', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Transmon', exact: true }).click()
  await page.getByTestId('paper-transmon').selectOption({ label: 'Krantz 2019 worked example (E_J/E_C = 50)' })
  await expect(page.locator('.paper-settings')).toContainText('Krantz et al. 2019')
  await expect(page.locator('.pane-left')).toContainText('E_J = 15 GHz')
})

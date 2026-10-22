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
  await page.getByRole('tab', { name: 'Transmon' }).click()
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

test('cat vs transmon shows the noise bias at measured numbers', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Cat vs transmon' }).click()
  await expect(page.locator('.chart-main .js-plotly-plot')).toBeVisible({ timeout: 300_000 })
  await expect(page.getByTestId('compare-facts')).toContainText('Google Quantum AI')
  await expect(page.locator('[data-formula="qubit.flip_probability"]')).toBeVisible()
  await expect(page.locator('.hud-corner')).toHaveCount(0, { timeout: 300_000 })
  await page.screenshot({ path: `${IMG}/cat_vs_transmon.png` })
})

test('checks page reports no surprises', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('tab', { name: 'Checks' }).click()
  await expect(page.locator('.lede')).toContainText('0 surprises', { timeout: 600_000 })
  await page.screenshot({ path: `${IMG}/checks.png` })
})

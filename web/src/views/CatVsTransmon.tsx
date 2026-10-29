// Noise bias: a cat and a transmon at measured hardware numbers: chance of a bit-flip and a phase-flip per idle
// step, and the noise bias between them. The cat curve comes from the Liouvillian (simulation,
// rates in units of kappa_2); every device number is a cited fact.

import { useEffect, useState } from "react";
import {
  type CompareDevice,
  useCompare,
  useFacts,
  useRates,
  useReferences,
} from "../api";
import { CONFIG, Plot, SERIES, layout } from "../components/Plot";
import { CopyPython, ErrorBanner, FormulaCard, PaperSettings, shortCite, Slider, Solving } from "../components/ui";
import { fmt } from "../lib/colormap";
import { useExplorer } from "../store";
import { useShared } from "../lib/share";

const SIZES = [1, 2, 3, 4, 5, 6, 7];
const WILLOW_GATE_NS = 25; // willow2025, Supplement Sec. I; also a fact in qubitlab.facts

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return v;
}

/** Phase-flip over bit-flip probability. The API returns it; this only picks the device. */
const bias = (d: CompareDevice | undefined) => d?.theory.bias ?? null;

export function CatVsTransmon() {
  const theme = useExplorer((s) => s.theme);
  const c = SERIES[theme];
  const [logT, setLogT] = useShared("compare.log_t", Math.log10(WILLOW_GATE_NS));
  const [kappa1, setKappa1] = useShared("compare.kappa_1", 0.01);
  const tNs = Math.round(10 ** useDebounced(logT, 300) * 10) / 10;
  const cmp = useCompare(tNs);
  const rates = useRates(
    {
      alpha_sq: 4,
      kappa_1: useDebounced(kappa1, 500),
      kappa_phi: 0,
      n_th: 0,
      scheme: "dissipative",
    },
    SIZES,
  );
  const facts = useFacts().data ?? [];
  const refs = useReferences().data;
  const devices = cmp.data?.devices ?? [];
  const cats = devices.filter((d) => d.qubit === "cat");
  const transmons = devices.filter((d) => d.qubit === "transmon");
  const rows = rates.data?.rows ?? [];
  const factOf = (key: string) => facts.find((f) => f.key === key);
  const willow = devices.find((d) => d.key === "willow2025");
  const bigCat = devices.find((d) => d.key === "reglade2024.11.3");
  const xMax = 12;
  const transmonColors = [c.c, c.muted];

  return (
    <div className="view">
      <aside className="pane pane-left">
        <PaperSettings view="compare" onPick={(s) => setLogT(Math.log10(Number(s.t_ns)))} />
        <h3>Idle step</h3>
        <Slider
          label="idle time t"
          value={logT}
          min={1}
          max={4}
          step={0.02}
          onChange={setLogT}
          format={(v) => `${fmt(10 ** v, 3)} ns`}
          testId="idle-time"
        />
        <p className="small muted">
          Starts at {WILLOW_GATE_NS} ns, the single-qubit gate time of Acharya et al. 2025. Each
          qubit idles for t with the measured decay times below.
        </p>
        <h3>Cat model</h3>
        <Slider
          label="photon loss"
          tex="\kappa_1/\kappa_2"
          value={kappa1}
          min={0.001}
          max={0.1}
          step={0.001}
          onChange={setKappa1}
        />
        <p className="small muted">
          A model input for the simulated curve, not a hardware number. The
          measured cats are the points.
        </p>
        <h3>Measured on hardware</h3>
        <table className="facts" data-testid="compare-facts">
          <tbody>
            {devices.flatMap((d) =>
              [d.t_bit, d.t_phase].map((k) => {
                const f = factOf(k);
                if (!f) return null;
                return (
                  <tr key={k}>
                    <td>
                      {d.label}: {f.label}
                    </td>
                    <td className="mono">
                      {fmt(f.value)} {f.unit}
                    </td>
                    <td className="small">
                      <span className={`kind kind-${f.kind}`}>
                        {f.kind.replace("_", " ")}
                      </span>{" "}
                      {shortCite(refs?.[f.cite], f.cite)}, {f.locator}
                    </td>
                  </tr>
                );
              }),
            )}
          </tbody>
        </table>
        <CopyPython view="compare" inputs={{ t_ns: tNs }} />
      </aside>

      <main className="pane pane-center">
        <ErrorBanner error={cmp.error ?? rates.error} />
        <div className="chart-main">
          {cmp.data ? (
            <Plot
              data={[
                {
                  x: rows.map((r) => r.alpha_sq),
                  y: rows.map((r) =>
                    r.sim.bit_flip_rate && r.sim.phase_flip_rate !== null
                      ? r.sim.phase_flip_rate / r.sim.bit_flip_rate
                      : null,
                  ),
                  name: `cat, simulation (κ₁/κ₂ = ${fmt(kappa1)})`,
                  mode: "lines+markers",
                  line: { color: c.a, width: 2 },
                },
                {
                  x: cats.map((d) => factOf(d.t_bit)?.alpha_sq ?? null),
                  y: cats.map(bias),
                  name: "cat, measured (Réglade 2024)",
                  mode: "markers",
                  marker: { color: c.a, size: 11, symbol: "diamond" },
                },
                ...transmons.map((d, i) => ({
                  x: [0, xMax],
                  y: [bias(d), bias(d)],
                  name: `${d.label}, measured`,
                  mode: "lines",
                  line: { color: transmonColors[i], width: 2, dash: "dot" },
                })),
              ]}
              layout={layout(theme, {
                xaxis: {
                  title: {
                    text: "cat size |α|² (a transmon has no such knob)",
                  },
                  range: [0, xMax],
                },
                yaxis: {
                  title: { text: "noise bias: p_phase / p_bit" },
                  type: "log",
                  exponentformat: "power",
                },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: "100%", height: "100%" }}
            />
          ) : (
            <Solving boot what="Solving the transmon idle and the cat Liouvillians" />
          )}
          {cmp.isFetching || rates.isFetching ? (
            <div className="hud-corner">
              <Solving what={rates.isFetching && rates.data ? `${rates.data.rows.length} of ${rates.data.total} cat sizes` : undefined} />
            </div>
          ) : null}
        </div>
        <div className="dock">
          {cmp.data ? (
            <Plot
              data={[
                {
                  type: "bar",
                  x: devices.map((d) => d.label),
                  y: devices.map((d) => d.theory.p_bit),
                  name: "bit-flip chance per step",
                  marker: { color: c.a },
                },
                {
                  type: "bar",
                  x: devices.map((d) => d.label),
                  y: devices.map((d) => d.theory.p_phase),
                  name: "phase-flip chance per step",
                  marker: { color: c.b },
                },
              ]}
              layout={layout(theme, {
                barmode: "group",
                yaxis: {
                  title: { text: `chance in ${fmt(tNs)} ns` },
                  type: "log",
                  exponentformat: "power",
                },
              })}
              config={CONFIG}
              useResizeHandler
              style={{ width: "100%", height: "100%" }}
            />
          ) : null}
        </div>
      </main>

      <aside className="pane pane-right">
        <h3>Why bias matters</h3>
        <p className="small" data-testid="bias-summary">
          In {fmt(tNs)} ns the |α|² = 11.3 cat is {fmt(bias(bigCat), 2)} times
          more likely to phase-flip than to bit-flip. The Acharya 2025 transmon's
          ratio is {fmt(bias(willow), 2)}: both errors matter. A code that only
          has to fix one error type can be a line of qubits (Guillaud 2019). The
          price: the cat's phase-flips are more common than the transmon's.
        </p>
        <p className="small" data-testid="idle-error">
          Acharya 2025 error per idle step, 1 − F̄: theory{" "}
          {fmt(willow?.theory.idle_error)}, simulation{" "}
          {fmt(willow?.sim?.idle_error)}.
        </p>
        <p className="small muted">
          The simulated cat sits above the measured points: in the ideal model
          bit-flips fall by e² per photon, on hardware by about 4.2 (Réglade
          2024, Fig. 4b).
        </p>
        <FormulaCard
          formulaKey="qubit.flip_probability"
          theory={willow?.theory.p_phase}
          sim={willow?.sim?.p_phase}
          simLabel="Acharya 2025 phase-flip, 4-level transmon"
        />
        <FormulaCard formulaKey="qubit.idle_fidelity" />
        <FormulaCard formulaKey="cat.bit_flip_suppression" />
      </aside>
    </div>
  );
}

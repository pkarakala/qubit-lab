# Letting the simulation design the pulse

DRAG is a formula: a Gaussian plus a quadrature drive that cancels leakage to first order. A short pulse is past first order. So ask the simulation for the gradient.

**How it works.** Cut the pulse into 0.25 ns slices, each with its own Ωx and Ωy. Simulate the 4-level transmon and score the result with the gate error 1 − F, the average over every input state (Pedersen et al. 2007, Eq. 3). Population left in level 2 counts as error. Dynamiqs runs on JAX, so the gradient for every slice costs about one extra simulation. Step downhill from DRAG. This is GRAPE (Khaneja et al. 2005). No drive may exceed Ω_max.

**What it finds.** For a 6 ns X gate with α = −200 MHz:

| | DRAG | optimized |
| --- | --- | --- |
| gate error | 8.7% | below 10⁻¹⁴ |
| leakage | 0.36% | below 10⁻¹⁴ |
| with T1 = T2 = 50 µs | 8.7% | 7.4 × 10⁻⁵ |

Most of DRAG's error is not leakage. The pulse rotates the qubit by the wrong angle and phase, because level 2 shifts the qubit frequency during the pulse. The optimizer fixes both, leaving only T1 and T2. At 4 ns with Ω_max = 300 MHz it stalls at 1.6%: there isn't enough drive to rotate the qubit that fast without leaking.

**Limits.** The optimizer sees no noise. It has no bandwidth limit, but real lines round off edges. The model has one transmon and no neighbors.

**Wrong model.** A pulse fits the α and frequency it was made for. A 10 ns pulse run 1 MHz off resonance has error 1.5 × 10⁻⁴, growing as the mismatch squared. Averaging 1 − F over ±2 MHz (Kobzar et al. 2004) cuts a 20 ns pulse's worst error from 0.37% to 0.01%.

**Try it.** In the explorer's *Pulse optimizer* view, shorten the gate until the optimizer stalls, then raise Ω_max.

Sources: pedersen2007 (Eq. 3), khaneja2005, motzoi2009 (Eq. 9), kobzar2004 (Eq. 1).

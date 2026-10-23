"""Copy as Python: a short script that reproduces what an explorer view shows.

Each script uses qubitlab's own functions (`pip install qubitlab`), with the view's current
inputs filled in, and prints the simulation next to the theory it is checked against.
`tests/test_snippets.py` runs every one, so they cannot drift from the code.
"""

from __future__ import annotations

from typing import Any, Literal

View = Literal["phase", "rates", "transmon", "qubit", "pulse", "qec", "compare", "readout"]

HEADER = "# Qubit Lab: {title}. Reproduce it with `pip install qubitlab`.\n"

TEMPLATES: dict[View, tuple[str, str]] = {
    "phase": (
        "one cat qubit in phase space",
        """from qubitlab.sim import cat
from qubitlab.theory import cat as theory

params = cat.CatParams(
    alpha_sq={alpha_sq!r},
    kappa_1={kappa_1!r},
    n_th={n_th!r},
    kappa_phi={kappa_phi!r},
    scheme={scheme!r},
)
ev = cat.evolve(params, {initial!r}, t_final={t_final!r}, n_frames={n_frames!r}, grid={grid!r})

# ev.wigner[k] is the Wigner function at ev.times[k] on the grid ev.xvec x ev.xvec.
print("photon number at the end:", ev.photon_number[-1])
print("parity <X> at the end:   ", ev.parity[-1])
if params.scheme == "dissipative" and params.kappa_1 > 0:
    alpha = params.alpha_sq ** 0.5
    n_theory = theory.stabilized_photon_number(alpha=alpha, kappa_1=params.kappa_1, kappa_2=1.0)
    print("theory, photon number:   ", float(n_theory))
""",
    ),
    "rates": (
        "bit-flip and phase-flip rates of a cat",
        """from qubitlab.sim import cat
from qubitlab.theory import cat as theory

base = cat.CatParams(
    kappa_1={kappa_1!r}, n_th={n_th!r}, kappa_phi={kappa_phi!r}, scheme={scheme!r}
)
sizes = {alpha_sq_values!r}  # |alpha|^2; rates are in units of kappa_2
for n, r in zip(sizes, cat.sweep_rates(base, sizes)):
    alpha = n**0.5
    line = f"|alpha|^2 = {{n:4}}  bit-flip {{r.bit_flip_rate:.3e}}"
    line += f"  phase-flip {{r.phase_flip_rate:.3e}}"
    if base.kappa_1 > 0:
        th = float(theory.phase_flip_rate(kappa_1=base.kappa_1, alpha=alpha))
        line += f"  theory phase-flip {{th:.3e}}"
    print(line)
""",
    ),
    "transmon": (
        "a transmon's spectrum",
        """from qubitlab.sim import transmon
from qubitlab.theory import transmon as theory

params = transmon.TransmonParams(E_J={E_J!r}, E_C={E_C!r})
levels = transmon.levels(params)  # exact, from scqubits, GHz above the ground state
print("qubit frequency, exact:  ", levels[1] - levels[0])
print("qubit frequency, Koch:   ", float(theory.qubit_frequency(E_J=params.E_J, E_C=params.E_C)))
print("anharmonicity, exact:    ", levels[2] - 2 * levels[1] + levels[0])
print("anharmonicity, -E_C:     ", float(theory.anharmonicity(E_C=params.E_C)))
""",
    ),
    "qubit": (
        "a lab experiment on a transmon",
        """from qubitlab.sim import transmon_dynamics as qb

q = qb.TransmonQubit(
    alpha_ghz={alpha_ghz!r}, t1_ns={t1_ns!r}, t2_ns={t2_ns!r}, sigma_ghz={sigma_ghz!r}
)
traj = qb.run(
    {experiment!r},
    q,
    delta_ghz={delta_ghz!r},
    omega_ghz={omega_ghz!r},
    duration_ns={duration_ns!r},
    drag={drag!r},
    t_max_ns={t_max_ns!r},
    n_points={n_points!r},
)

# traj.bloch[k] is (<X>, <Y>, <Z>) at traj.times_ns[k]; traj.populations[k] the level populations.
print("final populations:", traj.populations[-1])
print("leaked above |1>: ", traj.leakage[-1])
""",
    ),
    "pulse": (
        "a pulse found by gradient descent",
        """from qubitlab.sim import pulse_optimizer as po
from qubitlab.sim import transmon_dynamics as qb

q = qb.TransmonQubit(alpha_ghz={alpha_ghz!r}, t1_ns={t1_ns!r}, t2_ns={t2_ns!r})
slices = round(4 * {duration_ns!r})  # 0.25 ns per slice
edges, ox, oy = po.drag_slices(q, {duration_ns!r}, slices)
print("DRAG gate error:     ", po.gate_error(q, edges, ox, oy)[0])

omega_max, steps = {omega_max_ghz!r}, {steps!r}
best = po.optimize(q, {duration_ns!r}, slices=slices, omega_max_ghz=omega_max, steps=steps)
print("optimized gate error:", best.error)
H = qb.driven_h(q, 0.0, best.edges_ns, best.omega_x, best.omega_y)
print("with T1 and T2:      ", 1 - qb.average_gate_fidelity(q, H, {duration_ns!r}, po.X_GATE))
""",
    ),
    "qec": (
        "error correction with Stim and PyMatching",
        """from qubitlab.sim import qec
from qubitlab.theory import qec as theory

for d in (3, 5, 7):
    m = qec.memory("surface", d, {p!r}, shots=100_000)
    fowler = float(theory.surface_code_logical_error(p={p!r}, p_th=0.0057, d=float(d)))
    print(f"surface d = {{d}}: P_L per round {{m.per_round:.2e}}, Fowler {{fowler:.2e}}")

# A repetition code of cats: corrects phase-flips, adds d bit-flip chances per round.
for d in (3, 5, 7, 9):
    m = qec.memory("repetition", d, {p_phase!r}, shots=100_000)
    floor = float(theory.repetition_code_bit_flip_floor(d=float(d), p_bit={p_bit!r}))
    print(f"cat repetition d = {{d}}: phase-flips {{m.per_round:.2e}}, bit-flips {{floor:.2e}}")
""",
    ),
    "compare": (
        "cat vs transmon at measured numbers",
        """from qubitlab.facts import DEVICES, FACT
from qubitlab.theory import qubit as theory

t_ns = {t_ns!r}  # idle step
for d in DEVICES:
    t_bit, t_phase = FACT[d.t_bit].value * 1e9, FACT[d.t_phase].value * 1e9  # facts are in s
    p_bit = float(theory.flip_probability(t=t_ns, T=t_bit))
    p_phase = float(theory.flip_probability(t=t_ns, T=t_phase))
    bias = p_phase / p_bit
    print(f"{{d.label:28}} bit-flip {{p_bit:.2e}}  phase-flip {{p_phase:.2e}}  bias {{bias:.3g}}")
""",
    ),
    "readout": (
        "dispersive readout of a transmon",
        """import math

from qubitlab.sim import readout
from qubitlab.theory import readout as theory

E_C = {E_C!r}  # GHz
d = readout.dressed_chi({ratio!r} * E_C, E_C, f_r={f_r!r}, g_n={g_n!r})
delta = d.f01 - {f_r!r}
koch = theory.chi_transmon(g=d.g01, g_12=d.g12, Delta=delta, alpha=d.alpha)
print(f"chi, dressed spectrum: {{d.chi * 1e3:.4f}} MHz")
print(f"chi, Koch Eqs. 27-28:  {{float(koch) * 1e3:.4f}} MHz")

kappa = 2 * math.pi * {kappa_mhz!r}  # rad/us
chi = 2 * math.pi * d.chi * 1e3
eps = math.sqrt({n_bar!r}) * kappa / 2
print("dephasing, Dynamiqs:   ", readout.dephasing_rate(chi, eps, kappa), "per us")
gamma = theory.dephasing_rate(kappa=kappa, chi=chi, epsilon=eps)
print("dephasing, Gambetta:   ", float(gamma), "per us")
""",
    ),
}


def script(view: View, inputs: dict[str, Any]) -> str:
    """The script for `view`, with `inputs` (already validated by the API) filled in."""
    title, body = TEMPLATES[view]
    return HEADER.format(title=title) + "\n" + body.format(**inputs)

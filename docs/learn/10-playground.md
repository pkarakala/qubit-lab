# The playground

Every other view runs a fixed model. Here you write your own.

**One oscillator mode.** You give a Hamiltonian H and a list of losses. Each loss is a rate and an operator L, and adds rate × D[L] to the Lindblad equation. Dynamiqs solves it and draws the Wigner function over time.

**How to write it.** Use a, ad (a†), n (a†a) and I, your own parameters, and numbers (1j for i). Products of operators are matrix products, so a*ad is not ad*a. A number added to an operator means the number times I. Powers must be whole numbers: a**2.

**Examples.**

| Model | H | Losses |
| --- | --- | --- |
| Two-photon cat | 0 | 1 × D[a² − α²], κ₁ × D[a] |
| Kerr cat | −K a†²a² + ε₂(a†² + a²) | none |
| Driven cavity | ε(a + a†) | κ × D[a] |

The first is the model behind the cat views (Mirrahimi et al. 2014). The second is the Kerr cat of Grimm et al. 2020, whose cat size is α² = ε₂/K.

**Is it right?** A test runs the two-photon preset through the playground and through the built-in cat solver and gets the same Wigner function to 10⁻¹⁰. Two more tests match hand calcs: photon loss from a coherent state, and the driven cavity's steady state, ⟨n⟩ = (2ε/κ)².

**Watch the truncation.** The mode keeps N Fock states. If the state reaches the top ones, the answer is wrong, and the view says so.

**Take it with you.** *Copy as Python* gives a short Dynamiqs script that runs the same model.

**Safety.** What you type is parsed against a whitelist, never run as code.

Sources: dynamiqs, mirrahimi2014 (Sec. 2), grimm2020 (Eq. (1)).

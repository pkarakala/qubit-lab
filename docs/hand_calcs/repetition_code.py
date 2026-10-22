"""Hand calc: what a repetition code does for a biased qubit such as a cat.

A distance-d repetition code uses d data qubits and d - 1 ancillas: 2d - 1 qubits in all.
It corrects only one error type. For a cat, that is the phase-flip, the common error.

1. Corrected errors. Majority vote fails when at least (d + 1)/2 of the d data qubits flip
   (d odd). For small p the leading term is C(d, (d+1)/2) p^((d+1)/2), so log P_L against
   log p has slope (d + 1)/2. The same counting gives the surface code's d_e (fowler2012,
   Eq. (10)).
2. Uncorrected errors. A bit-flip on any one data qubit flips the logical state, and the code
   cannot see it. To first order, P_L,bit = d * p_bit per round. So the code only helps while
   d * p_bit stays below the corrected error: that is why it needs a strongly biased qubit.

Run this file to print the leading coefficient and check the union bound against the exact
probability of at least one flip.
"""

import sympy as sp

p, pb = sp.symbols("p p_bit", positive=True)
for d in (3, 5, 7):
    k = (d + 1) // 2
    exact = sum(sp.binomial(d, j) * p**j * (1 - p) ** (d - j) for j in range(k, d + 1))
    lead = sp.series(exact, p, 0, k + 1).removeO()
    print(f"d = {d}: P_L = {sp.expand(lead)} + ...  (qubits: {2 * d - 1})")
    at_least_one = 1 - (1 - pb) ** d
    first_order = sp.series(at_least_one, pb, 0, 2).removeO()
    print(f"        bit-flip: 1-(1-p_bit)^d = {first_order} + O(p_bit^2)")

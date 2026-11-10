# Error correction: why cats need fewer qubits

One physical qubit is never good enough. Error correction spreads one logical qubit over many physical ones, so that a few errors can be spotted and undone.

**Distance.** A code of distance d survives up to (d − 1)/2 errors. Below a threshold error rate, adding qubits helps: the logical error falls as p^((d+1)/2). Above it, more qubits only add more ways to fail. Our Stim simulation finds that exponent to within 5%.

**Surface code.** A transmon flips its bit and its phase about equally often, so it needs a code that catches both. The surface code does, with a 2D grid. In Stim's layout a distance-d patch uses 2d² − 1 qubits: 49 for d = 5.

**Space-time picture.** Each round, measure qubits check their neighbors. A changed result is a detection event. One error lights up a pair of events; the decoder (PyMatching) guesses which errors made which pairs. The 3D view stacks the rounds so you can see the pairs.

**Repetition code.** A cat's bit-flips are rare, so a line of cats only needs to catch phase-flips: 2d − 1 qubits for distance d, so 9 cats for d = 5 against 49 transmons for a d = 5 surface code. The catch is that the bit-flips it ignores grow as d × p_X. So there is a best size, and it grows with the noise bias η = p_Z / p_X.

**Try it.** In *Error correction*, raise p past about 1% and watch the decoder fail. Then lower η and watch the best cat code shrink.

Sources: fowler2012 (Eqs. 10 and 11, Sec. VII.A), qubitlab_hand_calcs (repetition_code.py).

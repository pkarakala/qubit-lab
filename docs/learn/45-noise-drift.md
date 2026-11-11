# Noise drift: when the decoder's model goes stale

A decoder is only as good as its noise model. A lab measures the noise, builds the decoder, and runs. Then the chip changes.

**What moves.** A two-level-system defect can drift onto a qubit's frequency and cut its T1 by up to ten times. These defects move over minutes to hours. High-energy impacts are faster and rarer: on one chip they came about every 10 s and raised errors across it for about 25 ms. Labs recalibrate between runs to keep up.

**Why it costs.** The decoder weighs each possible error by how likely it thinks it is. When one qubit gets worse, errors near it are likelier than the decoder believes, so it pairs detection events the wrong way more often.

**Finding it.** The detection events give the bad qubit away: the measure qubits around it fire more often. This needs no logical labels, so it can run while the code runs.

**Fixing it.** Refit the decoder from recent detection events. A sliding-window estimate of how often pairs of detectors fire together tracks drift that is slower than its window. A refit wins back part of the loss. The bad qubit still makes more errors, so only recalibrating it wins back all of it.

**Try it.** In *Error correction*, drag *one qubit gets worse by* to 10x. Watch its four neighbours light up, then compare the old decoder with one that knows the new noise.

Sources: klimov2018 (Fig. 2, Supplement S8), mcewen2022, willow2025 (Sec. III), spitz2018.

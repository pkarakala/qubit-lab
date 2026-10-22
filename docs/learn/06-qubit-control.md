# Controlling a transmon

A qubit is only useful if you can rotate it and it remembers where it was. Two numbers judge both.

**Rotating it.** A microwave drive at the qubit frequency turns |0⟩ into |1⟩ and back: a Rabi oscillation. Detune the drive by Δ and the rotation tilts, so it never fully reaches |1⟩. A π pulse is a short, shaped burst that does exactly half a turn.

**Leakage.** A transmon is not a perfect two-level system. Its 1→2 transition sits only |α| ≈ 200 to 300 MHz below the 0→1 drive frequency. A pulse that is too short has frequency content that reaches it. In our 4-level simulation, a 6 ns Gaussian π pulse with α = −200 MHz ends with 24% of the population in level 2.

**DRAG.** Add a second drive, 90° out of phase and proportional to the slope of the first. That cancels the leakage path to first order. The same 6 ns pulse now leaks 0.4%. It still misses |1⟩ by 13%: the pulse is too fast for first-order DRAG, and level 2 also shifts the qubit frequency. Real labs tune the DRAG coefficient, the detuning and the amplitude together.

**Forgetting.** T1 is how long |1⟩ survives before it decays to |0⟩. T2 is how long a superposition keeps its phase, and 1/T2 = 1/(2T1) + 1/T_φ, so T2 is at most 2T1. Slow frequency noise makes Ramsey fringes die faster, so Ramsey measures T2*, which is shorter than T2. A Hahn echo flips the qubit halfway and cancels that slow noise; it measures T2E, closer to T2.

**One finding.** The two-level Rabi formula misses by 1.8% for a 10 MHz drive detuned by 10 MHz. Level 2 shifts the resonance by about Ω²/(2|α|). The checks page shows it.

**Try it.** In the explorer's *Qubit control* view, run the lessons in order, then drag the DRAG slider.

Sources: motzoi2009 (Eq. 9), nielsen2002 (Eq. 3), qubitlab_hand_calcs (qubit_decoherence.py).

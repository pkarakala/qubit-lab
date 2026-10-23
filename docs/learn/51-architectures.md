# Inside the chip

Every design in this lab is a few metal shapes on a crystal. The shapes set the Hamiltonian.

**The layers.** A substrate of sapphire or silicon, a few hundred micrometres thick. A superconducting film on top, a few hundred nanometres: aluminium, niobium or tantalum. The film is patterned into a ground plane, capacitor pads, resonators and control lines. Josephson junctions, two aluminium layers with a nanometre of oxide between them, are the only nonlinear parts.

**What each shape does.** Pad area sets the charging energy E_C. A junction's area sets E_J. A loop of junctions lets flux tune E_J. A resonator reads the qubit, and a flux or drive line controls it.

**The designs in the gallery.**

- *Xmon transmon* (Barends et al. 2013): a cross to ground and a SQUID, so its frequency is tunable.
- *Tantalum transmon* (Place et al. 2021): the same idea in tantalum on sapphire. A cleaner surface oxide pushed T1 past 0.3 ms.
- *Tunable coupler* (Yan et al. 2018): a third transmon between two qubits that turns their coupling off or on.
- *Fluxonium* (Manucharyan et al. 2009): one small junction shunted by a chain of 43 large ones. It is low in frequency and protected at half flux.
- *Dissipative cat* (Réglade et al. 2024): a memory resonator, a lossy buffer, and an ATS that exchanges photon pairs.
- *Kerr cat* (Grimm et al. 2020): a SNAIL in a 3D cavity, squeezed into a double well.

**Honest sizes.** Click a part to see its real size and where it comes from. *Measured* means a paper states it, with the figure or table. *Schematic* means no paper gives it, and the view says why. Parts too small to see are drawn larger, and flagged.

**Try it.** In *Architectures*, explode the fluxonium and compare its level plot with the Xmon's.

Sources: barends2013 (Table I), place2021 (Supplement S1), yan2018, manucharyan2009, reglade2024 (Supplement S1), grimm2020 (Supplement VI).

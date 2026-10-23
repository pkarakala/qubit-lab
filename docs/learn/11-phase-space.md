# States and the Wigner function

Every qubit in this lab is built from oscillators, so start with how to picture an oscillator's state.

**Phase space.** A resonator has two quadratures, x and p, like position and momentum. We write a point as β = x + ip. A classical oscillator is one point circling the origin. A quantum state is spread out.

**Fock and coherent states.** The Fock state |n⟩ holds exactly n photons. A coherent state |α⟩ is what a classical drive makes: a blob centered at β = α, holding |α|² photons on average.

**The Wigner function.** W(x, p) is the closest thing to a probability density over phase space. It integrates to 1, and its shadows on the x and p axes are the measured distributions of x and p. A coherent state is a Gaussian bump. In our convention its peak height is 2/π.

**Negative values.** Unlike a probability, W can dip below zero. That only happens for non-classical states. A cat state, |α⟩ + |−α⟩, shows two blobs plus fringes between them that swing negative. At the origin W equals 2/π times the photon-number parity, so an even cat sits at +2/π and an odd cat at −2/π.

**Why it matters here.** A cat qubit's 0 and 1 are the two blobs. Its phase lives in the fringes. Watching W shows at a glance which kind of error a noise process causes.

**Try it.** In *Phase space*, pick *Even cat* and raise |α|². Then switch to *Odd cat* and watch the center flip sign.

Sources: gerry2005 (Ch. 3, 7), checked in `cat.wigner` and `cat.photon_number`.

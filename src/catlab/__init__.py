"""catlab: cat qubit and transmon physics, theory vs simulation."""

import jax

# Physics checks compare numbers to 1e-6 and below, so everything runs in double precision.
jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

__version__ = "0.1.0"

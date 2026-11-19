"""animalbench: animal-inspired multi-agent benchmark. JAX physics core is float64-capable."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU only (project rule: never GPU)
import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

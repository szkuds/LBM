from functools import partial

import jax.numpy as jnp
from jax import jit

from main import LBM


class BoundaryCondition(LBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.indices = kwargs.get("indices")
        self.name = None
        self.isSolid = False
        self.implementationStep = None

    @partial(jit, static_argnums=0, inline=True)
    def prepare(self, f_out, f_in):
        """Prepare the boundary condition attributes."""
        return f_out

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, value):
        if not isinstance(value, (list, tuple, jnp.ndarray)):
            raise ValueError("indices must be a list, tuple, or jnp.ndarray.")
        if isinstance(value, jnp.ndarray) and value.dtype != bool:
            raise ValueError("indices must contain boolean values (True for boundary indices, False for bulk).")
        self._indices = jnp.array(value, dtype=bool)


class BounceBackBoundaryCondition(BoundaryCondition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Bounce Back"
        self.isSolid = True
        self.implementationStep = "Post Streaming"

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def apply_bc(self, f):
        def bounce_back_boundary_conditions(f_i):
            # Bounce-back top wall
            f_i = f_i.at[:, -1, 7].set(f_i[:, -1, 5])
            f_i = f_i.at[:, -1, 4].set(f_i[:, -1, 2])
            f_i = f_i.at[:, -1, 8].set(f_i[:, -1, 6])
            # Bounce-back bottom wall
            f_i = f_i.at[:, 0, 6].set(f_i[:, 0, 8])
            f_i = f_i.at[:, 0, 2].set(f_i[:, 0, 4])
            f_i = f_i.at[:, 0, 5].set(f_i[:, 0, 7])
            return f_i

        return bounce_back_boundary_conditions(f)

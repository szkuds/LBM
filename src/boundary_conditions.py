from functools import partial

import jax.numpy as jnp
from jax import jit


class BoundaryCondition:
    def __init__(self, lattice, nx, ny, nz, indices):
        self.lattice = lattice
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.d = lattice.d
        self.indices = indices
        self.name = None
        self.isSolid = False
        self.implementationStep = None

    @partial(jit, static_argnums=0, inline=True)
    def prepare(self, f_out, f_in):
        """Prepare the boundary condition attributes."""
        return f_out

    @jit
    def apply(self, f):
        """Apply the boundary condition."""
        return f


class BounceBackBoundaryCondition(BoundaryCondition):
    def __init__(self, lattice, nx, ny, nz, indices):
        super().__init__(lattice, nx, ny, nz, indices)
        self.name = "Bounce Back"
        self.isSolid = True
        self.implementationStep = "Post Streaming"

    @jit
    def apply(self, f):
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
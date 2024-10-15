from functools import partial

import jax.numpy as jnp
from jax import jit

from main import LBM


class BGK(LBM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        rho, u = self.macro_vars(f)
        f_eq = self.equilibrium(rho, u)
        f_neq = f - f_eq
        f_post_col = f - 1/self.tau * f_neq
        if self.calculate_force is not None:
            f_post_col = self.apply_force(f_post_col, f_eq, rho, u)
        return f_post_col
